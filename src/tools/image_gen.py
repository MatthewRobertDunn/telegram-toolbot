"""Image generation tool: render images via pluggable backends.

Each backend is described by a :class:`_Provider` record in ``PROVIDERS``:
it owns its credentials, defaults, step validation, request format and
response parsing, and returns raw image bytes. All network I/O goes through
the shared async client in :mod:`http_client`, so a generation never blocks
the event loop, and every generation is bounded by an overall deadline.
"""

import asyncio
import base64
import logging
import os
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import config
import http_client
from chat_update_context import AgentContext, ChatUpdateContext
from sandbox import manager
from tool_registry import SUBAGENT, agent_tool
from tool_result import ImageAttachment, ToolCallResult

logger = logging.getLogger(__name__)

MIN_DIM = 512
MAX_DIM = 1568
DIM_STEP = 16
DEFAULT_RESOLUTION = "1024x1024"
# Cap on the endpoint error body echoed back to the model as the tool result.
MAX_ERROR_CHARS = 2000

# Overall deadline for one image generation (submit, polling and any image
# download together), so no single generation can run unbounded. It
# dominates the per-request budgets below.
IMAGE_GENERATION_DEADLINE = 300.0

# Image backends selectable via the tools' ``provider`` parameter.
PROVIDER_NVIDIA_FLUX3 = "nvidia-flux3"
PROVIDER_NVIDIA_FLUX2 = "nvidia-flux2"
PROVIDER_HORDE = "horde"
PROVIDER_CLOUDFLARE = "cloudflare"
DEFAULT_PROVIDER = PROVIDER_NVIDIA_FLUX3

# Per-request budgets for the backends' individual calls; every one is also
# covered by IMAGE_GENERATION_DEADLINE.
NVIDIA_REQUEST_TIMEOUT = 180.0
HORDE_REQUEST_TIMEOUT = 30.0
HORDE_IMAGE_URL_TIMEOUT = 60.0
CLOUDFLARE_REQUEST_TIMEOUT = 120.0

# AI Horde async polling behaviour.
HORDE_POLL_INTERVAL = 3.0
HORDE_CLIENT_AGENT = "telegram-toolbot/1.0"


def parse_resolution(resolution: str) -> tuple[int, int]:
    """Parse a 'WIDTHxHEIGHT' string into validated dimensions."""
    parts = re.split(r"[x×*]", resolution.strip().lower().replace(" ", ""))
    if len(parts) != 2:
        raise ValueError(
            f"Invalid resolution '{resolution}'. "
            "Use WIDTHxHEIGHT, e.g. '1024x1024', '1568x672', '768x1344'")
    try:
        width, height = int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(
            f"Invalid resolution '{resolution}'. "
            "Use WIDTHxHEIGHT, e.g. '1024x1024', '1568x672', '768x1344'") from None
    for name, dim in (("width", width), ("height", height)):
        if not (MIN_DIM <= dim <= MAX_DIM) or dim % DIM_STEP:
            raise ValueError(
                f"Invalid resolution {width}x{height}: the image endpoint only "
                f"accepts width and height values that are multiples of "
                f"{DIM_STEP} between {MIN_DIM} and {MAX_DIM} "
                f"(got {name}={dim})")
    return width, height


# ---------------------------------------------------------------------------
# Provider layer
#
# Every backend renders a validated request to raw image bytes; its URL,
# headers, payload and response parsing live in its render function, so
# ``generate_image`` only resolves the provider, applies the shared
# validation and routes.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Provider:
    """One image backend's contract with :func:`generate_image`."""

    name: str
    # (config attribute, human-readable name) pairs that must be set to use
    # this backend; all of them are checked, so a backend never
    # half-configures.
    requires: tuple[tuple[str, str], ...]
    # Default and accepted (inclusive) range for the step count.
    default_steps: int
    steps_range: tuple[int, int]
    # Renders one image; returns the raw bytes in whatever container the
    # backend produces (PNG, JPEG or WEBP).
    render: Callable[[str, int, int, int], Awaitable[bytes]]


async def _render_nvidia(url: str, prompt: str, width: int, height: int,
                         steps: int) -> bytes:
    """Render via an NVIDIA FLUX NIM endpoint.

    FLUX.3-Klein and FLUX.2-Klein expose the identical JSON contract and
    differ only in URL and default step count, so one renderer serves both.
    The endpoints reject a ``seed`` field (HTTP 422 ``extra_forbidden``),
    so none is ever sent.
    """
    headers = {
        "Authorization": f"Bearer {config.NVIDIA_API_KEY}",
        "Accept": "application/json",
    }
    payload = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "steps": steps,
    }
    response = await http_client.post(url, headers=headers, json_body=payload,
                                      timeout=NVIDIA_REQUEST_TIMEOUT)
    response.raise_for_status()
    artifact = response.json()["artifacts"][0]
    if artifact.get("finishReason") == "CONTENT_FILTERED":
        raise RuntimeError("Generation blocked by content filter")
    return base64.b64decode(artifact["base64"])


def _nvidia_provider(name: str, url_attr: str, default_steps: int) -> _Provider:
    """Build the provider record for one FLUX NIM endpoint.

    The URL is read from ``config`` at render time, so the environment can
    retarget an endpoint without a code change.
    """

    async def render(prompt: str, width: int, height: int,
                     steps: int) -> bytes:
        return await _render_nvidia(getattr(config, url_attr), prompt,
                                    width, height, steps)

    return _Provider(
        name=name,
        requires=(("NVIDIA_API_KEY", "NVIDIA API key"),),
        default_steps=default_steps,
        steps_range=(1, 50),
        render=render)


async def _render_horde(prompt: str, width: int, height: int,
                        steps: int) -> bytes:
    """Render via the AI Horde crowdsourced backend (async submit + poll).

    Defaults to 25 steps: larger requests need the account to already hold
    the kudos upfront (HTTP 403 ``KudosUpfront`` otherwise). Every wait is
    an ``await``, so polling shares the loop with everything else; the
    overall render is bounded by IMAGE_GENERATION_DEADLINE.
    """
    base_url = config.AI_HORDE_API_URL.rstrip("/")
    headers = {
        "apikey": config.AI_HORDE_KEY,
        "Client-Agent": HORDE_CLIENT_AGENT,
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": prompt,
        "params": {
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": 7,
            "sampler_name": "k_euler",
            "n": 1,
        },
        "nsfw": False,
        "models": [],
        "r2": False,
    }
    response = await http_client.post(f"{base_url}/generate/async",
                                      headers=headers, json_body=payload,
                                      timeout=HORDE_REQUEST_TIMEOUT)
    response.raise_for_status()
    request_id = response.json()["id"]

    while True:
        status_resp = await http_client.get(
            f"{base_url}/generate/check/{request_id}",
            headers=headers, timeout=HORDE_REQUEST_TIMEOUT)
        status_resp.raise_for_status()
        status = status_resp.json()
        if status.get("finished", 0) >= 1:
            break
        if status.get("faulted"):
            raise RuntimeError("AI Horde reported a generation fault.")
        await asyncio.sleep(HORDE_POLL_INTERVAL)

    result_resp = await http_client.get(
        f"{base_url}/generate/status/{request_id}",
        headers=headers, timeout=HORDE_REQUEST_TIMEOUT)
    result_resp.raise_for_status()
    generations = result_resp.json().get("generations", [])
    if not generations:
        raise RuntimeError("AI Horde returned no images.")
    image = generations[0]["img"]
    if image.startswith(("http://", "https://")):
        image_resp = await http_client.get(
            image, timeout=HORDE_IMAGE_URL_TIMEOUT)
        image_resp.raise_for_status()
        return image_resp.content
    if image.startswith("data:"):
        # A data URI without a comma separator is malformed; report that
        # instead of raising a bare IndexError.
        _, sep, encoded = image.partition(",")
        if not sep:
            raise RuntimeError(
                "AI Horde returned a malformed image data URI.")
        image = encoded
    # Decoding here both strips the base64 wrapper and proves the payload
    # really is image data before it reaches the shared store path.
    return base64.b64decode(image)


async def _render_cloudflare(prompt: str, width: int, height: int,
                             steps: int) -> bytes:
    """Render via Cloudflare Workers AI (default: Leonardo Phoenix 1.0).

    Phoenix returns raw JPEG bytes (``image/jpeg``); other Workers AI image
    models return ``{"result": {"image": "<base64>"}}``. Both shapes are
    accepted and normalised to raw bytes.
    """
    url = (f"https://api.cloudflare.com/client/v4/accounts/"
           f"{config.CLOUDFLARE_ACCOUNT_ID}/ai/run/"
           f"{config.CLOUDFLARE_IMAGE_MODEL}")
    headers = {
        "Authorization": f"Bearer {config.CLOUDFLARE_API_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_steps": steps,
    }
    response = await http_client.post(url, headers=headers, json_body=payload,
                                      timeout=CLOUDFLARE_REQUEST_TIMEOUT)
    response.raise_for_status()
    if "application/json" in response.header("Content-Type"):
        result = response.json().get("result", {})
        image = result.get("image") if isinstance(result, dict) else None
        if not image:
            raise RuntimeError(
                "Cloudflare Workers AI returned no image.")
        return base64.b64decode(image)
    return response.content


PROVIDERS: dict[str, _Provider] = {
    PROVIDER_NVIDIA_FLUX3: _nvidia_provider(
        PROVIDER_NVIDIA_FLUX3, "IMAGE_GEN_URL", default_steps=50),
    PROVIDER_NVIDIA_FLUX2: _nvidia_provider(
        PROVIDER_NVIDIA_FLUX2, "NVIDIA_FLUX2_GEN_URL", default_steps=4),
    PROVIDER_HORDE: _Provider(
        name=PROVIDER_HORDE,
        requires=(("AI_HORDE_KEY", "AI Horde API key"),),
        default_steps=25,
        steps_range=(1, 50),
        render=_render_horde),
    PROVIDER_CLOUDFLARE: _Provider(
        name=PROVIDER_CLOUDFLARE,
        requires=(("CLOUDFLARE_ACCOUNT_ID", "Cloudflare account ID"),
                  ("CLOUDFLARE_API_TOKEN", "Cloudflare API token")),
        default_steps=25,
        steps_range=(1, 50),
        render=_render_cloudflare),
}
# Backends in schema/enum order (the tool schemas and tests rely on it).
IMAGE_PROVIDERS = tuple(PROVIDERS)


async def generate_image(prompt: str, width: int = 1024, height: int = 1024,
                         steps: int | None = None,
                         provider: str = DEFAULT_PROVIDER) -> bytes:
    """Render an image with the selected backend, returning the raw bytes.

    ``steps`` defaults per backend (25 for Cloudflare Phoenix and AI Horde,
    where larger requests need upfront kudos, 50 for NVIDIA FLUX.3-Klein, 4
    for NVIDIA FLUX.2-Klein) and is validated against the backend's
    accepted range. The whole generation — submit, polling and any image
    download — is bounded by IMAGE_GENERATION_DEADLINE seconds.
    """
    try:
        spec = PROVIDERS[provider]
    except KeyError:
        raise ValueError(
            f"Unknown image provider '{provider}'. "
            f"Supported providers: {', '.join(IMAGE_PROVIDERS)}") from None
    for attr, label in spec.requires:
        if not getattr(config, attr):
            raise RuntimeError(f"{label} is not configured (set {attr})")
    resolved_steps = spec.default_steps if steps is None else steps
    low, high = spec.steps_range
    if not low <= resolved_steps <= high:
        raise ValueError(
            f"Invalid steps {resolved_steps} for provider '{provider}': "
            f"must be {low}-{high}")
    try:
        async with asyncio.timeout(IMAGE_GENERATION_DEADLINE):
            return await spec.render(prompt, width, height, resolved_steps)
    except TimeoutError:
        raise RuntimeError(
            "image generation timed out after "
            f"{IMAGE_GENERATION_DEADLINE:g}s") from None


@dataclass
class GeneratedImage:
    """A rendered image and where it landed in the sandbox."""
    # Raw image bytes (PNG, JPEG or WEBP; see _sniff_format).
    data: bytes
    # None when storing it in the sandbox failed.
    path: str | None
    # Sniffed from the bytes: NVIDIA returns PNG, Cloudflare JPEG,
    # AI Horde WEBP.
    mime: str


def _sniff_format(data: bytes) -> tuple[str, str]:
    """Guess (extension, mime) from magic bytes; Cloudflare is JPEG,
    NVIDIA PNG, Horde WEBP.

    Unknown containers are an error rather than a guessed PNG: a mislabeled
    image would be undeliverable (sendPhoto only takes PNG/JPEG) and
    unopenable in the sandbox under the wrong extension.
    """
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png", "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg", "image/jpeg"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "webp", "image/webp"
    raise ValueError(
        "Image backend returned an unsupported image format "
        f"({len(data)} bytes, unrecognized container)")


def _error_text(e: BaseException) -> str:
    """The single truncating formatter for every generation failure.

    The endpoint's own error body (e.g. a content-policy rejection or a bad
    parameter) is the useful part for the model, so it is passed back;
    everything is capped at MAX_ERROR_CHARS so a chatty endpoint cannot
    flood the model's context.
    """
    if isinstance(e, http_client.HttpStatusError):
        detail = e.body.strip() or f"HTTP {e.status} from {e.url}"
    else:
        detail = str(e).strip() or type(e).__name__
    return detail[:MAX_ERROR_CHARS]


async def _generate_image(ctx: AgentContext, prompt: str,
                          resolution: str,
                          provider: str = DEFAULT_PROVIDER) -> GeneratedImage | str:
    """Shared business logic: render the image and store it in the sandbox.

    Returns the bytes plus their sandbox path, or an error string fit for
    the model. Sending the image to the chat is the caller's job.
    """
    if provider not in PROVIDERS:
        return (f"Unknown image provider '{provider}'. "
                f"Supported providers: {', '.join(IMAGE_PROVIDERS)}")
    try:
        width, height = parse_resolution(resolution)
    except ValueError as e:
        logger.warning("Invalid image resolution requested: %s", e)
        return str(e)
    try:
        data = await generate_image(prompt, width=width, height=height,
                                    provider=provider)
        ext, mime = _sniff_format(data)
        path, error = await manager.store_file(
            ctx.user_id, data, f"generated_{int(time.time())}.{ext}")
        if error is not None:
            logger.error("Error storing generated image in sandbox: %s",
                         error)
            return GeneratedImage(data=data, path=None, mime=mime)
        return GeneratedImage(data=data, path=path, mime=mime)
    except Exception as e:
        logger.exception("Error generating image: %s", e)
        return f"Failed to generate image: {_error_text(e)}"


def _describe(image: GeneratedImage, prompt: str, sent: bool) -> str:
    """The tool result the model reads, worded for what actually happened."""
    if image.path is None:
        delivered = "sent to chat" if sent else "generated"
        return f"Image {delivered} (prompt: {prompt})"
    delivered = "sent to chat and " if sent else ""
    return (f"Image {delivered}stored in the sandbox at {image.path} "
            f"(prompt: {prompt})")


def _as_result(text: str, image: GeneratedImage,
               add_to_context: bool) -> str | ToolCallResult:
    if not add_to_context:
        return text
    return ToolCallResult(
        text=text,
        image=ImageAttachment(data=image.data, mime=image.mime))


_RESOLUTION_PARAM = {
    "type": "string",
    "description": ("Image size as WIDTHxHEIGHT, e.g. '1024x1024'. "
                    f"Each dimension must be a multiple of {DIM_STEP} "
                    f"between {MIN_DIM} and {MAX_DIM}."),
}

_ADD_TO_CONTEXT_PARAM = {
    "type": "boolean",
    "description": ("Attach generated image to this conversation so you can "
                    "see it (requires vision-capable model). Defaults to "
                    "false."),
}

_PROVIDER_PARAM = {
    "type": "string",
    "enum": list(IMAGE_PROVIDERS),
    "description": ("Image backend: 'nvidia-flux3' (default), "
                    "'nvidia-flux2' (fast few-step), 'cloudflare' (Leonardo "
                    "Phoenix 1.0) or 'horde' (crowdsourced)."),
}


@agent_tool(
    name="generate_image",
    description=("Generate an image from a text prompt, sent to the user in "
                 "chat and stored in /workspace/uploads for sandbox "
                 "processing. Set send_to_chat to false to keep it private, "
                 "add_to_context to true to view it yourself (needs vision). "
                 "Content filtering is by word list."
                 "You may need to avoid sexual, polistical, religious words."
                 "Set provider to 'nvidia-flux2', 'cloudflare' or 'horde' "
                 "to override the default 'nvidia-flux3'."),
    parameters={
        "prompt": {
            "type": "string",
            "description": "Text description of image to generate",
        },
        "resolution": _RESOLUTION_PARAM,
        "provider": _PROVIDER_PARAM,
        "send_to_chat": {
            "type": "boolean",
            "description": ("Send image to user in chat (default true). Set "
                            "false to keep private; still stored in sandbox."),
        },
        "add_to_context": _ADD_TO_CONTEXT_PARAM,
    })
async def generate_and_send_image(
        ctx: ChatUpdateContext, prompt: str,
        resolution: str = DEFAULT_RESOLUTION,
        provider: str = DEFAULT_PROVIDER,
        send_to_chat: bool = True,
        add_to_context: bool = False) -> str | ToolCallResult:
    result = await _generate_image(ctx, prompt, resolution, provider)
    if isinstance(result, str):
        return result
    if send_to_chat:
        if result.mime in ("image/png", "image/jpeg"):
            await ctx.messaging.send_image(result.data)
        else:
            # sendPhoto only reliably takes PNG/JPEG; anything else the
            # backend returns (e.g. Horde's WEBP) goes as a document.
            if result.path is not None:
                filename = os.path.basename(result.path)
            else:
                filename = f"generated.{result.mime.split('/')[-1]}"
            await ctx.messaging.send_document(result.data, filename)
    return _as_result(_describe(result, prompt, sent=send_to_chat),
                      result, add_to_context)


@agent_tool(
    name="generate_image",
    scope=SUBAGENT,
    description=("Generate an image from a text prompt, stored in "
                 "/workspace/uploads for sandbox processing. Never shown to "
                 "the user: report its path to the parent assistant to share. "
                 "Set add_to_context to true to view it yourself (needs "
                 "vision). Set provider to 'nvidia-flux2', "
                 "'cloudflare' or 'horde' to override the default "
                 "'nvidia-flux3'."),
    parameters={
        "prompt": {
            "type": "string",
            "description": "Text description of image to generate",
        },
        "resolution": _RESOLUTION_PARAM,
        "provider": _PROVIDER_PARAM,
        "add_to_context": _ADD_TO_CONTEXT_PARAM,
    })
async def generate_image_subagent(
        ctx: AgentContext, prompt: str,
        resolution: str = DEFAULT_RESOLUTION,
        provider: str = DEFAULT_PROVIDER,
        add_to_context: bool = False) -> str | ToolCallResult:
    result = await _generate_image(ctx, prompt, resolution, provider)
    if isinstance(result, str):
        return result
    return _as_result(_describe(result, prompt, sent=False),
                      result, add_to_context)
