"""Image generation tool: backend renderers in ``PROVIDERS`` return raw image
bytes through the shared async client, under one overall deadline.
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
MAX_ERROR_CHARS = 2000

# Overall deadline for one generation: the request plus any image download.
IMAGE_GENERATION_DEADLINE = 300.0

PROVIDER_NVIDIA_FLUX3 = "nvidia-flux3"
PROVIDER_NVIDIA_FLUX2 = "nvidia-flux2"
PROVIDER_PHOENIX = "phoenix"
PROVIDER_RECRAFT = "recraft"
DEFAULT_PROVIDER = PROVIDER_RECRAFT

NVIDIA_REQUEST_TIMEOUT = 180.0
CLOUDFLARE_REQUEST_TIMEOUT = 120.0
OPENROUTER_REQUEST_TIMEOUT = 120.0

VALID_ASPECT_RATIOS = ("1:1", "2:1", "1:2", "3:2", "2:3", "4:3", "3:4",
                       "5:4", "4:5", "16:9", "9:16", "auto")

# OpenRouter's images endpoint accepts n in 1-6. The tool delivers a single
# image, so it always asks for the maximum and keeps the first result.
OPENROUTER_N = 6


def parse_resolution(resolution: str) -> tuple[int, int]:
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


@dataclass(frozen=True)
class _Provider:
    name: str
    requires: tuple[tuple[str, str], ...]
    default_steps: int
    steps_range: tuple[int, int]
    # Renders one image to raw bytes. Width, height and steps are positional;
    # ``aspect_ratio`` is a keyword used only by the OpenRouter backend.
    render: Callable[..., Awaitable[bytes]]


async def _render_nvidia(url: str, prompt: str, width: int, height: int,
                         steps: int) -> bytes:
    """Render via an NVIDIA FLUX NIM endpoint.

    FLUX.3-Klein and FLUX.2-Klein share one JSON contract, so one renderer
    serves both. The endpoints reject a ``seed`` field (HTTP 422
    ``extra_forbidden``), so none is sent.
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
                     steps: int, aspect_ratio: str | None = None) -> bytes:
        return await _render_nvidia(getattr(config, url_attr), prompt,
                                    width, height, steps)

    return _Provider(
        name=name,
        requires=(("NVIDIA_API_KEY", "NVIDIA API key"),),
        default_steps=default_steps,
        steps_range=(1, 50),
        render=render)


async def _render_cloudflare(prompt: str, width: int, height: int,
                             steps: int, aspect_ratio: str | None = None) -> bytes:
    """Render via Cloudflare Workers AI.

    Phoenix returns raw JPEG; other Workers AI image models return
    ``{"result": {"image": "<base64>"}}``. Both are normalised to raw bytes.
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


async def _render_openrouter(prompt: str, width: int = 1024, height: int = 1024,
                             steps: int = 1,
                             aspect_ratio: str | None = None) -> bytes:
    """Render via the OpenRouter images API.

    Only ``aspect_ratio`` reaches the endpoint; width, height and steps are
    ignored, so ``generate_image`` rejects a ``resolution`` here.
    """
    aspect_ratio = aspect_ratio or "auto"
    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.OPENROUTER_IMAGE_MODEL,
        "prompt": prompt,
        "response_format": "b64_json",
        "aspect_ratio": aspect_ratio,
        "n": OPENROUTER_N,
    }
    response = await http_client.post(
        "https://openrouter.ai/api/v1/images",
        headers=headers, json_body=payload,
        timeout=OPENROUTER_REQUEST_TIMEOUT)
    response.raise_for_status()
    images = response.json().get("data", [])
    if not images:
        raise RuntimeError("OpenRouter returned no images.")
    first = images[0]
    if not first.get("b64_json"):
        raise RuntimeError("OpenRouter returned no b64_json image data.")
    return base64.b64decode(first["b64_json"])


PROVIDERS: dict[str, _Provider] = {
    PROVIDER_NVIDIA_FLUX3: _nvidia_provider(
        PROVIDER_NVIDIA_FLUX3, "IMAGE_GEN_URL", default_steps=50),
    PROVIDER_NVIDIA_FLUX2: _nvidia_provider(
        PROVIDER_NVIDIA_FLUX2, "NVIDIA_FLUX2_GEN_URL", default_steps=4),
    PROVIDER_PHOENIX: _Provider(
        name=PROVIDER_PHOENIX,
        requires=(("CLOUDFLARE_ACCOUNT_ID", "Cloudflare account ID"),
                  ("CLOUDFLARE_API_TOKEN", "Cloudflare API token")),
        default_steps=25,
        steps_range=(1, 50),
        render=_render_cloudflare),
    PROVIDER_RECRAFT: _Provider(
        name=PROVIDER_RECRAFT,
        requires=(("OPENROUTER_API_KEY", "OpenRouter API key"),),
        default_steps=1,
        steps_range=(1, 1),
        render=_render_openrouter),
}
# Backends in schema/enum order (the tool schemas and tests rely on it).
IMAGE_PROVIDERS = tuple(PROVIDERS)


async def generate_image(prompt: str, width: int = 1024, height: int = 1024,
                         steps: int | None = None,
                         provider: str = DEFAULT_PROVIDER,
                         aspect_ratio: str | None = None) -> bytes:
    """Render an image with the selected backend, returning the raw bytes.

    ``steps`` defaults per backend and is validated against the backend's
    accepted range; Recraft takes no step count, so any value but its single
    accepted one is rejected. The whole generation is bounded by
    IMAGE_GENERATION_DEADLINE seconds.
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
            return await spec.render(prompt, width, height, resolved_steps,
                                     aspect_ratio=aspect_ratio)
    except TimeoutError:
        raise RuntimeError(
            "image generation timed out after "
            f"{IMAGE_GENERATION_DEADLINE:g}s") from None


@dataclass
class GeneratedImage:
    data: bytes
    path: str | None
    mime: str


def _sniff_format(data: bytes) -> tuple[str, str]:
    """Guess (extension, mime) from magic bytes.

    Unknown containers error rather than default to PNG: a mislabeled image
    is unopenable in the sandbox under the wrong extension.
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
    """Format a generation failure for the model, capped at MAX_ERROR_CHARS.

    The endpoint's own error body (a content-policy rejection, a bad
    parameter) is the useful part, so it is passed back.
    """
    if isinstance(e, http_client.HttpStatusError):
        detail = e.body.strip() or f"HTTP {e.status} from {e.url}"
    else:
        detail = str(e).strip() or type(e).__name__
    return detail[:MAX_ERROR_CHARS]


async def _generate_image(ctx: AgentContext, prompt: str,
                          resolution: str | None,
                          provider: str = DEFAULT_PROVIDER,
                          aspect_ratio: str | None = None) -> GeneratedImage | str:
    """Render the image and store it in the sandbox.

    Returns the bytes plus their sandbox path, or an error string fit for
    the model; sending it to chat is the caller's job. ``resolution`` and
    ``aspect_ratio`` are both optional because the backends disagree on which
    they accept, and passing the wrong one is reported to the model.
    """
    if provider not in PROVIDERS:
        return (f"Unknown image provider '{provider}'. "
                f"Supported providers: {', '.join(IMAGE_PROVIDERS)}")
    if provider == PROVIDER_RECRAFT:
        if resolution is not None:
            return ("OpenRouter does not support the 'resolution' "
                    "parameter; use 'aspect_ratio' instead.")
        resolved_aspect = aspect_ratio or "auto"
        if resolved_aspect not in VALID_ASPECT_RATIOS:
            return (f"Invalid aspect_ratio '{resolved_aspect}'. "
                    f"Must be one of: {', '.join(VALID_ASPECT_RATIOS)}.")
        image_kwargs: dict = {'provider': provider,
                              'aspect_ratio': resolved_aspect}
    else:
        try:
            width, height = parse_resolution(resolution or DEFAULT_RESOLUTION)
        except ValueError as e:
            logger.warning("Invalid image resolution requested: %s", e)
            return str(e)
        image_kwargs = {'width': width, 'height': height,
                        'provider': provider}
    try:
        data = await generate_image(prompt, **image_kwargs)
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
                    f"between {MIN_DIM} and {MAX_DIM}. Not accepted by "
                    "'recraft', which uses aspect_ratio instead."),
}

_ASPECT_RATIO_PARAM = {
    "type": "string",
    "description": ("Aspect ratio for the image: '1:1', '2:1', '1:2', '3:2', "
                    "'2:3', '4:3', '3:4', '5:4', '4:5', '16:9', '9:16' or "
                    "'auto'. Only used by the 'recraft' provider."),
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
    "description": ("Image backend: 'nvidia-flux3', "
                    "'nvidia-flux2' (fast few-step), 'phoenix' (Leonardo "
                    "Phoenix 1.0 via Cloudflare) or 'recraft' (OpenRouter "
                    "images API)."),
}


@agent_tool(
    name="generate_image",
    description=("Generate an image from a text prompt, sent to the user in "
                 "chat and stored in /workspace/uploads for sandbox "
                 "processing. Set send_to_chat to false to keep it private, "
                 "add_to_context to true to view it yourself (needs vision). "
                 "Content filtering is by word list; avoid sexual, political, "
                 "religious words."
                 f"Set provider to override the default '{DEFAULT_PROVIDER}'."),
    parameters={
        "prompt": {
            "type": "string",
            "description": "Text description of image to generate",
        },
        "resolution": _RESOLUTION_PARAM,
        "aspect_ratio": _ASPECT_RATIO_PARAM,
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
        resolution: str | None = None,
        provider: str = DEFAULT_PROVIDER,
        send_to_chat: bool = True,
        add_to_context: bool = False,
        aspect_ratio: str | None = None) -> str | ToolCallResult:
    result = await _generate_image(ctx, prompt, resolution, provider,
                                   aspect_ratio=aspect_ratio)
    if isinstance(result, str):
        return result
    if send_to_chat:
        if result.mime in ("image/png", "image/jpeg"):
            await ctx.messaging.send_image(result.data)
        else:
            # sendPhoto only reliably takes PNG/JPEG; anything else the
            # backend returns (e.g. WEBP) goes as a document.
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
                 "vision). Set provider to 'nvidia-flux2', 'phoenix' or "
                 "'recraft' to override the default 'nvidia-flux3'."),
    parameters={
        "prompt": {
            "type": "string",
            "description": "Text description of image to generate",
        },
        "resolution": _RESOLUTION_PARAM,
        "aspect_ratio": _ASPECT_RATIO_PARAM,
        "provider": _PROVIDER_PARAM,
        "add_to_context": _ADD_TO_CONTEXT_PARAM,
    })
async def generate_image_subagent(
        ctx: AgentContext, prompt: str,
        resolution: str | None = None,
        provider: str = DEFAULT_PROVIDER,
        add_to_context: bool = False,
        aspect_ratio: str | None = None) -> str | ToolCallResult:
    result = await _generate_image(ctx, prompt, resolution, provider,
                                   aspect_ratio=aspect_ratio)
    if isinstance(result, str):
        return result
    return _as_result(_describe(result, prompt, sent=False),
                      result, add_to_context)
