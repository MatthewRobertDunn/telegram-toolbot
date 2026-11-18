import base64
import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass

import requests
import config
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

# Image backends selectable via the tools' ``provider`` parameter.
PROVIDER_NVIDIA_FLUX3 = "nvidia-flux3"
PROVIDER_NVIDIA_FLUX2 = "nvidia-flux2"
PROVIDER_HORDE = "horde"
PROVIDER_CLOUDFLARE = "cloudflare"
IMAGE_PROVIDERS = (PROVIDER_NVIDIA_FLUX3, PROVIDER_NVIDIA_FLUX2,
                   PROVIDER_HORDE, PROVIDER_CLOUDFLARE)
DEFAULT_PROVIDER = PROVIDER_NVIDIA_FLUX3

# AI Horde async polling behaviour.
HORDE_POLL_INTERVAL = 3.0
HORDE_TIMEOUT = 300.0
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


async def _generate_nvidia_flux3(prompt: str, width: int = 1024,
                             height: int = 1024, steps: int = 50) -> str:
    invoke_url = config.IMAGE_GEN_URL
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
    response = requests.post(invoke_url, headers=headers, json=payload)
    response.raise_for_status()
    artifact = response.json()["artifacts"][0]
    if artifact.get("finishReason") == "CONTENT_FILTERED":
        raise RuntimeError("Generation blocked by content filter")
    return artifact["base64"]


async def _generate_nvidia_flux2(prompt: str, width: int = 1024,
                                 height: int = 1024, steps: int = 4) -> str:
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
    response = requests.post(config.NVIDIA_FLUX2_GEN_URL, headers=headers,
                             json=payload, timeout=180)
    response.raise_for_status()
    artifact = response.json()["artifacts"][0]
    if artifact.get("finishReason") == "CONTENT_FILTERED":
        raise RuntimeError("Generation blocked by content filter")
    return artifact["base64"]


def _horde_headers() -> dict[str, str]:
    return {
        "apikey": config.AI_HORDE_KEY,
        "Client-Agent": HORDE_CLIENT_AGENT,
        "Content-Type": "application/json",
    }


def _horde_base_url() -> str:
    return config.AI_HORDE_API_URL.rstrip("/")


async def _generate_horde(prompt: str, width: int = 1024, height: int = 1024,
                          seed: int = 0, steps: int = 25) -> str:
    """Render via the AI Horde crowdsourced backend (async submit + poll).

    Defaults to 25 steps: larger requests need the account to already hold
    the kudos upfront (HTTP 403 ``KudosUpfront`` otherwise).
    """
    if not config.AI_HORDE_KEY:
        raise RuntimeError(
            "AI Horde API key is not configured "
            "(set AI_HORDE_KEY)")
    base_url = _horde_base_url()
    headers = _horde_headers()
    params: dict[str, object] = {
        "width": width,
        "height": height,
        "steps": steps,
        "cfg_scale": 7,
        "sampler_name": "k_euler",
        "n": 1,
    }
    if seed:
        params["seed"] = str(seed)
    payload = {
        "prompt": prompt,
        "params": params,
        "nsfw": False,
        "models": [],
        "r2": False,
    }
    response = requests.post(f"{base_url}/generate/async", headers=headers,
                             json=payload, timeout=30)
    response.raise_for_status()
    request_id = response.json()["id"]

    start = time.monotonic()
    while True:
        if time.monotonic() - start > HORDE_TIMEOUT:
            raise TimeoutError("AI Horde image generation timed out.")
        status_resp = requests.get(f"{base_url}/generate/check/{request_id}",
                                   headers=headers, timeout=30)
        status_resp.raise_for_status()
        status = status_resp.json()
        if status.get("finished", 0) >= 1:
            break
        if status.get("faulted"):
            raise RuntimeError("AI Horde reported a generation fault.")
        await asyncio.sleep(HORDE_POLL_INTERVAL)

    result_resp = requests.get(f"{base_url}/generate/status/{request_id}",
                               headers=headers, timeout=30)
    result_resp.raise_for_status()
    generations = result_resp.json().get("generations", [])
    if not generations:
        raise RuntimeError("AI Horde returned no images.")
    image_data = generations[0]["img"]
    if image_data.startswith("data:"):
        image_data = image_data.split(",", 1)[1]
    if image_data.startswith("http://") or image_data.startswith("https://"):
        image_resp = requests.get(image_data, timeout=60)
        image_resp.raise_for_status()
        return base64.b64encode(image_resp.content).decode()
    # Validate it decodes before returning it to the shared store path.
    base64.b64decode(image_data)
    return image_data


async def _generate_cloudflare(prompt: str, width: int = 1024, height: int = 1024,
                               seed: int = 0, steps: int = 25,
                               guidance: float | None = None,
                               negative_prompt: str | None = None) -> str:
    """Render via Cloudflare Workers AI (default: Leonardo Phoenix 1.0).

    Phoenix params: ``prompt`` (required), ``guidance`` (2-10, default 2),
    ``seed`` (>= 0), ``width``/``height`` (0-2048), ``num_steps`` (1-50,
    default 25), ``negative_prompt``. Phoenix returns raw JPEG bytes
    (``image/jpeg``); other Workers AI image models return
    ``{"result": {"image": "<base64>"}}``. Both shapes are accepted and
    normalised to a base64 string.
    """
    if not config.CLOUDFLARE_ACCOUNT_ID:
        raise RuntimeError(
            "Cloudflare account ID is not configured "
            "(set CLOUDFLARE_ACCOUNT_ID)")
    if not config.CLOUDFLARE_API_TOKEN:
        raise RuntimeError(
            "Cloudflare API token is not configured "
            "(set CLOUDFLARE_API_TOKEN)")
    if not 1 <= steps <= 50:
        raise ValueError(f"Invalid num_steps {steps}: must be 1-50")
    if guidance is not None and not 2 <= guidance <= 10:
        raise ValueError(f"Invalid guidance {guidance}: must be 2-10")
    url = (f"https://api.cloudflare.com/client/v4/accounts/"
           f"{config.CLOUDFLARE_ACCOUNT_ID}/ai/run/"
           f"{config.CLOUDFLARE_IMAGE_MODEL}")
    headers = {
        "Authorization": f"Bearer {config.CLOUDFLARE_API_TOKEN}",
        "Content-Type": "application/json",
    }
    payload: dict[str, object] = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_steps": steps,
    }
    if seed:
        payload["seed"] = seed
    if guidance is not None:
        payload["guidance"] = guidance
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    if "application/json" in content_type:
        result = response.json().get("result", {})
        image_data = result.get("image") if isinstance(result, dict) else None
        if not image_data:
            raise RuntimeError(
                "Cloudflare Workers AI returned no image.")
        # Validate it decodes before returning it to the shared store path.
        base64.b64decode(image_data)
        return image_data
    return base64.b64encode(response.content).decode()


async def generate_image(prompt: str, width: int = 1024, height: int = 1024,
                         seed: int = 0, steps: int | None = None,
                         provider: str = DEFAULT_PROVIDER,
                         guidance: float | None = None,
                         negative_prompt: str | None = None) -> str:
    """Render an image with the selected backend, returning base64 PNG data.

    ``steps`` defaults per backend (25 for Cloudflare Phoenix and AI Horde,
    where larger requests need upfront kudos, 50 for NVIDIA FLUX-3, 4 for
    NVIDIA FLUX.2-Klein). ``guidance`` (2-10) and ``negative_prompt`` are
    Phoenix-only and omitted for the other backends; ``seed`` is ignored
    for the NVIDIA backends (their endpoints reject it).
    """
    if provider == PROVIDER_NVIDIA_FLUX2:
        return await _generate_nvidia_flux2(
            prompt, width=width, height=height,
            steps=4 if steps is None else steps)
    if provider == PROVIDER_CLOUDFLARE:
        return await _generate_cloudflare(
            prompt, width=width, height=height, seed=seed,
            steps=25 if steps is None else steps,
            guidance=guidance, negative_prompt=negative_prompt)
    if provider == PROVIDER_NVIDIA_FLUX3:
        return await _generate_nvidia_flux3(prompt, width=width, height=height,
                                            steps=50 if steps is None
                                            else steps)
    if provider == PROVIDER_HORDE:
        return await _generate_horde(prompt, width=width, height=height,
                                     seed=seed,
                                     steps=25 if steps is None else steps)
    raise ValueError(
        f"Unknown image provider '{provider}'. "
        f"Supported providers: {', '.join(IMAGE_PROVIDERS)}")


@dataclass
class GeneratedImage:
    """A rendered image and where it landed in the sandbox."""
    png: bytes
    # None when storing it in the sandbox failed.
    path: str | None
    # Sniffed from the bytes: NVIDIA returns PNG, Cloudflare JPEG,
    # AI Horde WEBP.
    mime: str = "image/png"


def _sniff_format(data: bytes) -> tuple[str, str]:
    """Guess (extension, mime) from magic bytes; Cloudflare is JPEG,
    NVIDIA PNG, Horde WEBP."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png", "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg", "image/jpeg"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "webp", "image/webp"
    return "png", "image/png"


async def _generate_image(ctx: AgentContext, prompt: str,
                          resolution: str,
                          provider: str = DEFAULT_PROVIDER) -> GeneratedImage | str:
    """Shared business logic: render the image and store it in the sandbox.

    Returns the bytes plus their sandbox path, or an error string fit for
    the model. Sending the image to the chat is the caller's job.
    """
    if provider not in IMAGE_PROVIDERS:
        return (f"Unknown image provider '{provider}'. "
                f"Supported providers: {', '.join(IMAGE_PROVIDERS)}")
    try:
        width, height = parse_resolution(resolution)
    except ValueError as e:
        logger.warning("Invalid image resolution requested: %s", e)
        return str(e)
    try:
        b64 = await generate_image(prompt, width=width, height=height,
                                   provider=provider)
        image_bytes = base64.b64decode(b64)
        ext, mime = _sniff_format(image_bytes)
        path, error = await manager.store_file(
            ctx.user_id, image_bytes, f"generated_{int(time.time())}.{ext}")
        if error is not None:
            logger.error("Error storing generated image in sandbox: %s",
                         error)
            return GeneratedImage(png=image_bytes, path=None, mime=mime)
        return GeneratedImage(png=image_bytes, path=path, mime=mime)
    except requests.HTTPError as e:
        # The endpoint's error body (e.g. a content-policy rejection or a bad
        # parameter) is the useful part for the model, so pass it back.
        logger.exception("Error generating image: %s", e)
        detail = e.response.text if e.response is not None else str(e)
        return f"Failed to generate image: {detail.strip()[:MAX_ERROR_CHARS]}"
    except Exception as e:
        logger.exception("Error generating image: %s", e)
        return f"Failed to generate image: {e}"


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
        image=ImageAttachment(data=image.png, mime=image.mime))


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
            await ctx.messaging.send_image(result.png)
        else:
            # sendPhoto only reliably takes PNG/JPEG; anything else the
            # backend returns (e.g. Horde's WEBP) goes as a document.
            if result.path is not None:
                filename = os.path.basename(result.path)
            else:
                filename = f"generated.{result.mime.split('/')[-1]}"
            await ctx.messaging.send_document(result.png, filename)
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
