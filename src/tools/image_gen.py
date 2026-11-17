import base64
import logging
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


async def generate_image(prompt: str, width: int = 1024, height: int = 1024, seed: int = 0, steps: int = 50) -> str:
    invoke_url = config.IMAGE_GEN_URL
    headers = {
        "Authorization": f"Bearer {config.NVIDIA_API_KEY}",
        "Accept": "application/json",
    }
    payload = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "seed": seed,
        "steps": steps,
    }
    response = requests.post(invoke_url, headers=headers, json=payload)
    response.raise_for_status()
    artifact = response.json()["artifacts"][0]
    if artifact.get("finishReason") == "CONTENT_FILTERED":
        raise RuntimeError("Generation blocked by content filter")
    return artifact["base64"]


@dataclass
class GeneratedImage:
    """A rendered image and where it landed in the sandbox."""
    png: bytes
    # None when storing it in the sandbox failed.
    path: str | None


async def _generate_image(ctx: AgentContext, prompt: str,
                          resolution: str) -> GeneratedImage | str:
    """Shared business logic: render the image and store it in the sandbox.

    Returns the bytes plus their sandbox path, or an error string fit for
    the model. Sending the image to the chat is the caller's job.
    """
    try:
        width, height = parse_resolution(resolution)
    except ValueError as e:
        logger.warning("Invalid image resolution requested: %s", e)
        return str(e)
    try:
        b64 = await generate_image(prompt, width=width, height=height)
        image_bytes = base64.b64decode(b64)
        path, error = await manager.store_file(
            ctx.user_id, image_bytes, f"generated_{int(time.time())}.png")
        if error is not None:
            logger.error("Error storing generated image in sandbox: %s",
                         error)
            return GeneratedImage(png=image_bytes, path=None)
        return GeneratedImage(png=image_bytes, path=path)
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
        image=ImageAttachment(data=image.png, mime="image/png"))


_RESOLUTION_PARAM = {
    "type": "string",
    "description": ("Image dimensions as WIDTHxHEIGHT, e.g. '1024x1024'. "
                    f"Width and height must each be a multiple of {DIM_STEP} "
                    f"between {MIN_DIM} and {MAX_DIM}, e.g. '1024x1024', "
                    "'1568x672', '768x1344', '1344x768'"),
}

_ADD_TO_CONTEXT_PARAM = {
    "type": "boolean",
    "description": ("Whether to attach the generated image to this conversation "
                    "so you can see it. Defaults to false. Requires a "
                    "vision-capable model."),
}


@agent_tool(
    name="generate_image",
    description=("Generate an image from a text prompt. The image is sent to "
                 "the user in the chat and stored in the sandbox at "
                 "/workspace/uploads, so it can be processed with sandbox "
                 "tools. Set send_to_chat to false to generate an image "
                 "without showing it to the user. Set add_to_context to true "
                 "to attach the generated image to this conversation so you "
                 "can look at it (requires a vision-capable model)."),
    parameters={
        "prompt": {
            "type": "string",
            "description": "A text description of the image to generate",
        },
        "resolution": _RESOLUTION_PARAM,
        "send_to_chat": {
            "type": "boolean",
            "description": ("Whether to send the generated image to the user "
                            "in the chat. Defaults to true. Set to false to "
                            "generate the image privately: it is still stored "
                            "in the sandbox, but not shown to the user."),
        },
        "add_to_context": _ADD_TO_CONTEXT_PARAM,
    })
async def generate_and_send_image(
        ctx: ChatUpdateContext, prompt: str,
        resolution: str = DEFAULT_RESOLUTION,
        send_to_chat: bool = True,
        add_to_context: bool = False) -> str | ToolCallResult:
    result = await _generate_image(ctx, prompt, resolution)
    if isinstance(result, str):
        return result
    if send_to_chat:
        await ctx.messaging.send_image(result.png)
    return _as_result(_describe(result, prompt, sent=send_to_chat),
                      result, add_to_context)


@agent_tool(
    name="generate_image",
    scope=SUBAGENT,
    description=("Generate an image from a text prompt and store it in the "
                 "sandbox at /workspace/uploads, where it can be processed "
                 "with the sandbox tools. The image is never shown to the "
                 "user: report its path to the parent assistant, which can "
                 "share it. Set add_to_context to true to attach the image to "
                 "this conversation so you can look at it (requires a "
                 "vision-capable model)."),
    parameters={
        "prompt": {
            "type": "string",
            "description": "A text description of the image to generate",
        },
        "resolution": _RESOLUTION_PARAM,
        "add_to_context": _ADD_TO_CONTEXT_PARAM,
    })
async def generate_image_subagent(
        ctx: AgentContext, prompt: str,
        resolution: str = DEFAULT_RESOLUTION,
        add_to_context: bool = False) -> str | ToolCallResult:
    result = await _generate_image(ctx, prompt, resolution)
    if isinstance(result, str):
        return result
    return _as_result(_describe(result, prompt, sent=False),
                      result, add_to_context)
