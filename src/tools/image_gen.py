import base64
import logging
import re
import time

import requests
import config
from chat_update_context import context_tool, ChatUpdateContext
from messaging import send_image
from sandbox import manager

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


@context_tool
async def generate_and_send_image(ctx: ChatUpdateContext, prompt: str,
                                  resolution: str = DEFAULT_RESOLUTION,
                                  send_to_chat: bool = True) -> str:
    try:
        width, height = parse_resolution(resolution)
    except ValueError as e:
        logger.warning("Invalid image resolution requested: %s", e)
        return str(e)
    try:
        b64 = await generate_image(prompt, width=width, height=height)
        image_bytes = base64.b64decode(b64)
        if send_to_chat:
            await send_image(ctx, image_bytes)
        path, error = await manager.store_file(
            ctx.user_id, image_bytes, f"generated_{int(time.time())}.png")
        if error is not None:
            logger.error("Error storing generated image in sandbox: %s",
                         error)
            delivered = "sent to chat" if send_to_chat else "generated"
            return f"Image {delivered} (prompt: {prompt})"
        delivered = "sent to chat and " if send_to_chat else ""
        return (f"Image {delivered}stored in the sandbox at {path} "
                f"(prompt: {prompt})")
    except requests.HTTPError as e:
        # The endpoint's error body (e.g. a content-policy rejection or a bad
        # parameter) is the useful part for the model, so pass it back.
        logger.exception("Error generating image: %s", e)
        detail = e.response.text if e.response is not None else str(e)
        return f"Failed to generate image: {detail.strip()[:MAX_ERROR_CHARS]}"
    except Exception as e:
        logger.exception("Error generating image: %s", e)
        return f"Failed to generate image: {e}"

TOOL_NAME = "generate_image"    
TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "strict": False,
            "description": "Generate an image from a text prompt. The image is sent to the user in the chat and stored in the sandbox at /workspace/uploads, so it can be processed with sandbox tools. Set send_to_chat to false to generate an image without showing it to the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "A text description of the image to generate"
                    },
                    "resolution": {
                        "type": "string",
                        "description": ("Image dimensions as WIDTHxHEIGHT, e.g. '1024x1024'. "
                                        f"Width and height must each be a multiple of {DIM_STEP} "
                                        f"between {MIN_DIM} and {MAX_DIM}, e.g. '1024x1024', "
                                        "'1568x672', '768x1344', '1344x768'")
                    },
                    "send_to_chat": {
                        "type": "boolean",
                        "description": ("Whether to send the generated image to the user in the chat. "
                                        "Defaults to true. Set to false to generate the image privately: "
                                        "it is still stored in the sandbox, but not shown to the user.")
                    }
                },
                "required": ["prompt"],
                "additionalProperties": False
            }
        }
    },
]


def register(tools_list: list[dict], tool_map: dict):
    tools_list.extend(TOOL_DEFINITIONS)
    tool_map[TOOL_NAME] = generate_and_send_image
