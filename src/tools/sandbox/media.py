"""Images and file delivery.

The parent agent and its subagents get the same tool name from this module;
only the implementation differs. A subagent has no messaging handle, so its
variant reads or generates and reports the path for the parent to share.
"""

import base64
import logging
import posixpath
import shlex
from dataclasses import dataclass

from chat_update_context import AgentContext, ChatUpdateContext
from sandbox import manager
from tool_registry import SUBAGENT, agent_tool
from tool_result import ImageAttachment, ToolCallResult
from upload_cache import was_recently_uploaded

from .common import _resolve_path

logger = logging.getLogger(__name__)

MAX_SEND_FILE_BYTES = 45 * 1024 * 1024

MAX_IMAGE_DISPLAY_BYTES = 5 * 1024 * 1024

IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

_FILE_PATH_PARAM = {
    "type": "string",
    "description": "The path to the image file to view.",
}

_SEND_TO_CHAT_PARAM = {
    "type": "boolean",
    "description": ("Whether to also send the image to the user in the chat. "
                    "Defaults to true. Set to false to inspect the image "
                    "yourself without showing it to the user."),
}


@dataclass
class LoadedImage:
    """An image read out of the sandbox."""
    data: bytes
    mime: str
    path: str


async def _load_image(ctx: AgentContext,
                      filePath: str) -> LoadedImage | str:
    """Shared business logic: read an image out of the sandbox.

    Returns the bytes plus their resolved path, or an error string fit for
    the model. Showing the image to the user is the caller's job.
    """
    path = _resolve_path(filePath)
    mime = IMAGE_MIME_TYPES.get(posixpath.splitext(path)[1].lower())
    if mime is None:
        return ("Error: not a supported image type. Supported formats: "
                "png, jpg, jpeg, gif, webp.")
    stat = await manager.exec(
        ctx.user_id, f"stat -c %s {shlex.quote(path)}")
    if stat["returncode"] != 0:
        return f"Error: File not found: {path}"
    size = int(stat["stdout"].strip() or 0)
    if size > MAX_IMAGE_DISPLAY_BYTES:
        return (f"Error: image is too large to display "
                f"({size:,} bytes; limit is {MAX_IMAGE_DISPLAY_BYTES:,}).")

    encoded = await manager.exec(
        ctx.user_id, f"base64 -w0 {shlex.quote(path)}")
    if encoded["returncode"] != 0:
        return f"Error: {encoded['stderr'].strip()[:500]}"
    try:
        data = base64.b64decode(encoded["stdout"].strip())
    except ValueError:
        return "Error: could not read the image contents."
    return LoadedImage(data=data, mime=mime, path=path)


def _image_result(text: str, image: LoadedImage) -> ToolCallResult:
    return ToolCallResult(
        text=(f"{text}: {image.path} ({len(image.data):,} bytes). "
              "The image is attached to this result."),
        image=ImageAttachment(data=image.data, mime=image.mime),
    )


@agent_tool(
    name="read_image",
    description=(
        "Read an image file (png, jpg, jpeg, gif, webp) from the "
        "sandbox and attach it to the conversation so you can see "
        "it. Requires a vision-capable model. The image is also "
        "sent to the user's chat, unless send_to_chat is false, or "
        "the user uploaded that file recently (it is already "
        "visible in their chat window from the upload itself). "
        "Use this instead of the read tool for image "
        "files; the read tool rejects binary files. Images larger "
        "than 5 MB cannot be displayed."),
    parameters={
        "filePath": _FILE_PATH_PARAM,
        "send_to_chat": _SEND_TO_CHAT_PARAM,
    })
async def read_image(ctx: ChatUpdateContext, filePath: str,
                     send_to_chat: bool = True) -> str | ToolCallResult:
    loaded = await _load_image(ctx, filePath)
    if isinstance(loaded, str):
        return loaded

    # send_to_chat lets the model decide whether the image is sent to the chat.
    # A recently uploaded file is already visible in the chat window from the
    # user's own send, so displaying it again would be redundant.
    recently_uploaded = was_recently_uploaded(ctx.user_id, loaded.path)
    shown = send_to_chat and not recently_uploaded
    if shown:
        try:
            await ctx.messaging.send_image(loaded.data)
        except Exception as e:
            logger.warning("Could not send image to chat: %s", e)

    if recently_uploaded and send_to_chat:
        label = ("Image read (not re-displayed in the chat; the user "
                 "recently uploaded this file there)")
    else:
        label = "Image shown in the chat" if shown else "Image read"
    return _image_result(label, loaded)


@agent_tool(
    name="read_image",
    scope=SUBAGENT,
    description=(
        "Read an image file (png, jpg, jpeg, gif, webp) from the "
        "sandbox and attach it to your conversation so you can see "
        "it. Requires a vision-capable model. The image is not shown "
        "to the user: report its path to the parent assistant if it "
        "should be shared. Use this instead of the read tool for "
        "image files; the read tool rejects binary files. Images "
        "larger than 5 MB cannot be displayed."),
    parameters={"filePath": _FILE_PATH_PARAM})
async def read_image_subagent(ctx: AgentContext,
                              filePath: str) -> str | ToolCallResult:
    loaded = await _load_image(ctx, filePath)
    if isinstance(loaded, str):
        return loaded
    return _image_result("Image read (not shown to the user)", loaded)


@agent_tool(
    description=(
        "Send a file from the sandbox to the user as a Telegram "
        "document. Usage:\n"
        "- The file can be anywhere in the sandbox; paths are "
        "relative to the sandbox workspace root /workspace, or "
        "absolute.\n"
        "- Use this to share files generated in the sandbox with "
        "the user (reports, data files, images, binaries, etc.).\n"
        "- Files larger than 45 MB cannot be sent."),
    parameters={
        "filePath": {
            "type": "string",
            "description": "The path to the file to send.",
        },
    })
async def send_file(ctx: ChatUpdateContext, filePath: str) -> str:
    path = _resolve_path(filePath)
    stat = await manager.exec(
        ctx.user_id, f"stat -c %s {shlex.quote(path)}")
    if stat["returncode"] != 0:
        return f"Error: File not found: {path}"
    size = int(stat["stdout"].strip() or 0)
    if size > MAX_SEND_FILE_BYTES:
        return (f"Error: file is too large to send "
                f"({size:,} bytes; limit is {MAX_SEND_FILE_BYTES:,}).")
    encoded = await manager.exec(
        ctx.user_id, f"base64 -w0 {shlex.quote(path)}")
    if encoded["returncode"] != 0:
        return f"Error: {encoded['stderr'].strip()[:500]}"
    try:
        data = base64.b64decode(encoded["stdout"].strip())
    except ValueError:
        return "Error: could not decode the file contents."
    filename = posixpath.basename(path)
    await ctx.messaging.send_document(data, filename)
    return f"Sent {filename} ({len(data):,} bytes)."
