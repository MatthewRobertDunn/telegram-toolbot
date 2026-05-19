import base64
import requests
import config
from chat_update_context import context_tool, ChatUpdateContext
from messaging import send_image


async def generate_image(prompt: str, width: int = 1024, height: int = 1024, seed: int = 0, steps: int = 4) -> str:
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
    data = response.json()
    return data["artifacts"][0]["base64"]


@context_tool
async def generate_and_send_image(ctx: ChatUpdateContext, prompt: str) -> str:
    try:
        b64 = await generate_image(prompt)
        image_bytes = base64.b64decode(b64)
        await send_image(ctx, image_bytes)
        return f"Image sent to chat (prompt: {prompt})"
    except Exception as e:
        print(f"Error generating image: {e}")
        return "Failed to generate image"          

TOOL_NAME = "generate_image"    
TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "strict": False,
            "description": "Generate an image from a text prompt. The image is sent directly to the user",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "A text description of the image to generate"
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
