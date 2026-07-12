from types import SimpleNamespace
from chat_update_context import ChatUpdateContext


class RecordingBot:
    """Mock bot that captures all send_* calls for test assertions."""

    def __init__(self):
        self.sent = []
        self.username = "test_bot"

    async def send_message(self, *args, **kwargs):
        text = args[1] if len(args) >= 2 else kwargs.get("text")
        self.sent.append({
            "text": text or "",
            "parse_mode": kwargs.get("parse_mode"),
            "kind": "text",
        })
        return SimpleNamespace(message_id=len(self.sent))

    async def send_chat_action(self, *args, **kwargs):
        return None

    async def send_photo(self, *args, **kwargs):
        caption = kwargs.get("caption", "")
        self.sent.append({"text": caption or "", "kind": "photo"})
        return SimpleNamespace(message_id=len(self.sent))

    async def send_document(self, *args, **kwargs):
        caption = kwargs.get("caption", "")
        self.sent.append({"text": caption or "", "kind": "document"})
        return SimpleNamespace(message_id=len(self.sent))

    async def get_file(self, *args, **kwargs):
        return SimpleNamespace(
            file_path="mock://file",
            download_to_drive=lambda *a, **kw: None,
        )


def create_mock_bot() -> RecordingBot:
    return RecordingBot()


def create_mock_context(
    text: str = "",
    user_id: int = 1,
    chat_id: int = 1,
    command_text: str = "",
    command_name: str = "",
    chat_type: str = "private",
) -> ChatUpdateContext:
    bot = create_mock_bot()
    context = SimpleNamespace(
        bot=bot,
        bot_data={"allowed_ids": []},
        user_data={},
        chat_data={},
    )
    message = SimpleNamespace(
        text=text,
        caption=None,
        chat=SimpleNamespace(id=chat_id, type=chat_type),
        from_user=SimpleNamespace(
            id=user_id, first_name="Test", last_name="User"),
    )
    return ChatUpdateContext(
        update=None,
        context=context,
        bot=bot,
        chat=message.chat,
        user=message.from_user,
        message=message,
        user_data=context.user_data,
        chat_data=context.chat_data,
        command_text=command_text,
        command_name=command_name,
    )
