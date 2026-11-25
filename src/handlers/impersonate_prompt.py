"""The prompt that makes a cloned conversation write the user's next
message."""

IMPERSONATE_PROMPT = """<impersonation-task>
The conversation above is between an assistant and a user. Do not answer as the assistant. Instead, write the user's next message, impersonating the user.

The system prompt and any embedded user profile tell you who the user is; the conversation history shows how they talk. Match their voice: vocabulary, tone, sentence style, interests, and habits, including how they handle other languages, tools, or topics. Stay in character even if the conversation steers somewhere the user would not normally go.

Rules:
- Write only the message's text: no preamble, no quotes around it, no stage directions, no commentary about impersonating.
- Keep it at the natural length for how this user writes.
- Do not answer your own message or continue the conversation; the real assistant will reply next.
- Never mention this task, the impersonation, or that you are not the user.
</impersonation-task>"""
