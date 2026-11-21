"""Per-user, per-thread LLM settings.

The user keeps an override for each setting or leaves it unset (None in the
serialized form); reading a setting resolves it against the module default.
The defaults live here: the module-level constants below are the one
resolution point, and ``UserSettings`` properties fall back to them (or to
the matching config value) when the stored value is unset.

``DEFAULT_SYSTEM_PROMPT`` stays a plain module global rather than a frozen
default object because chatbot.py replaces it at startup with the contents
of the ./system_prompt file; the property reads it at call time so a
replacement takes effect for settings created before and after.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, asdict, fields
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import config
DEFAULT_SYSTEM_PROMPT: str = 'You are a helpful assistant running as a Telegram chat bot. Answer the user\'s questions to the best of your ability.'
# Fallback when the user never set one (matches the datetime tool's UTC).
DEFAULT_TIMEZONE: str = 'UTC'

@dataclass
class UserSettingsSerialized:
    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    system_prompt: str | None = None
    user_profile: str | None = None
    show_tool_output: bool | None = None
    thinking_keep: bool | None = None
    subagent_model: str | None = None
    timezone: str | None = None

    @classmethod
    def from_json(cls, json_str: str) -> UserSettingsSerialized:
        data = json.loads(json_str)
        valid_fields = {f.name for f in fields(cls)}
        cleaned = {
            k: v
            for k, v in data.items()
            if k in valid_fields and v is not None
        }

        return cls(**cleaned)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


class UserSettings:
    """One user's (or one thread's) settings, resolved against the defaults."""

    def __init__(self, raw_user_settings: UserSettingsSerialized) -> None:
        self._settings = raw_user_settings

    @staticmethod
    def _text(name: str, value: str | None) -> str | None:
        """None clears the setting; what is set must not be blank."""
        if value is not None and not value.strip():
            raise ValueError(f"{name} must not be empty")
        return value

    @staticmethod
    def _bounded(name: str, value: float | None,
                 low: float, high: float) -> float | None:
        if value is not None and not (low <= value <= high):
            raise ValueError(f"{name} must be between {low} and {high}")
        return value

    @property
    def model(self) -> str:
        return self._settings.model or config.LLM_MODEL

    @model.setter
    def model(self, value: str | None):
        self._settings.model = self._text('model', value)

    @property
    def subagent_model(self) -> str:
        # Falls back to the user's current model (/model) when no subagent
        # model override is set.
        return (self._settings.subagent_model
                or config.SUBAGENT_MODEL
                or self.model)

    @subagent_model.setter
    def subagent_model(self, value: str | None):
        self._settings.subagent_model = self._text('subagent_model', value)

    @property
    def timezone(self) -> str:
        return self._settings.timezone or DEFAULT_TIMEZONE

    @timezone.setter
    def timezone(self, value: str | None):
        if value is not None:
            try:
                ZoneInfo(value)
            except (ValueError, ZoneInfoNotFoundError):
                raise ValueError(
                    f"Unknown timezone '{value}'. "
                    "Use an IANA name like 'Australia/Sydney'.")
        self._settings.timezone = value

    @property
    def temperature(self) -> float:
        return (config.temperature if self._settings.temperature is None
                else self._settings.temperature)

    @temperature.setter
    def temperature(self, value: float | None):
        self._settings.temperature = self._bounded(
            'temperature', value, 0.0, 2.0)

    @property
    def top_p(self) -> float:
        return (config.top_p if self._settings.top_p is None
                else self._settings.top_p)

    @top_p.setter
    def top_p(self, value: float | None):
        self._settings.top_p = self._bounded('top_p', value, 0.0, 1.0)

    @property
    def user_profile(self) -> str | None:
        return self._settings.user_profile

    @user_profile.setter
    def user_profile(self, value: str | None):
        self._settings.user_profile = value

    @property
    def system_prompt(self) -> str:
        return self._settings.system_prompt or DEFAULT_SYSTEM_PROMPT

    @system_prompt.setter
    def system_prompt(self, value: str | None):
        self._settings.system_prompt = self._text('system_prompt', value)

    @property
    def show_tool_output(self) -> bool:
        return True if self._settings.show_tool_output is None else self._settings.show_tool_output

    @show_tool_output.setter
    def show_tool_output(self, value: bool):
        self._settings.show_tool_output = value

    @property
    def thinking_keep(self) -> bool:
        return False if self._settings.thinking_keep is None else self._settings.thinking_keep

    @thinking_keep.setter
    def thinking_keep(self, value: bool):
        self._settings.thinking_keep = value

    @classmethod
    def from_json(cls, json_str: str) -> UserSettings:
        raw = UserSettingsSerialized.from_json(json_str)
        return cls(raw)

    def to_json(self) -> str:
        return self._settings.to_json()

    @classmethod
    def default(cls) -> UserSettings:
        """Every setting unset: the module defaults above apply."""
        return cls(UserSettingsSerialized())
