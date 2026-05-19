from __future__ import annotations
import json
from dataclasses import dataclass, asdict, fields
import config
DEFAULT_SYSTEM_PROMPT: str = 'You are a helpful assistant running as a Telegram chat bot. Answer the user\'s questions to the best of your ability.'

@dataclass
class UserSettingsSerialized:
    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    system_prompt: str | None = None
    user_profile: str | None = None
    show_tool_output: bool | None = None

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
    def __init__(self, raw_user_settings: UserSettingsSerialized) -> None:
        self._settings = raw_user_settings
    @property
    def model(self) -> str:
        return self._settings.model or config.LLM_MODEL
    
    @model.setter
    def model(self, value: str | None):
        self._settings.model = value

    @property
    def temperature(self) -> float:
        return config.temperature if self._settings.temperature is None else self._settings.temperature
    
    @temperature.setter
    def temperature(self, value: float | None):
        if value is not None and not (0.0 <= value <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")
        self._settings.temperature = value

    @property
    def top_p(self) -> float:
         return config.top_p if self._settings.top_p is None else self._settings.top_p
    
    @top_p.setter
    def top_p(self, value: float | None):
        if value is not None and not (0.0 <= value <= 1.0):
            raise ValueError("top_p must be between 0.0 and 1.0")
        self._settings.top_p = value

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
        self._settings.system_prompt = value  
    
    @property
    def show_tool_output(self) -> bool:
        return True if self._settings.show_tool_output is None else self._settings.show_tool_output

    @show_tool_output.setter
    def show_tool_output(self, value: bool):
        self._settings.show_tool_output = value

    @classmethod
    def from_json(cls, json_str: str) -> UserSettings:
        raw = UserSettingsSerialized.from_json(json_str)
        return cls(raw) 
    
    def to_json(self) -> str:
        return self._settings.to_json()
    
    @classmethod
    def default(cls) -> UserSettings:
        return cls(UserSettingsSerialized())
        