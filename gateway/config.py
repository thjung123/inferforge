from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    triton_url: str

    model_config = SettingsConfigDict(env_file=".env")
