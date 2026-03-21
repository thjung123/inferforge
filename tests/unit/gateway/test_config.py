from gateway.config import get_settings, Settings


def test_default_settings_values():
    settings = Settings()
    assert settings.triton_url == "triton:8001"
    assert settings.redis_url == "redis://localhost:6379"
    assert settings.jwt_secret == "default_secret"
    assert isinstance(settings.api_key_whitelist, list)
    assert settings.rate_limit == 20
    assert settings.rate_window == 3
    assert settings.triton_max_retries == 3


def test_env_file_loading(monkeypatch):
    monkeypatch.setenv("TRITON_URL", "http://mock-triton:9000")
    monkeypatch.setenv("REDIS_URL", "redis://mock-redis:6379")
    monkeypatch.setenv("JWT_SECRET", "super_secret")
    monkeypatch.setenv("API_KEY_WHITELIST", '["key1", "key2"]')

    s = Settings()
    assert s.triton_url == "http://mock-triton:9000"
    assert s.redis_url == "redis://mock-redis:6379"
    assert s.jwt_secret == "super_secret"


def test_lru_cache_singleton_behavior():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
