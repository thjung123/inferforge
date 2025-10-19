import pytest
import requests
import time

BASE_URL = "http://localhost:8000"


def wait_for_service(url, timeout=40):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url)
            if r.status_code == 200:
                return True
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"Service {url} did not become ready in {timeout}s")


@pytest.fixture(scope="session", autouse=True)
def setup_compose_environment():
    wait_for_service(f"{BASE_URL}/health")


def test_health_check():
    resp = requests.get(f"{BASE_URL}/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    time.sleep(1)


def test_version_endpoint():
    headers = {"x-api-key": "test-key"}
    resp = requests.get(f"{BASE_URL}/version", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert "version" in body
    assert "commit" in body
    assert "build_time" in body
    time.sleep(1)
