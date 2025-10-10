import os
import pytest
import requests
import time

BASE_URL = "http://localhost:8000"
os.environ.setdefault("API_KEY_WHITELIST", '["test-key"]')
HEADERS = {"x-api-key": "test-key"}

RATE_LIMIT = 10


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
def setup_environment():
    wait_for_service(f"{BASE_URL}/health")


def test_rate_limiter_blocks_excessive_requests():
    success, blocked = 0, 0
    for i in range(30):
        resp = requests.get(f"{BASE_URL}/health", headers=HEADERS)
        if resp.status_code == 200:
            success += 1
        elif resp.status_code == 429:
            blocked += 1
    assert blocked > 0, f"Rate limiter did not block any requests (Allowed={success})"
    time.sleep(3)


def test_retry_policy_handles_temporary_failures(monkeypatch):
    a = requests.get(f"{BASE_URL}/health/reset", headers=HEADERS)
    print(a.text)

    for i in range(3):
        resp = requests.get(f"{BASE_URL}/health/unstable", headers=HEADERS)
        if resp.status_code == 200:
            assert resp.json()["status"] == "ok_after_retry"
            return
        print(resp.json())
        time.sleep(0.5)
    pytest.fail("Retry policy did not recover from temporary failures")


def test_circuit_breaker_opens_and_recovers(monkeypatch):
    fail_url = f"{BASE_URL}/health/fail"
    ok_url = f"{BASE_URL}/health"

    for _ in range(5):
        r = requests.get(fail_url, headers=HEADERS)
        print(r.json())
        assert r.status_code in (500, 503)

    blocked = requests.get(ok_url, headers=HEADERS)
    assert (
        blocked.status_code == 503
    ), f"Expected breaker open (503), got {blocked.status_code}"

    time.sleep(3)
    recovered = requests.get(ok_url, headers=HEADERS)
    assert recovered.status_code == 200
