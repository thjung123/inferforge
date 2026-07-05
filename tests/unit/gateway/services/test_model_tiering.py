import math

import pytest

from gateway.services.model_tiering import (
    ModelTieringService,
    Tier,
    TieringThresholds,
    decide_tier,
)

T = TieringThresholds(
    window_s=60,
    hot_min_rpm=2.0,
    warm_min_rpm=1.0,
    cold_after_idle_s=100.0,
    archive_after_idle_s=1000.0,
)


def test_decide_tier_hot_on_high_rate():
    assert decide_tier(rpm=5.0, idle_seconds=0.0, t=T) == Tier.HOT


def test_decide_tier_warm_on_moderate_rate():
    assert decide_tier(rpm=1.5, idle_seconds=5000.0, t=T) == Tier.WARM


def test_decide_tier_warm_when_recently_used():
    # low rate but accessed within cold_after_idle → stay warm
    assert decide_tier(rpm=0.0, idle_seconds=50.0, t=T) == Tier.WARM


def test_decide_tier_cold_when_idle():
    assert decide_tier(rpm=0.0, idle_seconds=300.0, t=T) == Tier.COLD


def test_decide_tier_archive_when_very_idle():
    assert decide_tier(rpm=0.0, idle_seconds=5000.0, t=T) == Tier.ARCHIVE


def test_decide_tier_never_seen_archives():
    assert decide_tier(rpm=0.0, idle_seconds=math.inf, t=T) == Tier.ARCHIVE


class _FakeRedis:
    def __init__(self):
        self.kv = {}
        self.sets = {}
        self.expire_calls = []

    async def incr(self, key):
        self.kv[key] = int(self.kv.get(key, 0)) + 1
        return self.kv[key]

    async def expire(self, key, ttl):
        self.expire_calls.append((key, ttl))
        return True

    async def get(self, key):
        v = self.kv.get(key)
        return None if v is None else str(v)

    async def set(self, key, value, ex=None):
        self.kv[key] = value
        return True

    async def sadd(self, key, *values):
        self.sets.setdefault(key, set()).update(values)
        return len(values)

    async def smembers(self, key):
        return set(self.sets.get(key, set()))


@pytest.mark.asyncio
async def test_record_access_then_hot():
    redis = _FakeRedis()
    svc = ModelTieringService(redis, T)
    for _ in range(3):  # rpm = 3/(60/60) = 3 >= hot_min 2
        await svc.record_access("bert_ensemble", now=1000.0)

    tier = await svc.evaluate("bert_ensemble", now=1000.0)
    assert tier == Tier.HOT


@pytest.mark.asyncio
async def test_record_access_registers_model():
    redis = _FakeRedis()
    svc = ModelTieringService(redis, T)
    await svc.record_access("bert_ensemble", now=1.0)
    assert await svc.known_models() == {"bert_ensemble"}


@pytest.mark.asyncio
async def test_reap_demotes_on_transition_only():
    redis = _FakeRedis()
    svc = ModelTieringService(redis, T)
    await redis.set("tier:last:m", "0.0")  # idle, no recent count → rpm 0

    unloaded = []

    async def unload(x):
        unloaded.append(x)

    async def archive(x):
        pass

    await svc.reap(["m"], now=300.0, unload=unload, archive=archive)  # HOT→COLD
    await svc.reap(["m"], now=300.0, unload=unload, archive=archive)  # stays COLD

    assert unloaded == ["m"]  # unloaded once, not every cycle
    assert await svc.get_tier("m") == Tier.COLD


@pytest.mark.asyncio
async def test_ensure_hot_loads_cold_model():
    redis = _FakeRedis()
    svc = ModelTieringService(redis, T)
    await svc.set_tier("m", Tier.COLD)

    loaded, restored = [], []

    async def load(x):
        loaded.append(x)

    async def restore(x):
        restored.append(x)

    prev = await svc.ensure_hot("m", now=1.0, restore=restore, load=load)
    assert prev == Tier.COLD
    assert loaded == ["m"]
    assert restored == []  # cold engine still in the repo; no restore needed
    assert await svc.get_tier("m") == Tier.HOT


@pytest.mark.asyncio
async def test_ensure_hot_restores_archived_model():
    redis = _FakeRedis()
    svc = ModelTieringService(redis, T)
    await svc.set_tier("m", Tier.ARCHIVE)

    loaded, restored = [], []

    async def load(x):
        loaded.append(x)

    async def restore(x):
        restored.append(x)

    prev = await svc.ensure_hot("m", now=1.0, restore=restore, load=load)
    assert prev == Tier.ARCHIVE
    assert restored == ["m"]
    assert loaded == ["m"]


@pytest.mark.asyncio
async def test_ensure_hot_noop_for_hot_model():
    redis = _FakeRedis()
    svc = ModelTieringService(redis, T)

    loaded = []

    async def load(x):
        loaded.append(x)

    async def restore(x):
        pass

    prev = await svc.ensure_hot("m", now=1.0, restore=restore, load=load)
    assert prev == Tier.HOT  # default when there's no recorded state
    assert loaded == []  # a hot model isn't reloaded


@pytest.mark.asyncio
async def test_record_access_sets_ttl_once_per_window():
    # Fixed window: TTL is set only when the counter is first created, not refreshed
    # on every request (otherwise rpm would grow unbounded and pin the model HOT).
    redis = _FakeRedis()
    svc = ModelTieringService(redis, T)
    for _ in range(5):
        await svc.record_access("m", now=1.0)

    assert redis.expire_calls == [("tier:count:m", T.window_s)]


@pytest.mark.asyncio
async def test_reap_unloads_cold_and_archives_stale():
    redis = _FakeRedis()
    svc = ModelTieringService(redis, T)
    # last access recorded, but the request-count window has since expired (rpm=0)
    await redis.set("tier:last:cold_model", "0.0")
    await redis.set("tier:last:stale_model", "0.0")

    unloaded, archived = [], []

    async def unload(m):
        unloaded.append(m)

    async def archive(m):
        archived.append(m)

    r1 = await svc.reap(["cold_model"], now=300.0, unload=unload, archive=archive)
    assert r1["cold_model"] == Tier.COLD  # idle 300s ∈ [100, 1000)
    assert unloaded == ["cold_model"]

    r2 = await svc.reap(["stale_model"], now=5000.0, unload=unload, archive=archive)
    assert r2["stale_model"] == Tier.ARCHIVE  # idle 5000s ≥ 1000
    assert archived == ["stale_model"]
