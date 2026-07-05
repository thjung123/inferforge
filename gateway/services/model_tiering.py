"""Usage-based lifecycle tiering for embedding models (hot/warm/cold/archive)."""

import math
from dataclasses import dataclass
from enum import Enum

from gateway.utils.logger import gateway_logger as logger

_COUNT_KEY = "tier:count:"
_LAST_KEY = "tier:last:"
_MODELS_KEY = "tier:models"
_STATE_KEY = "tier:state:"


class Tier(str, Enum):
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"
    ARCHIVE = "archive"


@dataclass
class TieringThresholds:
    window_s: int = 300
    hot_min_rpm: float = 10.0
    warm_min_rpm: float = 1.0
    cold_after_idle_s: float = 900.0
    archive_after_idle_s: float = 86400.0


def decide_tier(rpm: float, idle_seconds: float, t: TieringThresholds) -> Tier:
    """Pure tiering policy: request rate first, then idle time."""
    if rpm >= t.hot_min_rpm:
        return Tier.HOT
    if rpm >= t.warm_min_rpm or idle_seconds < t.cold_after_idle_s:
        return Tier.WARM
    if idle_seconds < t.archive_after_idle_s:
        return Tier.COLD
    return Tier.ARCHIVE


class ModelTieringService:
    def __init__(self, redis, thresholds: TieringThresholds | None = None):
        self._redis = redis
        self.t = thresholds or TieringThresholds()

    async def known_models(self) -> set[str]:
        return set(await self._redis.smembers(_MODELS_KEY))

    async def get_tier(self, model: str) -> Tier | None:
        raw = await self._redis.get(f"{_STATE_KEY}{model}")
        return Tier(raw) if raw else None

    async def set_tier(self, model: str, tier: Tier) -> None:
        await self._redis.set(f"{_STATE_KEY}{model}", tier.value)

    async def record_access(self, model: str, now: float) -> None:
        """Best-effort usage record on each inference call."""
        await self._redis.sadd(_MODELS_KEY, model)
        count = await self._redis.incr(f"{_COUNT_KEY}{model}")
        if count == 1:
            await self._redis.expire(f"{_COUNT_KEY}{model}", self.t.window_s)
        await self._redis.set(f"{_LAST_KEY}{model}", str(now))

    async def evaluate(self, model: str, now: float) -> Tier:
        raw_count = await self._redis.get(f"{_COUNT_KEY}{model}")
        raw_last = await self._redis.get(f"{_LAST_KEY}{model}")
        count = int(raw_count or 0)
        last = float(raw_last) if raw_last else None
        rpm = count / (self.t.window_s / 60.0)
        idle = (now - last) if last is not None else math.inf
        return decide_tier(rpm, idle, self.t)

    async def reap(self, models, now, *, unload, archive) -> dict:
        """Evaluate all models and apply demotions on tier transitions only."""
        result = {}
        for model in models:
            try:
                target = await self.evaluate(model, now)
                current = await self.get_tier(model) or Tier.HOT
                if target != current and target in (Tier.COLD, Tier.ARCHIVE):
                    await self.set_tier(model, target)
                    if target == Tier.COLD:
                        await unload(model)
                    elif target == Tier.ARCHIVE:
                        await archive(model)
                    logger.info(f"[Tiering] {model}: {current.value} → {target.value}")
                result[model] = target
            except Exception as e:
                logger.error(f"[Tiering] reap failed for {model}: {e}")
        return result

    async def ensure_hot(self, model: str, now: float, *, restore, load) -> Tier:
        """Promote a cold/archived model back to hot; returns its prior tier."""
        previous = await self.get_tier(model) or Tier.HOT
        if previous == Tier.ARCHIVE:
            await restore(model)
            await load(model)
        elif previous == Tier.COLD:
            await load(model)
        if previous in (Tier.COLD, Tier.ARCHIVE):
            await self.set_tier(model, Tier.HOT)
            logger.info(f"[Tiering] {model}: {previous.value} → hot (on-access)")
        await self.record_access(model, now)
        return previous
