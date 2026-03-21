from typing import Any, Awaitable, cast

from redis.asyncio import Redis

from builder.schemas import JobState


class JobTracker:
    _instance: "JobTracker | None" = None

    @classmethod
    def initialize(cls, redis: Redis) -> "JobTracker":
        cls._instance = cls(redis)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "JobTracker":
        if cls._instance is None:
            raise RuntimeError("JobTracker not initialized")
        return cls._instance

    def __init__(self, redis: Redis):
        self._redis = redis

    def _key(self, job_id: str) -> str:
        return f"build_job:{job_id}"

    async def create(self, job_id: str, model_name: str) -> None:
        await cast(
            Awaitable[int],
            self._redis.hset(
                self._key(job_id),
                mapping={
                    "job_id": job_id,
                    "model_name": model_name,
                    "status": JobState.PENDING.value,
                    "error": "",
                },
            ),
        )

    async def update_status(self, job_id: str, status: JobState) -> None:
        await cast(
            Awaitable[int],
            self._redis.hset(self._key(job_id), "status", status.value),
        )

    async def set_failed(self, job_id: str, error: str) -> None:
        await cast(
            Awaitable[int],
            self._redis.hset(
                self._key(job_id),
                mapping={"status": JobState.FAILED.value, "error": error},
            ),
        )

    async def get(self, job_id: str) -> dict[str, Any] | None:
        data: dict[str, Any] = await cast(
            Awaitable[dict[str, Any]],
            self._redis.hgetall(self._key(job_id)),
        )
        return data if data else None


def get_job_tracker() -> JobTracker:
    return JobTracker.get_instance()
