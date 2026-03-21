from typing import Any, Awaitable, cast

from redis.asyncio import Redis

from builder.schemas import JobState

_JOB_PREFIX = "build_job:"
_TERMINAL_TTL_SEC = 3600  # 1 hour TTL for completed/failed jobs


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
        return f"{_JOB_PREFIX}{job_id}"

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
        key = self._key(job_id)
        await cast(
            Awaitable[int],
            self._redis.hset(key, "status", status.value),
        )
        if status in (JobState.READY, JobState.FAILED):
            await self._redis.expire(key, _TERMINAL_TTL_SEC)

    async def set_failed(self, job_id: str, error: str) -> None:
        key = self._key(job_id)
        await cast(
            Awaitable[int],
            self._redis.hset(
                key,
                mapping={"status": JobState.FAILED.value, "error": error},
            ),
        )
        await self._redis.expire(key, _TERMINAL_TTL_SEC)

    async def get(self, job_id: str) -> dict[str, Any] | None:
        data: dict[str, Any] = await cast(
            Awaitable[dict[str, Any]],
            self._redis.hgetall(self._key(job_id)),
        )
        return data if data else None

    async def list_all(self) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        cursor: int = 0
        while True:
            cursor, keys = await cast(
                Awaitable[tuple[int, list[str]]],
                self._redis.scan(cursor, match=f"{_JOB_PREFIX}*", count=100),
            )
            for key in keys:
                data: dict[str, Any] = await cast(
                    Awaitable[dict[str, Any]],
                    self._redis.hgetall(key),
                )
                if data:
                    jobs.append(data)
            if cursor == 0:
                break
        return jobs


def get_job_tracker() -> JobTracker:
    return JobTracker.get_instance()
