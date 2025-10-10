import time
from typing import Optional


class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int = 5, recovery_time: int = 30):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self.fail_count = 0
        self.last_failure: Optional[float] = None
        self.open = False

    def record_failure(self):
        self.fail_count += 1
        self.last_failure = time.time()
        if self.fail_count >= self.failure_threshold:
            self.open = True

    def record_success(self):
        self.fail_count = 0
        self.open = False
        self.last_failure = None

    def allow_request(self) -> bool:
        if not self.open:
            return True
        if self.last_failure and time.time() - self.last_failure > self.recovery_time:
            self.open = False
            self.fail_count = 0
            return True
        return False
