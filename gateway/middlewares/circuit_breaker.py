import time


class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_time=30):
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self.fail_count = 0
        self.last_failure = 0
        self.open = False

    def record_failure(self):
        self.fail_count += 1
        self.last_failure = time.time()
        if self.fail_count >= self.failure_threshold:
            self.open = True

    def allow_request(self):
        if not self.open:
            return True
        if time.time() - self.last_failure > self.recovery_time:
            self.open = False
            self.fail_count = 0
            return True
        return False
