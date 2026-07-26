import time
from collections import defaultdict, deque


class RateLimiter:
    """Sliding-window rate limiter: at most `max_per_min` events per `window`
    seconds per key (e.g. client IP). In-memory, so it's per-instance — fine for
    a single free-tier web service. `clock` is injectable for tests.
    """

    def __init__(self, max_per_min, window=60, clock=time.time):
        self.max = max_per_min
        self.window = window
        self.clock = clock
        self._hits = defaultdict(deque)

    def allow(self, key):
        now = self.clock()
        dq = self._hits[key]
        while dq and now - dq[0] > self.window:
            dq.popleft()
        if len(dq) >= self.max:
            return False
        dq.append(now)
        return True

    def retry_after(self, key):
        dq = self._hits.get(key)
        if not dq:
            return 0
        return max(1, int(self.window - (self.clock() - dq[0])) + 1)
