import time


class FPSCounter:
    def __init__(self):
        self._prev = 0.0
        self._fps = 0.0

    def tick(self) -> float:
        now = time.time()
        if self._prev != 0.0:
            self._fps = 1.0 / (now - self._prev)
        self._prev = now
        return self._fps