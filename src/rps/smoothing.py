from collections import deque, Counter


class MajorityVoteSmoother:
    def __init__(self, window_size: int = 9):
        self.window = deque(maxlen=window_size)

    def update(self, label: str) -> str:
        self.window.append(label)
        return Counter(self.window).most_common(1)[0][0]