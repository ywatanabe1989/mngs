#!/usr/bin/env python3

import time


class TimeStamper:
    """
    A class for generating timestamps with optional comments, tracking both the time since object creation and since the last call.

    Attributes:
        id (int): An identifier for each timestamp, starting from 0.
        start_time (float): The time when the TimeStamper object was created.

    Arguments:
        is_simple (bool): Determines the output format. Defaults to True for a simpler format.

    Returns:
        str: The generated timestamp string.

    Example:
        >>> ts = TimeStamper(is_simple=True)
        >>> ts("Starting process")
        ID:0 | 00:00:00 Starting process |
        >>> time.sleep(1)
        >>> ts("One second later")
        ID:1 | 00:00:01 One second later |
    """

    def __init__(self, is_simple=True):
        self.id = -1
        self.start_time = time.time()
        self._is_simple = is_simple
        self._time = time
        self._prev = self.start_time

    def __call__(self, comment=""):
        # Calculation
        now = self._time.time()
        from_start = now - self.start_time
        from_prev = now - self._prev

        # Saves as attributes
        self._from_start_hhmmss = self._time.strftime(
            "%H:%M:%S", self._time.gmtime(from_start)
        )
        self._from_prev_hhmmss = self._time.strftime(
            "%H:%M:%S", self._time.gmtime(from_prev)
        )

        # Incrementation
        self.id += 1
        self.prev = now

        # Text construction
        if self._is_simple:
            self.text = (
                f"ID:{self.id} | {self._from_start_hhmmss} {comment} | "
            )
        else:
            self.text = f"Time (id:{self.id}): total {self._from_start_hhmmss}, prev {self._from_prev_hhmmss} [hh:mm:ss]: {comment}\n"

        print(self.text)
        return self.text


if __name__ == "__main__":
    ts = TimeStamper(is_simple=True)
    ts("Starting process")
    time.sleep(1)
    ts("One second later")
    time.sleep(2)
    ts("Two seconds later")
