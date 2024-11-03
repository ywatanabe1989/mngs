#!/usr/bin/env python3

import time

import pandas as pd


class TimeStamper:
    """
    A class for generating timestamps with optional comments, tracking both the time since object creation and since the last call.
    """

    def __init__(self, is_simple=True):
        self.id = -1
        self.start_time = time.time()
        self._is_simple = is_simple
        self._prev = self.start_time
        self._df_record = pd.DataFrame(
            columns=[
                "timestamp",
                "elapsed_since_start",
                "elapsed_since_prev",
                "comment",
                "formatted_text",
            ]
        )

    def __call__(self, comment="", verbose=False):
        now = time.time()
        from_start = now - self.start_time
        from_prev = now - self._prev
        formatted_from_start = time.strftime(
            "%H:%M:%S", time.gmtime(from_start)
        )
        formatted_from_prev = time.strftime("%H:%M:%S", time.gmtime(from_prev))
        self.id += 1
        self._prev = now
        text = (
            f"ID:{self.id} | {formatted_from_start} {comment} | "
            if self._is_simple
            else f"Time (id:{self.id}): total {formatted_from_start}, prev {formatted_from_prev} [hh:mm:ss]: {comment}\n"
        )

        # Update DataFrame directly
        self._df_record.loc[self.id] = [
            now,
            from_start,
            from_prev,
            comment,
            text,
        ]

        if verbose:
            print(text)
        return text

    @property
    def record(self):
        return self._df_record[
            [
                "timestamp",
                "elapsed_since_start",
                "elapsed_since_prev",
                "comment",
            ]
        ]

    def delta(self, id1, id2):
        """
        Calculate the difference in seconds between two timestamps identified by their IDs.

        Parameters:
            id1 (int): The ID of the first timestamp.
            id2 (int): The ID of the second timestamp.

        Returns:
            float: The difference in seconds between the two timestamps.

        Raises:
            ValueError: If either id1 or id2 is not in the DataFrame index.
        """
        # Adjust for negative indices, similar to negative list indexing in Python
        if id1 < 0:
            id1 = len(self._df_record) + id1
        if id2 < 0:
            id2 = len(self._df_record) + id2

        # Check if both IDs exist in the DataFrame
        if (
            id1 not in self._df_record.index
            or id2 not in self._df_record.index
        ):
            raise ValueError(
                "One or both of the IDs do not exist in the record."
            )

        # Compute the difference in timestamps
        time_diff = (
            self._df_record.loc[id1, "timestamp"]
            - self._df_record.loc[id2, "timestamp"]
        )
        return time_diff


# class TimeStamper:
#     """
#     A class for generating timestamps with optional comments, tracking both the time since object creation and since the last call.

#     Attributes:
#         id (int): An identifier for each timestamp, starting from 0.
#         start_time (float): The time when the TimeStamper object was created.

#     Arguments:
#         is_simple (bool): Determines the output format. Defaults to True for a simpler format.

#     Returns:
#         str: The generated timestamp string.

#     Example:
#         >>> ts = TimeStamper(is_simple=True)
#         >>> print(ts("Starting process"))
#         ID:0 | 00:00:00 Starting process |
#         >>> time.sleep(1)
#         >>> print(ts("One second later"))
#         ID:1 | 00:00:01 One second later |
#     """

#     def __init__(self, is_simple=True):
#         self.id = -1
#         self.start_time = time.time()
#         self._is_simple = is_simple
#         self._prev = self.start_time
#         self._record = []

#     def __call__(self, comment=""):
#         # Calculation
#         now = time.time()
#         from_start = now - self.start_time
#         from_prev = now - self._prev

#         # Format time strings for display
#         formatted_from_start = time.strftime(
#             "%H:%M:%S", time.gmtime(from_start)
#         )
#         formatted_from_prev = time.strftime("%H:%M:%S", time.gmtime(from_prev))

#         # Increment ID
#         self.id += 1
#         self._prev = now

#         # Text construction
#         if self._is_simple:
#             text = f"ID:{self.id} | {formatted_from_start} {comment} | "
#         else:
#             text = f"Time (id:{self.id}): total {formatted_from_start}, prev {formatted_from_prev} [hh:mm:ss]: {comment}\n"

#         # Update record with structured data
#         self._record.append(
#             {
#                 "id": self.id,
#                 "timestamp": now,
#                 "elapsed_since_start": from_start,
#                 "elapsed_since_prev": from_prev,
#                 "comment": comment,
#                 "formatted_text": text,
#             }
#         )

#         print(text)
#         return text

#     @property
#     def record(
#         self,
#     ):
#         cols = [
#             "timestamp",
#             "elapsed_since_start",
#             "elapsed_since_prev",
#             "comment",
#         ]
#         return pd.DataFrame(self._record).set_index("id")[cols]


# class TimeStamper:
#     """
#     A class for generating timestamps with optional comments, tracking both the time since object creation and since the last call.

#     Attributes:
#         id (int): An identifier for each timestamp, starting from 0.
#         start_time (float): The time when the TimeStamper object was created.

#     Arguments:
#         is_simple (bool): Determines the output format. Defaults to True for a simpler format.

#     Returns:
#         str: The generated timestamp string.

#     Example:
#         >>> ts = TimeStamper(is_simple=True)
#         >>> ts("Starting process")
#         ID:0 | 00:00:00 Starting process |
#         >>> time.sleep(1)
#         >>> ts("One second later")
#         ID:1 | 00:00:01 One second later |
#     """

#     def __init__(self, is_simple=True):
#         self.id = -1
#         self.start_time = time.time()
#         self._is_simple = is_simple
#         self._time = time
#         self._prev = self.start_time
#         self.record = []

#     def __call__(self, comment=""):
#         # Calculation
#         now = self._time.time()
#         from_start = now - self.start_time
#         from_prev = now - self._prev

#         # Saves as attributes
#         self._from_start_hhmmss = self._time.strftime(
#             "%H:%M:%S", self._time.gmtime(from_start)
#         )
#         self._from_prev_hhmmss = self._time.strftime(
#             "%H:%M:%S", self._time.gmtime(from_prev)
#         )

#         # Incrementation
#         self.id += 1
#         self.prev = now

#         # Text construction
#         if self._is_simple:
#             self.text = (
#                 f"ID:{self.id} | {self._from_start_hhmmss} {comment} | "
#             )
#         else:
#             self.text = f"Time (id:{self.id}): total {self._from_start_hhmmss}, prev {self._from_prev_hhmmss} [hh:mm:ss]: {comment}\n"

#         print(self.text)
#         self.record.append(self.text)
#         return self.text


if __name__ == "__main__":
    ts = TimeStamper(is_simple=True)
    ts("Starting process")
    time.sleep(1)
    ts("One second later")
    time.sleep(2)
    ts("Two seconds later")
