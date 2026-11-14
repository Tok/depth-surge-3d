"""Thread-based parallel map for per-frame image work (OpenCV releases the GIL)."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

MAX_FRAME_THREADS = 8


def frame_thread_count(num_items: int) -> int:
    """PURE-ish: Threads to use for per-frame work, leaving two cores for the system."""
    return max(1, min((os.cpu_count() or 1) - 2, MAX_FRAME_THREADS, num_items))


def map_frames(function: Callable[[Any], Any], items: Iterable[Any]) -> Iterator[Any]:
    """
    Apply a function to items in parallel threads, yielding results in input order.

    If the consumer stops early (e.g. on a stop request), work not started yet is dropped.

    Args:
        function: Function taking one item
        items: Items to process

    Yields:
        Results in the order of the items

    Side effects:
        - Runs the function in worker threads
    """
    work = list(items)
    workers = frame_thread_count(len(work))
    if workers <= 1:
        yield from map(function, work)
        return

    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        yield from executor.map(function, work)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
