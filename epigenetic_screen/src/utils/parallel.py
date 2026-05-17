from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
U = TypeVar("U")


def map_parallel(items: Iterable[T], fn: Callable[[T], U], n_jobs: int) -> list[U]:
    if n_jobs <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        return list(ex.map(fn, items))

