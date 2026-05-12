"""Process-pool helpers for the rest of the package.

Stdlib-only (uses ``concurrent.futures.ProcessPoolExecutor``).  Falls back to
serial execution when ``n_jobs <= 1`` or there is only one item.

Workers are CPU-bound (numpy / pandas / cumsum), so processes — not threads —
are the right tool: we need to escape the GIL.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Iterable, List, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def cpu_count(reserve: int = 1) -> int:
    """Available CPUs minus ``reserve`` (so the host stays responsive)."""
    return max(1, mp.cpu_count() - reserve)


def resolve_n_jobs(n_jobs: int | str | None) -> int:
    """Translate the ``n_jobs`` arg used across the package into a real worker count.

    * ``None`` / ``1``  → 1 (serial)
    * ``"auto"``        → cpu_count() − 1
    * positive int      → that many workers (clamped to ≥ 1)
    * negative int      → cpu_count() + n_jobs + 1  (joblib convention; -1 = all CPUs)
    """
    if n_jobs is None:
        return 1
    if isinstance(n_jobs, str):
        if n_jobs == "auto":
            return cpu_count(reserve=1)
        raise ValueError(f"Unknown n_jobs string: {n_jobs!r}")
    if isinstance(n_jobs, int):
        if n_jobs == 0:
            return 1
        if n_jobs < 0:
            return max(1, mp.cpu_count() + n_jobs + 1)
        return max(1, n_jobs)
    raise TypeError(f"n_jobs must be int, str or None, got {type(n_jobs).__name__}")


def parallel_map(
    func: Callable[[T], R],
    items: Iterable[T],
    n_jobs: int | str | None = "auto",
    progress: bool = False,
    desc: str = "",
) -> List[R]:
    """Map ``func`` over ``items``, returning results in input order.

    Falls back to a plain Python loop when ``n_jobs <= 1`` or when ``items``
    has 0/1 elements — avoids paying the process-spawn cost for trivial work.

    ``progress=True`` prints a one-line counter after each completion (cheap,
    no tqdm dependency).  Use ``desc`` to label the counter.

    Notes
    -----
    * ``func`` and every element of ``items`` must be picklable.  Define ``func``
      at module scope (top-level), not as a closure / lambda / inner function.
    * Each worker process inherits the parent's memory at fork time on Linux,
      so for memory-light tasks this is essentially free.  When the parent
      already holds large DataFrames, prefer to pass *paths* rather than the
      DataFrames themselves and let the worker materialize its own slice.
    """
    items_list = list(items)
    n = len(items_list)
    workers = resolve_n_jobs(n_jobs)
    if workers <= 1 or n <= 1:
        out: List[R] = []
        for i, x in enumerate(items_list, start=1):
            out.append(func(x))
            if progress:
                _print_progress(desc, i, n)
        return out

    workers = min(workers, n)
    results: List[R] = [None] * n  # type: ignore[list-item]
    # ``fork`` keeps notebooks and stdin-fed scripts working (Python 3.14 made
    # ``forkserver`` the default on Linux, which can't re-import a ``<stdin>``
    # main module).  fork is also cheaper for our workload — workers get the
    # parent's memory copy-on-write rather than re-importing it.
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        future_to_idx = {ex.submit(func, x): i for i, x in enumerate(items_list)}
        done = 0
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            results[idx] = fut.result()
            done += 1
            if progress:
                _print_progress(desc, done, n)
    return results


def _print_progress(desc: str, done: int, total: int) -> None:
    label = f"{desc} " if desc else ""
    end = "\n" if done == total else "\r"
    print(f"  {label}[{done}/{total}]", end=end, flush=True)
