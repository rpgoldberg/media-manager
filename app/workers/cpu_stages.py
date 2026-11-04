"""MASK5: the mask batch's CPU stages on worker processes.

The pipelined WSL batch (app.cli.mask_batch) decodes and hashes each
original's picture, encodes its clean copy, and judges and encodes its mask
on its workers' threads, where much of that work holds Python's lock (the
ThumbHash's pure-Python loops, the arrays' bookkeeping): ~1.6 of the box's
24 cores busy. ``ProcessStages`` runs those calls on ``workers`` processes
instead; ``INLINE`` runs them on the caller's thread, as before (one image
at a time, and ``--cpu-workers 0``).

- A CLEAN FORK. The workers are forked from multiprocessing's fork server, a
  fresh interpreter that imports the stage modules once (``PRELOAD``, or the
  caller's ``preload``), never from the driver, which holds the model, its
  threads and its database connections; they share the server's imported
  pages. Each worker also runs its parent's main module again (multiprocessing
  does), so a preload that covers that module's imports keeps each worker's
  own memory small (MASK5, 16 idle workers in the bench container: 976 MiB
  with no preload, 326 MiB with the stage modules, 194 MiB with the batch's
  command-line module).
- THE SAME ANSWER. A call is a module-level function of bytes and arrays
  (no session, no bucket, nothing kept between calls), so its answer is the
  one the caller's thread would get, and so is its exception, by class: one
  that cannot cross the process boundary is raised as a ``WorkerError``
  carrying its class name and message, never a broken pool.
- BOUNDED MEMORY. Each call takes units of a budget of ``BUDGET_UNITS``, one
  per ``UNIT_PIXELS`` of its picture begun (up to every unit: it then runs
  alone), in the order the calls came: the pixels decoded in the workers at
  once stay bounded whatever the pictures' sizes. The callers are the
  pipeline's threads, each waiting for its own call, so at most writers +
  prefetch calls queue, each holding only the bytes its thread holds anyway.
- THE OOM KILLER TAKES A WORKER. A cgroup OOM kills the process with the
  highest score, which would be the driver (it holds the model) and with it
  the batch; each worker raises its own score to the most
  (``WORKER_OOM_SCORE_ADJ``), so the kernel takes a worker instead.
- A WORKER THAT DIES (the OOM killer, a crash in a decoder) breaks the
  pool, and every call in flight on it fails; whose worker died is not
  told. The pool is replaced (each replacement reported: ``on_replaced``).
  A call that held part of the budget (others may have run beside it) is
  not charged for the break: it runs again alone, holding every unit,
  before any later call. A call that held every unit ran alone, so the
  break is its own: it is charged, and its ``ATTEMPTS``-th such break is
  raised (``BrokenProcessPool``). So a call that kills its worker spends
  its own tries, never another call's (the MASK5 challenge, SHOULD-2).
"""

from __future__ import annotations

import contextlib
import ctypes
import gc
import itertools
import multiprocessing
import os
import pickle
import signal
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import forkserver
from pathlib import Path
from typing import Any, TypeVar

from .mask_pipeline import Slots

__all__ = [
    "ATTEMPTS",
    "BUDGET_UNITS",
    "INLINE",
    "MAX_CPU_WORKERS",
    "UNIT_PIXELS",
    "BrokenProcessPool",
    "CpuStages",
    "ProcessStages",
    "WorkerError",
    "default_cpu_workers",
]

T = TypeVar("T")

#: A unit of the workers' pixel budget: a megapixel. Measured (MASK5, the
#: bench container): a worker's peak grows ~50 MiB a megapixel over a
#: picture's calls (0.7 MP +61 MiB, 2.2 MP +131, 12 MP +592, 24 MP +1184).
UNIT_PIXELS = 1_000_000
#: The budget, in units: the largest picture the image limits let in (64 MP,
#: app.workers.image_limits), which then runs alone; ~3.2 GB of the workers'
#: calls at once at most, whatever the workers.
BUDGET_UNITS = 64
#: The most worker processes (each ~35 MiB of its own between calls, MASK5).
MAX_CPU_WORKERS = 16
#: The CPUs left to the driver (the model's thread, its pre- and
#: post-processing) and the pipeline's own threads.
RESERVED_CPUS = 2
#: The breaks a call is charged before its BrokenProcessPool is raised: each
#: one while it held every unit, alone on the pool (a break while it held
#: part of the budget is not charged).
ATTEMPTS = 3
#: The modules whose functions the workers run, imported once by the fork
#: server (a caller adds its own main module's: ``ProcessStages(preload=)``).
PRELOAD = ("app.workers.mask_seed", "app.workers.mask")


#: A process's own OOM score adjustment, and a worker's (MASK5 challenge,
#: SHOULD-1): the most, so a cgroup OOM kills a worker, never the driver,
#: which holds the model and is otherwise the biggest process. Raising one's
#: own needs no privilege. Measured in a 2g container with the driver the
#: bigger (as in prod): without it the driver was killed (exit 137); with it
#: the worker was, the pool was replaced and the driver lived.
OOM_SCORE_ADJ_PATH = Path("/proc/self/oom_score_adj")
WORKER_OOM_SCORE_ADJ = 1000


#: The numeric libraries' own thread pools in a worker: one thread each (the
#: workers are the parallelism). Measured (MASK5): numpy's OpenBLAS starts a
#: thread per CPU in every process, so 16 workers on the 24-CPU box asked for
#: ~400 threads and hit the bench container's pids limit.
WORKER_THREAD_ENV = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def _start_fork_server() -> None:
    """The fork server, started now with ``WORKER_THREAD_ENV`` in its
    environment, which every worker forked from it inherits (this process's
    own environment as it was); one already running is left as it is."""
    saved = {name: os.environ.get(name) for name in WORKER_THREAD_ENV}
    os.environ.update(WORKER_THREAD_ENV)
    try:
        forkserver.ensure_running()
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def default_cpu_workers(writers: int, prefetch: int, *, cpus: int | None = None) -> int:
    """The workers for a run of ``writers`` (0: none, one image at a time
    runs its stages on its own thread): the CPUs this process may run on
    less ``RESERVED_CPUS``, no more than the calls that can be in flight at
    once (each worker's image and each original read ahead) or
    ``MAX_CPU_WORKERS``, and at least one."""
    if not writers:
        return 0
    usable = cpus if cpus is not None else len(os.sched_getaffinity(0))
    return max(1, min(MAX_CPU_WORKERS, usable - RESERVED_CPUS, writers + prefetch))


class WorkerError(Exception):
    """An exception a worker raised that could not cross the process
    boundary (its class takes other arguments than it keeps): its class
    name and message."""

    def __init__(self, name: str, message: str) -> None:
        super().__init__(f"{name}: {message}")
        self.name = name
        self.message = message

    def __reduce__(self) -> tuple[type[WorkerError], tuple[str, str]]:
        return type(self), (self.name, self.message)


def _worker_start() -> None:
    """A worker's start. A Ctrl-C reaches the terminal's whole process group,
    and the driver stops at its own pace (its images past the GPU finish),
    so no worker dies of it under a call. The objects it shares with the
    fork server are frozen out of its collections, which would otherwise
    walk them and so copy their pages (measured, MASK5: with each call's
    memory given back too, the 16 workers' memory peaked ~500 MiB lower
    and sat ~750 MiB lower idle on the bench). Its OOM score is raised to
    the most (``WORKER_OOM_SCORE_ADJ``); nothing where /proc has none."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    with contextlib.suppress(OSError):
        OOM_SCORE_ADJ_PATH.write_text(str(WORKER_OOM_SCORE_ADJ))
    gc.freeze()


#: glibc, once looked up (None: another C library, with no malloc_trim).
_LIBC: list[Any] = []


def _trim() -> None:
    """Give the memory a call freed back to the system: glibc keeps it
    otherwise (``malloc_trim``); nothing where there is no glibc."""
    if not _LIBC:
        try:
            _LIBC.append(ctypes.CDLL("libc.so.6"))
        except OSError:  # pragma: no cover - not glibc
            _LIBC.append(None)
    trim = getattr(_LIBC[0], "malloc_trim", None)
    if trim is not None:
        trim(0)


def _call(fn: Callable[..., T], args: tuple[Any, ...]) -> T:
    """``fn(*args)`` in a worker; its exception as it is, or by name when
    the caller could not unpickle it; then its freed memory given back."""
    try:
        return fn(*args)
    except Exception as exc:
        try:
            pickle.loads(pickle.dumps(exc))
        except Exception:
            raise WorkerError(type(exc).__name__, str(exc)) from None
        raise
    finally:
        _trim()


class CpuStages:
    """The CPU stages on the caller's own thread (``INLINE``)."""

    workers = 0

    def run(self, fn: Callable[..., T], *args: Any, pixels: int | None = None) -> T:
        """``fn(*args)``; ``pixels``, its picture's size, sizes its share of
        the workers' memory (none here)."""
        return fn(*args)

    def close(self) -> None:
        """Let the workers go (none here)."""


INLINE = CpuStages()


class ProcessStages(CpuStages):
    """The CPU stages on ``workers`` processes (module docstring)."""

    def __init__(
        self,
        workers: int,
        *,
        unit_pixels: int = UNIT_PIXELS,
        budget_units: int = BUDGET_UNITS,
        on_replaced: Callable[[str], None] | None = None,
        preload: Sequence[str] = PRELOAD,
        start: bool = True,
    ) -> None:
        if workers < 1:
            raise ValueError(f"a process pool needs a worker, got {workers}")
        self.workers = workers
        self._unit_pixels = unit_pixels
        self._budget = budget_units
        self._on_replaced = on_replaced
        self._preload = list(preload)
        self._units = Slots(budget_units)
        self._tickets = itertools.count()
        self._lock = threading.Lock()
        self.replaced = 0
        self._executor: ProcessPoolExecutor | None = self._start() if start else None

    def _start(self) -> ProcessPoolExecutor:
        context = multiprocessing.get_context("forkserver")
        context.set_forkserver_preload(self._preload)
        _start_fork_server()
        return ProcessPoolExecutor(self.workers, mp_context=context, initializer=_worker_start)

    def units(self, pixels: int | None) -> int:
        """The units a call on a picture of ``pixels`` takes: one per
        ``unit_pixels`` begun, up to the whole budget (it then runs alone);
        one when its size is unknown (it fails before its decode)."""
        if pixels is None:
            return 1
        return max(1, min(self._budget, -(-pixels // self._unit_pixels)))

    def run(self, fn: Callable[..., T], *args: Any, pixels: int | None = None) -> T:
        units = self.units(pixels)
        with self._lock:
            ticket = next(self._tickets)
        self._units.acquire(ticket, units)
        charged = 0
        try:
            while True:
                executor = self._executor
                assert executor is not None, "a pool made with start=False runs nothing"
                try:
                    return executor.submit(_call, fn, args).result()
                except BrokenProcessPool as exc:
                    self._replace(executor, exc)
                    if units < self._budget:
                        # others may have run beside it: not charged; again alone, its ticket first
                        self._units.release(units)
                        units = 0
                        self._units.acquire(ticket, self._budget)
                        units = self._budget
                        continue
                    charged += 1
                    if charged == ATTEMPTS:
                        raise
        finally:
            self._units.release(units)

    def _replace(self, broken: ProcessPoolExecutor, exc: BrokenProcessPool) -> None:
        """A new pool for ``broken``, once, whichever of its callers sees it first."""
        with self._lock:
            if self._executor is not broken:
                return
            broken.shutdown(wait=False, cancel_futures=True)
            self._executor = self._start()
            self.replaced += 1
        if self._on_replaced is not None:
            self._on_replaced(f"{type(exc).__name__}: {exc}")

    def close(self) -> None:
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
