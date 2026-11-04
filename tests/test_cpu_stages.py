"""MASK5: the mask batch's CPU stages on worker processes.

The batch's decodes, encodes and the mask's judgment hold Python's lock for
much of their time, so on the workers' threads the WSL batch used ~1.6 of 24
cores. ``ProcessStages`` runs them on ``workers`` processes (forked from a
clean fork server: never from the driver, which holds the model, its threads
and its database connections); ``INLINE`` runs them on the caller's thread,
as before. A call's answer and its exception (by class) are the same either
way. Each call takes units of a pixel budget by the size of its picture, so
the decoded pixels in the workers at once are bounded; a worker that dies
(the kernel's OOM killer) is replaced and the call tried again.
"""

from __future__ import annotations

import os
import pickle
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from app.workers import cpu_stages as cs
from app.workers.image_limits import ImageLimitError


def _pid() -> int:
    return os.getpid()


def _square(n: int) -> int:
    return n * n


def _refuse(code: str) -> None:
    raise ImageLimitError(code, f"refused: {code}")


class _Unpicklable(Exception):
    def __init__(self, a: str, b: str) -> None:
        super().__init__(f"{a}/{b}")


def _raise_unpicklable() -> None:
    raise _Unpicklable("x", "y")


def _die_once(flag: str) -> int:
    """Kill this worker process the first time (the OOM killer's SIGKILL)."""
    path = Path(flag)
    if not path.exists():
        path.write_text("died")
        os._exit(137)
    return os.getpid()


def _always_die() -> None:
    os._exit(137)


def _hold(seconds: float) -> float:
    started = time.monotonic()
    time.sleep(seconds)
    return started


@pytest.fixture(scope="module")
def two() -> Iterator[cs.ProcessStages]:
    stages = cs.ProcessStages(2)
    yield stages
    stages.close()


def test_inline_runs_the_call_on_the_callers_thread() -> None:
    assert cs.INLINE.workers == 0
    assert cs.INLINE.run(_pid) == os.getpid()
    assert cs.INLINE.run(_square, 7, pixels=10**9) == 49
    cs.INLINE.close()  # nothing to close


def test_a_process_pool_runs_the_call_in_a_worker_process(two: cs.ProcessStages) -> None:
    assert two.workers == 2
    assert two.run(_pid) != os.getpid()
    assert two.run(_square, 12) == 144


def test_a_workers_exception_is_raised_with_its_class_and_fields(two: cs.ProcessStages) -> None:
    with pytest.raises(ImageLimitError) as raised:
        two.run(_refuse, "too_many_pixels")
    assert (raised.value.code, str(raised.value)) == ("too_many_pixels", "refused: too_many_pixels")


def test_the_image_limits_error_survives_pickling() -> None:
    copy = pickle.loads(pickle.dumps(ImageLimitError("decoded_too_large", "too big")))
    assert (type(copy), copy.code, str(copy)) == (ImageLimitError, "decoded_too_large", "too big")


def test_an_exception_that_cannot_cross_processes_is_raised_by_name(
    two: cs.ProcessStages,
) -> None:
    """Never a broken pool: the class name and message still say what failed."""
    with pytest.raises(cs.WorkerError, match=r"^_Unpicklable: x/y$") as raised:
        two.run(_raise_unpicklable)
    assert raised.value.name == "_Unpicklable"
    assert two.run(_square, 3) == 9  # the pool still serves


def test_a_worker_killed_mid_call_is_replaced_and_the_call_tried_again(tmp_path: Path) -> None:
    replaced: list[str] = []
    stages = cs.ProcessStages(1, on_replaced=replaced.append)
    try:
        first = stages.run(_pid)
        again = stages.run(_die_once, str(tmp_path / "flag"))
        assert again != first  # a new worker answered
        assert stages.replaced == 1
        assert replaced and "BrokenProcessPool" in replaced[0]
        assert stages.run(_square, 4) == 16
    finally:
        stages.close()


def test_a_call_that_kills_every_worker_is_given_up_after_its_attempts() -> None:
    stages = cs.ProcessStages(1)
    try:
        with pytest.raises(cs.BrokenProcessPool):
            stages.run(_always_die)
        # a unit of the budget: its first break is not charged, its tries alone are
        assert stages.replaced == 1 + cs.ATTEMPTS
        assert stages.run(_square, 5) == 25  # the next call has a working pool
    finally:
        stages.close()


def _die_after(seconds: float) -> None:
    time.sleep(seconds)
    os._exit(137)


def test_a_call_that_kills_its_worker_spends_its_own_tries_never_another_calls() -> None:
    """The MASK5 challenge (SHOULD-2): a dead worker fails every call in
    flight on its pool, and whose worker died is not told. A call broken
    beside others is not charged for it: it runs again alone (every unit),
    in the order the calls came. A call broken alone is charged; the third
    time its BrokenProcessPool is raised. So the innocent calls answer, and
    the one that kills its worker fails alone."""
    stages = cs.ProcessStages(4)
    try:
        results: dict[str, tuple[str, object]] = {}

        def call(name: str, fn: Callable[..., object], *args: object) -> None:
            try:
                results[name] = ("ok", stages.run(fn, *args))
            except Exception as exc:
                results[name] = ("raised", type(exc).__name__)

        threads = [
            threading.Thread(target=call, args=(f"innocent{n}", _hold, 0.6)) for n in range(3)
        ]
        for thread in threads:
            thread.start()
        time.sleep(0.15)  # the three in flight
        threads.append(threading.Thread(target=call, args=("poison", _die_after, 0.15)))
        threads[-1].start()
        for thread in threads:
            thread.join(timeout=60)
        assert [results[f"innocent{n}"][0] for n in range(3)] == ["ok", "ok", "ok"]
        assert results["poison"] == ("raised", "BrokenProcessPool")
        # the break beside the innocent calls, then its own tries alone
        assert stages.replaced == 1 + cs.ATTEMPTS
        assert stages.run(_square, 6) == 36  # the pool serves on
    finally:
        stages.close()


def test_a_call_that_takes_every_unit_is_alone_so_each_break_is_its_own() -> None:
    stages = cs.ProcessStages(1)
    try:
        with pytest.raises(cs.BrokenProcessPool):
            stages.run(_always_die, pixels=10**9)
        assert stages.replaced == cs.ATTEMPTS
    finally:
        stages.close()


def test_the_units_a_call_takes_are_its_megapixels_up_to_the_whole_budget() -> None:
    """Measured (MASK5): a worker's peak grows ~50 MiB a megapixel over its
    three calls of a picture (0.7 MP +61 MiB, 2.2 MP +131, 12 MP +592, 24 MP
    +1184). A call takes a unit per megapixel begun of a budget of 64, the
    largest picture the limits let in (it then runs alone): ~3.2 GB of the
    workers' calls at once at most, whatever the workers."""
    stages = cs.ProcessStages(16, start=False)
    assert (cs.UNIT_PIXELS, cs.BUDGET_UNITS) == (1_000_000, 64)
    assert stages.units(None) == 1  # unknown: it fails before its decode
    assert stages.units(1) == stages.units(1_000_000) == 1
    assert stages.units(2_160_000) == 3  # the anitoys median: sixteen at once
    assert stages.units(12_000_000) == 12  # five at once
    assert stages.units(64_000_000) == stages.units(10**9) == 64  # alone


def test_a_big_picture_runs_alone_and_small_ones_share_the_workers() -> None:
    """The decoded pixels in the workers at once are bounded: a call of
    every unit waits for the others to end and holds back the later ones."""
    stages = cs.ProcessStages(2, unit_pixels=100, budget_units=2)
    try:
        started: dict[str, float] = {}

        def call(name: str, pixels: int, seconds: float) -> None:
            started[name] = stages.run(_hold, seconds, pixels=pixels)

        stages.run(_square, 1)  # both workers up
        stages.run(_square, 2)
        threads = [threading.Thread(target=call, args=("small-a", 100, 0.6))]
        threads[0].start()
        time.sleep(0.2)
        for name, pixels in (("big", 200), ("small-b", 100)):
            threads.append(threading.Thread(target=call, args=(name, pixels, 0.3)))
            threads[-1].start()
            time.sleep(0.05)
        for thread in threads:
            thread.join(timeout=20)
        assert started["big"] >= started["small-a"] + 0.5  # waited for small-a to end
        assert started["small-b"] >= started["big"] + 0.25  # held back behind the big one
    finally:
        stages.close()


@pytest.mark.parametrize(
    ("writers", "prefetch", "cpus", "expected"),
    [
        (16, 2, 24, 16),  # the WSL box: bounded by MAX_CPU_WORKERS
        (4, 2, 24, 6),  # no more than the calls that can be in flight at once
        (16, 2, 8, 6),  # the host's CPUs less two (the driver and its threads)
        (16, 2, 2, 1),  # at least one
        (0, 2, 24, 0),  # one image at a time: its CPU stages on its own thread
    ],
)
def test_the_default_cpu_workers_are_sized_to_the_host_and_bounded(
    writers: int, prefetch: int, cpus: int, expected: int
) -> None:
    assert cs.MAX_CPU_WORKERS == 16
    assert cs.default_cpu_workers(writers, prefetch, cpus=cpus) == expected


def test_the_default_reads_the_cpus_this_process_may_run_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cs.os, "sched_getaffinity", lambda _pid: set(range(5)))
    assert cs.default_cpu_workers(16, 2) == 3


def _interrupt_handler() -> str:
    import signal

    return repr(signal.getsignal(signal.SIGINT))


def test_a_worker_ignores_an_interrupt_the_driver_decides_when_to_stop(
    two: cs.ProcessStages,
) -> None:
    """A Ctrl-C reaches every process of the terminal's group: the driver
    stops at its own pace (its images past the GPU finish), so a worker
    must not die of it under a call."""
    import signal

    assert two.run(_interrupt_handler) == repr(signal.SIG_IGN)


def _oom_score_adj() -> str:
    return Path("/proc/self/oom_score_adj").read_text().strip()


@pytest.mark.skipif(not Path("/proc/self/oom_score_adj").exists(), reason="Linux's /proc")
def test_a_worker_is_the_oom_killers_first_choice_and_the_driver_keeps_its_own(
    two: cs.ProcessStages,
) -> None:
    """The MASK5 challenge (SHOULD-1): a cgroup OOM kills the process with
    the highest score, which was the driver (it holds the model; 8.26 GB
    once) and so the whole batch (exit 137). A worker raises its own score
    to the most (no privilege needed), so the kernel takes a worker, the
    pool replaces it and the driver lives; the driver's own is untouched."""
    own = _oom_score_adj()
    assert two.run(_oom_score_adj) == "1000"
    assert _oom_score_adj() == own


def _threads() -> tuple[str | None, int]:
    from pathlib import Path

    status = Path("/proc/self/status").read_text()
    threads = next(
        int(line.split()[1]) for line in status.splitlines() if line.startswith("Threads:")
    )
    return os.environ.get("OPENBLAS_NUM_THREADS"), threads


def test_a_worker_runs_its_numeric_libraries_on_one_thread(two: cs.ProcessStages) -> None:
    """Measured in the bench container (MASK5): numpy's OpenBLAS starts a
    thread per CPU in every process, so 16 workers on the 24-CPU box asked
    for ~400 threads and hit the container's pids limit. The workers ARE
    the parallelism: each runs its numeric libraries on one thread."""
    blas, threads = two.run(_threads)
    assert blas == "1"
    assert threads <= 2


def test_starting_the_fork_server_leaves_this_processs_environment_as_it_was(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    cs._start_fork_server()
    assert os.environ["OMP_NUM_THREADS"] == "8"
    assert "OPENBLAS_NUM_THREADS" not in os.environ and "MKL_NUM_THREADS" not in os.environ


def _frozen() -> tuple[int, int]:
    import gc

    return gc.get_freeze_count(), len(gc.get_objects())


def test_a_worker_never_collects_the_objects_it_shares_with_the_fork_server(
    two: cs.ProcessStages,
) -> None:
    """Measured (MASK5): a collection walks every object the worker shares
    with the fork server and so copies its pages; frozen at the worker's
    start, the 16 workers' memory peaked ~350 MiB lower on the bench."""
    frozen, collected = two.run(_frozen)
    assert frozen > 10 * collected  # the fork server's objects, out of every collection


def test_each_call_gives_its_freed_memory_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whether it answers or raises, a call ends with the worker's freed
    memory given back (glibc keeps it otherwise: measured, ~400 MiB more
    idle across 16 workers after 600 photos)."""
    trims: list[str] = []
    monkeypatch.setattr(cs, "_trim", lambda: trims.append("trim"))
    assert cs._call(_square, (6,)) == 36
    with pytest.raises(ImageLimitError):
        cs._call(_refuse, ("too_many_pixels",))
    assert trims == ["trim", "trim"]


def test_the_trim_is_glibcs_or_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    class _Glibc:
        def malloc_trim(self, pad: int) -> int:
            calls.append(pad)
            return 1

    monkeypatch.setattr(cs, "_LIBC", [_Glibc()])
    cs._trim()
    assert calls == [0]
    monkeypatch.setattr(cs, "_LIBC", [None])  # no glibc here (musl, macOS)
    cs._trim()
    assert calls == [0]
    monkeypatch.setattr(cs, "_LIBC", [])  # not looked up yet: looked up once, then kept
    cs._trim()
    assert len(cs._LIBC) == 1


# The paths below run in a worker in production, where coverage does not
# follow them; each is also exercised here, in this process.


def test_a_worker_error_survives_pickling() -> None:
    error = pickle.loads(pickle.dumps(cs.WorkerError("_Unpicklable", "x/y")))
    assert (type(error), error.name, error.message, str(error)) == (
        cs.WorkerError,
        "_Unpicklable",
        "x/y",
        "_Unpicklable: x/y",
    )


def test_a_call_whose_exception_cannot_cross_raises_it_by_name() -> None:
    with pytest.raises(cs.WorkerError, match=r"^_Unpicklable: x/y$"):
        cs._call(_raise_unpicklable, ())


def test_a_workers_start_ignores_interrupts_and_freezes_what_it_shares(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    done: list[object] = []
    monkeypatch.setattr(cs.signal, "signal", lambda sig, handler: done.append((sig, handler)))
    monkeypatch.setattr(cs.gc, "freeze", lambda: done.append("freeze"))
    adj = tmp_path / "oom_score_adj"  # never this test process's own
    adj.write_text("0")
    monkeypatch.setattr(cs, "OOM_SCORE_ADJ_PATH", adj)
    cs._worker_start()
    assert done == [(cs.signal.SIGINT, cs.signal.SIG_IGN), "freeze"]
    assert adj.read_text() == "1000"


def test_a_workers_start_goes_on_where_it_cannot_set_its_oom_score(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cs.signal, "signal", lambda sig, handler: None)
    monkeypatch.setattr(cs.gc, "freeze", lambda: None)
    monkeypatch.setattr(cs, "OOM_SCORE_ADJ_PATH", tmp_path / "no-proc" / "oom_score_adj")
    cs._worker_start()  # no /proc: nothing raised


def test_a_pool_needs_a_worker() -> None:
    with pytest.raises(ValueError, match="needs a worker, got 0"):
        cs.ProcessStages(0, start=False)


def test_a_pool_another_caller_already_replaced_is_left_as_it_is() -> None:
    """Two callers see the same broken pool: the first replaces it, the
    second finds a new one in its place and leaves it."""
    told: list[str] = []
    stages = cs.ProcessStages(1, start=False, on_replaced=told.append)
    current = object()
    stages._executor = current  # type: ignore[assignment]
    stages._replace(object(), cs.BrokenProcessPool("gone"))  # type: ignore[arg-type]
    assert (stages._executor, stages.replaced, told) == (current, 0, [])


def test_closing_a_pool_twice_lets_its_workers_go_once() -> None:
    stages = cs.ProcessStages(1)
    assert stages.run(_square, 3) == 9
    stages.close()
    stages.close()
    assert stages._executor is None
