"""One image's seconds by stage (MASK-PERF-1): where the mask batch's time goes.

The mask batch (app.cli.mask_batch) times every image's stages with its own
injectable clock and reports their means and medians per chunk. A stage is a
context manager around the code it times: it reads the clock on the way in
and out and lets every exception through untouched, so timing never changes
an answer, an order or a transaction.

A stage inside another counts only in the inner one (the outer keeps its own
seconds), so the stages of an image add up to the time spent in them, each
second once. The database's seconds are timed by SQLAlchemy's events instead
(DatabaseSeconds): every statement, commit and rollback, whichever stage runs
it, counted once under "db" by the same rule.

Pipelined (MASK-PERF-2), each image runs on a worker thread of its own with
a session of its own, and its stages are its own wall time: the waits the
pipeline adds are stages too ("claim_register", "gpu_wait", "turn_wait"), and what a
read-ahead thread measured for it is added (``StageTimings.add``).
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import TYPE_CHECKING, NamedTuple

from sqlalchemy import event

if TYPE_CHECKING:
    from sqlalchemy import Engine
    from sqlalchemy.orm import Session, SessionTransaction, sessionmaker

#: The stages, in the order a chunk reports them: the pipeline's order, the
#: database (reached all along the way) last.
STAGES = (
    "fetch",  # the original from the raw bucket, and its decode
    # pipelined, the claim (app.workers.mask_pipeline), told apart (MASKRATE):
    # the image's picture keys: its pixel digest read ahead (its decode) and,
    # for a picture with no copy yet, its clean copy encoded in its claim (MASK5),
    "claim_keys",
    # its footprint read but for the read's SQL ("db"): the ORM's own work,
    "claim_footprint",
    # and its wait to register behind every earlier image's footprint read
    "claim_register",
    # the clean copy minted and registered (app.workers.mask_seed), but for its
    # PUTs and its SQL: its decode, encode and phash, and the ORM's own work
    "seed",
    "readback",  # a media-bucket GET: the lane re-reading the copy it hashes
    "gpu_wait",  # pipelined: the model call queued behind other images'
    "inference",  # the mask model's answer (BiRefNet)
    "base_keep",  # the base-keep pass: the IS-Net donors and the merge
    "analysis",  # the copy's decode, grounding and appearances
    "encode",  # the mask PNG
    "upload",  # a media-bucket PUT: the original, its clean copy, the mask
    # pipelined: waiting for an earlier image of the same picture to end, and
    # for the turn to write the batch row (work-list order)
    "turn_wait",
    # every statement, commit and rollback (DatabaseSeconds), and the pool's
    # checkout and the session's rollback work around them
    "db",
)


class Mark(NamedTuple):
    """A moment on an image's clock, and where its stages stood then."""

    at: float
    entered: int  # stages entered so far
    depth: int  # stages open


class StageTimings:
    """The seconds of one image, per stage, by ``clock``."""

    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self._inner: list[float] = []  # per open stage: the seconds of the stages inside it
        self._entered = 0  # stages entered so far
        self.seconds: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = self._clock()
        self._entered += 1
        self._inner.append(0.0)
        try:
            yield
        finally:
            took = self._clock() - start
            own = took - self._inner.pop()
            self.seconds[name] = self.seconds.get(name, 0.0) + own
            if self._inner:
                self._inner[-1] += took

    def add(self, name: str, seconds: float) -> None:
        """``seconds`` measured elsewhere (a read-ahead thread) as ``name``:
        inside no stage of this image's, so carved out of none."""
        self.seconds[name] = self.seconds.get(name, 0.0) + seconds

    def carve(self, name: str, seconds: float) -> None:
        """``seconds`` of the open stage's spent as ``name`` elsewhere (MASK5:
        a CPU worker process's part of its call, the mask's encode): counted
        as ``name`` and carved out of that stage, as a nested stage is (the
        nesting rule); with no stage open, added alone."""
        self.seconds[name] = self.seconds.get(name, 0.0) + seconds
        if self._inner:
            self._inner[-1] += seconds

    def mark(self) -> Mark:
        """Now, for ``since``: the start of seconds two events bound rather
        than a block (DatabaseSeconds)."""
        return Mark(self._clock(), self._entered, len(self._inner))

    def since(self, name: str, mark: Mark) -> None:
        """The seconds from ``mark`` to now as ``name``, inside the stage open
        at both ends: counted there once, under ``name`` (the nesting rule).
        Across a stage's start or end they are already that stage's, and
        stay so."""
        if (mark.entered, mark.depth) != (self._entered, len(self._inner)):
            return
        took = self._clock() - mark.at
        self.seconds[name] = self.seconds.get(name, 0.0) + took
        if self._inner:
            self._inner[-1] += took


def timed(timings: StageTimings | None, name: str) -> AbstractContextManager[None]:
    """``timings.stage(name)``, or nothing at all when nobody is timing."""
    return nullcontext() if timings is None else timings.stage(name)


class _Charge:
    """One thread's image and what of its database work is open."""

    def __init__(self) -> None:
        self.timings: StageTimings | None = None
        self.statement: Mark | None = None  # a statement, or a connection's rollback
        self.by_cursor = False  # that statement has no before_execute: its cursor run ends it
        self.commit: Mark | None = None  # the session's commit, its statements inside


class DatabaseSeconds:
    """Every second one session spends in its database as "db", wherever its
    SQL runs: each statement (from its execute to its result, all its cursor
    runs inside), each commit (from the session's commit, its flush
    included, to the transaction's end) and each rollback, carved out of
    whichever stage is open (the nesting rule), for the image ``charging``
    names; nothing between images.

    It listens on the session and its engine from construction to ``close``
    and only reads the clock: no statement, result, transaction or error
    changes. A statement is timed from ``before_execute`` and a commit from
    ``before_commit``, so a listener another added earlier runs inside them
    (one executed straight on the driver, which has no ``before_execute``,
    from its cursor run). What no event spans stays in the stage around it:
    the pool's checkout (its pre-ping is a SELECT 1 straight on the DBAPI
    cursor) and a session rollback's own work (it expires all it holds). The
    batch's "db" stages around a transaction's first query and around each
    rollback hold those.

    Pipelined (MASK-PERF-2), ``sessions`` makes each worker's session: its
    sessions are listened to as well, and each thread charges its own image
    (an image's SQL all runs on its worker's thread), so images timed at once
    are each charged their own statements and commits, never another's.
    """

    def __init__(self, session: Session, sessions: sessionmaker[Session] | None = None) -> None:
        engine = session.get_bind().engine
        self._local = threading.local()
        per_session: list[tuple[str, Callable[..., None]]] = [
            ("before_commit", self._committing),
            ("after_rollback", self._ended),
            ("after_transaction_end", self._transaction_ended),
        ]
        targets: list[Session | sessionmaker[Session]] = [session]
        if sessions is not None:
            targets.append(sessions)
        self._hooks: list[
            tuple[Session | sessionmaker[Session] | Engine, str, Callable[..., None]]
        ] = [
            *((target, name, fn) for target in targets for name, fn in per_session),
            (engine, "before_execute", self._starting),
            (engine, "after_execute", self._answered),
            (engine, "before_cursor_execute", self._cursor_starting),
            (engine, "after_cursor_execute", self._cursor_answered),
            (engine, "rollback", self._starting),
            (engine, "handle_error", self._ended),
        ]
        for target, name, fn in self._hooks:
            event.listen(target, name, fn)

    def _charge(self) -> _Charge:
        charge: _Charge | None = getattr(self._local, "charge", None)
        if charge is None:
            charge = self._local.charge = _Charge()
        return charge

    @contextmanager
    def charging(self, timings: StageTimings) -> Iterator[None]:
        """The database's seconds go to ``timings`` (one image's) meanwhile,
        for the SQL this thread runs."""
        charge = self._charge()
        charge.timings = timings
        try:
            yield
        finally:
            charge.timings = None
            charge.statement = None
            charge.by_cursor = False
            charge.commit = None

    def close(self) -> None:
        """Stop listening: the session and its engine as they were."""
        for target, name, fn in self._hooks:
            event.remove(target, name, fn)

    def _starting(self, *_: object) -> None:
        # a commit open already spans its statements and its flush's rollback
        charge = self._charge()
        if charge.timings is not None and charge.commit is None:
            charge.statement = charge.timings.mark()
            charge.by_cursor = False

    def _cursor_starting(self, *_: object) -> None:
        charge = self._charge()
        if charge.timings is not None and charge.commit is None and charge.statement is None:
            charge.statement = charge.timings.mark()
            charge.by_cursor = True

    def _cursor_answered(self, *_: object) -> None:
        if self._charge().by_cursor:
            self._answered()

    def _answered(self, *_: object) -> None:
        charge = self._charge()
        if charge.timings is not None and charge.statement is not None:
            charge.timings.since("db", charge.statement)
        charge.statement = None
        charge.by_cursor = False

    def _committing(self, *_: object) -> None:
        charge = self._charge()
        charge.statement = None
        if charge.timings is not None:
            charge.commit = charge.timings.mark()

    def _ended(self, *_: object) -> None:
        # the transaction's end or an error: whatever was open is done
        self._answered()
        charge = self._charge()
        if charge.timings is not None and charge.commit is not None:
            charge.timings.since("db", charge.commit)
        charge.commit = None

    def _transaction_ended(self, session: Session, transaction: SessionTransaction) -> None:
        if transaction.parent is None:  # a flush's own ends inside its commit
            self._ended()
