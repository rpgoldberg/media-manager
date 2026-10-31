"""The media.v1 replay: the media DB's clean copies and masks, reported to the
spine over SpineMedia (``python -m app.cli.mask_report``; MM-REPORT-1,
fc-aggregation DEPLOY.md section 4 (a)).

THE PLAN, read from the media DB. Every ORIGINAL behind a LIVE clean copy
(``sanitized`` public object; live as app.publication.images_behind judges
it: its version and every image behind it not soft-deleted), mapped to one
copy: the copy minted from that image, else the lowest-addressed copy it is
linked behind. Every mask row with bytes, judged per MASK BYTES, as the
origin serves them: PUBLISHED when some ``ok`` row of a live copy carries
them; otherwise WITHDRAWN -- a ``withdrawn`` row with its reason and instant,
or an ``ok`` row of a dead copy (reason ``deleted``, as
app.publication.mask_refusal says). ``held`` and other rows are never sent.
A published row whose mask bytes have no ``mask`` public object is never
reported (the origin would answer 404 for it): it is refused locally by name
(``mask_object``).
Store eligibility is NOT re-judged here: run ``masks reevaluate --apply``
first, as for the origin.

THE INDEX AND THE DETAILS. The run holds only the plan's index for its whole
length (each original's copy, the live copies, the published mask bytes, the
masked copies), read in streamed columns, and reads each batch's copies (their
records and mask rows) when the batch comes: its memory is the index plus one
batch, never the whole media DB (the whole plan measured 2.37 GB at 282k
originals, GKPH-S3-2). The mask batch may write while a run goes: a mask row
written after the index was read waits for the next run.

PER BATCH of at most 200 originals, in sha order:
  1. LeasePendingImages{origin_sha256: the batch}, repeated until the batch's
     pending depictions are drained. Every returned origin_sha256 is checked
     against the batch. Anything else means the spine is not honouring the
     filter (a server before 0.10.0 ignores it): a DEPLOYMENT FAULT. The
     batch's own depictions in that answer are reported, the strays are left
     to lapse, and the run STOPS (exit 3) rather than walk the shared queue.
  2. ReportDerivative for each depiction: the clean copy, its mime and size,
     the clean pipeline's version, its instant, and its own appearance
     (thumbhash, dominant colour; a grounding only for a transparent copy).
  3. ReportMask for each published mask row of the copies whose LAST
     original is in this batch (so their derivatives have been reported).
     FAILED_PRECONDITION (no derivative registered for the copy): when this
     run's report STORED that derivative it is retried, and if it persists
     the mask is ``unregistered`` (exit 1); when this run reported it and the
     spine kept another derivative for each of those originals (stored=false:
     its registry is write-once per original, fc-aggregation
     derivative-writer.ts, so a copy minted after an original was derived
     never replaces the one it shows) it is ``kept_other``: counted, never
     retried, never a failure; when no original of the copy had a depiction
     to lease (the spine never captured it, it is suppressed, or it is
     already derived) it is an ``orphan``: expected, counted, the originals
     named in the batch line, never a failure.
  4. A restore (WithdrawMask restore=true) for each published mask whose
     withdrawal this replay once sent (``spine_mask_withdrawals``).
  5. WithdrawMask for each withdrawn mask, recorded in
     ``spine_mask_withdrawals`` BEFORE the call.
A last TAIL batch carries the copies no original planned (dead copies): their
withdrawals only.

IDEMPOTENT AND RESUMABLE. Every call is retry-safe by contract, so a rerun
stores nothing new. The run ledger is ``spine_report_runs``: the cursor is
the last original of the last finished batch. Before EVERY lease call
(each round of a batch), ``resume_not_before`` is set and committed to when
that call's hold will have lapsed, whichever attempt inside it takes it: the
lease seconds asked for, a margin, and the call's worst case (every attempt
at its deadline and every backoff). Once the call has answered or failed it
is set again, from that instant. It is cleared when the batch finishes. A run
that stops mid-batch -- a fatal status, a spine that stays down, a lease with
a failed attempt (which may hold rows a retry cannot see), or a killed pod --
leaves it set, and the rerun waits for it before it
leases that batch again: a held depiction is invisible to a lease, and
reading that as "nothing pending" would skip it.

Exit status: 0 done; 1 done, but not everything landed (a spine refusal, a
value the field rule would not send, a mask with no public object, or an
``unregistered`` mask); 3 stopped, resumable (rerun); 2 is the CLI's (bad
arguments). Exit 1 is read from the run's PERSISTED counts, so a resumed run
reports what an earlier attempt of it could not land. Orphans never make it 1.
"""

from __future__ import annotations

import datetime as dt
import socket
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import grpc
from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session, aliased

from ..models import Image as ImageModel
from ..models import (
    ImageMask,
    ImageVersion,
    PublicObject,
    PublicObjectImage,
    SpineMaskWithdrawal,
    SpineReportRun,
)
from ..workers.grounding import AlphaGrounding
from ..workers.mask import MASK_PIPELINE_VERSION
from ..workers.mask_seed import CLEAN_PIPELINE_VERSION
from . import messages as msg
from .field_rules import MAX_ORIGIN_FILTER, FieldRefused
from .media_client import DEFAULT_LEASE_SECONDS, SpineCallError, SpineMediaClient, Verdict

EXIT_DONE = 0
EXIT_REFUSED = 1
EXIT_STOPPED = 3

#: What one lease asks for; the contract's cap.
LEASE_MAX_ITEMS = 200
#: A lease is repeated at most this many times per batch.
MAX_LEASE_ROUNDS = 50
#: Seconds added to the lease's own when a stopped batch is resumed.
LEASE_MARGIN_SECONDS = 5
#: Copies whose details one query reads (an IN list of their shas).
DETAIL_CHUNK = 500
#: Rows the index fetches at a time (a server-side cursor on Postgres).
INDEX_FETCH = 10_000
#: Warnings quoted per batch line (all are counted).
WARNINGS_QUOTED = 5
#: The reason sent for a withdrawn row that records none: the spine refuses
#: a withdrawal without one, and the withdrawal must still reach it.
NO_REASON = "withdrawn (no reason recorded)"

GROUNDING_FIELDS = tuple(f.name for f in AlphaGrounding.__dataclass_fields__.values())

__all__ = ["CLEAN_PIPELINE_VERSION", "ReplayConfig", "load_plan", "make_batches", "run_replay"]


# --------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Withdrawal:
    mask_sha256: str
    reason: str | None
    #: None = withdrawn by the plan itself (a dead copy's ok row): the run's start.
    withdrawn_at: dt.datetime | None


@dataclass
class Plan:
    """The INDEX, held for the whole run (``copy_of``, ``live``,
    ``published``, ``masked``), and the DETAILS of the copies last
    :meth:`load`-ed (``copies``, ``reports``, ``withdrawals``,
    ``objectless``): the run loads one batch's copies at a time, so its
    memory is the index plus one batch, whatever the size of the media DB."""

    #: origin sha -> the clean copy reported for its depictions (live copies only)
    copy_of: dict[str, str]
    #: the live clean copies' records (loaded copies only)
    copies: dict[str, msg.CopyRecord]
    live: set[str]
    #: mask bytes some ok row of a live copy carries
    published: set[str]
    reports: dict[str, list[msg.MaskRecord]] = field(default_factory=dict)
    withdrawals: dict[str, list[Withdrawal]] = field(default_factory=dict)
    #: every copy with a mask row (live or not)
    masked: set[str] = field(default_factory=set)
    #: copy -> published mask bytes with no ``mask`` public object (never sent)
    objectless: dict[str, list[str]] = field(default_factory=dict)
    #: the newest mask row the index read: a row written after it (the mask
    #: batch runs alongside) waits for the next run, as it did when the whole
    #: plan was read at once, so a withdrawal is never judged against a
    #: ``published`` that has not seen the row
    last_mask_id: int = 0
    _originals: dict[str, list[str]] | None = field(default=None, init=False, repr=False)

    def originals_of(self, copy: str) -> list[str]:
        """The planned originals whose depictions report ``copy``."""
        if self._originals is None:
            inverse: dict[str, list[str]] = {}
            for origin, mapped in self.copy_of.items():
                inverse.setdefault(mapped, []).append(origin)
            self._originals = inverse
        return self._originals.get(copy, [])

    def reports_for(self, copy: str) -> list[msg.MaskRecord]:
        return self.reports.get(copy, [])

    def withdrawals_for(self, copy: str) -> list[Withdrawal]:
        return self.withdrawals.get(copy, [])

    def load(self, db: Session, copies: Iterable[str]) -> None:
        """Read the details of ``copies`` (read only), replacing the last
        ones: their records when live, their reports, withdrawals and
        objectless masks."""
        self.copies, self.reports, self.withdrawals, self.objectless = {}, {}, {}, {}
        wanted = sorted(set(copies))
        for i in range(0, len(wanted), DETAIL_CHUNK):
            self._load_chunk(db, wanted[i : i + DETAIL_CHUNK])

    def _load_chunk(self, db: Session, chunk: list[str]) -> None:
        mask_rows = list(
            db.execute(
                select(ImageMask)
                .where(ImageMask.source_sha256.in_(chunk), ImageMask.id <= self.last_mask_id)
                .order_by(ImageMask.id)
            ).scalars()
        )
        by_copy: dict[str, list[ImageMask]] = {}
        for row in mask_rows:
            by_copy.setdefault(row.source_sha256, []).append(row)
        mask_shas = sorted({r.mask_sha256 for r in mask_rows if r.mask_sha256})
        mask_objects = {
            o.sha256: o
            for o in db.execute(
                select(PublicObject).where(
                    PublicObject.kind == "mask", PublicObject.sha256.in_(mask_shas)
                )
            ).scalars()
        }
        for obj in db.execute(
            select(PublicObject).where(
                PublicObject.kind == "sanitized", PublicObject.sha256.in_(chunk)
            )
        ).scalars():
            if obj.sha256 not in self.live:
                continue
            look = _appearance(by_copy.get(obj.sha256, []))
            self.copies[obj.sha256] = msg.CopyRecord(
                sha256=obj.sha256,
                mime=obj.mime,
                width=obj.width,
                height=obj.height,
                generated_at=obj.created_at,
                thumbhash=look.plain_thumbhash if look else None,
                dominant_color=look.plain_dominant_color if look else None,
                own_grounding=look.own_alpha_grounding if look else None,
            )
        for row in mask_rows:
            if not row.mask_sha256 or row.status not in ("ok", "withdrawn"):
                continue
            if row.status == "ok" and row.source_sha256 in self.live:
                mask_obj = mask_objects.get(row.mask_sha256)
                if mask_obj is None:
                    self.objectless.setdefault(row.source_sha256, []).append(row.mask_sha256)
                    continue
                self.reports.setdefault(row.source_sha256, []).append(
                    msg.MaskRecord(
                        clean_sha256=row.source_sha256,
                        pipeline_version=row.pipeline_version,
                        mask_sha256=row.mask_sha256,
                        mime=mask_obj.mime,
                        generated_at=mask_obj.created_at,
                        grounding={name: getattr(row, name) for name in GROUNDING_FIELDS},
                        thumbhash=row.masked_thumbhash,
                        dominant_color=row.masked_dominant_color,
                    )
                )
            elif row.mask_sha256 not in self.published:
                if row.status == "withdrawn":
                    withdrawn = Withdrawal(row.mask_sha256, row.reason or NO_REASON, row.updated_at)
                else:
                    withdrawn = Withdrawal(row.mask_sha256, "deleted", None)
                self.withdrawals.setdefault(row.source_sha256, []).append(withdrawn)


def _dead_copies(db: Session) -> set[str]:
    """Clean copies app.publication.images_behind calls not live, in one query:
    no version, a deleted version, or a deleted image behind it. Each EXISTS
    selects a named column, never ``*``: the Job's role holds column grants
    on ``images`` and Postgres checks every column a query names."""
    owner = aliased(ImageModel)
    linked = aliased(ImageModel)
    query = (
        select(PublicObject.sha256)
        .outerjoin(ImageVersion, ImageVersion.id == PublicObject.image_version_id)
        .where(
            PublicObject.kind == "sanitized",
            or_(
                ImageVersion.id.is_(None),
                ImageVersion.deleted_at.is_not(None),
                select(owner.id)
                .where(owner.id == ImageVersion.image_id, owner.deleted_at.is_not(None))
                .exists(),
                select(linked.id)
                .where(
                    PublicObjectImage.sha256 == PublicObject.sha256,
                    linked.id == PublicObjectImage.image_id,
                    linked.deleted_at.is_not(None),
                )
                .exists(),
            ),
        )
    )
    return set(db.execute(query).scalars())


def _appearance(rows: list[ImageMask]) -> ImageMask | None:
    """The row whose plain appearance describes the copy: the current mask
    version's when it has one, else the latest measured."""
    measured = [r for r in rows if r.plain_thumbhash or r.plain_dominant_color]
    if not measured:
        return None
    current = [r for r in measured if r.pipeline_version == MASK_PIPELINE_VERSION]
    return (current or sorted(measured, key=lambda r: (r.updated_at, r.id)))[-1]


def load_plan(db: Session, *, details: bool = True) -> Plan:
    """Everything the replay sends, read from the media DB (read only).
    ``details=False``: the index only; :meth:`Plan.load` reads a batch's
    copies when the batch comes (the run's way)."""
    dead = _dead_copies(db)
    # Columns, never entities, streamed: the index reads one row per clean
    # copy, per original and per mask row of the whole media DB.
    live: set[str] = set()
    #: image -> the lowest live copy minted from it
    owned: dict[int, str] = {}
    for copy, image_id in db.execute(
        select(PublicObject.sha256, ImageVersion.image_id)
        .join(ImageVersion, ImageVersion.id == PublicObject.image_version_id)
        .where(PublicObject.kind == "sanitized")
        .execution_options(yield_per=INDEX_FETCH)
    ):
        if copy in dead:
            continue
        live.add(copy)
        if image_id not in owned or copy < owned[image_id]:
            owned[image_id] = copy
    #: image -> the lowest live copy it is linked behind (only the lowest is
    #: ever reported, when the image minted none)
    behind: dict[int, str] = {}
    for copy, image_id in db.execute(
        select(PublicObjectImage.sha256, PublicObjectImage.image_id)
        .join(
            PublicObject,
            and_(PublicObject.sha256 == PublicObjectImage.sha256, PublicObject.kind == "sanitized"),
        )
        .execution_options(yield_per=INDEX_FETCH)
    ):
        if copy in live and (image_id not in behind or copy < behind[image_id]):
            behind[image_id] = copy
    # Every image's sha, filtered here: an IN list of every id behind a copy
    # is one bound value per original.
    copy_of: dict[str, str] = {}
    for image_id, origin in db.execute(
        select(ImageModel.id, ImageModel.sha256).execution_options(yield_per=INDEX_FETCH)
    ):
        mapped = owned.get(image_id) or behind.get(image_id)
        if mapped:
            copy_of[origin] = mapped
    del behind, owned

    published: set[str] = set()
    masked: set[str] = set()
    last_mask_id = 0
    for mask_id, source, status, mask_sha in db.execute(
        select(
            ImageMask.id, ImageMask.source_sha256, ImageMask.status, ImageMask.mask_sha256
        ).execution_options(yield_per=INDEX_FETCH)
    ):
        last_mask_id = max(last_mask_id, mask_id)
        if not mask_sha or status not in ("ok", "withdrawn"):
            continue
        masked.add(source)
        if status == "ok" and source in live:
            published.add(mask_sha)
    plan = Plan(
        copy_of=copy_of,
        copies={},
        live=live,
        published=published,
        masked=masked,
        last_mask_id=last_mask_id,
    )
    if details:
        plan.load(db, live | masked)
    return plan


@dataclass
class Batch:
    index: int
    origins: list[str]
    copies_due: list[str]
    tail: bool = False


def make_batches(plan: Plan, *, size: int, after: str | None = None) -> list[Batch]:
    """The plan's originals in sha order, ``size`` at a time (1-200), each
    copy's masks due in the batch of its LAST original, then the tail: the
    copies with mask rows that no planned original reaches. ``after``: only
    the originals after this sha (a resumed run's cursor)."""
    if not 1 <= size <= MAX_ORIGIN_FILTER:
        raise ValueError(f"a batch holds 1-200 originals, not {size}")
    last_of: dict[str, str] = {}
    for origin, copy in plan.copy_of.items():
        if origin > last_of.get(copy, ""):
            last_of[copy] = origin
    due_at: dict[str, list[str]] = {}
    for copy, origin in last_of.items():
        due_at.setdefault(origin, []).append(copy)
    origins = sorted(o for o in plan.copy_of if after is None or o > after)
    batches: list[Batch] = []
    for i in range(0, len(origins), size):
        chunk = origins[i : i + size]
        due = sorted(c for o in chunk for c in due_at.get(o, []))
        batches.append(Batch(len(batches), chunk, due))
    tail = sorted(plan.masked - set(last_of))
    batches.append(Batch(len(batches), [], tail, tail=True))
    return batches


# --------------------------------------------------------------------------
# a run
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayConfig:
    batch_size: int = MAX_ORIGIN_FILTER
    lease_seconds: int = DEFAULT_LEASE_SECONDS
    dry_run: bool = False
    restart: bool = False
    pipeline_version: str = CLEAN_PIPELINE_VERSION

    def __post_init__(self) -> None:
        if not 1 <= self.batch_size <= MAX_ORIGIN_FILTER:
            raise ValueError(f"batch_size must be 1-{MAX_ORIGIN_FILTER}")
        if not 1 <= self.lease_seconds <= 3600:
            raise ValueError("lease_seconds must be 1-3600 (the contract's cap)")


class _Stop(Exception):
    """The run stops here; it resumes on a rerun."""


def _rpc_counts() -> dict[str, Any]:
    return {
        "calls": 0,
        "ok": 0,
        "stored": 0,
        "matched": 0,
        "warnings": 0,
        "refused": Counter(),
    }


@dataclass
class _BatchReport:
    batch: Batch
    leased: int = 0
    matched: int = 0
    unrequested: int = 0
    lease_rounds: int = 0
    lease_failed_attempts: int = 0
    rpc: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            r: _rpc_counts()
            for r in ("ReportDerivative", "ReportMask", "RestoreMask", "WithdrawMask")
        }
    )
    states: Counter[str] = field(default_factory=Counter)
    unregistered: int = 0
    #: masks of a copy the spine kept another derivative out for (stored=false)
    kept_other: int = 0
    #: masks the spine would not take because no original of their copy had
    #: a depiction to lease, and those originals
    orphan_masks: int = 0
    orphans: set[str] = field(default_factory=set)
    local_refused: Counter[str] = field(default_factory=Counter)
    local_warnings: int = 0
    quoted: list[str] = field(default_factory=list)

    def warn(self, texts: Iterable[str]) -> None:
        for text in texts:
            if len(self.quoted) < WARNINGS_QUOTED and text not in self.quoted:
                self.quoted.append(text)

    def line(self, run_id: int | None) -> dict[str, Any]:
        out: dict[str, Any] = {
            "event": "batch",
            "run": run_id,
            "batch": self.batch.index,
            "tail": self.batch.tail,
            "origins": len(self.batch.origins),
            "copies_due": len(self.batch.copies_due),
            "leased": self.leased,
            "matched": self.matched,
            "unrequested": self.unrequested,
            "lease_rounds": self.lease_rounds,
            "lease_failed_attempts": self.lease_failed_attempts,
            "local": {"refused": dict(self.local_refused), "warnings": self.local_warnings},
            "orphans": sorted(self.orphans),
            "warnings_sample": self.quoted,
        }
        for rpc, c in self.rpc.items():
            out[rpc] = {**c, "refused": dict(c["refused"])}
        out["ReportDerivative"]["states"] = dict(self.states)
        out["ReportMask"]["unregistered"] = self.unregistered
        out["ReportMask"]["kept_other"] = self.kept_other
        out["ReportMask"]["orphan"] = self.orphan_masks
        return out

    def flat(self) -> Counter[str]:
        flat: Counter[str] = Counter()
        for rpc, c in self.rpc.items():
            for key in ("calls", "ok", "stored", "matched", "warnings"):
                flat[f"{rpc}.{key}"] += c[key]
            for code, n in c["refused"].items():
                flat[f"{rpc}.refused.{code}"] += n
        for state, n in self.states.items():
            flat[f"ReportDerivative.state.{state}"] += n
        for name, n in self.local_refused.items():
            flat[f"local.refused.{name}"] += n
        flat["ReportMask.unregistered"] += self.unregistered
        flat["ReportMask.kept_other"] += self.kept_other
        flat["ReportMask.orphan"] += self.orphan_masks
        flat["orphan"] += len(self.orphans)
        flat["leased"] += self.leased
        flat["unrequested"] += self.unrequested
        flat["local.warnings"] += self.local_warnings
        return +flat


@dataclass
class _Run:
    db: Session
    plan: Plan
    config: ReplayConfig
    client: SpineMediaClient | None
    emit: Callable[[dict[str, Any]], None]
    now: Callable[[], dt.datetime]
    started: dt.datetime
    ledger: SpineReportRun | None
    owner: str
    #: copies a report of this run STORED (the spine registered these bytes)
    #: whose masks are not yet due: dropped once they are, so the run holds no
    #: more as it passes
    landed: set[str] = field(default_factory=set)
    #: copies this run reported OK, stored or not, whose masks are not yet due
    reported: set[str] = field(default_factory=set)
    withdrawn_now: set[str] = field(default_factory=set)

    # ----- one call, counted -------------------------------------------------

    def _send(
        self,
        report: _BatchReport,
        rpc: str,
        call: Callable[[], Any],
        *,
        copy: str | None = None,
    ) -> Any | None:
        """``copy``: the clean copy a ReportMask is for."""
        counts = report.rpc[rpc]
        counts["calls"] += 1
        try:
            ack = call()
        except SpineCallError as exc:
            if copy is not None and exc.code is grpc.StatusCode.FAILED_PRECONDITION:
                self._unlanded(report, copy)
            elif exc.verdict is Verdict.REFUSED:
                counts["refused"][exc.code.name] += 1
            else:
                raise _Stop(str(exc)) from exc
            report.warn([f"{rpc} {exc.code.name}: {exc.details[:200]}"])
            return None
        counts["ok"] += 1
        counts["warnings"] += len(ack.warnings)
        report.warn(f"{rpc}: {w}" for w in ack.warnings)
        return ack

    def _unlanded(self, report: _BatchReport, copy: str) -> None:
        """A mask the spine would not take: it has no derivative registered
        for ``copy`` (FAILED_PRECONDITION). A report of this run stored that
        derivative (the call was retried): the mask never landed,
        ``unregistered``, exit 1. This run reported it, never stored: the
        spine kept the derivative it already holds for each of those
        originals, ``kept_other``. Otherwise no original of the copy had a
        depiction to lease -- the spine never captured it, it is suppressed,
        or it is already derived -- and that is expected: an ``orphan``, its
        originals named in the batch line."""
        if copy in self.landed:
            report.unregistered += 1
        elif copy in self.reported:
            report.kept_other += 1
        else:
            report.orphan_masks += 1
            report.orphans.update(self.plan.originals_of(copy))

    def _local(self, report: _BatchReport, exc: FieldRefused) -> None:
        report.local_refused[exc.field] += 1
        report.warn([f"not sent: {exc}"])

    def _objectless(self, report: _BatchReport, copy: str) -> None:
        for sha in self.plan.objectless.get(copy, []):
            self._local(report, FieldRefused("mask_object", f"{sha} has no public object"))

    def _local_warnings(self, report: _BatchReport, warnings: list[str]) -> None:
        report.local_warnings += len(warnings)
        report.warn(f"local: {w}" for w in warnings)

    # ----- the steps ----------------------------------------------------------

    def _hold(self, *, call_seconds: float = 0.0) -> None:
        """Set, and commit, when a lease taken now (or within ``call_seconds``)
        has lapsed: a rerun waits for it."""
        assert self.ledger is not None
        self.ledger.resume_not_before = self.now() + dt.timedelta(
            seconds=call_seconds + self.config.lease_seconds + LEASE_MARGIN_SECONDS
        )
        self.db.commit()

    def _lease_and_report(self, report: _BatchReport) -> bool:
        """Steps 1-2. Returns False when a lease attempt failed (the batch
        may have rows held that no lease can see until they lapse)."""
        assert self.client is not None
        assert self.ledger is not None
        wanted = {bytes.fromhex(o): o for o in report.batch.origins}
        clean = True
        for _ in range(MAX_LEASE_ROUNDS):
            # Every round holds rows of its own, and so may every attempt
            # inside its call: a rerun must wait out the LAST hold. Before the
            # call, the hold covers the call's worst case (a pod killed during
            # it); once it has answered or failed, it runs from that instant.
            self._hold(call_seconds=self.client.worst_case_seconds)
            try:
                out = self.client.lease(
                    list(wanted),
                    max_items=LEASE_MAX_ITEMS,
                    lease_owner=self.owner,
                    lease_seconds=self.config.lease_seconds,
                )
            except SpineCallError as exc:
                self._hold()
                report.lease_failed_attempts += exc.attempts
                raise _Stop(f"lease: {exc}") from exc
            self._hold()
            report.lease_rounds += 1
            report.lease_failed_attempts += out.failed_attempts
            clean = clean and out.failed_attempts == 0
            self._local_warnings(report, out.warnings)
            report.warn(f"LeasePendingImages: {w}" for w in out.response.warnings)
            images = list(out.response.images)
            report.leased += len(images)
            mine = [i for i in images if i.origin_sha256 in wanted]
            stray = len(images) - len(mine)
            report.unrequested += stray
            report.matched += len(mine)
            for image in mine:
                self._report_derivative(report, image.product_image_id, wanted[image.origin_sha256])
            if stray:
                raise _Stop(
                    f"lease: the spine returned {stray} depiction(s) of originals not "
                    "requested, so it is not honouring origin_sha256 (a server before "
                    "0.10.0?): a deployment fault. They were left to lapse; rerun once "
                    "the spine honours the filter"
                )
            if len(images) < LEASE_MAX_ITEMS:
                return clean
        report.warn([f"lease still returning after {MAX_LEASE_ROUNDS} rounds"])
        return False

    def _report_derivative(self, report: _BatchReport, product_image_id: str, origin: str) -> None:
        assert self.client is not None
        copy = self.plan.copies[self.plan.copy_of[origin]]
        try:
            built = msg.derivative_report(
                product_image_id, copy, pipeline_version=self.config.pipeline_version
            )
        except FieldRefused as exc:
            self._local(report, exc)
            return
        self._local_warnings(report, built.warnings)
        client = self.client
        ack = self._send(
            report, "ReportDerivative", partial(client.report_derivative, built.message)
        )
        if ack is None:
            return
        report.rpc["ReportDerivative"]["stored"] += int(ack.stored)
        report.states[ack.product_image_state] += 1
        self.reported.add(copy.sha256)
        if ack.stored:  # this call registered these bytes; stored=false may be other bytes kept
            self.landed.add(copy.sha256)

    def _report_masks(self, report: _BatchReport, copy: str) -> None:
        assert self.client is not None
        client = self.client
        self._objectless(report, copy)
        for mask in self.plan.reports_for(copy):
            try:
                built = msg.mask_report(mask)
            except FieldRefused as exc:
                self._local(report, exc)
                continue
            self._local_warnings(report, built.warnings)
            landed = copy in self.landed
            ack = self._send(
                report,
                "ReportMask",
                partial(client.report_mask, built.message, derivative_landed=landed),
                copy=copy,
            )
            if ack is not None:
                report.rpc["ReportMask"]["stored"] += ack.stored
                report.rpc["ReportMask"]["matched"] += ack.matched

    def _restores(self, report: _BatchReport, copy: str) -> None:
        assert self.client is not None
        client = self.client
        for mask in self.plan.reports_for(copy):
            sha = mask.mask_sha256
            sent = self.db.get(SpineMaskWithdrawal, sha)
            if sent is None:
                continue
            ack = self._send(report, "RestoreMask", partial(client.withdraw_mask, msg.restore(sha)))
            if ack is not None:
                report.rpc["RestoreMask"]["stored"] += ack.stored
                report.rpc["RestoreMask"]["matched"] += ack.matched
                self.db.delete(sent)
                self.db.commit()

    def _withdrawals(self, report: _BatchReport, copy: str) -> None:
        assert self.client is not None
        client = self.client
        for w in self.plan.withdrawals_for(copy):
            if w.mask_sha256 in self.withdrawn_now:
                continue
            at = w.withdrawn_at or self.started
            try:
                built = msg.withdrawal(w.mask_sha256, reason=w.reason, withdrawn_at=at)
            except FieldRefused as exc:
                self._local(report, exc)
                continue
            self._local_warnings(report, built.warnings)
            if self.db.get(SpineMaskWithdrawal, w.mask_sha256) is None:
                self.db.add(
                    SpineMaskWithdrawal(
                        mask_sha256=w.mask_sha256,
                        reason=built.message.reason,
                        withdrawn_at=at,
                        sent_at=self.now(),
                    )
                )
                self.db.commit()
            ack = self._send(report, "WithdrawMask", partial(client.withdraw_mask, built.message))
            if ack is not None:
                report.rpc["WithdrawMask"]["stored"] += ack.stored
                report.rpc["WithdrawMask"]["matched"] += ack.matched
                self.withdrawn_now.add(w.mask_sha256)

    # ----- a batch ------------------------------------------------------------

    def batch(self, batch: Batch) -> tuple[_BatchReport, str | None]:
        """Run one batch. Returns its report and why the run must stop (None:
        it finished)."""
        report = _BatchReport(batch)
        clean = True
        try:
            if batch.origins:
                clean = self._lease_and_report(report)
            for copy in batch.copies_due:
                self._report_masks(report, copy)
                self.landed.discard(copy)  # its masks are due once per run: now
                self.reported.discard(copy)
                self._restores(report, copy)
                self._withdrawals(report, copy)
        except _Stop as stop:
            return report, str(stop)
        if not clean:
            return report, (
                "lease: an attempt failed, so rows of this batch may be held; "
                "rerun after the hold lapses"
            )
        return report, None

    def plan_only(self, batch: Batch) -> _BatchReport:
        """A dry run's batch: one ``would_send`` line per call."""
        report = _BatchReport(batch)

        def would(rpc: str, **fields: Any) -> None:
            key = "RestoreMask" if fields.get("restore") else rpc
            if key in report.rpc:
                report.rpc[key]["calls"] += 1
            else:
                report.lease_rounds += 1
            self.emit({"event": "would_send", "batch": batch.index, "rpc": rpc, **fields})

        if batch.origins:
            would("LeasePendingImages", origin_sha256=batch.origins, max_items=LEASE_MAX_ITEMS)
            for origin in batch.origins:
                would(
                    "ReportDerivative",
                    origin_sha256=origin,
                    derivative_sha256=self.plan.copy_of[origin],
                    pipeline_version=self.config.pipeline_version,
                    per="each pending depiction of the origin",
                )
        for copy in batch.copies_due:
            self._objectless(report, copy)
            for mask in self.plan.reports_for(copy):
                would(
                    "ReportMask",
                    derivative_sha256=copy,
                    pipeline_version=mask.pipeline_version,
                    mask_sha256=mask.mask_sha256,
                )
                if self.db.get(SpineMaskWithdrawal, mask.mask_sha256) is not None:
                    would("WithdrawMask", mask_sha256=mask.mask_sha256, restore=True)
            for w in self.plan.withdrawals_for(copy):
                would("WithdrawMask", mask_sha256=w.mask_sha256, restore=False, reason=w.reason)
        return report


def _open_ledger(
    db: Session, config: ReplayConfig, now: dt.datetime
) -> tuple[SpineReportRun, bool]:
    """The run to continue (resumed=True), or a new one."""
    running = (
        db.execute(
            select(SpineReportRun)
            .where(SpineReportRun.status == "running")
            .order_by(SpineReportRun.id.desc())
        )
        .scalars()
        .first()
    )
    if running is not None and not config.restart:
        return running, True
    if running is not None:
        running.status, running.finished_at = "abandoned", now
    run = SpineReportRun(status="running", counts={}, batch_size=config.batch_size, started_at=now)
    if running is not None:
        run.resume_not_before = running.resume_not_before  # its holds still bind
    db.add(run)
    db.commit()
    return run, False


def _failed(counts: dict[str, int]) -> bool:
    """Whether the run's PERSISTED counts -- every attempt of it, a resumed
    one included -- hold something that did not land: a refusal, the spine's
    or a local one, or an unregistered mask. An orphan is not a failure."""
    return any(
        n > 0 and (".refused." in key or key == "ReportMask.unregistered")
        for key, n in counts.items()
    )


def _aware(value: dt.datetime) -> dt.datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=dt.UTC)


def run_replay(
    db: Session,
    config: ReplayConfig,
    *,
    connect: Callable[[], SpineMediaClient],
    emit: Callable[[dict[str, Any]], None],
    now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
    sleep: Callable[[float], None] = time.sleep,
    item: Callable[[int, Callable[[], int | None]], int | None] = lambda _index, fn: fn(),
) -> int:
    """Replay the media DB into the spine. ``connect`` opens the client, and
    is never called on a dry run. Returns the exit status. Each batch -- its
    lease, its calls and its line -- runs inside ``item(index, fn)``: mm-report
    makes each one an item of its run (its own trace, linked to the run)."""
    started = now()
    plan = load_plan(db, details=False)
    ledger: SpineReportRun | None = None
    cursor: str | None = None
    if config.dry_run:
        running = (
            db.execute(select(SpineReportRun).where(SpineReportRun.status == "running"))
            .scalars()
            .first()
        )
        cursor = None if running is None or config.restart else running.cursor_sha256
    else:
        ledger, resumed = _open_ledger(db, config, started)
        cursor = ledger.cursor_sha256
        if resumed or ledger.resume_not_before is not None:
            wait = 0.0
            if ledger.resume_not_before is not None:
                wait = (_aware(ledger.resume_not_before) - started).total_seconds()
            emit(
                {
                    "event": "resume",
                    "run": ledger.id,
                    "cursor": cursor,
                    "wait_seconds": max(wait, 0.0),
                }
            )
            if wait > 0:
                sleep(wait)
    run = _Run(
        db=db,
        plan=plan,
        config=config,
        client=None if config.dry_run else connect(),
        emit=emit,
        now=now,
        started=started,
        ledger=ledger,
        owner=f"mm-report run {ledger.id if ledger else 'dry'} on {socket.gethostname()}",
    )
    batches = make_batches(plan, size=config.batch_size, after=cursor)
    emit(
        {
            "event": "plan",
            "run": ledger.id if ledger else None,
            "dry_run": config.dry_run,
            "originals": len(plan.copy_of),
            "live_copies": len(plan.live),
            "published_masks": len(plan.published),
            "batches": len(batches),
            "cursor": cursor,
        }
    )

    def one(batch: Batch) -> int | None:
        plan.load(db, {plan.copy_of[o] for o in batch.origins}.union(batch.copies_due))
        if config.dry_run:
            emit(run.plan_only(batch).line(None))
            return None
        assert ledger is not None
        report, stop = run.batch(batch)
        ledger.counts = dict(Counter(ledger.counts) + report.flat())
        emit(report.line(ledger.id))
        if stop is not None:
            db.commit()
            emit({"event": "stopped", "run": ledger.id, "batch": batch.index, "why": stop})
            return EXIT_STOPPED
        if batch.origins:
            ledger.cursor_sha256 = batch.origins[-1]
        ledger.resume_not_before = None
        db.commit()
        return None

    for batch in batches:
        if item(batch.index, partial(one, batch)) is not None:
            return EXIT_STOPPED
    code = EXIT_REFUSED if ledger is not None and _failed(ledger.counts) else EXIT_DONE
    if ledger is not None:
        ledger.status, ledger.finished_at = "done", now()
        db.commit()
    emit(
        {
            "event": "done",
            "run": ledger.id if ledger else None,
            "dry_run": config.dry_run,
            "exit": code,
            "counts": ledger.counts if ledger else None,
        }
    )
    return code
