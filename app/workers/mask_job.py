"""MASK-1 batch job: masks for every clean image that has none yet.

DRY RUN BY DEFAULT. A dry run computes everything -- eligibility, the mask,
its grounding -- and reports it, but writes nothing: no object, no row, no
batch record. ``dry_run=False`` is the only way anything is stored, and it
refuses a backend that is not a mask model (``publishes_masks``; the stub,
which is also the configured default) before writing anything: a stub mask
would otherwise be published and, being a result, settle its image for good.

IDEMPOTENT. A clean image is selected only while it has no image_masks row
at the current MASK_PIPELINE_VERSION, so a re-run is a no-op and a version
bump selects everything again (the old masks stay addressable). Mask bytes
are content-addressed, so two images with the same mask share one object.
Every capture of a picture is linked behind every clean copy of it
(app.workers.mask_seed): a copy is judged over every image behind it
(app.publication) and waits while any is soft-deleted.

RESUMABLE. Each clean image is settled in its own transaction -- its object
written first, then its row, its public registration and the batch's cursor
and counts committed together -- so a killed worker loses at most the image
in flight. ``resume_batch_id`` continues a batch that never finished from
its cursor (candidates are taken in sha order, a keyset rather than an
offset, so nothing is skipped or seen twice).

FAILURES are per image -- in the judgment as in the mask -- recorded as
``failed`` with the exception's CLASS only (never a message, which can carry
a bucket or key), and re-driven with ``retry_failed=True``. An
``ineligible`` image is re-judged with ``recheck_ineligible=True`` (a store
granted later, origins that changed). An image soft-deleted after the batch
selected it is ``skipped`` (reason ``deleted``) and nothing is recorded, so
restoring it makes it a candidate again.

RE-EVALUATION. ``reevaluate_masks`` re-judges every published (``ok``) and
``withdrawn`` row against its image's current state (app.publication): the
origin already refuses such a mask on every request; this keeps the lane's
own record in line and restores a mask whose image qualifies again.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import functools
import hashlib
import logging
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sqlalchemy import Select, func, select, union
from sqlalchemy.orm import Session, aliased

from ..models import (
    FigureGallery,
    ImageMask,
    ImageVersion,
    MaskBatch,
    MaskStoreRevocation,
    PublicObject,
    PublicObjectImage,
)
from ..models import Image as ImageModel
from ..object_store import ObjectStore, derived_key
from ..publication import judged_origins, mask_refusal, store_revocations
from .grounding import AlphaGrounding
from .mask import MASK_MIME, MASK_PIPELINE_VERSION, MaskOutcome, build_mask_artifact
from .mask_policy import Revoked, StoreGrants, configured_revocations, mask_eligibility, source_key
from .mask_seed import register_public_object
from .matting import MattingBackend, get_matting_backend
from .stage_timing import StageTimings, timed

if TYPE_CHECKING:
    from .cpu_stages import CpuStages

logger = logging.getLogger(__name__)

COUNT_KEYS = ("ok", "held", "empty", "ineligible", "failed", "skipped")

DEFAULT_PAGE_SIZE = 50


@dataclass(frozen=True)
class ItemReport:
    source_sha256: str
    status: str
    reason: str | None
    origin_class: str | None
    mask_sha256: str | None = None
    grounding: AlphaGrounding | None = None


def _zero_counts() -> dict[str, int]:
    return dict.fromkeys(COUNT_KEYS, 0)


@dataclass
class BatchReport:
    batch_id: int | None
    dry_run: bool
    items: list[ItemReport] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=_zero_counts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "dry_run": self.dry_run,
            "pipeline_version": MASK_PIPELINE_VERSION,
            "counts": dict(self.counts),
            "items": [dataclasses.asdict(i) for i in self.items],
        }


class _Refused(Exception):
    """A per-image refusal whose reason is safe to record verbatim."""


def _of_source(source: str) -> Select[tuple[str]]:
    """The clean copies with an image behind them -- its own or a linked
    capture -- that holds a live origin of ``source`` (as ``source_key``)."""
    of_source = (
        func.lower(func.trim(FigureGallery.source)) == source,
        FigureGallery.deleted_at.is_(None),
    )
    linked = (
        select(PublicObjectImage.sha256)
        .join(FigureGallery, FigureGallery.image_id == PublicObjectImage.image_id)
        .where(*of_source)
    )
    owned = (
        select(PublicObject.sha256)
        .join(ImageVersion, ImageVersion.id == PublicObject.image_version_id)
        .join(FigureGallery, FigureGallery.image_id == ImageVersion.image_id)
        .where(*of_source)
    )
    return select(union(linked, owned).subquery().c[0])


def _candidates(
    db: Session,
    *,
    after: str | None,
    page_size: int,
    redrive: tuple[str, ...],
    source: str | None = None,
    only: Sequence[str] | None = None,
) -> list[PublicObject]:
    # NOT EXISTS, never NOT IN (subquery) (MASKRATE): Postgres cannot make a NOT
    # IN an anti-join (a NULL in the subquery would change its answer), so it
    # reads the whole subquery -- every mask row of the version, every dead
    # capture -- on every call, for one copy as for a page. Every key compared
    # here is NOT NULL, so the two answer alike. Each EXISTS names a column,
    # never ``*``: a role with column grants is checked on every column named.
    settled = (
        select(ImageMask.source_sha256)
        .where(
            ImageMask.source_sha256 == PublicObject.sha256,
            ImageMask.pipeline_version == MASK_PIPELINE_VERSION,
            ImageMask.status.not_in(redrive),
        )
        .exists()
    )
    # A clean copy shared by several captures of a picture waits while ANY of them
    # is soft-deleted (app.publication would not serve it).
    dead = aliased(ImageModel)
    dead_capture = (
        select(PublicObjectImage.sha256)
        .join(dead, dead.id == PublicObjectImage.image_id)
        .where(PublicObjectImage.sha256 == PublicObject.sha256, dead.deleted_at.is_not(None))
        .exists()
    )
    query = (
        select(PublicObject)
        .join(ImageVersion, ImageVersion.id == PublicObject.image_version_id)
        .join(ImageModel, ImageModel.id == ImageVersion.image_id)
        .where(
            PublicObject.kind == "sanitized",
            ImageVersion.deleted_at.is_(None),
            ImageModel.deleted_at.is_(None),
            ~settled,
            ~dead_capture,
        )
        .order_by(PublicObject.sha256)
        .limit(page_size)
    )
    if after is not None:
        query = query.where(PublicObject.sha256 > after)
    if source is not None:
        query = query.where(PublicObject.sha256.in_(_of_source(source)))
    if only is not None:
        query = query.where(PublicObject.sha256.in_(only))
    return list(db.execute(query).scalars())


def _row_fields(outcome: MaskOutcome) -> dict[str, Any]:
    g = outcome.grounding
    fields: dict[str, Any] = {
        "backend": outcome.backend,
        "width": outcome.width,
        "height": outcome.height,
        "mask_sha256": outcome.mask_sha256,
        "plain_thumbhash": outcome.unmasked.thumbhash,
        "plain_dominant_color": outcome.unmasked.dominant_color,
        "masked_thumbhash": outcome.masked.thumbhash if outcome.masked else None,
        "masked_dominant_color": outcome.masked.dominant_color if outcome.masked else None,
        "own_alpha_grounding": (
            dataclasses.asdict(outcome.own_alpha_grounding) if outcome.own_alpha_grounding else None
        ),
    }
    for name in (f.name for f in dataclasses.fields(AlphaGrounding)):
        fields[name] = getattr(g, name) if g else None
    return fields


_RESULT_COLUMNS = (
    "backend",
    "width",
    "height",
    "mask_sha256",
    "mask_storage_key",
    *(f.name for f in dataclasses.fields(AlphaGrounding)),
    "masked_thumbhash",
    "masked_dominant_color",
    "plain_thumbhash",
    "plain_dominant_color",
    "own_alpha_grounding",
)


def _write_row(
    db: Session,
    *,
    source_sha256: str,
    batch: MaskBatch,
    status: str,
    reason: str | None,
    origin_class: str | None,
    timings: StageTimings | None = None,
    turn: Callable[[], None] | None = None,
    **fields: Any,
) -> None:
    if turn is not None:
        turn()
    with timed(timings, "db"):
        now = dt.datetime.now(dt.UTC)
        row = db.execute(
            select(ImageMask).where(
                ImageMask.source_sha256 == source_sha256,
                ImageMask.pipeline_version == MASK_PIPELINE_VERSION,
            )
        ).scalar_one_or_none()
        if row is None:
            row = ImageMask(source_sha256=source_sha256, pipeline_version=MASK_PIPELINE_VERSION)
            row.created_at = now
            db.add(row)
        row.status = status
        row.reason = reason
        row.origin_class = origin_class
        row.batch_id = batch.id
        row.updated_at = now
        # Every result column is written on every settle, so a re-driven row
        # never keeps a value from the attempt it replaces.
        for name in _RESULT_COLUMNS:
            setattr(row, name, fields.get(name))
        batch.cursor_sha256 = source_sha256
        batch.counts = {**batch.counts, status: batch.counts.get(status, 0) + 1}
        db.commit()


def _settle(
    db: Session,
    store: ObjectStore,
    obj: PublicObject,
    backend: MattingBackend,
    batch: MaskBatch | None,
    revoked: Revoked | None = None,
    timings: StageTimings | None = None,
    turn: Callable[[], None] | None = None,
    slot: Callable[[], AbstractContextManager[None]] | None = None,
    cpu: CpuStages | None = None,
) -> ItemReport:
    """Settle one clean image. ``batch`` is None on a dry run: nothing is
    written then, the report is the whole result. ``revoked`` defaults to
    the revocations as they stand when this image is judged. ``timings``,
    when given, gets the seconds of each stage (MASK-PERF-1); its SQL's
    seconds are the caller's to time (app.workers.stage_timing.DatabaseSeconds).
    ``turn``, when given, is called before the image's first write after its
    judgment -- a mask registered or its row and the batch's -- and waits
    for this image's turn to write them (MASK-PERF-2: the pipelined batch
    writes the batch row in work-list order). ``slot``, when given, is held
    from the copy's decode to its mask (the pipelined batch bounds the
    copies decoded at once: app.workers.mask_pipeline.Slots). ``cpu``, when
    given, runs the mask's judgment once the model answered (MASK5:
    ``build_mask_artifact``), the slot still held."""
    sha = obj.sha256
    try:
        # Judged over every image behind the clean copy (app.publication).
        origins = judged_origins(db, sha)
        if origins is None:
            # Soft-deleted since it was selected: skipped and NOT recorded,
            # as when it is deleted before selection, so a restore makes it
            # a candidate again for a plain batch.
            return ItemReport(sha, "skipped", "deleted", None)
        eligibility = mask_eligibility(
            origins, revoked=store_revocations(db) if revoked is None else revoked
        )
    except Exception as exc:
        # A judgment that raises is this image's failure, never the batch's.
        with timed(timings, "db"):
            db.rollback()
        reason = type(exc).__name__
        logger.warning("mask_judgment_failed", extra={"source_sha256": sha, "reason": reason})
        if batch is not None:
            _write_row(
                db,
                source_sha256=sha,
                batch=batch,
                status="failed",
                reason=reason,
                origin_class=None,
                timings=timings,
                turn=turn,
            )
        return ItemReport(sha, "failed", reason, None)
    if not eligibility.eligible:
        if batch is not None:
            _write_row(
                db,
                source_sha256=sha,
                batch=batch,
                status="ineligible",
                reason=eligibility.reason,
                origin_class=eligibility.origin_class,
                timings=timings,
                turn=turn,
            )
        return ItemReport(sha, "ineligible", eligibility.reason, eligibility.origin_class)

    try:
        with timed(timings, "readback"):
            data = store.get(obj.storage_key)
        if hashlib.sha256(data).hexdigest() != sha:
            raise _Refused("source_hash_mismatch")
        decoded = nullcontext() if slot is None else slot()
        with timed(timings, "analysis"), decoded:  # the model and the encode count apart
            outcome = build_mask_artifact(data, backend=backend, timings=timings, cpu=cpu)
            fields = _row_fields(outcome)
        if batch is not None and outcome.mask_png is not None and outcome.mask_sha256 is not None:
            if outcome.status == "ok":
                key = derived_key(outcome.mask_sha256, MASK_MIME)
                with timed(timings, "upload"):
                    store.put(key, outcome.mask_png, MASK_MIME)
                if turn is not None:
                    turn()  # a mask another image shares is registered once, in order
                register_public_object(
                    db,
                    sha256=outcome.mask_sha256,
                    kind="mask",
                    mime=MASK_MIME,
                    size=len(outcome.mask_png),
                    width=outcome.width,
                    height=outcome.height,
                    storage_key=key,
                )
            else:
                key = f"held-masks/{outcome.mask_sha256}.png"
                with timed(timings, "upload"):
                    store.put(key, outcome.mask_png, MASK_MIME)
            fields["mask_storage_key"] = key
    except Exception as exc:
        with timed(timings, "db"):
            db.rollback()
        reason = str(exc) if isinstance(exc, _Refused) else type(exc).__name__
        logger.warning("mask_failed", extra={"source_sha256": sha, "reason": reason})
        if batch is not None:
            _write_row(
                db,
                source_sha256=sha,
                batch=batch,
                status="failed",
                reason=reason,
                origin_class=eligibility.origin_class,
                timings=timings,
                turn=turn,
            )
        return ItemReport(sha, "failed", reason, eligibility.origin_class)

    if batch is not None:
        _write_row(
            db,
            source_sha256=sha,
            batch=batch,
            status=outcome.status,
            reason=outcome.reason,
            origin_class=eligibility.origin_class,
            timings=timings,
            turn=turn,
            **fields,
        )
    return ItemReport(
        sha,
        outcome.status,
        outcome.reason,
        eligibility.origin_class,
        outcome.mask_sha256,
        outcome.grounding,
    )


def _open_batch(
    db: Session,
    *,
    resume_batch_id: int | None,
    backend_name: str,
    limit: int | None,
    source: str | None,
) -> MaskBatch:
    if resume_batch_id is not None:
        batch = db.get(MaskBatch, resume_batch_id)
        if (
            batch is None
            or batch.status != "running"
            or batch.pipeline_version != MASK_PIPELINE_VERSION
        ):
            raise ValueError(f"mask batch {resume_batch_id} is not resumable")
        if source is not None and source != batch.source:
            raise ValueError(
                f"mask batch {resume_batch_id} is limited to source {batch.source!r}:"
                " resume it with that source or none"
            )
        return batch
    batch = MaskBatch(
        pipeline_version=MASK_PIPELINE_VERSION,
        backend=backend_name,
        dry_run=False,
        status="running",
        item_limit=limit,
        source=source,
        counts=_zero_counts(),
    )
    db.add(batch)
    db.commit()
    return batch


def _redrive(retry_failed: bool, recheck_ineligible: bool) -> tuple[str, ...]:
    return ("failed",) * retry_failed + ("ineligible",) * recheck_ineligible


def open_mask_batch(db: Session, *, backend_name: str) -> MaskBatch:
    """A new applied batch for ``settle_clean_copy`` (MASK-BATCH-1: one per
    chunk); close it with ``finish_mask_batch``."""
    return _open_batch(db, resume_batch_id=None, backend_name=backend_name, limit=None, source=None)


def finish_mask_batch(db: Session, batch: MaskBatch) -> None:
    batch.status = "done"
    batch.finished_at = dt.datetime.now(dt.UTC)
    db.commit()


def settle_clean_copy(
    db: Session,
    store: ObjectStore,
    sha256: str,
    *,
    backend: MattingBackend,
    batch: MaskBatch,
    retry_failed: bool = False,
    recheck_ineligible: bool = False,
    timings: StageTimings | None = None,
    turn: Callable[[], None] | None = None,
    slot: Callable[[], AbstractContextManager[None]] | None = None,
    cpu: CpuStages | None = None,
) -> ItemReport | None:
    """Settle ONE clean copy, by its sha, exactly as ``run_mask_batch`` settles
    each of its candidates (MASK-BATCH-1 drives the lane image by image, in its
    own order). None when the copy is not a candidate: already settled at this
    pipeline version (unless re-driven), soft-deleted, or not a clean copy.
    Applied only: a backend that is not a mask model is refused. ``timings``,
    when given, gets the seconds of each stage (MASK-PERF-1); its SQL's
    seconds are the caller's to time (app.workers.stage_timing.DatabaseSeconds).
    ``turn``, ``slot``: as ``_settle`` takes them (MASK-PERF-2); ``cpu``
    too (MASK5)."""
    if not backend.publishes_masks:
        raise ValueError(f"backend {backend.name!r} is not a mask model: dry run only")
    with timed(timings, "db"):  # the lane's first query: the pool's checkout too
        page = _candidates(
            db,
            after=None,
            page_size=1,
            redrive=_redrive(retry_failed, recheck_ineligible),
            only=(sha256,),
        )
    if not page:
        return None
    return _settle(
        db, store, page[0], backend, batch, timings=timings, turn=turn, slot=slot, cpu=cpu
    )


def original_settled(
    db: Session, original_sha256: str, *, retry_failed: bool = False
) -> str | None:
    """The mask result, at this pipeline version, of a clean copy behind the
    original whose bytes have ``original_sha256`` -- ``ok`` when any copy has
    one -- or None when it has no copy yet or no result yet (a ``failed``
    result counts as none when ``retry_failed``). MASK-BATCH-1's idempotence
    check: an original whose copy is settled is skipped without its bytes
    being read."""
    copies = (
        select(PublicObjectImage.sha256)
        .join(ImageModel, ImageModel.id == PublicObjectImage.image_id)
        .join(PublicObject, PublicObject.sha256 == PublicObjectImage.sha256)
        .where(ImageModel.sha256 == original_sha256, PublicObject.kind == "sanitized")
    )
    statuses = set(
        db.execute(
            select(ImageMask.status).where(
                ImageMask.source_sha256.in_(copies),
                ImageMask.pipeline_version == MASK_PIPELINE_VERSION,
                ImageMask.status.not_in(_redrive(retry_failed, False)),
            )
        ).scalars()
    )
    if not statuses:
        return None
    return "ok" if "ok" in statuses else min(statuses)


def run_mask_batch(
    db: Session,
    store: ObjectStore,
    *,
    backend: MattingBackend | None = None,
    dry_run: bool = True,
    limit: int | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    resume_batch_id: int | None = None,
    retry_failed: bool = False,
    recheck_ineligible: bool = False,
    source: str | None = None,
    revoked: Revoked | None = None,
) -> BatchReport:
    """Settle up to ``limit`` clean images that have no mask result yet --
    plus, on request, those whose result was ``failed`` (``retry_failed``) or
    ``ineligible`` (``recheck_ineligible``: after a store is granted or
    restored, or an image's origins changed). ``source`` limits the batch to
    clean copies with a live origin of that source, and is recorded: a
    resumed batch keeps its own (another ``source`` is refused); ``revoked``
    overrides the revocations (``restore_store``'s dry run judges without
    its own)."""
    if resume_batch_id is not None and dry_run:
        raise ValueError("a dry run cannot resume a batch: it records nothing to resume")
    active = backend or get_matting_backend()
    if not dry_run and not active.publishes_masks:
        raise ValueError(f"backend {active.name!r} is not a mask model: dry run only")
    key = None if source is None else source_key(source)
    batch: MaskBatch | None = None
    if not dry_run:
        batch = _open_batch(
            db,
            resume_batch_id=resume_batch_id,
            backend_name=active.name,
            limit=limit,
            source=key,
        )
        key = batch.source
        if resume_batch_id is not None and batch.item_limit is not None:
            limit = max(0, batch.item_limit - sum(batch.counts.values()))

    redrive = _redrive(retry_failed, recheck_ineligible)
    report = BatchReport(batch_id=batch.id if batch else None, dry_run=dry_run)
    cursor = batch.cursor_sha256 if batch else None
    remaining = limit
    while remaining is None or remaining > 0:
        size = page_size if remaining is None else min(page_size, remaining)
        page = _candidates(db, after=cursor, page_size=size, redrive=redrive, source=key)
        if not page:
            break
        for obj in page:
            item = _settle(db, store, obj, active, batch, revoked)
            report.items.append(item)
            report.counts[item.status] += 1
            cursor = obj.sha256
        if remaining is not None:
            remaining -= len(page)

    if batch is not None:
        finish_mask_batch(db, batch)
    return report


#: Rows judged per batch by reevaluate_masks (the masks CLI makes each batch an item of its run).
REEVALUATE_BATCH = 500


def reevaluate_masks(
    db: Session,
    *,
    dry_run: bool = True,
    store_grants: StoreGrants | None = None,
    source: str | None = None,
    revoked: Revoked | None = None,
    batch_size: int | None = None,
    batch_scope: Callable[[int, Callable[[], dict[str, Any]]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Withdraw every ``ok`` row whose clean image is deleted or no longer
    mask-eligible (reason: the refusal) and restore every ``withdrawn`` row
    whose image qualifies again. Every pipeline version is judged: an old
    version's mask is as addressable as the current one's. Dry run unless
    ``dry_run=False``; held, empty, ineligible and failed rows are never
    touched. Each change is committed on its own, and a row whose judgment
    raises is judged refused (reason: the exception's class) -- one bad row
    never stops the rest. ``source`` judges only the rows of clean copies
    with a live origin of that source -- nothing else is touched -- and
    ``revoked`` overrides the revocations, as in ``run_mask_batch``. The
    rows are judged ``batch_size`` (REEVALUATE_BATCH) at a time, each batch
    inside ``batch_scope(index, judge)`` (1-based), which returns judge's
    part of the report."""
    query = select(ImageMask).where(ImageMask.status.in_(("ok", "withdrawn")))
    if source is not None:
        query = query.where(ImageMask.source_sha256.in_(_of_source(source_key(source))))
    rows = list(db.execute(query.order_by(ImageMask.id)).scalars())
    size = batch_size or REEVALUATE_BATCH
    now = dt.datetime.now(dt.UTC)

    def judge(batch: list[ImageMask]) -> dict[str, Any]:
        part: dict[str, Any] = {"withdrawn": [], "restored": [], "unchanged": 0}
        for row in batch:
            sha = row.source_sha256
            try:
                refusal = mask_refusal(db, sha, store_grants, revoked)
            except Exception as exc:
                db.rollback()
                refusal = type(exc).__name__
                logger.warning(
                    "mask_reevaluate_failed", extra={"source_sha256": sha, "reason": refusal}
                )
            if row.status == "ok" and refusal is not None:
                part["withdrawn"].append(sha)
                status, reason = "withdrawn", refusal
            elif row.status == "withdrawn" and refusal is None:
                part["restored"].append(sha)
                status, reason = "ok", None
            else:
                part["unchanged"] += 1
                continue
            if not dry_run:
                row.status, row.reason, row.updated_at = status, reason, now
                db.commit()
        return part

    withdrawn: list[str] = []
    restored: list[str] = []
    unchanged = 0
    for index, start in enumerate(range(0, len(rows), size), start=1):
        batch = rows[start : start + size]
        part = (
            judge(batch)
            if batch_scope is None
            else batch_scope(index, functools.partial(judge, batch))
        )
        withdrawn += part["withdrawn"]
        restored += part["restored"]
        unchanged += part["unchanged"]
    return {
        "dry_run": dry_run,
        "withdrawn": withdrawn,
        "restored": restored,
        "unchanged": unchanged,
    }


def _named(source: str) -> str:
    key = source_key(source)
    if not key:
        raise ValueError("no source named")
    return key


def revoke_store(db: Session, source: str, *, dry_run: bool = True) -> dict[str, Any]:
    """MASK-G: revoke every mask of ``source`` (a store asks for its cut-outs
    to be removed). Applied, the revocation is persisted and committed FIRST
    -- from then on the origin refuses every mask of that source on every
    request and no batch makes a new one -- and then every ``ok`` row of a
    clean copy with a live origin of that source is withdrawn (reason
    ``store_revoked``); no other row is judged. Idempotent. Dry run unless
    ``dry_run=False``: it reports what would be withdrawn and writes nothing."""
    key = _named(source)
    if not dry_run and db.get(MaskStoreRevocation, key) is None:
        db.add(MaskStoreRevocation(source=key))
        db.commit()
    revoked = store_revocations(db) | {key}
    report = reevaluate_masks(db, dry_run=dry_run, source=key, revoked=revoked)
    return {"source": key, "dry_run": dry_run, "reevaluate": report}


def restore_store(
    db: Session,
    store: ObjectStore,
    source: str,
    *,
    dry_run: bool = True,
    backend: MattingBackend | None = None,
) -> dict[str, Any]:
    """MASK-G: lift a ``revoke_store``. Applied, the persisted revocation is
    deleted, every withdrawn row of that source that qualifies again is
    restored (``reevaluate_masks``), and that source's refused images are
    re-checked (``run_mask_batch`` with ``recheck_ineligible``), so what was
    captured while it was revoked gets its mask. Re-run it if interrupted.
    Refused before anything changes: a backend that is not a mask model (on
    apply), or a source revoked by MASK_STORE_REVOKED, which only the
    deployment's environment lifts. Dry run unless ``dry_run=False``."""
    key = _named(source)
    if key in configured_revocations():
        raise ValueError(f"{key!r} is revoked by MASK_STORE_REVOKED: lift it in the environment")
    active = backend or get_matting_backend()
    if not dry_run and not active.publishes_masks:
        raise ValueError(f"backend {active.name!r} is not a mask model: dry run only")
    if not dry_run:
        revocation = db.get(MaskStoreRevocation, key)
        if revocation is not None:
            db.delete(revocation)
            db.commit()
    revoked = store_revocations(db) - {key}
    report = reevaluate_masks(db, dry_run=dry_run, source=key, revoked=revoked)
    batch = run_mask_batch(
        db,
        store,
        backend=active,
        dry_run=dry_run,
        recheck_ineligible=True,
        source=key,
        revoked=revoked,
    )
    return {"source": key, "dry_run": dry_run, "reevaluate": report, "batch": batch.as_dict()}
