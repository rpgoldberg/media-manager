"""MM-REPORT-1: the replay (app.spine.replay) from the media DB into SpineMedia.

Per batch of at most 200 originals: LeasePendingImages{origin_sha256} -> each
returned origin checked -> ReportDerivative per depiction -> ReportMask per
(clean sha, mask version) -> restores -> WithdrawMask per withdrawn mask.
Idempotent and resumable, with the run ledger in the media DB. Driven here
against a stateful fake spine over real gRPC (tests/spine_fakes.py)."""

from __future__ import annotations

import base64
import datetime as dt
from collections.abc import Iterator
from typing import Any

import grpc
import pytest
from app.models import (
    Image,
    ImageMask,
    ImageVersion,
    PublicObject,
    SpineMaskWithdrawal,
    SpineReportRun,
)
from app.spine import replay as rp
from app.spine.media_client import RetryPolicy, SpineMediaClient, open_channel
from sqlalchemy import select
from sqlalchemy.orm import Session

from spine_fakes import AT, GROUND, FakeSpine, LapsingSpine, seed_copy, serve, sha

C = grpc.StatusCode
T0 = dt.datetime(2026, 10, 15, 1, 0, tzinfo=dt.UTC)


class Clock:
    def __init__(self) -> None:
        self.now = T0
        self.slept: list[float] = []

    def __call__(self) -> dt.datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += dt.timedelta(seconds=seconds)


@pytest.fixture()
def spine() -> Iterator[tuple[FakeSpine, str]]:
    fake = FakeSpine()
    with serve(fake) as url:
        yield fake, url


def _run(
    db: Session,
    url: str | None,
    *,
    clock: Clock | None = None,
    attempts: int = 2,
    **config: Any,
) -> tuple[int, list[dict[str, Any]]]:
    lines: list[dict[str, Any]] = []
    clock = clock or Clock()

    def connect() -> SpineMediaClient:
        assert url is not None, "a dry run must not open a channel"
        return SpineMediaClient(
            open_channel(url),
            policy=RetryPolicy(max_attempts=attempts, initial_backoff=0.01),
            sleep=lambda s: None,
        )

    code = rp.run_replay(
        db,
        rp.ReplayConfig(**config),
        connect=connect,
        emit=lines.append,
        now=clock,
        sleep=clock.sleep,
    )
    return code, lines


def _batches(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [line for line in lines if line["event"] == "batch"]


def _total(lines: list[dict[str, Any]], rpc: str, key: str) -> int:
    return sum(b[rpc][key] for b in _batches(lines))


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------


def test_the_plan_maps_each_live_original_to_its_clean_copy(db_session: Session) -> None:
    a = seed_copy(db_session, "a", extra_originals=("a2",))
    gone = seed_copy(db_session, "gone", deleted=True)
    plan = rp.load_plan(db_session)
    assert plan.copy_of == {a["origin"]: a["clean"], a["others"][0]: a["clean"]}
    assert gone["clean"] not in plan.live
    assert a["clean"] in plan.live


def test_an_original_behind_two_copies_takes_the_one_minted_from_it(db_session: Session) -> None:
    a = seed_copy(db_session, "a")
    b = seed_copy(db_session, "b", extra_originals=())
    # a's original is also a capture of b's picture
    image = db_session.execute(select(Image).where(Image.sha256 == a["origin"])).scalar_one()
    from app.models import PublicObjectImage

    db_session.add(PublicObjectImage(sha256=b["clean"], image_id=image.id))
    db_session.commit()
    plan = rp.load_plan(db_session)
    assert plan.copy_of[a["origin"]] == a["clean"]
    assert plan.copy_of[b["origin"]] == b["clean"]


def test_the_plans_liveness_is_the_origins(db_session: Session) -> None:
    from app.publication import images_behind

    seed_copy(db_session, "live", extra_originals=("live2",))
    seed_copy(db_session, "dead", deleted=True)
    dead_version = seed_copy(db_session, "deadv")
    seed_copy(db_session, "deadlink", extra_originals=("deadlink2",))
    version = db_session.execute(
        select(ImageVersion).where(ImageVersion.storage_key == f"d/{dead_version['clean']}")
    ).scalar_one()
    version.deleted_at = AT
    linked = db_session.execute(
        select(Image).where(Image.sha256 == sha("orig:deadlink2"))
    ).scalar_one()
    linked.deleted_at = AT
    db_session.commit()
    plan = rp.load_plan(db_session)
    for tag in ("live", "dead", "deadv", "deadlink"):
        clean = sha(f"clean:{tag}")
        assert (clean in plan.live) is (images_behind(db_session, clean) is not None), tag
    assert plan.live == {sha("clean:live")}


def test_mask_status_is_judged_per_mask_bytes(db_session: Session) -> None:
    seed_copy(db_session, "a", masks=(("mask-v1", "ok", "ma"), ("mask-v0", "held", "mh")))
    seed_copy(db_session, "b", masks=(("mask-v1", "withdrawn", "mb"),))
    seed_copy(db_session, "c", masks=(("mask-v1", "ok", "mc"),), deleted=True)
    seed_copy(db_session, "d", masks=(("mask-v1", "withdrawn", "ma"),))  # same bytes as a's
    plan = rp.load_plan(db_session)
    assert plan.published == {sha("mask:ma")}
    reported = {(m.clean_sha256, m.mask_sha256) for m in plan.reports_for(sha("clean:a"))}
    assert reported == {(sha("clean:a"), sha("mask:ma"))}  # never the held one
    withdrawn = {w.mask_sha256: w.reason for w in plan.withdrawals_for(sha("clean:b"))}
    assert withdrawn == {sha("mask:mb"): "store_revoked"}
    # an ok mask of a deleted copy is withdrawn as 'deleted'
    assert [w.reason for w in plan.withdrawals_for(sha("clean:c"))] == ["deleted"]
    # bytes some live ok row still carries are never withdrawn
    assert plan.withdrawals_for(sha("clean:d")) == []


def test_batches_hold_at_most_the_contracts_200_originals(db_session: Session) -> None:
    for i in range(5):
        seed_copy(db_session, f"c{i}")
    plan = rp.load_plan(db_session)
    batches = rp.make_batches(plan, size=2)
    assert [len(b.origins) for b in batches] == [2, 2, 1, 0]
    assert batches[-1].tail is True
    assert sorted(o for b in batches for o in b.origins) == sorted(plan.copy_of)
    with pytest.raises(ValueError, match="1-200"):
        rp.make_batches(plan, size=201)
    with pytest.raises(ValueError, match="1-200"):
        rp.make_batches(plan, size=0)


def test_a_copys_masks_wait_for_the_batch_of_its_last_original(db_session: Session) -> None:
    seed_copy(db_session, "x", extra_originals=tuple(f"x{i}" for i in range(4)))
    plan = rp.load_plan(db_session)
    batches = rp.make_batches(plan, size=2)
    due = [b.copies_due for b in batches]
    assert due[:-1] == [[], [], [sha("clean:x")]]
    assert max(plan.copy_of) in batches[2].origins


def test_a_dead_copys_withdrawals_go_in_the_tail(db_session: Session) -> None:
    seed_copy(db_session, "dead", deleted=True)
    seed_copy(db_session, "live")
    batches = rp.make_batches(rp.load_plan(db_session), size=200)
    assert batches[-1].tail and batches[-1].copies_due == [sha("clean:dead")]
    assert batches[0].copies_due == [sha("clean:live")]


# ---------------------------------------------------------------------------
# a run
# ---------------------------------------------------------------------------


def test_a_run_reports_every_copy_and_mask_and_a_rerun_stores_nothing(
    db_session: Session, spine: tuple[FakeSpine, str]
) -> None:
    fake, url = spine
    a = seed_copy(db_session, "a", extra_originals=("a2",))
    b = seed_copy(db_session, "b", masks=(("mask-v1", "withdrawn", "mb"),))
    pa = fake.depict(a["origin"])
    pa2 = fake.depict(a["others"][0])
    pb = fake.depict(b["origin"])
    other = fake.depict(sha("orig:not-ours"))

    code, lines = _run(db_session, url, batch_size=200)
    assert code == rp.EXIT_DONE
    assert {fake.depictions[p]["state"] for p in (pa, pa2, pb)} == {"derived"}
    assert fake.depictions[other]["state"] == "captured"  # never named, never touched
    assert fake.published(a["masks"][0])
    assert bytes.fromhex(b["masks"][0]) in fake.tombstones
    batch = _batches(lines)[0]
    assert batch["leased"] == 3 and batch["matched"] == 3 and batch["unrequested"] == 0
    assert batch["ReportDerivative"]["calls"] == 3 and batch["ReportDerivative"]["stored"] == 2
    assert batch["ReportDerivative"]["states"] == {"derived": 3}
    assert batch["ReportMask"]["stored"] == 1 and batch["WithdrawMask"]["stored"] == 1
    run = db_session.execute(select(SpineReportRun)).scalar_one()
    assert run.status == "done" and run.resume_not_before is None
    assert run.counts["ReportDerivative.calls"] == 3
    assert lines[-1]["event"] == "done" and lines[-1]["exit"] == 0

    sent = fake.sent("ReportDerivative")[0]
    assert sent.derivative_sha256 == bytes.fromhex(a["clean"])
    assert sent.pipeline_version == rp.CLEAN_PIPELINE_VERSION and sent.derive_kind == "sanitized"
    assert sent.thumbhash and sent.dominant_color == "#112233"
    mask = fake.sent("ReportMask")[0]
    assert mask.thumbhash and mask.dominant_color == "#445566"
    assert mask.grounding.bottom_margin_frac == "0.0125"
    withdraw = fake.sent("WithdrawMask")[0]
    assert withdraw.reason == "store_revoked" and not withdraw.restore
    assert withdraw.withdrawn_at == "2026-10-14T10:00:00.000000Z"

    code, lines = _run(db_session, url, batch_size=200)
    assert code == rp.EXIT_DONE
    for rpc in ("ReportDerivative", "ReportMask", "WithdrawMask"):
        assert _total(lines, rpc, "stored") == 0, rpc
    assert _total(lines, "ReportDerivative", "calls") == 0  # nothing pending is leased
    assert len(db_session.execute(select(SpineReportRun)).scalars().all()) == 2


def test_a_spine_that_ignores_the_filter_is_a_deployment_fault_that_stops_the_run(
    db_session: Session, spine: tuple[FakeSpine, str]
) -> None:
    fake, url = spine
    fake.ignore_filter = True  # a server from before 0.10.0
    a = seed_copy(db_session, "a")
    stranger = fake.depict(sha("orig:stranger"))
    mine = fake.depict(a["origin"])
    code, lines = _run(db_session, url)
    assert code == rp.EXIT_STOPPED
    assert fake.depictions[mine]["state"] == "derived"  # ours in that answer still land
    assert fake.depictions[stranger]["state"] == "captured"  # never reported
    batch = _batches(lines)[0]
    assert batch["unrequested"] == 1 and batch["lease_rounds"] == 1
    assert batch["ReportMask"]["calls"] == 0  # the batch stops before its masks
    stop = lines[-1]
    assert stop["event"] == "stopped" and "not honouring origin_sha256" in stop["why"]
    run = db_session.execute(select(SpineReportRun)).scalar_one()
    assert run.status == "running" and run.cursor_sha256 is None
    assert run.resume_not_before is not None  # the stranger's hold is waited out


def test_the_lease_is_repeated_until_the_batchs_depictions_are_drained(
    db_session: Session, spine: tuple[FakeSpine, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, url = spine
    a = seed_copy(db_session, "a")
    for _ in range(5):
        fake.depict(a["origin"])
    monkeypatch.setattr(rp, "LEASE_MAX_ITEMS", 2)
    code, lines = _run(db_session, url)
    assert code == rp.EXIT_DONE
    assert _batches(lines)[0]["lease_rounds"] == 3
    assert {d["state"] for d in fake.depictions.values()} == {"derived"}


def test_a_lease_that_failed_an_attempt_stops_the_run_until_its_hold_lapses(
    db_session: Session, spine: tuple[FakeSpine, str]
) -> None:
    fake, url = spine
    a = seed_copy(db_session, "a")
    b = seed_copy(db_session, "b")
    fake.depict(a["origin"])
    fake.depict(b["origin"])
    fake.fail("LeasePendingImages", C.UNAVAILABLE)
    clock = Clock()
    code, lines = _run(db_session, url, clock=clock, batch_size=1, lease_seconds=60)
    assert code == rp.EXIT_STOPPED
    run = db_session.execute(select(SpineReportRun)).scalar_one()
    assert run.status == "running" and run.cursor_sha256 is None
    assert run.resume_not_before is not None
    stop = lines[-1]
    assert stop["event"] == "stopped" and "lease" in stop["why"]

    code, lines = _run(db_session, url, clock=clock, batch_size=1, lease_seconds=60)
    assert code == rp.EXIT_DONE
    assert clock.slept and clock.slept[0] == pytest.approx(60 + rp.LEASE_MARGIN_SECONDS)
    assert lines[0]["event"] == "resume" and lines[0]["run"] == run.id
    assert {d["state"] for d in fake.depictions.values()} == {"derived"}
    assert len(db_session.execute(select(SpineReportRun)).scalars().all()) == 1


def test_a_fatal_status_stops_the_run_and_a_rerun_resumes_after_its_cursor(
    db_session: Session, spine: tuple[FakeSpine, str]
) -> None:
    fake, url = spine
    copies = [seed_copy(db_session, f"c{i}") for i in range(3)]
    for c in copies:
        fake.depict(c["origin"])
    ordered = sorted(c["origin"] for c in copies)
    fake.fail("ReportDerivative", None, C.PERMISSION_DENIED)  # the second report
    clock = Clock()
    code, lines = _run(db_session, url, clock=clock, batch_size=1)
    assert code == rp.EXIT_STOPPED
    assert lines[-1]["event"] == "stopped" and "PERMISSION_DENIED" in lines[-1]["why"]
    assert len(fake.sent("ReportDerivative")) == 2  # a fatal status is never retried
    run = db_session.execute(select(SpineReportRun)).scalar_one()
    assert run.cursor_sha256 == ordered[0] and run.resume_not_before is not None

    fake.held.clear()  # the stopped batch's hold lapses while the rerun waits
    code, lines = _run(db_session, url, clock=clock, batch_size=1)
    assert code == rp.EXIT_DONE
    assert [b["origins"] for b in _batches(lines)] == [1, 1, 0]
    assert {d["state"] for d in fake.depictions.values()} == {"derived"}
    assert (
        db_session.execute(select(SpineReportRun)).scalar_one().counts["ReportDerivative.calls"]
        == 4
    )


def test_restart_closes_an_open_run_and_starts_over(
    db_session: Session, spine: tuple[FakeSpine, str]
) -> None:
    _, url = spine
    seed_copy(db_session, "a", masks=())
    open_run = SpineReportRun(
        status="running", cursor_sha256="f" * 64, counts={}, batch_size=200, started_at=AT
    )
    db_session.add(open_run)
    db_session.commit()
    code, lines = _run(db_session, url, restart=True)
    assert code == rp.EXIT_DONE
    runs = db_session.execute(select(SpineReportRun).order_by(SpineReportRun.id)).scalars().all()
    assert [r.status for r in runs] == ["abandoned", "done"]
    assert _batches(lines)[0]["origins"] == 1


def test_a_restart_inside_the_abandoned_runs_hold_waits_it_out_and_lands_the_mask(
    db_session: Session,
) -> None:
    clock = Clock()
    fake = LapsingSpine(clock)
    a = seed_copy(db_session, "a")
    pid = fake.depict(a["origin"])
    fake.hold_then_fail.append(C.UNAVAILABLE)  # run 1's lease holds the row, then fails
    with serve(fake) as url:
        code, _ = _run(db_session, url, clock=clock, attempts=1, lease_seconds=60)
        assert code == rp.EXIT_STOPPED
        held = db_session.execute(select(SpineReportRun)).scalar_one().resume_not_before
        assert held is not None
        hold = rp._aware(held)
        assert hold > clock.now
        code, lines = _run(db_session, url, clock=clock, attempts=1, lease_seconds=60, restart=True)
    # never 'done' while the depiction stays captured and the mask never landed
    assert not (code == rp.EXIT_DONE and fake.depictions[pid]["state"] == "captured")
    assert code == rp.EXIT_DONE and clock.now >= hold
    assert fake.depictions[pid]["state"] == "derived" and len(fake.masks) == 1
    assert _batches(lines)[0]["orphans"] == []


def test_a_withdrawal_is_recorded_before_it_is_sent_and_a_restore_follows_it(
    db_session: Session, spine: tuple[FakeSpine, str]
) -> None:
    fake, url = spine
    a = seed_copy(db_session, "a", masks=(("mask-v1", "withdrawn", "ma"),))
    fake.depict(a["origin"])
    fake.fail("WithdrawMask", C.UNAUTHENTICATED)
    code, _ = _run(db_session, url)
    assert code == rp.EXIT_STOPPED
    recorded = db_session.execute(select(SpineMaskWithdrawal)).scalar_one()
    assert recorded.mask_sha256 == a["masks"][0] and recorded.reason == "store_revoked"

    code, _ = _run(db_session, url)
    assert code == rp.EXIT_DONE
    assert not fake.published(a["masks"][0])

    row = db_session.execute(select(ImageMask)).scalar_one()
    row.status, row.reason = "ok", None  # restored in media-manager
    db_session.commit()
    code, lines = _run(db_session, url)
    assert code == rp.EXIT_DONE
    restore = [w for w in fake.sent("WithdrawMask") if w.restore]
    assert len(restore) == 1 and restore[0].mask_sha256 == bytes.fromhex(a["masks"][0])
    assert fake.published(a["masks"][0])
    assert db_session.execute(select(SpineMaskWithdrawal)).first() is None
    assert _total(lines, "RestoreMask", "calls") == 1


def test_an_original_the_spine_has_no_depiction_for_is_an_orphan_not_a_failure(
    db_session: Session, spine: tuple[FakeSpine, str]
) -> None:
    fake, url = spine
    # the spine never captured either original: nothing to lease, nothing registered
    a = seed_copy(
        db_session,
        "a",
        extra_originals=("a2", "a3", "a4", "a5"),
        masks=(("mask-v1", "ok", "m"), ("mask-v2", "ok", "n")),
    )
    code, lines = _run(db_session, url, attempts=5)
    assert code == rp.EXIT_DONE  # expected (ruling 10-15): counted, never exit 1
    assert lines[-1]["exit"] == rp.EXIT_DONE
    assert len(fake.sent("ReportMask")) == 2  # not retried
    batch = _batches(lines)[0]
    assert batch["orphans"] == sorted([a["origin"], *a["others"]])  # the blobs, once each
    assert batch["ReportMask"]["orphan"] == 2
    assert batch["ReportMask"]["unregistered"] == 0 and batch["ReportMask"]["refused"] == {}
    counts = db_session.execute(select(SpineReportRun)).scalar_one().counts
    assert counts["orphan"] == 5 and counts["ReportMask.orphan"] == 2
    assert "ReportMask.unregistered" not in counts


def test_a_mask_whose_landed_derivative_the_spine_still_lacks_is_unregistered_exit_1(
    db_session: Session, spine: tuple[FakeSpine, str]
) -> None:
    fake, url = spine
    a = seed_copy(db_session, "a")
    b = seed_copy(db_session, "b")
    fake.depict(a["origin"])
    fake.depict(b["origin"])
    fake.fail("ReportMask", C.FAILED_PRECONDITION, C.FAILED_PRECONDITION)  # the first mask's
    code, lines = _run(db_session, url, attempts=2)
    assert code == rp.EXIT_REFUSED  # this run reported the derivative: a mask never landed
    run = db_session.execute(select(SpineReportRun)).scalar_one()
    assert run.status == "done"  # counted, the run went on
    batch = _batches(lines)[0]
    assert batch["ReportMask"]["unregistered"] == 1 and batch["ReportMask"]["orphan"] == 0
    assert batch["orphans"] == [] and batch["ReportMask"]["ok"] == 1
    assert len(fake.sent("ReportMask")) == 3  # the first retried once, the second sent
    first_copy = min(a["clean"], b["clean"])  # copies are reported in sha order
    assert fake.sent("ReportMask")[0].derivative_sha256 == bytes.fromhex(first_copy)


def test_a_mask_whose_derivative_landed_is_retried_on_failed_precondition(
    db_session: Session, spine: tuple[FakeSpine, str]
) -> None:
    fake, url = spine
    a = seed_copy(db_session, "a")
    fake.depict(a["origin"])
    fake.fail("ReportMask", C.FAILED_PRECONDITION)
    code, lines = _run(db_session, url, attempts=3)
    assert code == rp.EXIT_DONE
    assert len(fake.sent("ReportMask")) == 2
    assert _batches(lines)[0]["ReportMask"]["stored"] == 1


def test_a_derivative_landed_in_an_earlier_batch_still_makes_its_mask_unregistered(
    db_session: Session, spine: tuple[FakeSpine, str]
) -> None:
    fake, url = spine
    a = seed_copy(db_session, "a", extra_originals=("a2",))
    first, last = sorted([a["origin"], *a["others"]])
    fake.depict(first)  # the copy lands in batch 0; its masks are due in batch 1
    fake.fail("ReportMask", C.FAILED_PRECONDITION, C.FAILED_PRECONDITION)
    code, lines = _run(db_session, url, attempts=2, batch_size=1)
    assert code == rp.EXIT_REFUSED
    landed, due = _batches(lines)[:2]
    assert landed["ReportDerivative"]["ok"] == 1 and landed["ReportMask"]["calls"] == 0
    assert due["batch"] == 1 and due["ReportMask"]["unregistered"] == 1
    assert due["ReportMask"]["orphan"] == 0 and due["orphans"] == []
    assert len(fake.sent("ReportMask")) == 2  # retried: the derivative had landed


def _held_then_newer(db: Session, fake: FakeSpine, url: str) -> dict[str, Any]:
    """REPLAY-REDERIV: original x, a capture behind copy ``held``, is derived
    on the spine with ``held`` and its mask; then media mints a NEWER copy
    from x, so the replay now maps x to it. The spine's registry is write-once
    per original: it keeps ``held``."""
    held = seed_copy(db, "held", extra_originals=("x",), masks=(("mask-v1", "ok", "m-held"),))
    x = held["others"][0]
    first = fake.depict(x)
    code, _ = _run(db, url)
    assert code == rp.EXIT_DONE and fake.published(held["masks"][0])
    newer = seed_copy(db, "newer", minted_by=x, masks=(("mask-v1", "ok", "m-newer"),))
    assert rp.load_plan(db, details=False).copy_of[x] == newer["clean"]
    return {"x": x, "held": held, "newer": newer, "first": first}


def _newer_masks_sent(fake: FakeSpine, newer: dict[str, Any]) -> int:
    return sum(
        m.derivative_sha256 == bytes.fromhex(newer["clean"]) for m in fake.sent("ReportMask")
    )


def test_a_newer_copy_the_spine_keeps_out_for_a_new_depiction_is_not_a_failure(
    db_session: Session, spine: tuple[FakeSpine, str]
) -> None:
    """A NEW depiction of x is leased: its report of the newer copy is absorbed
    (stored=false, warned: the held copy stands), so that copy's mask cannot
    land. Not ``unregistered`` (exit 1) and never retried: ``kept_other``."""
    fake, url = spine
    s = _held_then_newer(db_session, fake, url)
    second = fake.depict(s["x"])  # the same original on another product
    code, lines = _run(db_session, url, attempts=5)
    assert code == rp.EXIT_DONE, lines
    assert fake.depictions[second]["state"] == "derived"
    assert fake.by_origin[bytes.fromhex(s["x"])] == bytes.fromhex(s["held"]["clean"])
    assert fake.published(s["held"]["masks"][0]) and not fake.published(s["newer"]["masks"][0])
    assert _newer_masks_sent(fake, s["newer"]) == 1  # not retried: its derivative never landed
    assert _total(lines, "ReportDerivative", "warnings") == 1
    assert _total(lines, "ReportMask", "kept_other") == 1
    assert _total(lines, "ReportMask", "unregistered") == 0
    assert _total(lines, "ReportMask", "orphan") == 0
    assert all(b["orphans"] == [] for b in _batches(lines))
    counts = lines[-1]["counts"]
    assert counts["ReportMask.kept_other"] == 1 and "ReportMask.unregistered" not in counts


def test_a_newer_copy_of_an_original_already_derived_is_an_orphan_and_nothing_flips(
    db_session: Session, spine: tuple[FakeSpine, str]
) -> None:
    """No depiction of x is pending (the prod shape, 10-31): nothing is
    leased or reported for it, the newer copy's mask is refused as an orphan
    (never a failure, never a stop), and the spine keeps showing the held copy
    with its mask."""
    fake, url = spine
    s = _held_then_newer(db_session, fake, url)
    code, lines = _run(db_session, url, attempts=5)
    assert code == rp.EXIT_DONE, lines
    assert _total(lines, "ReportDerivative", "calls") == 0
    assert _newer_masks_sent(fake, s["newer"]) == 1
    assert _total(lines, "ReportMask", "orphan") == 1
    assert sorted(o for b in _batches(lines) for o in b["orphans"]) == [s["x"]]
    assert fake.depictions[s["first"]]["state"] == "derived"
    assert fake.published(s["held"]["masks"][0])


def test_a_refused_report_is_counted_by_status_and_the_run_ends_1(
    db_session: Session, spine: tuple[FakeSpine, str]
) -> None:
    fake, url = spine
    a = seed_copy(db_session, "a")
    fake.depict(a["origin"])
    fake.fail("ReportDerivative", C.INVALID_ARGUMENT)
    code, lines = _run(db_session, url)
    assert code == rp.EXIT_REFUSED
    assert _batches(lines)[0]["ReportDerivative"]["refused"] == {"INVALID_ARGUMENT": 1}
    assert db_session.execute(select(SpineReportRun)).scalar_one().status == "done"


def test_a_spine_that_stays_down_stops_the_run(
    db_session: Session, spine: tuple[FakeSpine, str]
) -> None:
    fake, url = spine
    a = seed_copy(db_session, "a")
    fake.depict(a["origin"])
    fake.fail("ReportMask", C.UNAVAILABLE, C.UNAVAILABLE)
    code, lines = _run(db_session, url, attempts=2)
    assert code == rp.EXIT_STOPPED
    assert "UNAVAILABLE" in lines[-1]["why"]


def test_a_mask_that_cannot_be_sent_whole_is_refused_locally(
    db_session: Session, spine: tuple[FakeSpine, str]
) -> None:
    fake, url = spine
    a = seed_copy(db_session, "a")
    fake.depict(a["origin"])
    row = db_session.execute(select(ImageMask)).scalar_one()
    row.box_top_frac = None
    db_session.commit()
    code, lines = _run(db_session, url)
    assert code == rp.EXIT_REFUSED
    assert fake.sent("ReportMask") == []
    assert _batches(lines)[0]["local"]["refused"] == {"box_top_frac": 1}


# ---------------------------------------------------------------------------
# the dry run
# ---------------------------------------------------------------------------


def test_a_dry_run_lists_what_it_would_send_and_opens_nothing(db_session: Session) -> None:
    a = seed_copy(db_session, "a")
    b = seed_copy(db_session, "b", masks=(("mask-v1", "withdrawn", "mb"),))
    code, lines = _run(db_session, None, dry_run=True)
    assert code == rp.EXIT_DONE
    would = [line for line in lines if line["event"] == "would_send"]
    rpcs = [w["rpc"] for w in would]
    assert rpcs.count("LeasePendingImages") == 1
    assert rpcs.count("ReportDerivative") == 2 and rpcs.count("ReportMask") == 1
    assert rpcs.count("WithdrawMask") == 1
    lease = next(w for w in would if w["rpc"] == "LeasePendingImages")
    assert sorted(lease["origin_sha256"]) == sorted([a["origin"], b["origin"]])
    withdraw = next(w for w in would if w["rpc"] == "WithdrawMask")
    assert withdraw["mask_sha256"] == b["masks"][0] and withdraw["restore"] is False
    assert db_session.execute(select(SpineReportRun)).first() is None
    assert db_session.execute(select(SpineMaskWithdrawal)).first() is None
    assert lines[-1]["event"] == "done" and lines[-1]["dry_run"] is True


def test_config_is_checked_before_anything_runs(db_session: Session) -> None:
    for bad in ({"batch_size": 201}, {"lease_seconds": 0}, {"lease_seconds": 3601}):
        with pytest.raises(ValueError):
            rp.ReplayConfig(**bad)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# edges
# ---------------------------------------------------------------------------


def test_a_withdrawn_row_with_no_reason_is_withdrawn_with_a_stated_one(
    db_session: Session, spine: tuple[FakeSpine, str]
) -> None:
    fake, url = spine
    seed_copy(db_session, "a", masks=(("mask-v1", "withdrawn", "ma"),))
    row = db_session.execute(select(ImageMask)).scalar_one()
    row.reason = None
    db_session.commit()
    code, lines = _run(db_session, url)
    assert code == rp.EXIT_DONE
    assert [w.reason for w in fake.sent("WithdrawMask")] == [rp.NO_REASON]
    assert _batches(lines)[0]["local"]["refused"] == {}


def test_a_copy_with_no_mask_rows_is_reported_without_an_appearance(
    db_session: Session, spine: tuple[FakeSpine, str]
) -> None:
    fake, url = spine
    a = seed_copy(db_session, "a", masks=())
    fake.depict(a["origin"])
    code, _ = _run(db_session, url)
    assert code == rp.EXIT_DONE
    sent = fake.sent("ReportDerivative")[0]
    assert not sent.HasField("thumbhash") and not sent.HasField("dominant_color")


def test_a_collectors_own_photo_judged_ineligible_still_has_its_clean_copy_reported(
    db_session: Session, spine: tuple[FakeSpine, str]
) -> None:
    """An MFC /upload/pictures/ photo gets no mask (mask_policy: user_photo),
    but its sanitized copy is public and must reach the spine, or it never
    shows in the picker."""
    fake, url = spine
    pic = seed_copy(db_session, "pic", masks=())
    db_session.add(
        ImageMask(
            source_sha256=pic["clean"],
            pipeline_version="mask-v1",
            status="ineligible",
            reason="user_photo",
            origin_class="user_photo",
            created_at=AT,
            updated_at=AT,
        )
    )
    db_session.commit()
    plan = rp.load_plan(db_session)
    assert plan.copy_of == {pic["origin"]: pic["clean"]} and pic["clean"] in plan.live
    assert plan.masked == set() and plan.reports_for(pic["clean"]) == []
    pid = fake.depict(pic["origin"])
    code, lines = _run(db_session, url)
    assert code == rp.EXIT_DONE
    sent = fake.sent("ReportDerivative")
    assert [r.derivative_sha256.hex() for r in sent] == [pic["clean"]]
    assert fake.depictions[pid]["state"] == "derived"
    assert fake.sent("ReportMask") == [] and fake.sent("WithdrawMask") == []
    assert _batches(lines)[0]["local"]["refused"] == {}


def test_a_lease_the_mesh_refuses_stops_the_run(
    db_session: Session, spine: tuple[FakeSpine, str]
) -> None:
    fake, url = spine
    seed_copy(db_session, "a")
    fake.fail("LeasePendingImages", C.PERMISSION_DENIED)
    code, lines = _run(db_session, url)
    assert code == rp.EXIT_STOPPED
    assert lines[-1]["why"].startswith("lease: LeasePendingImages PERMISSION_DENIED")
    assert _batches(lines)[0]["lease_failed_attempts"] == 1


def test_a_lease_that_never_drains_stops_the_run(
    db_session: Session, spine: tuple[FakeSpine, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, url = spine
    a = seed_copy(db_session, "a")
    for _ in range(3):
        fake.depict(a["origin"])
    monkeypatch.setattr(rp, "LEASE_MAX_ITEMS", 1)
    monkeypatch.setattr(rp, "MAX_LEASE_ROUNDS", 2)
    code, lines = _run(db_session, url)
    assert code == rp.EXIT_STOPPED
    batch = _batches(lines)[0]
    assert batch["lease_rounds"] == 2 and batch["matched"] == 2
    assert any("after 2 rounds" in w for w in batch["warnings_sample"])


def test_a_copy_the_field_rule_will_not_send_is_refused_locally(
    db_session: Session, spine: tuple[FakeSpine, str]
) -> None:
    from app.models import PublicObject

    fake, url = spine
    a = seed_copy(db_session, "a", masks=())
    db_session.get(PublicObject, a["clean"]).mime = "image/gif"  # type: ignore[union-attr]
    db_session.commit()
    fake.depict(a["origin"])
    code, lines = _run(db_session, url)
    assert code == rp.EXIT_REFUSED
    assert fake.sent("ReportDerivative") == []
    assert _batches(lines)[0]["local"]["refused"] == {"mime": 1}


def test_a_withdrawn_mask_with_unsendable_bytes_is_refused_locally(
    db_session: Session, spine: tuple[FakeSpine, str]
) -> None:
    fake, url = spine
    seed_copy(db_session, "a", masks=(("mask-v1", "withdrawn", "ma"),))
    row = db_session.execute(select(ImageMask)).scalar_one()
    row.mask_sha256 = row.mask_sha256.upper()  # type: ignore[union-attr]
    db_session.commit()
    code, lines = _run(db_session, url)
    assert code == rp.EXIT_REFUSED
    assert fake.sent("WithdrawMask") == []
    assert _batches(lines)[0]["local"]["refused"] == {"mask_sha256": 1}


def test_one_mask_shared_by_two_copies_is_withdrawn_and_restored_once(
    db_session: Session, spine: tuple[FakeSpine, str]
) -> None:
    fake, url = spine
    for tag in ("a", "b"):
        fake.depict(
            seed_copy(db_session, tag, masks=(("mask-v1", "withdrawn", "shared"),))["origin"]
        )
    code, _ = _run(db_session, url)
    assert code == rp.EXIT_DONE
    assert len(fake.sent("WithdrawMask")) == 1
    for row in db_session.execute(select(ImageMask)).scalars():
        row.status, row.reason = "ok", None
    db_session.commit()
    fake.calls.clear()
    code, _ = _run(db_session, url)
    assert code == rp.EXIT_DONE
    assert [w.restore for w in fake.sent("WithdrawMask")] == [True]


def test_refused_withdrawals_and_restores_are_counted_and_kept(
    db_session: Session, spine: tuple[FakeSpine, str]
) -> None:
    fake, url = spine
    a = seed_copy(db_session, "a", masks=(("mask-v1", "withdrawn", "ma"),))
    fake.fail("WithdrawMask", C.INVALID_ARGUMENT)
    code, lines = _run(db_session, url)
    assert code == rp.EXIT_REFUSED
    assert _batches(lines)[0]["WithdrawMask"]["refused"] == {"INVALID_ARGUMENT": 1}
    row = db_session.execute(select(ImageMask)).scalar_one()
    row.status, row.reason = "ok", None
    db_session.commit()
    fake.fail("WithdrawMask", C.INVALID_ARGUMENT)
    code, lines = _run(db_session, url)
    assert code == rp.EXIT_REFUSED
    assert sum(b["RestoreMask"]["refused"].get("INVALID_ARGUMENT", 0) for b in _batches(lines)) == 1
    # the restore is still owed: the ledger keeps the withdrawal
    assert db_session.get(SpineMaskWithdrawal, a["masks"][0]) is not None


def test_a_refusal_in_a_run_whose_batches_carry_no_originals_still_ends_1(
    db_session: Session, spine: tuple[FakeSpine, str]
) -> None:
    fake, url = spine
    seed_copy(db_session, "dead", deleted=True)  # only the tail: withdraw its mask
    fake.fail("WithdrawMask", C.INVALID_ARGUMENT)
    code, lines = _run(db_session, url)
    assert [b["origins"] for b in _batches(lines)] == [0]
    assert _batches(lines)[0]["WithdrawMask"]["refused"] == {"INVALID_ARGUMENT": 1}
    run = db_session.execute(select(SpineReportRun)).scalar_one()
    assert run.cursor_sha256 is None
    assert code == rp.EXIT_REFUSED and lines[-1]["exit"] == rp.EXIT_REFUSED


def test_a_dry_run_lists_a_restore_it_owes(db_session: Session) -> None:
    a = seed_copy(db_session, "a")
    db_session.add(SpineMaskWithdrawal(mask_sha256=a["masks"][0], reason="r", withdrawn_at=AT))
    db_session.commit()
    _, lines = _run(db_session, None, dry_run=True)
    restores = [x for x in lines if x.get("rpc") == "WithdrawMask" and x.get("restore")]
    assert [r["mask_sha256"] for r in restores] == [a["masks"][0]]


@pytest.mark.parametrize("pause", [None, -30.0])
def test_a_resumed_run_with_no_hold_left_does_not_wait(
    db_session: Session, spine: tuple[FakeSpine, str], pause: float | None
) -> None:
    _, url = spine
    seed_copy(db_session, "a", masks=())
    clock = Clock()
    resume = None if pause is None else T0 + dt.timedelta(seconds=pause)
    db_session.add(
        SpineReportRun(
            status="running",
            counts={},
            batch_size=200,
            started_at=AT,
            resume_not_before=resume,
        )
    )
    db_session.commit()
    code, lines = _run(db_session, url, clock=clock)
    assert code == rp.EXIT_DONE
    assert clock.slept == []
    assert lines[0] == {"event": "resume", "run": 1, "cursor": None, "wait_seconds": 0.0}


def test_leasing_stops_at_the_first_stray_even_in_a_full_answer(
    db_session: Session, spine: tuple[FakeSpine, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, url = spine
    fake.ignore_filter = True
    a = seed_copy(db_session, "a")
    fake.depict(sha("orig:stranger-1"))
    fake.depict(a["origin"])
    fake.depict(sha("orig:stranger-2"))
    monkeypatch.setattr(rp, "LEASE_MAX_ITEMS", 2)
    code, lines = _run(db_session, url)
    assert code == rp.EXIT_STOPPED
    batch = _batches(lines)[0]
    assert batch["lease_rounds"] == 1 and batch["unrequested"] == 1
    assert len(fake.sent("LeasePendingImages")) == 1  # never walks the stranger's queue


def test_the_cursor_is_the_last_original_of_the_last_finished_batch(
    db_session: Session, spine: tuple[FakeSpine, str]
) -> None:
    fake, url = spine
    for i in range(4):
        fake.depict(seed_copy(db_session, f"c{i}")["origin"])
    fake.fail("ReportDerivative", None, None, C.UNIMPLEMENTED)  # the second batch's first
    code, _ = _run(db_session, url, batch_size=2)
    assert code == rp.EXIT_STOPPED
    run = db_session.execute(select(SpineReportRun)).scalar_one()
    first_batch = sorted(fake.depictions[p]["origin"].hex() for p in fake.depictions)[:2]
    assert run.cursor_sha256 == first_batch[-1]
    fake.held.clear()
    code, lines = _run(db_session, url, batch_size=2)
    assert code == rp.EXIT_DONE
    assert [b["origins"] for b in _batches(lines)] == [2, 0]


# ---------------------------------------------------------------------------
# close-out 2: every lease round's hold, a spine that ignores the filter
# ---------------------------------------------------------------------------


class _PodKilled(BaseException):
    """The pod is killed: nothing after it runs, nothing uncommitted survives."""


class _KilledAtLease(SpineMediaClient):
    def __init__(self, *args: Any, kill_at: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.kill_at = kill_at
        self.leases = 0

    def lease(self, *args: Any, **kwargs: Any) -> Any:
        self.leases += 1
        out = super().lease(*args, **kwargs)
        if self.leases == self.kill_at:
            raise _PodKilled  # after the spine held the rows
        return out


def _three_depictions_of_one_original(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> tuple[Clock, LapsingSpine, list[str]]:
    clock = Clock()
    fake = LapsingSpine(clock, report_seconds=40)
    a = seed_copy(db, "a")
    pids = [fake.depict(a["origin"]) for _ in range(3)]
    monkeypatch.setattr(rp, "LEASE_MAX_ITEMS", 2)
    return clock, fake, pids


def test_a_rerun_waits_out_the_hold_of_a_later_lease_round_that_failed(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock, fake, pids = _three_depictions_of_one_original(db_session, monkeypatch)
    fake.hold_then_fail.extend([None, C.UNAVAILABLE])  # round 2 holds, then fails
    with serve(fake) as url:
        code, lines = _run(db_session, url, clock=clock, attempts=1, lease_seconds=60)
        assert code == rp.EXIT_STOPPED and "lease" in lines[-1]["why"]
        assert clock.now == T0 + dt.timedelta(seconds=80)  # two reports of 40 s
        code, lines = _run(db_session, url, clock=clock, attempts=1, lease_seconds=60)
    assert code == rp.EXIT_DONE
    # round 2 leased at T0+80: its hold lapses at T0+140, the rerun waits to T0+145
    assert lines[0]["event"] == "resume" and lines[0]["wait_seconds"] == 65.0
    assert {fake.depictions[p]["state"] for p in pids} == {"derived"}


def test_a_pod_killed_in_a_later_lease_round_has_that_rounds_hold_committed(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock, fake, pids = _three_depictions_of_one_original(db_session, monkeypatch)
    with serve(fake) as url:

        def connect() -> SpineMediaClient:
            return _KilledAtLease(
                open_channel(url),
                kill_at=2,
                policy=RetryPolicy(max_attempts=1),
                sleep=lambda s: None,
            )

        with pytest.raises(_PodKilled):
            rp.run_replay(
                db_session,
                rp.ReplayConfig(lease_seconds=60),
                connect=connect,
                emit=lambda line: None,
                now=clock,
                sleep=clock.sleep,
            )
        db_session.rollback()  # what the killed pod had not committed is gone
        run = db_session.execute(select(SpineReportRun)).scalar_one()
        assert run.resume_not_before is not None
        # round 2 leased at T0+80: 60 s of hold, 5 s of margin, and the call's
        # own worst case (one attempt of a 10 s deadline)
        assert rp._aware(run.resume_not_before) == T0 + dt.timedelta(seconds=155)
        code, lines = _run(db_session, url, clock=clock, attempts=1, lease_seconds=60)
    assert code == rp.EXIT_DONE
    assert lines[0]["wait_seconds"] == 75.0
    assert {fake.depictions[p]["state"] for p in pids} == {"derived"}


def test_a_spine_that_ignores_the_filter_never_has_its_queue_walked(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = Clock()
    fake = LapsingSpine(clock)
    fake.ignore_filter = True
    mine = [seed_copy(db_session, f"m{i}") for i in range(4)]
    strangers = [fake.depict(sha(f"orig:stranger-{i}")) for i in range(6)]
    for m in mine:
        fake.depict(m["origin"])
    monkeypatch.setattr(rp, "LEASE_MAX_ITEMS", 2)
    with serve(fake) as url:
        code, lines = _run(db_session, url, clock=clock, batch_size=1, lease_seconds=60)
    assert code == rp.EXIT_STOPPED
    assert len(fake.sent("LeasePendingImages")) == 1
    assert sum(fake.held_now(p) for p in strangers) == 2  # one answer's worth, no more
    assert lines[-1]["batch"] == 0 and "deployment fault" in lines[-1]["why"]


# ---------------------------------------------------------------------------
# close-out 2: what each report carries, pinned at the replay
# ---------------------------------------------------------------------------


def test_the_mask_carries_its_masked_appearance_and_the_copy_its_plain_one(
    db_session: Session, spine: tuple[FakeSpine, str]
) -> None:
    fake, url = spine
    a = seed_copy(db_session, "a")
    fake.depict(a["origin"])
    mask_obj = db_session.get(PublicObject, a["masks"][0])
    assert mask_obj is not None
    mask_obj.created_at = AT + dt.timedelta(minutes=2)  # neither the row's created nor updated
    mask_obj.mime = "image/webp"  # the object's own type, never assumed
    db_session.commit()
    code, _ = _run(db_session, url)
    assert code == rp.EXIT_DONE
    derived = fake.sent("ReportDerivative")[0]
    assert derived.thumbhash == base64.b64decode("AAECAwQFBgcICQ==")
    assert derived.dominant_color == "#112233"
    assert derived.generated_at == "2026-10-14T09:00:00.000000Z"
    assert not derived.HasField("grounding")  # an opaque copy
    mask = fake.sent("ReportMask")[0]
    assert mask.thumbhash == base64.b64decode("CQgHBgUEAwIBAA==")
    assert mask.dominant_color == "#445566"
    assert mask.generated_at == "2026-10-14T09:02:00.000000Z"  # the mask object's instant
    assert mask.mime == "image/webp"


def test_a_transparent_copy_is_reported_with_its_own_grounding(
    db_session: Session, spine: tuple[FakeSpine, str]
) -> None:
    fake, url = spine
    own = {**GROUND, "bottom_margin_frac": 0.05}
    a = seed_copy(db_session, "a", mime="image/png", own_grounding=own)
    fake.depict(a["origin"])
    code, _ = _run(db_session, url)
    assert code == rp.EXIT_DONE
    derived = fake.sent("ReportDerivative")[0]
    assert derived.grounding.bottom_margin_frac == "0.05"
    assert fake.sent("ReportMask")[0].grounding.bottom_margin_frac == "0.0125"


def test_a_held_mask_is_never_sent(db_session: Session, spine: tuple[FakeSpine, str]) -> None:
    fake, url = spine
    a = seed_copy(db_session, "a", masks=(("mask-v1", "held", "mh"),))
    fake.depict(a["origin"])
    code, lines = _run(db_session, url)
    assert code == rp.EXIT_DONE
    assert fake.sent("ReportMask") == [] and fake.sent("WithdrawMask") == []
    assert len(fake.sent("ReportDerivative")) == 1


def test_the_copys_appearance_is_the_current_versions_else_the_latest_measured(
    db_session: Session,
) -> None:
    def rows(tag: str, *spec: tuple[str, str | None, int]) -> str:
        """(pipeline version, plain thumbhash or None = unmeasured, hours after AT)"""
        seed = seed_copy(db_session, tag, masks=tuple((v, "ok", f"{tag}{v}") for v, _, _ in spec))
        found = db_session.execute(
            select(ImageMask).where(ImageMask.source_sha256 == seed["clean"]).order_by(ImageMask.id)
        ).scalars()
        for row, (_, thumb, hours) in zip(found, spec, strict=True):
            row.plain_thumbhash = thumb
            row.plain_dominant_color = None if thumb is None else "#000000"
            row.updated_at = AT + dt.timedelta(hours=hours)
        db_session.commit()
        return str(seed["clean"])

    current = rows("cur", ("mask-v1", "Y3VycmVudA==", 1), ("mask-v0", "b2xkZXJyb3c=", 2))
    latest = rows("lat", ("mask-v0", "ZWFybGllcg==", 1), ("mask-v00", "bGF0ZXN0IQ==", 2))
    unmeasured = rows("unm", ("mask-v0", "bWVhc3VyZWQ=", 1), ("mask-v00", None, 2))
    copies = rp.load_plan(db_session).copies
    assert copies[current].thumbhash == "Y3VycmVudA=="  # the current version wins
    assert copies[latest].thumbhash == "bGF0ZXN0IQ=="  # else the latest measured
    assert copies[unmeasured].thumbhash == "bWVhc3VyZWQ="  # an unmeasured row never
    assert copies[unmeasured].dominant_color == "#000000"


def test_a_dry_run_lists_only_what_follows_an_unfinished_runs_cursor(
    db_session: Session,
) -> None:
    origins = sorted(seed_copy(db_session, f"c{i}")["origin"] for i in range(3))
    db_session.add(
        SpineReportRun(
            status="running",
            counts={},
            batch_size=200,
            started_at=AT,
            cursor_sha256=origins[0],
        )
    )
    db_session.commit()

    def leased(**config: Any) -> list[str]:
        _, lines = _run(db_session, None, dry_run=True, **config)
        lease = next(w for w in lines if w.get("rpc") == "LeasePendingImages")
        return sorted(lease["origin_sha256"])

    assert leased() == origins[1:]
    assert leased(restart=True) == origins


def test_an_ok_mask_with_no_public_object_is_refused_by_name_not_given_a_mime(
    db_session: Session, spine: tuple[FakeSpine, str]
) -> None:
    fake, url = spine
    a = seed_copy(db_session, "a")
    fake.depict(a["origin"])
    db_session.delete(db_session.get(PublicObject, a["masks"][0]))
    db_session.commit()
    assert rp.load_plan(db_session).reports_for(a["clean"]) == []
    code, lines = _run(db_session, None, dry_run=True)
    assert code == rp.EXIT_DONE
    assert _batches(lines)[0]["local"]["refused"] == {"mask_object": 1}
    assert not [w for w in lines if w.get("rpc") == "ReportMask"]
    code, lines = _run(db_session, url)
    assert code == rp.EXIT_REFUSED
    assert fake.sent("ReportMask") == [] and fake.sent("WithdrawMask") == []
    batch = _batches(lines)[0]
    assert batch["local"]["refused"] == {"mask_object": 1}
    assert any(a["masks"][0] in w and "no public object" in w for w in batch["warnings_sample"])


def test_a_negative_zero_fraction_is_sent_as_zero(
    db_session: Session, spine: tuple[FakeSpine, str]
) -> None:
    fake, url = spine
    a = seed_copy(db_session, "a")
    fake.depict(a["origin"])
    row = db_session.execute(select(ImageMask)).scalar_one()
    row.bottom_margin_frac = -0.0
    db_session.commit()
    code, _ = _run(db_session, url)
    assert code == rp.EXIT_DONE
    assert fake.sent("ReportMask")[0].grounding.bottom_margin_frac == "0.0"


def test_a_repeated_warning_is_counted_every_time_and_quoted_once(
    db_session: Session, spine: tuple[FakeSpine, str]
) -> None:
    fake, url = spine
    a = seed_copy(db_session, "a", masks=())
    seed_copy(db_session, "b", masks=(("mask-v1", "ok", "mb"),))
    for row in db_session.execute(select(ImageMask)).scalars():
        row.plain_dominant_color = "red"  # off the contract spelling: dropped, warned
    db_session.commit()
    b_origin = sha("orig:b")
    for _ in range(3):
        fake.depict(b_origin)
    fake.depict(a["origin"])
    code, lines = _run(db_session, url)
    assert code == rp.EXIT_DONE
    batch = _batches(lines)[0]
    assert batch["local"]["warnings"] == 3
    assert (
        batch["warnings_sample"].count("local: dominant_color off the contract spelling: dropped")
        == 1
    )


# ---------------------------------------------------------------------------
# close-out 3: a hold taken inside one lease call's retries; the exit read
# from the persisted ledger; an orphan original
# ---------------------------------------------------------------------------


class _SlowLeaseSpine(LapsingSpine):
    """The first lease attempt is slow: ``slow_seconds`` pass on the clock,
    the rows are held from then, and the client's deadline has passed
    (DEADLINE_EXCEEDED). Later attempts answer."""

    def __init__(self, clock: Clock, slow_seconds: float) -> None:
        super().__init__(clock)
        self.slow = [slow_seconds]

    def LeasePendingImages(self, request: Any, context: grpc.ServicerContext) -> Any:  # noqa: N802
        if self.slow:
            self.clock.now += dt.timedelta(seconds=self.slow.pop())
            self.hold_then_fail.append(C.DEADLINE_EXCEEDED)
        return super().LeasePendingImages(request, context)


def _run_production_policy(db: Session, url: str, clock: Clock) -> tuple[int, list[dict[str, Any]]]:
    lines: list[dict[str, Any]] = []
    code = rp.run_replay(
        db,
        rp.ReplayConfig(lease_seconds=60),
        connect=lambda: SpineMediaClient(
            open_channel(url), policy=RetryPolicy(), sleep=clock.sleep
        ),
        emit=lines.append,
        now=clock,
        sleep=clock.sleep,
    )
    return code, lines


@pytest.mark.parametrize("slow", [10.0, 30.0])
def test_a_rerun_waits_out_a_hold_a_slow_attempt_took_inside_one_lease_call(
    db_session: Session, slow: float
) -> None:
    clock = Clock()
    fake = _SlowLeaseSpine(clock, slow)
    a = seed_copy(db_session, "a", masks=())
    pids = [fake.depict(a["origin"]) for _ in range(2)]
    with serve(fake) as url:
        code, lines = _run_production_policy(db_session, url, clock)
        assert code == rp.EXIT_STOPPED
        assert _batches(lines)[0]["lease_failed_attempts"] == 1
        run = db_session.execute(select(SpineReportRun)).scalar_one()
        held_until = rp._aware(run.resume_not_before)
        code, lines = _run_production_policy(db_session, url, clock)
    assert code == rp.EXIT_DONE
    assert {fake.depictions[p]["state"] for p in pids} == {"derived"}
    # the slow attempt holds until T0+slow+60; the call ended at T0+slow+0.5
    assert held_until == T0 + dt.timedelta(seconds=slow + 0.5 + 65)


def test_a_pod_killed_inside_a_lease_call_leaves_a_hold_covering_all_its_retries(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock, fake, _ = _three_depictions_of_one_original(db_session, monkeypatch)
    with serve(fake) as url:

        def connect() -> SpineMediaClient:
            return _KilledAtLease(
                open_channel(url), kill_at=1, policy=RetryPolicy(), sleep=lambda s: None
            )

        with pytest.raises(_PodKilled):
            rp.run_replay(
                db_session,
                rp.ReplayConfig(lease_seconds=60),
                connect=connect,
                emit=lambda line: None,
                now=clock,
                sleep=clock.sleep,
            )
        db_session.rollback()
    run = db_session.execute(select(SpineReportRun)).scalar_one()
    # 60 s of hold and 5 of margin, after 5 attempts of 10 s and 0.5+1+2+4 s of backoff
    assert rp._aware(run.resume_not_before) == T0 + dt.timedelta(seconds=122.5)


def test_after_a_clean_lease_call_the_hold_is_counted_from_its_answer(
    db_session: Session, spine: tuple[FakeSpine, str]
) -> None:
    fake, url = spine
    a = seed_copy(db_session, "a", masks=())
    fake.depict(a["origin"])
    fake.fail("ReportDerivative", C.UNIMPLEMENTED)  # the run stops after the lease
    code, _ = _run(db_session, url, attempts=2)
    assert code == rp.EXIT_STOPPED
    run = db_session.execute(select(SpineReportRun)).scalar_one()
    # the call has ended, so its retries no longer count: the default 120 s + 5
    assert rp._aware(run.resume_not_before) == T0 + dt.timedelta(
        seconds=rp.DEFAULT_LEASE_SECONDS + rp.LEASE_MARGIN_SECONDS
    )


def test_a_failed_attempt_in_an_early_full_round_still_stops_the_batch(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = Clock()
    fake = LapsingSpine(clock)
    a = seed_copy(db_session, "a", masks=())
    pids = [fake.depict(a["origin"]) for _ in range(5)]
    monkeypatch.setattr(rp, "LEASE_MAX_ITEMS", 2)
    fake.hold_then_fail.append(C.UNAVAILABLE)  # round 1, attempt 1: holds 2, answer lost
    with serve(fake) as url:
        code, lines = _run(db_session, url, clock=clock, attempts=2, lease_seconds=60)
    assert code == rp.EXIT_STOPPED
    assert sorted(fake.depictions[p]["state"] for p in pids) == [
        "captured",
        "captured",
        "derived",
        "derived",
        "derived",
    ]
    assert _batches(lines)[0]["lease_failed_attempts"] == 1


def _two_batches(db: Session, fake: FakeSpine, *, masked_first: bool) -> list[dict[str, Any]]:
    copies = sorted(
        (
            seed_copy(db, "a", masks=(("mask-v1", "ok", "m"),) if masked_first else ()),
            seed_copy(db, "b", masks=()),
        ),
        key=lambda c: c["origin"],
    )
    if masked_first and copies[0]["origin"] != sha("orig:a"):
        pytest.fail("the fixture expects orig:a to sort first")
    for c in copies:
        fake.depict(c["origin"])
    return copies


def test_a_resumed_run_ends_1_for_a_refusal_its_earlier_attempt_counted(
    db_session: Session, spine: tuple[FakeSpine, str]
) -> None:
    fake, url = spine
    _two_batches(db_session, fake, masked_first=False)
    fake.fail("ReportDerivative", C.INVALID_ARGUMENT)  # batch 0's derivative: refused
    fake.fail("LeasePendingImages", None, C.UNAVAILABLE)  # batch 1's lease: down
    clock = Clock()
    code, lines = _run(db_session, url, clock=clock, attempts=1, batch_size=1)
    assert code == rp.EXIT_STOPPED
    assert _batches(lines)[0]["ReportDerivative"]["refused"] == {"INVALID_ARGUMENT": 1}
    code, lines = _run(db_session, url, clock=clock, attempts=1, batch_size=1)
    assert lines[0]["event"] == "resume"
    assert [b["batch"] for b in _batches(lines)] == [0, 1]  # batch 1 of the plan, and the tail
    assert all(b["ReportDerivative"]["refused"] == {} for b in _batches(lines))
    assert code == rp.EXIT_REFUSED and lines[-1]["exit"] == rp.EXIT_REFUSED


def test_a_resumed_run_ends_1_for_a_mask_an_earlier_attempt_left_unregistered(
    db_session: Session, spine: tuple[FakeSpine, str]
) -> None:
    fake, url = spine
    _two_batches(db_session, fake, masked_first=True)
    fake.fail("ReportMask", C.FAILED_PRECONDITION, C.FAILED_PRECONDITION)
    fake.fail("LeasePendingImages", None, C.UNAVAILABLE, C.UNAVAILABLE)  # batch 1's lease
    clock = Clock()
    code, lines = _run(db_session, url, clock=clock, attempts=2, batch_size=1)
    assert code == rp.EXIT_STOPPED
    assert _batches(lines)[0]["ReportMask"]["unregistered"] == 1
    code, lines = _run(db_session, url, clock=clock, attempts=2, batch_size=1)
    assert lines[0]["event"] == "resume"
    assert all(b["ReportMask"]["unregistered"] == 0 for b in _batches(lines))
    assert code == rp.EXIT_REFUSED and lines[-1]["exit"] == rp.EXIT_REFUSED


def test_a_failed_precondition_on_any_other_call_is_a_refusal_not_an_orphan(
    db_session: Session, spine: tuple[FakeSpine, str]
) -> None:
    fake, url = spine
    a = seed_copy(db_session, "a", masks=())
    fake.depict(a["origin"])
    fake.fail("ReportDerivative", C.FAILED_PRECONDITION)
    code, lines = _run(db_session, url)
    assert code == rp.EXIT_REFUSED
    batch = _batches(lines)[0]
    assert batch["ReportDerivative"]["refused"] == {"FAILED_PRECONDITION": 1}
    assert batch["orphans"] == [] and batch["ReportMask"]["orphan"] == 0


@pytest.mark.parametrize(
    ("counts", "failed"),
    [
        ({}, False),
        ({"ReportMask.ok": 3, "orphan": 2, "ReportMask.orphan": 1, "leased": 4}, False),
        ({"ReportDerivative.refused.INVALID_ARGUMENT": 0, "ReportMask.unregistered": 0}, False),
        ({"WithdrawMask.refused.NOT_FOUND": 1}, True),
        ({"local.refused.mask_object": 1}, True),
        ({"ReportMask.unregistered": 2}, True),
    ],
)
def test_the_exit_reads_failures_from_the_persisted_counts(
    counts: dict[str, int], failed: bool
) -> None:
    assert rp._failed(counts) is failed


# ---------------------------------------------------------------------------
# GKPH-S3-2: the index for the run, a batch's details at a time
# ---------------------------------------------------------------------------


def test_a_mask_the_batch_writes_during_a_run_waits_for_the_next_run(
    db_session: Session, spine: tuple[FakeSpine, str]
) -> None:
    """The mask batch runs alongside the replay. A mask row written after the
    run read its index is left to the next run, which lands it: the run's
    details never hold a row its ``published`` has not seen."""
    fake, url = spine
    a = seed_copy(db_session, "a")
    b = seed_copy(db_session, "b", masks=())
    for seed in (a, b):
        fake.depict(seed["origin"])
    late = sha("mask:late")
    lines: list[dict[str, Any]] = []

    def batch_writes_after_the_index(line: dict[str, Any]) -> None:
        lines.append(line)
        if line.get("event") != "plan":
            return
        db_session.add(
            PublicObject(
                sha256=late,
                kind="mask",
                mime="image/png",
                bytes=5,
                width=100,
                height=200,
                storage_key=f"d/{late}",
                created_at=AT,
            )
        )
        db_session.add(
            ImageMask(
                source_sha256=b["clean"],
                pipeline_version="mask-v1",
                status="ok",
                mask_sha256=late,
                created_at=AT,
                updated_at=AT,
                **GROUND,
            )
        )
        db_session.commit()

    clock = Clock()
    client = SpineMediaClient(
        open_channel(url), policy=RetryPolicy(max_attempts=2, initial_backoff=0.01)
    )
    code = rp.run_replay(
        db_session,
        rp.ReplayConfig(),
        connect=lambda: client,
        emit=batch_writes_after_the_index,
        now=clock,
        sleep=clock.sleep,
    )
    assert code == rp.EXIT_DONE
    assert fake.published(a["masks"][0])
    assert not fake.published(late)  # written after the index: not this run's
    assert {d["state"] for d in fake.depictions.values()} == {"derived"}

    code, lines = _run(db_session, url, clock=clock)
    assert code == rp.EXIT_DONE
    assert fake.published(late)  # the next run lands it
    assert len(fake.sent("ReportDerivative")) == 2  # nothing pending was reported twice
    assert _batches(lines)[0]["ReportMask"]["stored"] == 1


def test_a_plans_details_are_read_a_chunk_of_copies_at_a_time(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeds = [seed_copy(db_session, f"k{i}", masks=(("mask-v1", "ok", f"k{i}"),)) for i in range(5)]
    monkeypatch.setattr(rp, "DETAIL_CHUNK", 2)
    plan = rp.load_plan(db_session)
    assert set(plan.copies) == {s["clean"] for s in seeds}
    for s in seeds:
        assert [m.mask_sha256 for m in plan.reports_for(s["clean"])] == s["masks"]
    index = rp.load_plan(db_session, details=False)
    assert index.copies == {} and index.reports == {} and index.copy_of == plan.copy_of
    index.load(db_session, [seeds[3]["clean"]])
    assert set(index.copies) == {seeds[3]["clean"]}


def test_an_image_that_minted_two_live_copies_reports_the_lowest_in_either_order(
    db_session: Session,
) -> None:
    """The index streams copies in the database's order; the copy an image
    minted is the lowest live one, whichever comes first."""
    expected = {}
    for tag in ("p", "q"):
        seed = seed_copy(db_session, tag, masks=())
        image = db_session.execute(select(Image).where(Image.sha256 == seed["origin"])).scalar_one()
        version = ImageVersion(image_id=image.id, version_no=3, storage_key=f"d/again-{tag}")
        db_session.add(version)
        db_session.flush()
        # p's second copy sorts below its first, q's above: one replaces, one keeps
        second = "0" * 64 if tag == "p" else "f" * 64
        db_session.add(
            PublicObject(
                sha256=second,
                kind="sanitized",
                mime="image/jpeg",
                bytes=10,
                width=100,
                height=200,
                storage_key=f"d/{second}",
                image_version_id=version.id,
                pixel_sha256=sha(f"px:again-{tag}"),
                own_pixel_sha256=sha(f"own:again-{tag}"),
                created_at=AT,
            )
        )
        db_session.commit()
        expected[seed["origin"]] = min(seed["clean"], second)
    plan = rp.load_plan(db_session, details=False)
    assert {o: plan.copy_of[o] for o in expected} == expected
