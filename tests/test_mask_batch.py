"""MASK-BATCH-1: the gated, chunked, resumable mask batch for the WSL GPU.

The driver (``app.cli.mask_batch``) reads a work list of ORIGINALS (the
spine's captured item images: blob sha, raw-bucket key, origin url, product),
Ross's holdings first, and per original: skips it when its clean copy already
has a mask result (idempotent per image sha), reads the bytes from the raw
bucket (read only) and checks their sha, seeds the clean copy
(``seed_clean_image``) and settles its mask (``settle_clean_copy``) in the
media database and bucket. Before EVERY chunk it checks Ross's gate -- the
scraper-quiet minute windows, the GPU (utilisation < 5 % OR free >= 9 GiB)
and the residential exit -- and waits while any fails, releasing the model so
a game gets the GPU back. Per chunk it reports count, s/img, peak VRAM and the
failures; its own progress file makes it resumable; ``--dry-run`` touches
nothing.

No GPU, no network, no real model here: SQLite, local directories, the
stub's arithmetic standing in for a mask model, fake clocks and fake
nvidia-smi output.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import gzip
import hashlib
import io
import json
import subprocess
import sys
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from app.cli import mask_batch as mb
from app.models import Base, ImageMask, MaskBatch, PublicObject
from app.object_store import LocalObjectStore
from app.workers import mask as mask_artifact
from app.workers import mask_job, mask_seed
from app.workers.basekept import BaseKeptMattingBackend
from app.workers.mask_policy import ImageOrigin
from app.workers.mask_seed import seed_clean_image
from app.workers.matting import MattingBackend, StubMattingBackend
from app.workers.stage_timing import DatabaseSeconds, StageTimings
from PIL import Image
from sqlalchemy import create_engine, func, select
from sqlalchemy import event as sa_event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool, StaticPool

ITEM = "https://static.myfigurecollection.net/upload/items/2/{n}-a1b2.jpg"
PICTURE = "https://static.myfigurecollection.net/upload/pictures/2024/01/02/{n}.jpeg"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _stdout_events(capsys: pytest.CaptureFixture[str]) -> list[dict[str, Any]]:
    """main's stdout (LG-U11): the estate's lines, each event under its own name again."""
    out = []
    for line in capsys.readouterr().out.splitlines():
        event = json.loads(line)
        event["event"] = event["event"].removeprefix("app.")
        out.append(event)
    return out


def _refusal(capsys: pytest.CaptureFixture[str]) -> str:
    """Why main refused: the run's job.end err (stderr stays empty)."""
    captured = capsys.readouterr()
    end = json.loads(captured.out.splitlines()[-1])
    assert (end["event"], end["code"], captured.err) == ("job.end", "error", "")
    return str(end["err"]["message"])


def _photo(n: int) -> bytes:
    arr = np.full((60, 40, 3), (250, 250, 248), dtype=np.uint8)
    arr[10:50, 12:28] = (31 + 10 * n % 200, 30, 90)
    arr[0, 0] = (250, 250, 248 - n % 3)
    arr[1, 0] = (n % 251, 250, 250)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


class _Model(StubMattingBackend):
    """The stub's arithmetic standing in for a mask MODEL (publishes)."""

    name = "stub-model"
    publishes_masks = True

    def __init__(self) -> None:
        self.loads = 0
        self.releases = 0

    def load(self) -> None:
        self.loads += 1

    def release(self) -> None:
        self.releases += 1

    @property
    def active_provider(self) -> str | None:
        return "FakeExecutionProvider"


class _Raw:
    """A raw bucket on disk, keyed the way the scraper's asset lane keys it."""

    def __init__(self, root: Path) -> None:
        self.store = LocalObjectStore(root)

    def add(
        self, n: int, *, codec: str = "identity", rank: int = 9, url: str = ITEM
    ) -> mb.WorkItem:
        data = _photo(n)
        sha = hashlib.sha256(data).hexdigest()
        key = f"raw-img/sha256/{sha[:2]}/{sha}.png"
        self.store.put(key, gzip.compress(data) if codec == "gzip" else data, "image/png")
        return mb.WorkItem(
            blob_sha256=sha,
            storage_key=key,
            url=url.format(n=n),
            product_id=f"00000000-0000-0000-0000-{n:012d}",
            head_id=f"00000000-0000-0000-0000-{n:012d}",
            rank=rank,
            codec=codec,
        )


class _Clock:
    """UTC wall clock + monotonic clock; ``sleep`` advances both."""

    def __init__(self, minute: int = 30) -> None:
        self.t = dt.datetime(2026, 10, 14, 15, minute, 0, tzinfo=dt.UTC)
        self.mono = 1000.0
        self.slept: list[float] = []

    def now(self) -> dt.datetime:
        return self.t

    def monotonic(self) -> float:
        return self.mono

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.advance(seconds)

    def advance(self, seconds: float) -> None:
        self.t += dt.timedelta(seconds=seconds)
        self.mono += seconds


class _Sampler:
    def __init__(self, peak: int | None = 3210) -> None:
        self.peak_mib = peak
        self.entered = 0

    def __enter__(self) -> _Sampler:
        self.entered += 1
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _runtime(
    db: Session,
    tmp_path: Path,
    raw: _Raw,
    *,
    clock: _Clock | None = None,
    gates: mb.Gates | None = None,
    backend: MattingBackend | None = None,
    progress: Path | None = None,
    events: list[dict[str, Any]] | None = None,
    stop: Callable[[], bool] | None = None,
) -> mb.Runtime:
    clock = clock or _Clock()
    model = backend or _Model()
    out = events if events is not None else []
    return mb.Runtime(
        gates=gates or mb.Gates(windows=(), gpu=None, residential=None, now=clock.now),
        open_db=lambda: db,
        media_store=LocalObjectStore(tmp_path / "media"),
        raw_store=mb.ReadOnlyStore(raw.store),
        make_backend=lambda: model,
        progress=mb.Progress(progress if progress is not None else tmp_path / "progress.jsonl"),
        vram=lambda: _Sampler(),
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        emit=out.append,
        stop_requested=stop or (lambda: False),
    )


def _events(events: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [e for e in events if e["event"] == name]


def _gpu(readings: Iterable[str]) -> Callable[[], str]:
    it = iter(readings)
    return lambda: next(it)


# --------------------------------------------------------------------------
# the work list
# --------------------------------------------------------------------------


def _line(sha: str, rank: int, **extra: Any) -> str:
    row = {
        "blob_sha256": sha,
        "storage_key": f"raw-img/sha256/{sha[:2]}/{sha}.jpg",
        "url": ITEM.format(n=1),
        "source": "mfc",
        "product_id": "p",
        "rank": rank,
        **extra,
    }
    return json.dumps(row)


def test_the_work_list_is_ross_first_then_file_order_each_original_once() -> None:
    a, b, c, d = "a" * 64, "b" * 64, "c" * 64, "d" * 64
    lines = [
        "##### worklist-catalog.sql",
        _line(a, 9),
        _line(b, 9),
        "(3 rows)",
        "",
        _line(c, 2),
        _line(b, 0),  # b is also one of Ross's: his rank wins
        _line(c, 9),  # a later, lower-priority copy of c changes nothing
        _line(d, 9),
        _line(a, 9),  # a repeat at the same rank keeps a's first place
    ]
    items, skipped = mb.parse_worklist(lines)
    assert [(i.blob_sha256, i.rank) for i in items] == [(b, 0), (c, 2), (a, 9), (d, 9)]
    assert skipped == 2  # the psql header and footer; blank lines are not counted
    assert items[0].codec == "identity"
    assert items[0].head_id is None


def test_a_bad_row_in_the_work_list_is_refused_with_its_line_number() -> None:
    with pytest.raises(ValueError, match="line 2: blob_sha256"):
        mb.parse_worklist([_line("a" * 64, 0), _line("A" * 64, 0)])
    with pytest.raises(ValueError, match="line 1: storage_key"):
        mb.parse_worklist([_line("a" * 64, 0, storage_key="")])
    with pytest.raises(ValueError, match="line 1: rank"):
        mb.parse_worklist([_line("a" * 64, -1)])
    with pytest.raises(ValueError, match="line 1: rank"):
        mb.parse_worklist([_line("a" * 64, True)])
    with pytest.raises(ValueError, match="line 1: url"):
        mb.parse_worklist([_line("a" * 64, 0, url=None)])
    with pytest.raises(ValueError, match="line 1: product_id"):
        mb.parse_worklist([_line("a" * 64, 0, product_id="")])
    with pytest.raises(ValueError, match="line 1: codec"):
        mb.parse_worklist([_line("a" * 64, 0, codec=7)])
    with pytest.raises(ValueError, match="line 1: not a JSON object"):
        mb.parse_worklist(['{"blob_sha256": '])


def test_read_worklists_concatenates_files_before_ordering(tmp_path: Path) -> None:
    (tmp_path / "catalog.jsonl").write_text(_line("a" * 64, 9) + "\n")
    (tmp_path / "ross.jsonl").write_text(_line("a" * 64, 1) + "\n" + _line("d" * 64, 0) + "\n")
    items, skipped = mb.read_worklists([tmp_path / "catalog.jsonl", tmp_path / "ross.jsonl"])
    assert [(i.blob_sha256[0], i.rank) for i in items] == [("d", 0), ("a", 1)]
    assert skipped == 0


def test_a_row_carries_the_source_class_the_spine_recorded() -> None:
    """GAL3: the class the capture lane gave the depiction (the spine's
    product_image.source_class), absent or null when it gave none."""
    a, b, c = "a" * 64, "b" * 64, "c" * 64
    items, _ = mb.parse_worklist(
        [
            _line(a, 0),
            _line(b, 0, source_class="manufacturer_press"),
            _line(c, 0, source_class=None),
        ]
    )
    assert [i.source_class for i in items] == [None, "manufacturer_press", None]
    assert items[1].as_row()["source_class"] == "manufacturer_press"


@pytest.mark.parametrize("bad", ["junk", "", "Manufacturer_Press", 7, True])
def test_a_source_class_outside_the_vocabulary_is_refused(bad: object) -> None:
    with pytest.raises(ValueError, match="line 1: source_class"):
        mb.parse_worklist([_line("a" * 64, 0, source_class=bad)])


def test_every_row_can_be_read_in_file_order(tmp_path: Path) -> None:
    """``masks official-gallery`` needs every depiction, not each original once."""
    (tmp_path / "a.jsonl").write_text(
        _line("a" * 64, 9, product_id="p1")
        + "\n"
        + "(2 rows)\n"
        + _line("a" * 64, 0, product_id="p2")
        + "\n"
    )
    (tmp_path / "b.jsonl").write_text(_line("d" * 64, 0) + "\n")
    items, skipped = mb.read_worklists([tmp_path / "a.jsonl", tmp_path / "b.jsonl"], every_row=True)
    assert [(i.blob_sha256[0], i.product_id) for i in items] == [
        ("a", "p1"),
        ("a", "p2"),
        ("d", "p"),
    ]
    assert skipped == 1


# --------------------------------------------------------------------------
# the gates
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reading", "ok"),
    [
        ("4, 0, 12000, 12282", True),  # idle GPU, whatever the memory
        ("5, 9216, 3000, 12282", True),  # busy, but 9 GiB free
        ("5, 9215, 3000, 12282", False),  # busy and one MiB short
        ("99, 100, 12000, 12282", False),  # a game
        ("0, 12000, 200, 12282", True),
    ],
)
def test_the_gpu_gate_is_ross_rule(reading: str, ok: bool) -> None:
    gate = mb.GpuGate(query=lambda: reading, samples=1, sleep=lambda s: None)
    result = gate.check()
    assert result.ok is ok
    assert result.gate == "gpu"
    assert "util" in result.detail and "free" in result.detail


def test_every_gpu_sample_must_pass() -> None:
    slept: list[float] = []
    gate = mb.GpuGate(
        query=_gpu(["0, 100, 1, 12282", "80, 100, 1, 12282", "0, 100, 1, 12282"]),
        samples=3,
        interval=0.5,
        sleep=slept.append,
    )
    assert gate.check().ok is False
    assert slept == [0.5]  # the second sample already failed: no third
    calm = mb.GpuGate(query=lambda: "0, 100, 1, 12282", samples=3, interval=0.5, sleep=slept.append)
    assert calm.check().ok is True
    assert slept == [0.5, 0.5, 0.5]


def test_the_gpu_gate_fails_closed() -> None:
    def broken() -> str:
        raise OSError("nvidia-smi: not found")

    assert mb.GpuGate(query=broken, samples=1).check().ok is False
    assert mb.GpuGate(query=lambda: "N/A, 1, 2, 3", samples=1).check().ok is False
    assert mb.GpuGate(query=lambda: "1, 2", samples=1).check().ok is False
    assert mb.GpuGate(query=lambda: "", samples=1).check().ok is False


def test_with_device_auto_a_box_without_nvidia_smi_passes_the_gpu_gate() -> None:
    def absent() -> str:
        raise FileNotFoundError("nvidia-smi")

    def broken() -> str:
        raise subprocess.CalledProcessError(9, "nvidia-smi")

    passed = mb.GpuGate(query=absent, samples=1, absent_ok=True).check()
    assert passed.ok is True and "no GPU" in passed.detail
    assert mb.GpuGate(query=absent, samples=1).check().ok is False  # cuda: closed
    assert (
        mb.GpuGate(query=broken, samples=1, absent_ok=True).check().ok is False
    )  # a GPU that errs


def test_the_gpu_reading_keeps_the_used_memory_for_the_baseline() -> None:
    reading = mb.GpuGate(query=lambda: "3, 10579, 1416, 12282", samples=1).read()
    assert reading == mb.GpuReading(util=3, free_mib=10579, used_mib=1416, total_mib=12282)


def test_nvidia_smi_query_names_one_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    class _Done:
        stdout = "1, 2, 3, 4\n"

    def fake_run(cmd: list[str], **kw: Any) -> _Done:
        seen.append(cmd)
        assert kw["check"] is True and kw["timeout"] > 0
        return _Done()

    monkeypatch.setattr(mb.subprocess, "run", fake_run)
    assert mb.nvidia_smi_query(1)() == "1, 2, 3, 4"
    assert seen[0][0] == "nvidia-smi" and seen[0][-2:] == ["-i", "1"]
    assert "--query-gpu=utilization.gpu,memory.free,memory.used,memory.total" in seen[0]


def test_the_busy_windows_are_the_scraper_rolls() -> None:
    windows = mb.parse_windows(mb.DEFAULT_BUSY_WINDOWS)
    assert windows == ((3, 26), (41, 43), (50, 56))
    busy = {m for m in range(60) if mb.busy_window(m, windows)}
    quiet = sorted(set(range(60)) - busy)
    assert quiet == [0, 1, 2, *range(27, 41), 44, 45, 46, 47, 48, 49, 57, 58, 59]
    assert mb.busy_window(26, windows) == (3, 26)
    assert mb.parse_windows("") == ()


@pytest.mark.parametrize("spec", ["3-", "26-3", "0-60", "-1-4", "a-b", "3"])
def test_a_malformed_window_is_refused(spec: str) -> None:
    with pytest.raises(ValueError, match="window"):
        mb.parse_windows(spec)


def test_the_residential_check_wants_the_home_address() -> None:
    clock = _Clock()
    answers = iter(["198.51.100.7\n", "203.0.113.9"])
    check = mb.ResidentialCheck(
        expected_ip="198.51.100.7", fetch=lambda: next(answers), ttl=300, clock=clock.monotonic
    )
    assert check.check().ok is True
    clock.advance(299)
    assert check.check().ok is True  # cached: no second fetch yet
    clock.advance(1)
    result = check.check()
    assert result.ok is False and result.gate == "residential"
    assert "203.0.113.9" in result.detail


def test_a_failed_residential_check_is_never_cached() -> None:
    calls: list[int] = []

    def fetch() -> str:
        calls.append(1)
        if len(calls) == 1:
            raise OSError("no route")
        return "198.51.100.7"

    check = mb.ResidentialCheck(expected_ip="198.51.100.7", fetch=fetch, ttl=300)
    assert check.check().ok is False
    assert check.check().ok is True
    assert len(calls) == 2


def test_public_ip_reads_the_echo_service(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resp:
        text = "198.51.100.7\n"

        def raise_for_status(self) -> None:
            return None

    seen: list[tuple[str, float]] = []

    def fake_get(url: str, *, timeout: float) -> _Resp:
        seen.append((url, timeout))
        return _Resp()

    monkeypatch.setattr(mb.httpx, "get", fake_get)
    assert mb.public_ip("https://echo.example/")() == "198.51.100.7\n"
    assert seen == [("https://echo.example/", 10.0)]


def test_the_gates_are_checked_window_then_gpu_then_residential() -> None:
    clock = _Clock(minute=10)
    gpu_calls: list[int] = []

    def gpu() -> str:
        gpu_calls.append(1)
        return "90, 10, 1, 12282"

    gates = mb.Gates(
        windows=((3, 26),),
        gpu=mb.GpuGate(query=gpu, samples=1),
        residential=mb.ResidentialCheck(expected_ip="198.51.100.7", fetch=lambda: "198.51.100.7"),
        now=clock.now,
    )
    first = gates.check()
    assert (first.ok, first.gate) == (False, "window")
    assert gpu_calls == []  # the window is decided before nvidia-smi is asked
    clock.advance(17 * 60)
    assert gates.check().gate == "gpu"
    gates.gpu = None
    assert gates.check().ok is True


# --------------------------------------------------------------------------
# progress, sampler, read-only store
# --------------------------------------------------------------------------


def test_progress_survives_a_torn_last_line(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "progress.jsonl"
    progress = mb.Progress(path)
    assert progress.load() == {}
    progress.append({"event": "item", "blob_sha256": "a" * 64, "status": "ok"})
    progress.append({"event": "chunk", "chunk": 1})
    with path.open("a") as fh:
        fh.write('{"event": "item", "blob_sha256": "b')  # the power went out here
    assert set(progress.load()) == {"a" * 64}
    progress.append({"event": "item", "blob_sha256": "c" * 64, "status": "ok"})
    assert set(progress.load()) == {"a" * 64, "c" * 64}  # the torn line swallowed nothing
    with path.open("a") as fh:
        fh.write(json.dumps({"event": "item", "blob_sha256": 7}) + "\n")
    assert set(progress.load()) == {"a" * 64, "c" * 64}
    assert mb.Progress(None).load() == {}
    mb.Progress(None).append({"event": "item"})  # no file: nothing written


def test_the_vram_sampler_keeps_the_peak() -> None:
    cmd = [sys.executable, "-c", "print(100); print(250); print('N/A'); print(200)"]
    with mb.VramSampler(cmd, grace=10) as sampler:
        pass
    assert sampler.peak_mib == 250
    with mb.VramSampler(None) as none:
        pass
    assert none.peak_mib is None
    with mb.VramSampler(["/nonexistent/nvidia-smi"]) as missing:
        pass
    assert missing.peak_mib is None
    forever = [sys.executable, "-c", "import time\nwhile True: time.sleep(1)"]
    with mb.VramSampler(forever) as endless:  # nvidia-smi -lms never ends by itself
        pass
    assert endless.peak_mib is None
    assert mb.vram_command(1)[0] == "nvidia-smi" and mb.vram_command(1)[-4:] == [
        "-i",
        "1",
        "-lms",
        "200",
    ]


def test_the_raw_bucket_is_read_only(tmp_path: Path) -> None:
    raw = _Raw(tmp_path / "raw")
    item = raw.add(1)
    ro = mb.ReadOnlyStore(raw.store)
    assert hashlib.sha256(ro.get(item.storage_key)).hexdigest() == item.blob_sha256
    with pytest.raises(PermissionError):
        ro.put("x", b"y", "image/png")


# --------------------------------------------------------------------------
# the mask lane, one clean copy at a time
# --------------------------------------------------------------------------


def _seed(db: Session, store: LocalObjectStore, n: int, url: str = ITEM) -> PublicObject:
    return seed_clean_image(
        db,
        store,
        _photo(n),
        origins=[ImageOrigin(url=url.format(n=n), source="mfc")],
        figure_id=f"f{n}",
    )


def test_settle_clean_copy_settles_once(db_session: Session, tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path / "media")
    one, two = _seed(db_session, store, 1), _seed(db_session, store, 2)
    obj = max(one, two, key=lambda o: o.sha256)  # not the first in sha order
    batch = mask_job.open_mask_batch(db_session, backend_name="stub-model")
    first = mask_job.settle_clean_copy(db_session, store, obj.sha256, backend=_Model(), batch=batch)
    assert first is not None and first.status == "ok"
    assert first.source_sha256 == obj.sha256  # that copy, and only that copy
    assert db_session.execute(select(ImageMask.source_sha256)).scalars().all() == [obj.sha256]
    assert (
        mask_job.settle_clean_copy(db_session, store, obj.sha256, backend=_Model(), batch=batch)
        is None
    )
    mask_job.finish_mask_batch(db_session, batch)
    assert db_session.get(MaskBatch, batch.id).status == "done"  # type: ignore[union-attr]
    assert db_session.get(MaskBatch, batch.id).counts["ok"] == 1  # type: ignore[union-attr]


def test_settle_clean_copy_refuses_a_backend_that_is_no_mask_model(
    db_session: Session, tmp_path: Path
) -> None:
    store = LocalObjectStore(tmp_path / "media")
    obj = _seed(db_session, store, 1)
    batch = mask_job.open_mask_batch(db_session, backend_name="stub")
    with pytest.raises(ValueError, match="not a mask model"):
        mask_job.settle_clean_copy(
            db_session, store, obj.sha256, backend=StubMattingBackend(), batch=batch
        )


def test_a_failed_copy_is_redriven_only_when_asked(db_session: Session, tmp_path: Path) -> None:
    class _Boom(_Model):
        def predict_alpha(self, image: Image.Image) -> np.ndarray:
            raise RuntimeError("boom")

    store = LocalObjectStore(tmp_path / "media")
    obj = _seed(db_session, store, 1)
    batch = mask_job.open_mask_batch(db_session, backend_name="boom")
    failed = mask_job.settle_clean_copy(db_session, store, obj.sha256, backend=_Boom(), batch=batch)
    assert failed is not None and failed.status == "failed"
    assert mask_job.original_settled(db_session, hashlib.sha256(_photo(1)).hexdigest()) == "failed"
    assert (
        mask_job.original_settled(
            db_session, hashlib.sha256(_photo(1)).hexdigest(), retry_failed=True
        )
        is None
    )  # a failure re-driven is not settled
    assert (
        mask_job.settle_clean_copy(db_session, store, obj.sha256, backend=_Model(), batch=batch)
        is None
    )
    again = mask_job.settle_clean_copy(
        db_session, store, obj.sha256, backend=_Model(), batch=batch, retry_failed=True
    )
    assert again is not None and again.status == "ok"
    assert (
        mask_job.original_settled(
            db_session, hashlib.sha256(_photo(1)).hexdigest(), retry_failed=True
        )
        == "ok"
    )


def test_original_settled_reads_the_copy_behind_an_original(
    db_session: Session, tmp_path: Path
) -> None:
    store = LocalObjectStore(tmp_path / "media")
    sha = hashlib.sha256(_photo(1)).hexdigest()
    assert mask_job.original_settled(db_session, sha) is None  # never seen
    _seed(db_session, store, 1)
    assert mask_job.original_settled(db_session, sha) is None  # a copy, no result yet
    batch = mask_job.open_mask_batch(db_session, backend_name="m")
    obj = db_session.execute(
        select(PublicObject).where(PublicObject.kind == "sanitized")
    ).scalar_one()
    mask_job.settle_clean_copy(db_session, store, obj.sha256, backend=_Model(), batch=batch)
    assert mask_job.original_settled(db_session, sha) == "ok"
    failed_only = mask_job.original_settled(db_session, sha, retry_failed=True)
    assert failed_only == "ok"  # an ok result is never re-driven


# --------------------------------------------------------------------------
# the driver
# --------------------------------------------------------------------------


def test_mints_ross_first_and_reports_every_chunk(db_session: Session, tmp_path: Path) -> None:
    raw = _Raw(tmp_path / "raw")
    items, _ = mb.parse_worklist(
        [
            json.dumps(raw.add(1, rank=9).as_row()),
            json.dumps(raw.add(2, rank=0).as_row()),
            json.dumps(raw.add(3, rank=2).as_row()),
        ]
    )
    events: list[dict[str, Any]] = []
    rt = _runtime(db_session, tmp_path, raw, events=events)
    code = mb.run_batch(items, mb.BatchConfig(chunk_size=2), rt)
    assert code == 0
    chunks = _events(events, "chunk")
    assert [c["count"] for c in chunks] == [2, 1]
    assert [c["ranks"] for c in chunks] == [[0, 2], [9]]
    first = chunks[0]
    assert first["counts"]["ok"] == 2 and first["failures"] == []
    assert first["peak_vram_mib"] == 3210
    assert first["provider"] == "FakeExecutionProvider"
    assert first["seconds_per_image"] is not None and first["seconds_per_image"] >= 0
    assert first["mask_seconds_per_image"] is not None
    assert first["batch_id"] is not None
    rows = db_session.execute(select(ImageMask.status)).scalars().all()
    assert rows == ["ok", "ok", "ok"]
    done = _events(events, "done")[0]
    assert done["counts"]["ok"] == 3 and done["pending"] == 0
    progress = mb.Progress(tmp_path / "progress.jsonl").load()
    assert {r["status"] for r in progress.values()} == {"ok"}
    assert all(r["clean_sha256"] and r["mask_sha256"] for r in progress.values())
    statuses = db_session.execute(select(MaskBatch.status)).scalars().all()
    assert statuses == ["done", "done"]


def test_a_rerun_is_idempotent_per_image_sha(db_session: Session, tmp_path: Path) -> None:
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(1, rank=0), raw.add(2, rank=0)]
    assert mb.run_batch(items, mb.BatchConfig(), _runtime(db_session, tmp_path, raw)) == 0
    masks = db_session.execute(select(func.count()).select_from(ImageMask)).scalar_one()

    # the same progress file: nothing is even looked up
    events: list[dict[str, Any]] = []
    rt = _runtime(db_session, tmp_path, raw, events=events)
    assert mb.run_batch(items, mb.BatchConfig(), rt) == 0
    assert _events(events, "chunk") == []
    assert _events(events, "start")[0]["pending"] == 0

    # a lost progress file: the database says both are settled
    events2: list[dict[str, Any]] = []
    model = _Model()
    rt2 = _runtime(
        db_session, tmp_path, raw, events=events2, progress=tmp_path / "fresh.jsonl", backend=model
    )
    reads: list[str] = []

    class _Counting(mb.ReadOnlyStore):
        def get(self, key: str) -> bytes:
            reads.append(key)
            return super().get(key)

    rt2.raw_store = _Counting(raw.store)
    assert mb.run_batch(items, mb.BatchConfig(), rt2) == 0
    assert reads == []  # a settled original's bytes are not even read
    (chunk,) = _events(events2, "chunk")
    assert chunk["counts"]["already"] == 2 and chunk["count"] == 2
    assert chunk["seconds_per_image"] is None  # nothing was minted
    assert db_session.execute(select(func.count()).select_from(ImageMask)).scalar_one() == masks


def test_resume_skips_what_is_done_and_retries_failures_only_on_request(
    db_session: Session, tmp_path: Path
) -> None:
    raw = _Raw(tmp_path / "raw")
    ok, failed, fresh = raw.add(1, rank=0), raw.add(2, rank=0), raw.add(3, rank=0)
    progress = mb.Progress(tmp_path / "progress.jsonl")
    progress.append({"event": "item", "blob_sha256": ok.blob_sha256, "status": "ok"})
    progress.append({"event": "item", "blob_sha256": failed.blob_sha256, "status": "failed"})
    events: list[dict[str, Any]] = []
    rt = _runtime(db_session, tmp_path, raw, events=events)
    assert mb.run_batch([ok, failed, fresh], mb.BatchConfig(), rt) == 0
    start = _events(events, "start")[0]
    assert (start["done"], start["failed_before"], start["pending"]) == (1, 1, 1)
    assert _events(events, "chunk")[0]["count"] == 1

    events2: list[dict[str, Any]] = []
    rt2 = _runtime(db_session, tmp_path, raw, events=events2)
    assert mb.run_batch([ok, failed, fresh], mb.BatchConfig(retry_failed=True), rt2) == 0
    assert _events(events2, "start")[0]["pending"] == 1
    assert progress.load()[failed.blob_sha256]["status"] == "ok"


def test_a_failure_the_media_db_recorded_is_reminted_only_with_retry_failed(
    db_session: Session, tmp_path: Path
) -> None:
    """The progress file lost (another machine, a fresh state dir): the media
    database's failed row alone makes the image settled, until asked."""
    raw = _Raw(tmp_path / "raw")
    item = raw.add(1)

    class _Broken(_Model):
        def predict_alpha(self, image: Image.Image) -> np.ndarray:
            raise RuntimeError("model fault")

    first = _runtime(db_session, tmp_path, raw, backend=_Broken())
    assert mb.run_batch([item], mb.BatchConfig(), first) == 1
    assert mask_job.original_settled(db_session, item.blob_sha256) == "failed"

    def run(name: str, retry: bool) -> dict[str, Any]:
        rt = _runtime(db_session, tmp_path, raw, progress=tmp_path / name)
        mb.run_batch([item], mb.BatchConfig(retry_failed=retry), rt)
        return mb.Progress(tmp_path / name).load()[item.blob_sha256]

    plain = run("fresh-1.jsonl", False)
    assert (plain["status"], plain["mask_status"]) == ("already", "failed")
    assert run("fresh-2.jsonl", True)["status"] == "ok"


def test_the_seed_keeps_the_work_lists_source_class(db_session: Session, tmp_path: Path) -> None:
    """GAL3: a gallery-list photo is masked on its first pass; a collector's
    own upload under the same path, with no class, is not."""
    from app.models import FigureGallery

    raw = _Raw(tmp_path / "raw")
    gallery = dataclasses.replace(raw.add(1, url=PICTURE), source_class="manufacturer_press")
    collector = raw.add(2, url=PICTURE)
    rt = _runtime(db_session, tmp_path, raw)
    assert mb.run_batch([gallery, collector], mb.BatchConfig(chunk_size=2), rt) == 0
    rows = db_session.execute(
        select(ImageMask.status, ImageMask.origin_class).order_by(ImageMask.id)
    )
    assert sorted(rows.all()) == [("ineligible", "user_photo"), ("ok", "official_gallery")]
    links = db_session.execute(select(FigureGallery.source_url, FigureGallery.source_class))
    assert dict(links.all()) == {
        PICTURE.format(n=1): "manufacturer_press",
        PICTURE.format(n=2): None,
    }


def test_per_image_failures_are_counted_and_named(db_session: Session, tmp_path: Path) -> None:
    raw = _Raw(tmp_path / "raw")
    good = raw.add(1, rank=0)
    zipped = raw.add(2, rank=0, codec="gzip")
    missing = mb.WorkItem(
        blob_sha256="e" * 64, storage_key="raw-img/sha256/ee/none.png", url=ITEM, product_id="p"
    )
    wrong = raw.add(4, rank=0)
    wrong = mb.WorkItem(**{**wrong.as_row(), "blob_sha256": "f" * 64})
    odd = raw.add(5, rank=0)
    odd = mb.WorkItem(**{**odd.as_row(), "codec": "zstd"})
    picture = raw.add(6, rank=0, url=PICTURE)
    events: list[dict[str, Any]] = []
    rt = _runtime(db_session, tmp_path, raw, events=events)
    code = mb.run_batch(
        [good, zipped, missing, wrong, odd, picture], mb.BatchConfig(chunk_size=10), rt
    )
    assert code == 1  # finished, with failures
    (chunk,) = _events(events, "chunk")
    assert chunk["counts"]["ok"] == 2
    assert chunk["counts"]["ineligible"] == 1  # a collector's photo: no mask, no failure
    assert chunk["counts"]["failed"] == 3
    reasons = {f["blob_sha256"]: f["reason"] for f in chunk["failures"]}
    assert reasons == {
        "e" * 64: "fetch:FileNotFoundError",
        "f" * 64: "raw_hash_mismatch",
        odd.blob_sha256: "unsupported_codec:zstd",
    }
    # nothing was seeded for a failed original
    assert mask_job.original_settled(db_session, odd.blob_sha256) is None


def test_a_seed_failure_is_one_images_failure(db_session: Session, tmp_path: Path) -> None:
    raw = _Raw(tmp_path / "raw")
    junk = b"not an image at all"
    sha = hashlib.sha256(junk).hexdigest()
    raw.store.put(f"raw-img/sha256/{sha[:2]}/{sha}.png", junk, "image/png")
    item = mb.WorkItem(
        blob_sha256=sha, storage_key=f"raw-img/sha256/{sha[:2]}/{sha}.png", url=ITEM, product_id="p"
    )
    events: list[dict[str, Any]] = []
    good = raw.add(2, rank=9)
    assert (
        mb.run_batch(
            [item, good], mb.BatchConfig(), _runtime(db_session, tmp_path, raw, events=events)
        )
        == 1
    )
    (chunk,) = _events(events, "chunk")
    assert chunk["failures"][0]["reason"].startswith("seed:")
    assert chunk["counts"]["ok"] == 1


def test_database_trouble_is_named_per_image(
    db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _Raw(tmp_path / "raw")
    first, second = raw.add(1), raw.add(2)

    def lookup_down(*a: object, **k: object) -> str | None:
        raise ConnectionError("server closed the connection")

    def settle_down(*a: object, **k: object) -> None:
        raise TimeoutError("statement timeout")

    events: list[dict[str, Any]] = []
    monkeypatch.setattr(mb, "original_settled", lookup_down)
    assert (
        mb.run_batch([first], mb.BatchConfig(), _runtime(db_session, tmp_path, raw, events=events))
        == 1
    )
    monkeypatch.undo()
    monkeypatch.setattr(mb, "settle_clean_copy", settle_down)
    assert (
        mb.run_batch([second], mb.BatchConfig(), _runtime(db_session, tmp_path, raw, events=events))
        == 1
    )
    reasons = [f["reason"] for c in _events(events, "chunk") for f in c["failures"]]
    assert reasons == ["db:ConnectionError", "settle:TimeoutError"]


def test_a_second_capture_of_a_masked_picture_is_already_done(
    db_session: Session, tmp_path: Path
) -> None:
    raw = _Raw(tmp_path / "raw")
    first = raw.add(1)
    # the same pixels, other bytes: a re-save with a text chunk
    from PIL import PngImagePlugin

    meta = PngImagePlugin.PngInfo()
    meta.add_text("Comment", "captured again")
    buf = io.BytesIO()
    Image.open(io.BytesIO(_photo(1))).save(buf, format="PNG", pnginfo=meta)
    again = buf.getvalue()
    sha = hashlib.sha256(again).hexdigest()
    raw.store.put(f"raw-img/sha256/{sha[:2]}/{sha}.png", again, "image/png")
    twin = mb.WorkItem(
        blob_sha256=sha,
        storage_key=f"raw-img/sha256/{sha[:2]}/{sha}.png",
        url=ITEM.format(n=1),
        product_id="p2",
    )
    events: list[dict[str, Any]] = []
    rt = _runtime(db_session, tmp_path, raw, events=events)
    assert mb.run_batch([first, twin], mb.BatchConfig(), rt) == 0
    (chunk,) = _events(events, "chunk")
    assert chunk["counts"] == {"ok": 1, "already": 1}
    record = mb.Progress(tmp_path / "progress.jsonl").load()[sha]
    assert (record["status"], record["mask_status"]) == ("already", "ok")
    assert db_session.execute(select(func.count()).select_from(ImageMask)).scalar_one() == 1


def test_a_mask_failure_recorded_by_the_lane_counts_as_a_failure(
    db_session: Session, tmp_path: Path
) -> None:
    class _Boom(_Model):
        def predict_alpha(self, image: Image.Image) -> np.ndarray:
            raise RuntimeError("boom")

    raw = _Raw(tmp_path / "raw")
    events: list[dict[str, Any]] = []
    rt = _runtime(db_session, tmp_path, raw, events=events, backend=_Boom())
    assert mb.run_batch([raw.add(1)], mb.BatchConfig(), rt) == 1
    (chunk,) = _events(events, "chunk")
    assert chunk["failures"] == [
        {"blob_sha256": hashlib.sha256(_photo(1)).hexdigest(), "reason": "RuntimeError"}
    ]


def test_consecutive_failures_open_the_circuit(db_session: Session, tmp_path: Path) -> None:
    raw = _Raw(tmp_path / "raw")
    bucket = _Bucket(tmp_path / "raw")
    raw.store = bucket
    gone = [raw.add(n) for n in range(6)]
    bucket.down = {i.storage_key for i in gone}  # an outage, not the images
    events: list[dict[str, Any]] = []
    rt = _runtime(db_session, tmp_path, raw, events=events)
    assert mb.run_batch(gone, mb.BatchConfig(max_consecutive_failures=3), rt) == 3
    assert _events(events, "circuit_open")[0]["consecutive_failures"] == 3
    assert _events(events, "chunk")[0]["count"] == 3
    assert len(mb.Progress(tmp_path / "progress.jsonl").load()) == 3
    assert db_session.execute(select(MaskBatch.status)).scalars().all() == ["done"]


def test_a_success_resets_the_failure_streak(db_session: Session, tmp_path: Path) -> None:
    raw = _Raw(tmp_path / "raw")
    bucket = _Bucket(tmp_path / "raw")
    raw.store = bucket
    items = [raw.add(n) for n in range(1, 6)]
    bucket.down = {i.storage_key for i in items if i is not items[2]}
    assert (
        mb.run_batch(
            items, mb.BatchConfig(max_consecutive_failures=3), _runtime(db_session, tmp_path, raw)
        )
        == 1
    )


def test_a_failing_gate_pauses_releases_the_gpu_and_resumes(
    db_session: Session, tmp_path: Path
) -> None:
    raw = _Raw(tmp_path / "raw")
    clock = _Clock()
    model = _Model()
    readings = (
        ["2, 100, 1, 12282", "2, 2900, 9100, 12282"]  # before the load; after the first image
        + ["97, 0, 12282, 12282", "97, 3000, 9000, 12282"]  # a game: held, then released
        + ["1, 9000, 3000, 12282"] * 5
    )
    gates = mb.Gates(
        windows=(),
        gpu=mb.GpuGate(query=_gpu(readings), samples=1, sleep=lambda s: None),
        residential=None,
        now=clock.now,
    )
    events: list[dict[str, Any]] = []
    rt = _runtime(db_session, tmp_path, raw, clock=clock, gates=gates, backend=model, events=events)
    items = [raw.add(1), raw.add(2)]
    assert mb.run_batch(items, mb.BatchConfig(chunk_size=1, poll_seconds=60), rt) == 0
    paused = _events(events, "paused")
    assert len(paused) == 1  # one line per pause, not one per poll
    assert paused[0]["gate"] == "gpu" and "97%" in paused[0]["detail"]
    assert len(_events(events, "resumed")) == 1
    assert clock.slept == [60, 60]
    assert model.releases >= 1  # the game gets the GPU back
    assert model.loads == 2  # loaded for chunk 1, again after the pause
    assert [c["baseline_vram_mib"] for c in _events(events, "chunk")] == [1, 3000]


def test_waiting_longer_than_allowed_gives_up(db_session: Session, tmp_path: Path) -> None:
    raw = _Raw(tmp_path / "raw")
    clock = _Clock()
    gates = mb.Gates(
        windows=(),
        gpu=None,
        residential=mb.ResidentialCheck(expected_ip="198.51.100.7", fetch=lambda: "203.0.113.9"),
        now=clock.now,
    )
    events: list[dict[str, Any]] = []
    rt = _runtime(db_session, tmp_path, raw, clock=clock, gates=gates, events=events)
    config = mb.BatchConfig(poll_seconds=60, max_wait_seconds=120)
    assert mb.run_batch([raw.add(1)], config, rt) == 4
    assert _events(events, "gave_up")[0]["gate"] == "residential"
    assert clock.slept == [60, 60]  # gives up once it has waited the full 120 s


def test_a_chunk_never_starts_in_a_busy_window_and_yields_when_one_begins(
    db_session: Session, tmp_path: Path
) -> None:
    raw = _Raw(tmp_path / "raw")
    clock = _Clock(minute=2)
    clock.advance(50)  # 15:02:50, quiet until :03

    class _Slow(_Model):
        def predict_alpha(self, image: Image.Image) -> np.ndarray:
            clock.advance(6)  # each image takes 6 s
            return super().predict_alpha(image)

    gates = mb.Gates(windows=((3, 26),), gpu=None, residential=None, now=clock.now)
    events: list[dict[str, Any]] = []
    model = _Slow()
    rt = _runtime(db_session, tmp_path, raw, clock=clock, gates=gates, backend=model, events=events)
    items = [raw.add(n) for n in range(4)]
    assert mb.run_batch(items, mb.BatchConfig(chunk_size=10, poll_seconds=60), rt) == 0
    chunks = _events(events, "chunk")
    assert chunks[0]["count"] == 2 and chunks[0]["yielded"] is True  # :02:50 and :02:56
    assert _events(events, "paused")[0]["gate"] == "window"
    assert clock.now().minute >= 27
    assert sum(c["count"] for c in chunks) == 4
    # a scraper window can last 24 minutes: the GPU is not held through it
    # (the gate reads no GPU inside a window, so a game started then would
    # find ~9.5 GB taken), and the model is loaded again after it
    assert (model.releases, model.loads) == (1, 2)


def test_the_limit_caps_the_images_considered(db_session: Session, tmp_path: Path) -> None:
    raw = _Raw(tmp_path / "raw")
    events: list[dict[str, Any]] = []
    rt = _runtime(db_session, tmp_path, raw, events=events)
    items = [raw.add(n, rank=n) for n in range(5)]
    assert mb.run_batch(items, mb.BatchConfig(limit=2), rt) == 0
    assert _events(events, "start")[0]["pending"] == 2
    assert _events(events, "chunk")[0]["ranks"] == [0, 1]


def test_a_stop_request_finishes_the_image_in_flight(db_session: Session, tmp_path: Path) -> None:
    raw = _Raw(tmp_path / "raw")
    flag = {"stop": False}

    class _StopAfterOne(_Model):
        def predict_alpha(self, image: Image.Image) -> np.ndarray:
            flag["stop"] = True
            return super().predict_alpha(image)

    events: list[dict[str, Any]] = []
    rt = _runtime(
        db_session, tmp_path, raw, backend=_StopAfterOne(), events=events, stop=lambda: flag["stop"]
    )
    assert mb.run_batch([raw.add(1), raw.add(2)], mb.BatchConfig(), rt) == 130
    assert _events(events, "chunk")[0]["count"] == 1
    assert _events(events, "stopped")
    assert db_session.execute(select(MaskBatch.status)).scalars().all() == ["done"]


def test_a_stop_while_paused_exits_without_a_chunk(db_session: Session, tmp_path: Path) -> None:
    raw = _Raw(tmp_path / "raw")
    clock = _Clock(minute=10)
    gates = mb.Gates(windows=((3, 26),), gpu=None, residential=None, now=clock.now)
    events: list[dict[str, Any]] = []
    rt = _runtime(
        db_session,
        tmp_path,
        raw,
        clock=clock,
        gates=gates,
        events=events,
        stop=lambda: bool(clock.slept),
    )
    assert mb.run_batch([raw.add(1)], mb.BatchConfig(), rt) == 130
    assert _events(events, "chunk") == []


def test_a_model_that_will_not_load_is_refused_before_any_image(
    db_session: Session, tmp_path: Path
) -> None:
    class _NoCuda(_Model):
        def load(self) -> None:
            raise RuntimeError("onnxruntime session loaded without CUDAExecutionProvider")

    raw = _Raw(tmp_path / "raw")
    events: list[dict[str, Any]] = []
    rt = _runtime(db_session, tmp_path, raw, backend=_NoCuda(), events=events)
    assert mb.run_batch([raw.add(1)], mb.BatchConfig(), rt) == 2
    assert "CUDAExecutionProvider" in _events(events, "refused")[0]["detail"]
    assert mb.Progress(tmp_path / "progress.jsonl").load() == {}
    assert db_session.execute(select(func.count()).select_from(MaskBatch)).scalar_one() == 0


def test_a_backend_that_is_no_mask_model_is_refused(db_session: Session, tmp_path: Path) -> None:
    raw = _Raw(tmp_path / "raw")
    events: list[dict[str, Any]] = []
    rt = _runtime(db_session, tmp_path, raw, backend=StubMattingBackend(), events=events)
    assert mb.run_batch([raw.add(1)], mb.BatchConfig(), rt) == 2
    assert "not a mask model" in _events(events, "refused")[0]["detail"]


def test_the_dry_run_touches_nothing(tmp_path: Path) -> None:
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(n, rank=n % 3) for n in range(5)]

    def forbidden(*a: object, **k: object) -> Any:
        raise AssertionError("a dry run opened something")

    clock = _Clock(minute=10)
    events: list[dict[str, Any]] = []
    progress_path = tmp_path / "progress.jsonl"
    progress_path.write_text(
        json.dumps({"event": "item", "blob_sha256": items[0].blob_sha256, "status": "ok"}) + "\n"
    )
    before = progress_path.read_bytes()
    rt = mb.Runtime(
        gates=mb.Gates(windows=((3, 26),), gpu=None, residential=None, now=clock.now),
        open_db=forbidden,
        media_store=None,
        raw_store=None,
        make_backend=forbidden,
        progress=mb.Progress(progress_path),
        vram=forbidden,
        sleep=forbidden,
        monotonic=clock.monotonic,
        emit=events.append,
        stop_requested=lambda: False,
    )
    assert mb.run_batch(items, mb.BatchConfig(chunk_size=2, dry_run=True), rt) == 0
    (plan,) = _events(events, "plan")
    assert plan["pending"] == 4 and plan["chunks"] == 2
    assert plan["gate"] == {"ok": False, "gate": "window", "detail": plan["gate"]["detail"]}
    assert [i["rank"] for i in plan["first_chunk"]] == [0, 1]
    assert progress_path.read_bytes() == before
    assert not (tmp_path / "media").exists()


# --------------------------------------------------------------------------
# the work-list SQL (run by the orchestrator, read only)
# --------------------------------------------------------------------------

ROSS = "a7f14f89-0000-4000-8000-000000000001"


def _read_only(sql: str) -> None:
    lines = [ln.strip() for ln in sql.strip().splitlines() if ln.strip()]
    assert lines[0] == "BEGIN TRANSACTION READ ONLY;"
    assert lines[-1] == "ROLLBACK;"
    body = " ".join(lines).upper()
    for word in ("INSERT ", "UPDATE ", "DELETE ", "CREATE ", "DROP ", "ALTER ", "COPY ", "COMMIT"):
        assert word not in body, word


def test_the_ross_heads_query_is_read_only_and_names_one_user() -> None:
    sql = mb.ross_heads_sql(ROSS)
    _read_only(sql)
    assert f"'{ROSS}'::uuid" in sql
    assert "occ/%/head" in sql and "occ/%/status" not in sql
    with pytest.raises(ValueError, match="uuid"):
        mb.ross_heads_sql("x'; DROP TABLE holding; --")


def test_the_ross_worklist_query_carries_only_validated_heads() -> None:
    heads = [
        "##### ross-heads.sql",
        json.dumps({"head_id": "11111111-1111-4111-8111-111111111111", "rank": 0}),
        json.dumps({"head_id": "22222222-2222-4222-8222-222222222222", "rank": 2}),
        "(2 rows)",
    ]
    sql = mb.ross_worklist_sql(heads)
    _read_only(sql)
    assert "('11111111-1111-4111-8111-111111111111'::uuid, 0)" in sql
    assert "('22222222-2222-4222-8222-222222222222'::uuid, 2)" in sql
    with pytest.raises(ValueError, match="head"):
        mb.ross_worklist_sql([json.dumps({"head_id": "x'); DROP", "rank": 0})])
    with pytest.raises(ValueError, match="rank"):
        mb.ross_worklist_sql(
            [json.dumps({"head_id": "11111111-1111-4111-8111-111111111111", "rank": "0); --"})]
        )
    with pytest.raises(ValueError, match="rank"):
        mb.ross_worklist_sql(
            [json.dumps({"head_id": "11111111-1111-4111-8111-111111111111", "rank": 10})]
        )
    with pytest.raises(ValueError, match="no heads"):
        mb.ross_worklist_sql(["(0 rows)"])


def test_the_catalog_query_shards_by_the_first_byte_of_the_sha() -> None:
    whole = mb.catalog_worklist_sql()
    _read_only(whole)
    assert "get_byte" not in whole
    shard = mb.catalog_worklist_sql("3/8")
    _read_only(shard)
    assert "get_byte(pi.blob_sha256, 0) % 8 = 3" in shard
    for bad in ("8/8", "-1/8", "1/0", "a/b", "1"):
        with pytest.raises(ValueError, match="shard"):
            mb.catalog_worklist_sql(bad)


def test_every_work_list_carries_the_source_class() -> None:
    for sql in (mb.catalog_worklist_sql(), mb.official_gallery_worklist_sql()):
        assert "'source_class', pi.source_class" in sql


def test_the_official_gallery_query_is_the_gallery_lists_mfc_images_only() -> None:
    whole = mb.official_gallery_worklist_sql()
    _read_only(whole)
    assert "pi.source_class = 'manufacturer_press'" in whole
    assert "s.site = 'mfc'" in whole and "get_byte" not in whole
    shard = mb.official_gallery_worklist_sql("1/4")
    _read_only(shard)
    assert "get_byte(pi.blob_sha256, 0) % 4 = 1" in shard
    with pytest.raises(ValueError, match="shard"):
        mb.official_gallery_worklist_sql("4/4")


# --------------------------------------------------------------------------
# the command line
# --------------------------------------------------------------------------


def _worklist_file(tmp_path: Path, items: list[mb.WorkItem]) -> Path:
    path = tmp_path / "worklist.jsonl"
    path.write_text("".join(json.dumps(i.as_row()) + "\n" for i in items))
    return path


def test_main_refuses_without_the_residential_address(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MASK_BATCH_RESIDENTIAL_IP", raising=False)
    wl = _worklist_file(tmp_path, [])
    assert mb.main(["run", "--worklist", str(wl), "--device", "cpu", "--dry-run"]) == 2


def test_main_refuses_a_malformed_work_list(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wl = tmp_path / "bad.jsonl"
    wl.write_text(_line("Z" * 64, 0) + "\n")
    code = mb.main(["run", "--worklist", str(wl), "--no-residential-check", "--dry-run"])
    assert code == 2
    assert "blob_sha256" in _refusal(capsys)


def test_main_dry_run_prints_the_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = _Raw(tmp_path / "raw")
    wl = _worklist_file(tmp_path, [raw.add(1, rank=0), raw.add(2, rank=9)])
    monkeypatch.setattr(mb, "nvidia_smi_query", lambda index: lambda: "0, 11000, 1200, 12282")
    code = mb.main(
        [
            "run",
            "--worklist",
            str(wl),
            "--residential-ip",
            "198.51.100.7",
            "--ip-echo-url",
            "https://echo.invalid/",
            "--busy-windows",
            "",
            "--dry-run",
        ]
    )
    out = _stdout_events(capsys)
    plan = next(e for e in out if e["event"] == "plan")
    assert code == 0
    assert plan["pending"] == 2
    gate = json.loads(plan["gate"])
    assert gate["ok"] is False and gate["gate"] == "residential"
    assert (out[0]["event"], out[-1]["event"], out[-1]["exit"]) == ("job.start", "job.end", 0)


def test_main_runs_a_sandbox_batch_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = _Raw(tmp_path / "raw")
    wl = _worklist_file(tmp_path, [raw.add(1, rank=0), raw.add(2, rank=0)])
    monkeypatch.setattr(mb, "build_backend", lambda args: _Model())
    progress = tmp_path / "state" / "progress.jsonl"
    argv = [
        "--sandbox",
        str(tmp_path / "sandbox"),
        "run",
        "--worklist",
        str(wl),
        "--raw-dir",
        str(tmp_path / "raw"),
        "--progress",
        str(progress),
        "--device",
        "cpu",
        "--no-residential-check",
        "--busy-windows",
        "",
        "--chunk-size",
        "1",
    ]
    assert mb.main(argv) == 0
    out = _stdout_events(capsys)
    assert [e["count"] for e in out if e["event"] == "chunk"] == [1, 1]
    assert len(mb.Progress(progress).load()) == 2
    assert mb.main(argv) == 0  # resumed: nothing left
    again = _stdout_events(capsys)
    assert next(e for e in again if e["event"] == "start")["pending"] == 0


def test_main_runs_each_chunk_as_an_item_of_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """LG-U11: one root trace per run, one new trace per chunk (fc-shared's job
    model); every line carries run_id, the run's trace id."""
    raw = _Raw(tmp_path / "raw")
    wl = _worklist_file(tmp_path, [raw.add(1, rank=0), raw.add(2, rank=0)])
    monkeypatch.setattr(mb, "build_backend", lambda args: _Model())
    argv = ["--sandbox", str(tmp_path / "sandbox"), "run", "--worklist", str(wl)]
    argv += ["--raw-dir", str(tmp_path / "raw"), "--progress", str(tmp_path / "p.jsonl")]
    argv += ["--device", "cpu", "--no-residential-check", "--busy-windows", "", "--chunk-size", "1"]
    assert mb.main(argv) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    run_id = lines[0]["trace_id"]
    chunks = [line["trace_id"] for line in lines if line["event"] == "app.chunk"]
    assert len(chunks) == 2 and len({run_id, *chunks}) == 3
    assert {line["run_id"] for line in lines} == {run_id}
    assert {
        lines[-1]["trace_id"],
        next(x for x in lines if x["event"] == "app.done")["trace_id"],
    } == {run_id}


def test_main_reads_originals_from_the_raw_bucket_named_in_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    wl = _worklist_file(tmp_path, [])
    argv = ["--sandbox", str(tmp_path / "sb"), "run", "--worklist", str(wl), "--device", "cpu"]
    argv += ["--no-residential-check", "--busy-windows", ""]
    for name in ("RAW_S3_ENDPOINT_URL", "RAW_S3_BUCKET", "RAW_S3_ACCESS_KEY", "RAW_S3_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert mb.main(argv) == 2
    assert "RAW_S3_ENDPOINT_URL" in _refusal(capsys)
    monkeypatch.setenv("RAW_S3_ENDPOINT_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("RAW_S3_BUCKET", "raw")
    monkeypatch.setenv("RAW_S3_ACCESS_KEY", "k")
    monkeypatch.setenv("RAW_S3_SECRET_KEY", "s")
    args = mb._parser().parse_args(argv)
    store = mb._raw_store(args.__class__(**{**vars(args), "raw_dir": None}))
    assert isinstance(store, mb.ReadOnlyStore)
    with pytest.raises(PermissionError):
        store.put("raw-img/x", b"", "image/png")
    monkeypatch.setattr(mb, "build_backend", lambda args: _Model())
    assert mb.main(argv) == 0  # an empty work list touches no bucket


def test_main_stops_cleanly_on_sigterm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import signal

    class _Terminated(_Model):
        def predict_alpha(self, image: Image.Image) -> np.ndarray:
            signal.raise_signal(signal.SIGTERM)  # `docker stop` mid-image
            return super().predict_alpha(image)

    raw = _Raw(tmp_path / "raw")
    path = tmp_path / "worklist.jsonl"
    path.write_text(
        "##### banner\n" + "".join(json.dumps(raw.add(n).as_row()) + "\n" for n in (1, 2))
    )
    monkeypatch.setattr(mb, "build_backend", lambda args: _Terminated())
    before = signal.getsignal(signal.SIGTERM)
    argv = ["--sandbox", str(tmp_path / "sb"), "run", "--worklist", str(path), "--raw-dir"]
    argv += [str(tmp_path / "raw"), "--progress", str(tmp_path / "p.jsonl"), "--device", "cpu"]
    argv += ["--no-residential-check", "--busy-windows", ""]
    assert mb.main(argv) == 130
    out = _stdout_events(capsys)
    assert next(e for e in out if e["event"] == "worklist")["skipped_lines"] == 1
    assert (
        next(e for e in out if e["event"] == "chunk")["count"] == 1
    )  # the image in flight finished
    assert signal.getsignal(signal.SIGTERM) == before  # handlers put back


def test_main_prints_the_sql(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert mb.main(["sql", "ross-heads", "--user", ROSS]) == 0
    assert "occ/%/head" in capsys.readouterr().out
    heads = tmp_path / "heads.txt"
    heads.write_text(
        json.dumps({"head_id": "11111111-1111-4111-8111-111111111111", "rank": 0}) + "\n"
    )
    assert mb.main(["sql", "ross-worklist", "--heads", str(heads)]) == 0
    assert "'11111111-1111-4111-8111-111111111111'::uuid" in capsys.readouterr().out
    assert mb.main(["sql", "catalog-worklist", "--shard", "0/4"]) == 0
    assert "% 4 = 0" in capsys.readouterr().out
    assert mb.main(["sql", "official-gallery-worklist", "--shard", "2/4"]) == 0
    printed = capsys.readouterr().out
    assert "% 4 = 2" in printed and "'manufacturer_press'" in printed
    assert mb.main(["sql", "ross-heads", "--user", "nope"]) == 2


def test_build_backend_reads_the_weights_and_the_device(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.workers.basekept import BaseKeptMattingBackend
    from app.workers.matting import BiRefNetMattingBackend

    monkeypatch.setenv("MATTING_MODEL_PATH", "/b.onnx")
    monkeypatch.setenv("MATTING_ISNET_GENERAL_PATH", "/g.onnx")
    monkeypatch.setenv("MATTING_ISNET_ANIME_PATH", "/a.onnx")
    args = mb._parser().parse_args(
        ["run", "--worklist", "w", "--device", "auto", "--cpu-threads", "2"]
    )
    backend = mb.build_backend(args)
    assert isinstance(backend, BaseKeptMattingBackend)
    assert isinstance(backend.primary, BiRefNetMattingBackend)
    assert (backend.primary.device, backend.primary.cpu_threads) == ("auto", 2)


@pytest.mark.parametrize(("device", "gate_ok"), [("auto", True), ("cuda", False)])
def test_main_wires_the_gpu_less_pass_to_device_auto_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    device: str,
    gate_ok: bool,
) -> None:
    def absent() -> str:
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(mb, "nvidia_smi_query", lambda index: absent)
    wl = _worklist_file(tmp_path, [])
    argv = ["run", "--worklist", str(wl), "--device", device, "--no-residential-check"]
    assert mb.main([*argv, "--busy-windows", "", "--dry-run"]) == 0
    out = _stdout_events(capsys)
    plan = next(e for e in out if e["event"] == "plan")
    assert json.loads(plan["gate"])["ok"] is gate_ok


def test_a_copy_just_written_is_read_back_from_memory(tmp_path: Path) -> None:
    """The lane re-reads the clean copy it just stored (to hash it before the
    mask); over the WAN that is a Hetzner round trip per image saved."""
    gets: list[str] = []
    heads: list[str] = []

    class _Backing(LocalObjectStore):
        def get(self, key: str) -> bytes:
            gets.append(key)
            return super().get(key)

        def etag(self, key: str) -> str:
            heads.append(key)
            return super().etag(key)

    backing = _Backing(tmp_path / "media")
    store = mb.RecentWrites(backing, keep=2)
    for n in range(3):
        store.put(f"k{n}", bytes([n]), "image/png")
    assert store.get("k2") == b"\x02" and store.get("k1") == b"\x01"
    assert gets == []
    assert heads == ["k2", "k1"]  # the bucket's ETag vouches for each kept copy
    assert store.get("k0") == b"\x00"  # evicted: read from the bucket
    assert gets == ["k0"]
    assert backing.get("k2") == b"\x02"  # every write still reaches the bucket


def test_the_driver_reads_no_copy_back_from_the_bucket(db_session: Session, tmp_path: Path) -> None:
    raw = _Raw(tmp_path / "raw")
    gets: list[str] = []

    class _Backing(LocalObjectStore):
        def get(self, key: str) -> bytes:
            gets.append(key)
            return super().get(key)

    rt = _runtime(db_session, tmp_path, raw)
    rt.media_store = mb.RecentWrites(_Backing(tmp_path / "media"))
    assert mb.run_batch([raw.add(1), raw.add(2)], mb.BatchConfig(), rt) == 0
    assert gets == []


def test_the_residential_address_may_be_the_isps_block() -> None:
    """A DHCP renewal moves the box within its ISP's block (seen 2026-10-14:
    .66 -> .71); a VPN moves it out. A CIDR keeps the first from pausing the
    batch and still catches the second."""
    answers = iter(["198.51.100.71", "203.0.113.9", "not-an-address"])
    check = mb.ResidentialCheck(expected_ip="198.51.100.0/24", fetch=lambda: next(answers), ttl=0)
    assert check.check().ok is True
    assert check.check().ok is False
    assert check.check().ok is False
    with pytest.raises(ValueError, match="residential"):
        mb.ResidentialCheck(expected_ip="home", fetch=lambda: "x")


# --------------------------------------------------------------------------
# the gate while the model is held: another consumer pauses, our own never
# --------------------------------------------------------------------------

#: The card as nvidia-smi shows it on this box (MASK-BATCH-1 close-out 2,
#: measured): ~1.3 GB of desktop; the model's own 9,473 MiB once it has run
#: (MORE than the 9 GiB mark, so free + own can never be the test); a game's
#: ~1.5 GB and 97 % on top.
IDLE = "0, 10985, 1297, 12282"
HELD = "0, 1112, 10770, 12282"  # our model held, the chunk over: the card idle
HELD_TAIL = "6, 1112, 10770, 12282"  # the same, a reading that caught its own tail
GAME_ON_HELD = "97, 0, 12282, 12282"
GAME = "97, 7985, 4297, 12282"
OWN = 10770 - 1297


def _held_gate(*after: str, samples: int = 1) -> tuple[mb.GpuGate, list[float]]:
    """A gate that saw IDLE, loaded the model and measured it at HELD; then ``after``."""
    readings = iter([IDLE] * samples + [HELD, *after])
    slept: list[float] = []
    gate = mb.GpuGate(query=lambda: next(readings), samples=samples, sleep=slept.append)
    assert gate.check().ok is True  # IDLE: before the model is loaded
    gate.mark_loading()
    gate.measure_own()
    assert gate.own_mib == OWN
    slept.clear()
    return gate, slept


def test_the_measured_own_memory_is_above_the_9_gib_mark() -> None:
    # why crediting the model's own memory cannot be the rule on this card
    assert OWN > mb.DEFAULT_GPU_FREE_MIN_MIB


def test_while_the_model_is_held_its_own_memory_never_pauses_the_batch() -> None:
    gate, slept = _held_gate(HELD, HELD_TAIL, HELD, samples=3)
    held = gate.check()  # 1112 MiB free and the model's 9.2 GB: no pause
    assert held.ok is True
    assert "+0 MiB" in held.detail
    assert slept == [mb.DEFAULT_GPU_SETTLE_SECONDS, 0.5, 0.5]  # settle, then the samples
    assert mb.DEFAULT_GPU_SETTLE_SECONDS >= 2


def test_while_held_a_game_pauses_the_batch() -> None:
    gate, _ = _held_gate(GAME_ON_HELD)
    result = gate.check()
    assert result.ok is False and result.gate == "gpu"
    assert "97%" in result.detail and "+1512 MiB" in result.detail


@pytest.mark.parametrize(("grown", "ok"), [(0, True), (512, True), (513, False), (1200, False)])
def test_while_held_memory_another_process_takes_pauses_the_batch(grown: int, ok: bool) -> None:
    # a quiet consumer (util 2 %): only its memory shows it
    gate, _ = _held_gate(f"2, {1112 - grown}, {10770 + grown}, 12282")
    result = gate.check()
    assert result.ok is ok
    if not ok:
        assert f"+{grown} MiB" in result.detail


@pytest.mark.parametrize(
    ("utils", "ok"),
    [
        ((60, 60, 60), False),  # busy on every settled sample: someone else computes
        ((60, 4, 60), True),  # one quiet sample: our tail or a blip
        ((5, 5, 5), False),
        ((4, 4, 4), True),
    ],
)
def test_while_held_a_gpu_busy_on_every_settled_sample_pauses_the_batch(
    utils: tuple[int, int, int], ok: bool
) -> None:
    # a compute consumer with next to no memory (+100 MiB)
    gate, slept = _held_gate(*(f"{u}, 1012, 10870, 12282" for u in utils), samples=3)
    result = gate.check()
    assert result.ok is ok
    assert len(slept) == 3  # every sample was read: one busy reading alone never pauses
    if not ok:
        assert "every" in result.detail


def test_memory_another_process_freed_lowers_the_reference() -> None:
    # someone left (-500 MiB), then a quiet consumer takes 513 MiB: +13 MiB over
    # the snapshot, +513 over what the card held after they left
    gate, _ = _held_gate("2, 1612, 10270, 12282", "2, 1099, 10783, 12282")
    assert gate.check().ok is True
    assert gate.check().ok is False


@pytest.mark.parametrize(("excess", "ok"), [(0, True), (512, True), (513, False)])
def test_own_memory_beyond_the_caps_counts_as_another_consumer(excess: int, ok: bool) -> None:
    """A consumer that started while the model loaded is measured as ours; the
    model cannot hold more than its arena caps (8 + 2 GiB) and ~0.5 GiB of
    overhead, so the rest is someone else's."""
    assert mb.OWN_MAX_MIB == 8192 + 2048 + 512
    used = 300 + mb.OWN_MAX_MIB + excess
    readings = iter(["0, 11900, 300, 12282"] + [f"0, {12282 - used}, {used}, 12282"] * 2)
    gate = mb.GpuGate(query=lambda: next(readings), samples=1, sleep=lambda s: None)
    gate.check()
    gate.mark_loading()
    gate.measure_own()
    assert gate.own_mib == mb.OWN_MAX_MIB + excess  # reported as measured
    assert gate.check().ok is ok


def test_loaded_but_not_yet_measured_the_gate_is_ross_rule_after_the_settle() -> None:
    # a chunk of ineligible photos only: the weights loaded, the model never ran
    readings = iter([IDLE, "0, 8400, 3300, 12282", "97, 8400, 3300, 12282"])
    slept: list[float] = []
    gate = mb.GpuGate(query=lambda: next(readings), samples=1, sleep=slept.append)
    gate.check()
    gate.mark_loading()
    assert gate.check().ok is True  # idle, though 8400 MiB < 9 GiB free
    assert gate.check().ok is False  # busy and short of 9 GiB: Ross's rule
    assert slept == [mb.DEFAULT_GPU_SETTLE_SECONDS] * 2


def test_released_the_gate_is_ross_rule_again_without_a_settle() -> None:
    gate, slept = _held_gate(GAME, "97, 9300, 2690, 12282", IDLE)
    gate.forget_own()
    assert gate.own_mib == 0
    assert gate.check().ok is False  # the game, our model gone: busy, 7985 MiB free
    assert gate.check().ok is True  # busy, but 9 GiB free: Ross's rule passes it
    assert gate.check().ok is True
    assert slept == []


def test_a_new_load_is_measured_afresh() -> None:
    # loaded again without a release: Ross's rule until the model has run again
    gate, _ = _held_gate("0, 512, 11370, 12282")
    gate.mark_loading()
    assert gate.check().ok is True  # idle; +600 MiB over the old measurement is not growth


def test_the_own_memory_is_never_negative_and_needs_a_baseline() -> None:
    gate = mb.GpuGate(query=lambda: HELD, samples=1, sleep=lambda s: None)
    gate.mark_loading()  # no reading yet: no baseline
    gate.measure_own()
    assert gate.own_mib == 0
    shrinking = iter([HELD, IDLE])
    gate = mb.GpuGate(query=lambda: next(shrinking), samples=1, sleep=lambda s: None)
    gate.check()
    gate.mark_loading()  # baseline 10770 (someone else's memory)
    gate.measure_own()  # it went away: 1297 - 10770
    assert gate.own_mib == 0


@pytest.mark.parametrize(
    "error",
    [
        OSError("nvidia-smi"),
        subprocess.CalledProcessError(9, "nvidia-smi"),
        subprocess.TimeoutExpired("nvidia-smi", 15),
        ValueError("N/A"),
    ],
)
def test_an_unreadable_gpu_at_measure_time_leaves_ross_rule_in_force(error: Exception) -> None:
    def broken() -> str:
        raise error

    gate = mb.GpuGate(query=lambda: IDLE, samples=1, sleep=lambda s: None)
    gate.check()
    gate.mark_loading()
    gate._query = broken
    gate.measure_own()  # no exception: no measurement
    assert gate.own_mib == 0
    gate._query = lambda: HELD_TAIL  # util 6 %, 1112 MiB free: Ross's rule pauses
    assert gate.check().ok is False


class _Card:
    """nvidia-smi on the box: what it shows follows the model and a consumer."""

    def __init__(self, model: _Model, other: tuple[int, int] = (97, 1512)) -> None:
        self.model = model
        self.game = False
        self.other = other  # (util, MiB) of the other consumer
        self.reads = 0

    def query(self) -> str:
        self.reads += 1
        used = 1297 + (OWN if self.model.loads > self.model.releases else 0)
        util = 0
        if self.game:
            util, used = self.other[0], used + self.other[1]
        used = min(used, 12282)
        return f"{util}, {12282 - used}, {used}, 12282"


def test_with_the_model_loaded_and_nothing_else_on_the_gpu_the_batch_never_pauses(
    db_session: Session, tmp_path: Path
) -> None:
    raw = _Raw(tmp_path / "raw")
    clock = _Clock()
    model = _Model()
    card = _Card(model)
    gate = mb.GpuGate(query=card.query, samples=3, sleep=lambda s: None)
    gates = mb.Gates(windows=(), gpu=gate, residential=None, now=clock.now)
    events: list[dict[str, Any]] = []
    rt = _runtime(db_session, tmp_path, raw, clock=clock, gates=gates, backend=model, events=events)
    items = [raw.add(n) for n in range(4)]
    assert mb.run_batch(items, mb.BatchConfig(chunk_size=1, poll_seconds=60), rt) == 0
    assert _events(events, "paused") == []
    assert (model.loads, model.releases) == (1, 0)
    chunks = _events(events, "chunk")
    assert [c["held_vram_mib"] for c in chunks] == [OWN] * 4
    assert clock.slept.count(60) == 0


@pytest.mark.parametrize(
    ("other", "shows"),
    [
        ((97, 1512), "97%"),  # a game: busy and memory
        ((2, 1200), "+1200 MiB"),  # quiet, memory only
        ((90, 100), "every"),  # compute, next to no memory
    ],
)
def test_another_process_on_the_gpu_still_pauses_the_batch(
    db_session: Session, tmp_path: Path, other: tuple[int, int], shows: str
) -> None:
    raw = _Raw(tmp_path / "raw")
    clock = _Clock()
    events: list[dict[str, Any]] = []

    class _Gamer(_Model):
        def predict_alpha(self, image: Image.Image) -> np.ndarray:
            # it starts during the second chunk, the model long measured
            card.game = self.loads == 1 and len(_events(events, "chunk")) == 1
            return super().predict_alpha(image)

    gamer = _Gamer()
    card = _Card(gamer, other)

    def sleep(seconds: float) -> None:
        clock.sleep(seconds)
        card.game = False  # it ends while the batch waits

    gate = mb.GpuGate(query=card.query, samples=3, sleep=lambda s: None)
    gates = mb.Gates(windows=(), gpu=gate, residential=None, now=clock.now)
    rt = _runtime(db_session, tmp_path, raw, clock=clock, gates=gates, backend=gamer, events=events)
    rt.sleep = sleep
    items = [raw.add(n) for n in range(3)]
    assert mb.run_batch(items, mb.BatchConfig(chunk_size=1, poll_seconds=60), rt) == 0
    paused = _events(events, "paused")
    assert len(paused) == 1 and paused[0]["gate"] == "gpu"
    assert shows in paused[0]["detail"]
    assert clock.slept == [60]
    assert len(_events(events, "resumed")) == 1
    assert (gamer.loads, gamer.releases) == (2, 1)  # released for it, loaded again
    assert gate.own_mib == OWN  # measured again after the reload


# --------------------------------------------------------------------------
# the run options are validated before anything runs
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--chunk-size", "0"),  # would loop forever, a mask_batches row per pass
        ("--chunk-size", "-1"),
        ("--chunk-size", "two"),
        ("--max-consecutive-failures", "0"),  # would call one success an outage
        ("--max-consecutive-failures", "-3"),
        ("--gpu-samples", "0"),
        ("--gpu-samples", "-1"),
        ("--limit", "-1"),  # would silently drop the last image
        ("--limit", "1.5"),
        ("--poll-seconds", "0"),  # a tight loop on the address echo in a VPN window
        ("--poll-seconds", "0.5"),
        ("--poll-seconds", "-60"),
        ("--poll-seconds", "nan"),
        ("--poll-seconds", "inf"),
        ("--poll-seconds", "soon"),
        ("--poll-seconds", "3601"),  # an hour at most between two looks at the gate
        ("--poll-seconds", "1e308"),  # time.sleep raises OverflowError
        ("--max-wait-seconds", "nan"),
        ("--max-wait-seconds", "inf"),
        ("--max-wait-seconds", "-5"),
        ("--residential-ttl", "nan"),
        ("--residential-ttl", "inf"),
        ("--residential-ttl", "0"),
        ("--residential-ttl", "-1"),
        ("--gpu-free-min-mib", "0"),  # would pass a full card
        ("--gpu-free-min-mib", "1023"),
        ("--writers", "-1"),
        ("--writers", "24"),  # with the batch's own, more than 24 connections to the database
        ("--decode-slots", "0"),
        ("--decode-slots", "9"),  # ~0.55 GB each at 4000x3000
        ("--writers", "two"),
        ("--prefetch", "0"),
        ("--prefetch", "17"),  # originals held in memory
    ],
)
def test_a_bad_run_option_is_refused_by_the_parser(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], option: str, value: str
) -> None:
    wl = _worklist_file(tmp_path, [])
    argv = ["run", "--worklist", str(wl), "--no-residential-check", "--dry-run", option, value]
    with pytest.raises(SystemExit) as exc:
        mb.main(argv)
    assert exc.value.code == 2
    assert option in capsys.readouterr().err
    assert not (tmp_path / "mask-batch-progress.jsonl").exists()


@pytest.mark.parametrize(
    ("option", "value", "field", "expected"),
    [
        ("--chunk-size", "1", "chunk_size", 1),
        ("--max-consecutive-failures", "1", "max_consecutive_failures", 1),
        ("--gpu-samples", "1", "gpu_samples", 1),
        ("--limit", "0", "limit", 0),
        ("--poll-seconds", "1", "poll_seconds", 1.0),
        ("--poll-seconds", "2.5", "poll_seconds", 2.5),
        ("--poll-seconds", "3600", "poll_seconds", 3600.0),
        ("--max-wait-seconds", "0", "max_wait_seconds", 0.0),
        ("--residential-ttl", "0.5", "residential_ttl", 0.5),
        ("--gpu-free-min-mib", "1024", "gpu_free_min_mib", 1024),
        ("--writers", "0", "writers", 0),
        ("--writers", "16", "writers", 16),
        ("--writers", "23", "writers", 23),
        ("--decode-slots", "1", "decode_slots", 1),
        ("--decode-slots", "8", "decode_slots", 8),
        ("--prefetch", "1", "prefetch", 1),
        ("--prefetch", "16", "prefetch", 16),
        ("--cpu-workers", "0", "cpu_workers", 0),
        ("--cpu-workers", "16", "cpu_workers", 16),
    ],
)
def test_the_smallest_sane_run_options_are_accepted(
    option: str, value: str, field: str, expected: float
) -> None:
    args = mb._parser().parse_args(["run", "--worklist", "w", option, value])
    assert getattr(args, field) == expected


@pytest.mark.parametrize(
    "argv",
    [
        ["--sand", "box"],  # before the command
        ["--dev", "auto"],  # would run on the CPU in the wrapper's 8g (it looks for --device)
        ["--devi=cpu"],
        ["--chunk", "5"],
        ["--retry"],
    ],
)
def test_an_abbreviated_option_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    wl = _worklist_file(tmp_path, [])
    run = ["run", "--worklist", str(wl), "--no-residential-check", "--dry-run"]
    with pytest.raises(SystemExit) as exc:
        mb.main([*argv, *run] if argv[0] == "--sand" else [*run, *argv])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "unrecognized arguments" in err or "invalid choice: 'box'" in err


def test_run_batch_refuses_a_chunk_size_below_one(db_session: Session, tmp_path: Path) -> None:
    raw = _Raw(tmp_path / "raw")
    events: list[dict[str, Any]] = []
    model = _Model()
    calls = iter(range(100))  # without the guard the loop never ends: stop it, red not hung
    rt = _runtime(
        db_session, tmp_path, raw, backend=model, events=events, stop=lambda: next(calls) >= 50
    )
    for size in (0, -1):
        assert mb.run_batch([raw.add(1)], mb.BatchConfig(chunk_size=size), rt) == 2
    assert [e["event"] for e in events] == ["refused", "refused"]
    assert "chunk size" in events[0]["detail"]
    assert model.loads == 0
    assert db_session.execute(select(func.count()).select_from(MaskBatch)).scalar_one() == 0
    assert not (tmp_path / "progress.jsonl").exists()


# --------------------------------------------------------------------------
# an outage is not the images' failure: a plain rerun retries them
# --------------------------------------------------------------------------


class _Flaky(LocalObjectStore):
    """The media bucket through a port-forward that drops: while ``down``,
    every read fails (the lane records the image as failed in the media db)."""

    down = False

    def get(self, key: str) -> bytes:
        if self.down:
            raise ConnectionError("port-forward gone")
        return super().get(key)


def test_failures_that_open_the_circuit_are_interrupted_and_a_plain_rerun_retries_them(
    db_session: Session, tmp_path: Path
) -> None:
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(n) for n in range(6)]
    media = _Flaky(tmp_path / "media")

    class _Outage(_Model):
        def predict_alpha(self, image: Image.Image) -> np.ndarray:
            media.down = True  # the forward drops after the first image
            return super().predict_alpha(image)

    events: list[dict[str, Any]] = []
    rt = _runtime(db_session, tmp_path, raw, backend=_Outage(), events=events)
    rt.media_store = media
    config = mb.BatchConfig(chunk_size=2, max_consecutive_failures=3)
    assert mb.run_batch(items, config, rt) == 3
    progress = mb.Progress(tmp_path / "progress.jsonl").load()
    statuses = [progress[i.blob_sha256]["status"] for i in items[:4]]
    assert statuses == ["ok", "interrupted", "interrupted", "interrupted"]
    assert progress[items[1].blob_sha256]["reason"] == "ConnectionError"  # the cause is kept
    chunks = _events(events, "chunk")
    assert chunks[1]["counts"] == {"interrupted": 2}  # this chunk's share of the streak
    assert [f["reason"] for f in chunks[1]["failures"]] == ["ConnectionError"] * 2
    assert _events(events, "circuit_open")[0]["interrupted"] == 3
    # the lane recorded those three as failed in the media db
    assert mask_job.original_settled(db_session, items[1].blob_sha256) == "failed"

    media.down = False  # the operator restores the port-forward and reruns, no flags
    events2: list[dict[str, Any]] = []
    rt2 = _runtime(db_session, tmp_path, raw, events=events2)
    rt2.media_store = media
    assert mb.run_batch(items, mb.BatchConfig(chunk_size=2), rt2) == 0
    start = _events(events2, "start")[0]
    assert (start["done"], start["interrupted_before"], start["pending"]) == (1, 3, 5)
    assert _events(events2, "done")[0]["counts"] == {"ok": 5}
    progress = mb.Progress(tmp_path / "progress.jsonl").load()
    assert {progress[i.blob_sha256]["status"] for i in items} == {"ok"}


def test_a_failure_that_does_not_open_the_circuit_stays_failed(
    db_session: Session, tmp_path: Path
) -> None:
    raw = _Raw(tmp_path / "raw")
    gone = mb.WorkItem(blob_sha256="e" * 64, storage_key="raw-img/none", url=ITEM, product_id="p")
    rt = _runtime(db_session, tmp_path, raw)
    assert mb.run_batch([gone, raw.add(1)], mb.BatchConfig(max_consecutive_failures=2), rt) == 1
    assert mb.Progress(tmp_path / "progress.jsonl").load()["e" * 64]["status"] == "failed"


# --------------------------------------------------------------------------
# only an outage opens the circuit; an image's own fault stays failed
# --------------------------------------------------------------------------


def _client_error(code: str, status: int) -> Exception:
    """As boto3 raises it: a ClientError subclass named for the code when it has one."""
    from botocore.exceptions import ClientError

    cls = type(code, (ClientError,), {}) if code == "NoSuchKey" else ClientError
    body = {
        "Error": {"Code": code, "Message": code},
        "ResponseMetadata": {"HTTPStatusCode": status},
    }
    return cls(body, "GetObject")  # type: ignore[arg-type]


def _transport_errors() -> list[Exception]:
    from botocore.exceptions import (
        ConnectionClosedError,
        EndpointConnectionError,
        ReadTimeoutError,
    )
    from requests.exceptions import ReadTimeout
    from sqlalchemy.exc import InterfaceError, OperationalError

    return [
        ConnectionError("port-forward gone"),
        ConnectionResetError(),
        TimeoutError(),
        OperationalError("SELECT 1", {}, Exception("server closed the connection")),
        EndpointConnectionError(endpoint_url="https://hel1.example"),
        ReadTimeoutError(endpoint_url="https://hel1.example"),
        _client_error("SlowDown", 503),
        _client_error("InternalError", 500),
        _client_error("GatewayTimeout", 504),
        ConnectionClosedError(endpoint_url="https://hel1.example"),
        InterfaceError("SELECT 1", {}, Exception("connection already closed")),
        ReadTimeout(),
    ]


def _own_faults() -> list[Exception]:
    from PIL import UnidentifiedImageError

    return [
        _client_error("NoSuchKey", 404),
        _client_error("AccessDenied", 403),
        FileNotFoundError("raw-img/none"),
        UnidentifiedImageError("not a picture"),
        ValueError("bad"),
        RuntimeError("model"),
    ]


def test_an_outage_is_told_from_an_images_own_fault() -> None:
    assert [mb.is_outage(e) for e in _transport_errors()] == [True] * 12
    assert [mb.is_outage(e) for e in _own_faults()] == [False] * 6
    # the lane records the class name only
    for name in (
        "ConnectionError",
        "OperationalError",
        "InterfaceError",
        "ReadTimeoutError",
        "TimeoutError",
        "Timeout",
        "HTTPClientError",
        "ConnectionResetError",
        "ConnectionRefusedError",
        "ConnectionAbortedError",
        "BrokenPipeError",
        "EndpointConnectionError",
        "ConnectionClosedError",
        "ConnectTimeoutError",
        "ProxyConnectionError",
    ):
        assert mb.is_outage_reason(name) is True
    for name in ("ClientError", "RuntimeError", "unsupported_codec:zstd", "Fail", None):
        assert mb.is_outage_reason(name) is False


@pytest.mark.parametrize(
    ("status", "outage"), [(499, False), (500, True), (504, True), (599, True)]
)
def test_every_bucket_status_from_500_up_is_an_outage(status: int, outage: bool) -> None:
    assert mb.is_outage(_client_error("Err", status)) is outage


class _Bucket(LocalObjectStore):
    """The raw bucket as S3 answers: a missing key is NoSuchKey; a key in
    ``down`` is an outage (the connection drops)."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.down: set[str] = set()

    def get(self, key: str) -> bytes:
        if key in self.down:
            raise ConnectionError("connection dropped")
        try:
            return super().get(key)
        except FileNotFoundError:
            raise _client_error("NoSuchKey", 404) from None


def _own_fault_items(raw: _Raw) -> list[mb.WorkItem]:
    """Five originals that fail for their own reasons, every run the same."""
    zstd = dataclasses.replace(raw.add(101), codec="zstd")
    wrong = raw.add(102)
    raw.store.put(wrong.storage_key, b"other bytes", "image/png")
    gone = dataclasses.replace(raw.add(103), storage_key="raw-img/sha256/no/such.png")
    junk = b"not a picture at all"
    sha = hashlib.sha256(junk).hexdigest()
    raw.store.put(f"raw-img/{sha}", junk, "image/png")
    not_image = mb.WorkItem(blob_sha256=sha, storage_key=f"raw-img/{sha}", url=ITEM, product_id="p")
    gzip_junk = dataclasses.replace(raw.add(104), codec="gzip")
    raw.store.put(gzip_junk.storage_key, _photo(104), "image/png")  # not gzip at all
    return [dataclasses.replace(i, rank=0) for i in (zstd, wrong, gone, not_image, gzip_junk)]


def test_an_images_own_faults_never_open_the_circuit_and_a_plain_rerun_moves_on(
    db_session: Session, tmp_path: Path
) -> None:
    raw = _Raw(tmp_path / "raw")
    raw.store = _Bucket(tmp_path / "raw")
    bad = _own_fault_items(raw)
    good = [raw.add(n, rank=1) for n in range(3)]
    events: list[dict[str, Any]] = []
    rt = _runtime(db_session, tmp_path, raw, events=events)
    config = mb.BatchConfig(chunk_size=4, max_consecutive_failures=3)
    assert mb.run_batch(bad + good, config, rt) == 1  # done, with failures
    assert _events(events, "circuit_open") == []
    progress = mb.Progress(tmp_path / "progress.jsonl").load()
    assert [progress[i.blob_sha256]["reason"] for i in bad] == [
        "unsupported_codec:zstd",
        "raw_hash_mismatch",
        "fetch:NoSuchKey",
        "seed:UnidentifiedImageError",
        "fetch:BadGzipFile",
    ]
    assert {progress[i.blob_sha256]["status"] for i in bad} == {"failed"}
    assert {progress[i.blob_sha256]["status"] for i in good} == {"ok"}

    for _ in range(2):  # plain reruns: nothing left, never stuck on the five
        again: list[dict[str, Any]] = []
        rt2 = _runtime(db_session, tmp_path, raw, events=again)
        assert mb.run_batch(bad + good, config, rt2) == 0
        start = _events(again, "start")[0]
        assert (start["pending"], start["failed_before"]) == (0, 5)


def test_an_images_own_fault_neither_counts_toward_nor_resets_the_streak(
    db_session: Session, tmp_path: Path
) -> None:
    raw = _Raw(tmp_path / "raw")
    bucket = _Bucket(tmp_path / "raw")
    raw.store = bucket
    out = [raw.add(n, rank=0) for n in (1, 2, 4, 5)]
    bucket.down = {i.storage_key for i in out}
    zstd = dataclasses.replace(raw.add(3, rank=0), codec="zstd")
    items = [out[0], out[1], zstd, out[2], out[3], raw.add(6, rank=1)]
    events: list[dict[str, Any]] = []
    rt = _runtime(db_session, tmp_path, raw, events=events)
    assert mb.run_batch(items, mb.BatchConfig(max_consecutive_failures=3), rt) == 3
    (opened,) = _events(events, "circuit_open")
    assert (opened["consecutive_failures"], opened["interrupted"]) == (3, 3)
    progress = mb.Progress(tmp_path / "progress.jsonl").load()
    statuses = [progress[i.blob_sha256]["status"] for i in items[:4]]
    assert statuses == ["interrupted", "interrupted", "failed", "interrupted"]
    assert items[4].blob_sha256 not in progress  # stopped before it


def test_a_lone_outage_failure_is_interrupted_and_retried_by_a_plain_rerun(
    db_session: Session, tmp_path: Path
) -> None:
    raw = _Raw(tmp_path / "raw")
    bucket = _Bucket(tmp_path / "raw")
    raw.store = bucket
    items = [raw.add(n) for n in range(3)]
    bucket.down = {items[0].storage_key}
    rt = _runtime(db_session, tmp_path, raw)
    assert mb.run_batch(items, mb.BatchConfig(), rt) == 1  # done, one image interrupted
    progress = mb.Progress(tmp_path / "progress.jsonl").load()
    assert progress[items[0].blob_sha256]["status"] == "interrupted"
    assert progress[items[0].blob_sha256]["reason"] == "fetch:ConnectionError"
    bucket.down = set()
    events: list[dict[str, Any]] = []
    assert (
        mb.run_batch(items, mb.BatchConfig(), _runtime(db_session, tmp_path, raw, events=events))
        == 0
    )
    assert _events(events, "done")[0]["counts"] == {"ok": 1}


def test_an_interrupted_image_that_fails_for_its_own_fault_becomes_failed(
    db_session: Session, tmp_path: Path
) -> None:
    raw = _Raw(tmp_path / "raw")
    bucket = _Bucket(tmp_path / "raw")
    raw.store = bucket
    items = [raw.add(n) for n in range(4)]
    bucket.down = {i.storage_key for i in items}
    config = mb.BatchConfig(max_consecutive_failures=2)
    assert mb.run_batch(items, config, _runtime(db_session, tmp_path, raw)) == 3
    bucket.down = set()
    for item in items[:2]:  # their bytes went bad meanwhile: their own fault now
        raw.store.put(item.storage_key, b"rotted", "image/png")
    assert mb.run_batch(items, config, _runtime(db_session, tmp_path, raw)) == 1
    progress = mb.Progress(tmp_path / "progress.jsonl").load()
    assert [progress[i.blob_sha256]["status"] for i in items] == ["failed", "failed", "ok", "ok"]
    assert progress[items[0].blob_sha256]["reason"] == "raw_hash_mismatch"
    events: list[dict[str, Any]] = []
    assert mb.run_batch(items, config, _runtime(db_session, tmp_path, raw, events=events)) == 0
    assert _events(events, "start")[0]["pending"] == 0  # not re-driven again


class _Bucket5xx(LocalObjectStore):
    """The media bucket answering 503 to every PUT (or GET) once ``down``."""

    down = False
    reads_too = False

    def put(self, key: str, data: bytes, content_type: str) -> None:
        if self.down:
            raise _client_error("SlowDown", 503)
        super().put(key, data, content_type)

    def get(self, key: str) -> bytes:
        if self.reads_too:
            raise _client_error("ServiceUnavailable", 503)
        return super().get(key)


def test_a_bucket_5xx_on_the_lanes_read_is_an_outage(db_session: Session, tmp_path: Path) -> None:
    raw = _Raw(tmp_path / "raw")
    media = _Bucket5xx(tmp_path / "media")
    media.reads_too = True  # the lane reads the seeded copy back: 503
    rt = _runtime(db_session, tmp_path, raw)
    rt.media_store = media
    item = raw.add(1)
    assert mb.run_batch([item], mb.BatchConfig(), rt) == 1
    record = mb.Progress(tmp_path / "progress.jsonl").load()[item.blob_sha256]
    assert (record["status"], record["reason"]) == ("interrupted", "ClientError")


def test_a_database_blip_inside_the_lane_is_an_outage(
    db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lane catches it and records only the class name; no bucket error
    was seen, so the name alone tells the outage."""
    from sqlalchemy.exc import OperationalError

    def blip(*a: object, **k: object) -> None:
        raise OperationalError("INSERT", {}, Exception("server closed the connection"))

    monkeypatch.setattr(mask_job, "register_public_object", blip)
    raw = _Raw(tmp_path / "raw")
    item = raw.add(1)
    assert mb.run_batch([item], mb.BatchConfig(), _runtime(db_session, tmp_path, raw)) == 1
    record = mb.Progress(tmp_path / "progress.jsonl").load()[item.blob_sha256]
    assert (record["status"], record["reason"]) == ("interrupted", "OperationalError")


def test_a_bucket_5xx_inside_the_lane_is_an_outage(db_session: Session, tmp_path: Path) -> None:
    """The lane records only the class (ClientError, 4xx or 5xx alike); the
    driver saw the exception the bucket raised and tells the outage apart."""
    raw = _Raw(tmp_path / "raw")
    media = _Bucket5xx(tmp_path / "media")

    class _Then503(_Model):
        def predict_alpha(self, image: Image.Image) -> np.ndarray:
            media.down = True  # after the copy was seeded: the mask PUT gets the 503
            return super().predict_alpha(image)

    rt = _runtime(db_session, tmp_path, raw, backend=_Then503())
    rt.media_store = media
    items = [raw.add(n) for n in range(3)]
    assert mb.run_batch(items, mb.BatchConfig(max_consecutive_failures=5), rt) == 1
    progress = mb.Progress(tmp_path / "progress.jsonl").load()
    first = progress[items[0].blob_sha256]
    assert (first["status"], first["reason"]) == ("interrupted", "ClientError")
    assert [progress[i.blob_sha256]["reason"] for i in items[1:]] == ["seed:ClientError"] * 2
    assert {progress[i.blob_sha256]["status"] for i in items} == {"interrupted"}


class _Bucket403(LocalObjectStore):
    """The media bucket answering AccessDenied (403) to every PUT once ``down``."""

    down = False

    def put(self, key: str, data: bytes, content_type: str) -> None:
        if self.down:
            raise _client_error("AccessDenied", 403)
        super().put(key, data, content_type)


def test_a_bucket_403_inside_the_lane_stays_failed_and_opens_no_circuit(
    db_session: Session, tmp_path: Path
) -> None:
    """A 4xx the media bucket answers inside the mask step is the image's fault
    (the lane records ClientError for it, as for a 503): not an outage."""
    raw = _Raw(tmp_path / "raw")
    media = _Bucket403(tmp_path / "media")

    class _Then403(_Model):
        def predict_alpha(self, image: Image.Image) -> np.ndarray:
            media.down = True  # after the copy was seeded: the mask PUT gets the 403
            return super().predict_alpha(image)

    rt = _runtime(db_session, tmp_path, raw, backend=_Then403())
    rt.media_store = media
    item = raw.add(1)
    assert mb.run_batch([item], mb.BatchConfig(max_consecutive_failures=1), rt) == 1  # not 3
    record = mb.Progress(tmp_path / "progress.jsonl").load()[item.blob_sha256]
    assert (record["status"], record["reason"]) == ("failed", "ClientError")


class _ClosedRaw(LocalObjectStore):
    """The raw bucket whose connection closes in the middle of the response."""

    def get(self, key: str) -> bytes:
        from botocore.exceptions import ConnectionClosedError

        raise ConnectionClosedError(endpoint_url="https://hel1.example")


def test_a_raw_connection_closed_mid_response_is_an_outage(
    db_session: Session, tmp_path: Path
) -> None:
    raw = _Raw(tmp_path / "raw")
    rt = _runtime(db_session, tmp_path, raw)
    rt.raw_store = mb.ReadOnlyStore(_ClosedRaw(tmp_path / "closed"))
    item = raw.add(1)
    assert mb.run_batch([item], mb.BatchConfig(), rt) == 1
    record = mb.Progress(tmp_path / "progress.jsonl").load()[item.blob_sha256]
    assert (record["status"], record["reason"]) == ("interrupted", "fetch:ConnectionClosedError")


def _lane_raises(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    def boom(*a: object, **k: object) -> None:
        raise error

    monkeypatch.setattr(mask_job, "register_public_object", boom)


def test_a_database_interface_error_inside_the_lane_is_an_outage(
    db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sqlalchemy.exc import InterfaceError

    _lane_raises(monkeypatch, InterfaceError("INSERT", {}, Exception("connection already closed")))
    raw = _Raw(tmp_path / "raw")
    item = raw.add(1)
    assert mb.run_batch([item], mb.BatchConfig(), _runtime(db_session, tmp_path, raw)) == 1
    record = mb.Progress(tmp_path / "progress.jsonl").load()[item.blob_sha256]
    assert (record["status"], record["reason"]) == ("interrupted", "InterfaceError")


def test_a_requests_style_timeout_inside_the_lane_is_an_outage(
    db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from requests.exceptions import Timeout

    _lane_raises(monkeypatch, Timeout())
    raw = _Raw(tmp_path / "raw")
    item = raw.add(1)
    assert mb.run_batch([item], mb.BatchConfig(), _runtime(db_session, tmp_path, raw)) == 1
    record = mb.Progress(tmp_path / "progress.jsonl").load()[item.blob_sha256]
    assert (record["status"], record["reason"]) == ("interrupted", "Timeout")


@pytest.mark.parametrize(
    "error",
    [ConnectionResetError(), ConnectionRefusedError(), ConnectionAbortedError()],
    ids=lambda e: type(e).__name__,
)
def test_the_lane_records_only_the_class_so_its_connection_errors_are_named(
    db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    """The lane keeps the subclass name (not ConnectionError), so the name list must hold it."""
    _lane_raises(monkeypatch, error)
    raw = _Raw(tmp_path / "raw")
    item = raw.add(1)
    assert mb.run_batch([item], mb.BatchConfig(), _runtime(db_session, tmp_path, raw)) == 1
    record = mb.Progress(tmp_path / "progress.jsonl").load()[item.blob_sha256]
    assert (record["status"], record["reason"]) == ("interrupted", type(error).__name__)


# --------------------------------------------------------------------------
# a kept copy is served only when the bucket's ETag matches it
# --------------------------------------------------------------------------


def test_a_copy_the_bucket_stored_wrong_is_read_back_from_the_bucket(tmp_path: Path) -> None:
    """A PUT the bucket accepted but stored wrong: the kept bytes would hash
    right, so the lane's source_hash_mismatch could never fire. The ETag says
    otherwise, and the bucket's own bytes go to the lane's hash check."""

    class _Mangling(LocalObjectStore):
        def put(self, key: str, data: bytes, content_type: str) -> None:
            super().put(key, data[:-1], content_type)

    store = mb.RecentWrites(_Mangling(tmp_path / "media"))
    store.put("k", b"clean copy", "image/png")
    assert store.get("k") == b"clean cop"


def test_without_an_etag_the_copy_is_read_back_from_the_bucket(tmp_path: Path) -> None:
    class _Plain:
        def __init__(self) -> None:
            self.objects: dict[str, bytes] = {}
            self.gets = 0

        def get(self, key: str) -> bytes:
            self.gets += 1
            return self.objects[key]

        def put(self, key: str, data: bytes, content_type: str) -> None:
            self.objects[key] = data

    plain = _Plain()
    store = mb.RecentWrites(plain)
    store.put("k", b"x", "image/png")
    assert store.get("k") == b"x" and plain.gets == 1

    class _Unreachable(LocalObjectStore):
        gets = 0

        def get(self, key: str) -> bytes:
            self.gets += 1
            return super().get(key)

        def etag(self, key: str) -> str:
            raise ConnectionError("HEAD failed")

    bucket = _Unreachable(tmp_path / "u")
    unreachable = mb.RecentWrites(bucket)
    unreachable.put("k", b"y", "image/png")
    assert unreachable.get("k") == b"y" and bucket.gets == 1  # the bucket, which answers GET

    class _Multipart(LocalObjectStore):
        def etag(self, key: str) -> str:
            return "0" * 32 + "-2"

    multipart = mb.RecentWrites(_Multipart(tmp_path / "m"))
    multipart.put("k", b"z", "image/png")
    seen: list[str] = []
    multipart._store.get = lambda key: seen.append(key) or b"z"  # type: ignore[method-assign]
    assert multipart.get("k") == b"z" and seen == ["k"]


def test_a_put_that_raises_keeps_nothing(tmp_path: Path) -> None:
    class _Down(LocalObjectStore):
        def put(self, key: str, data: bytes, content_type: str) -> None:
            raise ConnectionError("bucket gone")

    store = mb.RecentWrites(_Down(tmp_path / "media"))
    with pytest.raises(ConnectionError):
        store.put("k", b"never stored", "image/png")
    assert store._recent == {}  # kept only once the PUT returned
    with pytest.raises(FileNotFoundError):
        store.get("k")  # nothing kept: the bucket is asked, and has nothing


def test_the_stores_etag_is_the_md5_of_the_stored_bytes(tmp_path: Path) -> None:
    from app.object_store import S3ObjectStore
    from moto import mock_aws

    md5 = hashlib.md5(b"bytes", usedforsecurity=False).hexdigest()
    local = LocalObjectStore(tmp_path)
    local.put("derived/ab/abc.png", b"bytes", "image/png")
    assert local.etag("derived/ab/abc.png") == md5
    with mock_aws():
        import boto3

        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="images")
        s3 = S3ObjectStore(client=client, bucket="images")
        s3.put("derived/ab/abc.png", b"bytes", "image/png")
        assert s3.etag("derived/ab/abc.png") == md5  # HEAD, quotes stripped


# --------------------------------------------------------------------------
# seconds per MASKED image, apart from the cheap ineligible ones
# --------------------------------------------------------------------------


def test_the_chunk_reports_seconds_per_masked_image_apart(
    db_session: Session, tmp_path: Path
) -> None:
    raw = _Raw(tmp_path / "raw")
    clock = _Clock()

    class _Timed(_Model):
        def predict_alpha(self, image: Image.Image) -> np.ndarray:
            clock.advance(3)  # the model's 3 s; an ineligible photo never reaches it
            return super().predict_alpha(image)

    events: list[dict[str, Any]] = []
    rt = _runtime(db_session, tmp_path, raw, clock=clock, backend=_Timed(), events=events)
    items = [raw.add(1, rank=0), raw.add(2, rank=1, url=PICTURE), raw.add(3, rank=2)]
    assert mb.run_batch(items, mb.BatchConfig(chunk_size=3), rt) == 0
    (chunk,) = _events(events, "chunk")
    assert chunk["counts"] == {"ok": 2, "ineligible": 1}
    assert chunk["seconds_per_image"] == 2.0  # (3 + 0 + 3) / 3
    assert chunk["masked"] == 2
    assert chunk["seconds_per_masked_image"] == 3.0

    events2: list[dict[str, Any]] = []
    rt2 = _runtime(db_session, tmp_path, raw, clock=clock, backend=_Timed(), events=events2)
    rt2.progress = mb.Progress(tmp_path / "other.jsonl")
    assert mb.run_batch([raw.add(4, url=PICTURE)], mb.BatchConfig(), rt2) == 0
    (only_ineligible,) = _events(events2, "chunk")
    assert only_ineligible["masked"] == 0
    assert only_ineligible["seconds_per_masked_image"] is None


# --------------------------------------------------------------------------
# --device cpu: no GPU gate, no VRAM sampler
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("device", "on_gpu"), [("cpu", False), ("auto", True), ("cuda", True)])
def test_only_a_gpu_run_builds_the_gpu_gate_and_the_vram_sampler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: str, on_gpu: bool
) -> None:
    seen: list[mb.Runtime] = []

    def capture(items: Any, config: mb.BatchConfig, rt: mb.Runtime) -> int:
        seen.append(rt)
        return 0

    monkeypatch.setattr(mb, "run_batch", capture)
    wl = _worklist_file(tmp_path, [])
    argv = ["run", "--worklist", str(wl), "--device", device, "--no-residential-check"]
    assert mb.main([*argv, "--dry-run"]) == 0
    (rt,) = seen
    assert (rt.gates.gpu is not None) is on_gpu
    sampler = rt.vram()
    assert isinstance(sampler, mb.VramSampler)
    assert (sampler._command is not None) is on_gpu


def test_the_own_memory_is_measured_after_the_model_has_run(
    db_session: Session, tmp_path: Path
) -> None:
    """An ineligible photo never reaches the model: the arena is not yet full
    after it, so measuring then would credit too little and pause the batch."""
    raw = _Raw(tmp_path / "raw")
    clock = _Clock()
    ran = {"since_load": False}

    class _Arena(_Model):
        def load(self) -> None:
            super().load()
            ran["since_load"] = False

        def predict_alpha(self, image: Image.Image) -> np.ndarray:
            ran["since_load"] = True
            return super().predict_alpha(image)

    model = _Arena()

    def query() -> str:
        if model.loads > model.releases:
            return HELD if ran["since_load"] else "0, 8400, 3300, 12282"  # weights only
        return IDLE

    gates = mb.Gates(
        windows=(),
        gpu=mb.GpuGate(query=query, samples=1, sleep=lambda s: None),
        residential=None,
        now=clock.now,
    )
    events: list[dict[str, Any]] = []
    rt = _runtime(db_session, tmp_path, raw, clock=clock, gates=gates, backend=model, events=events)
    items = [raw.add(1, rank=0, url=PICTURE), raw.add(2, rank=1), raw.add(3, rank=2)]
    assert mb.run_batch(items, mb.BatchConfig(chunk_size=2, poll_seconds=60), rt) == 0
    assert _events(events, "paused") == []
    assert gates.gpu is not None and gates.gpu.own_mib == OWN


# --------------------------------------------------------------------------
# MASK-PERF-1: where each image's seconds go
# --------------------------------------------------------------------------

#: The item record and the chunk event as they were before the stage timings:
#: the record gains nothing, the chunk only the two stage dicts.
_ITEM_KEYS = {
    "event", "blob_sha256", "product_id", "head_id", "rank", "batch_id", "clean_sha256",
    "mask_sha256", "reason", "status", "mask_seconds", "seconds", "chunk",
}  # fmt: skip
_CHUNK_KEYS = {
    "event", "chunk", "batch_id", "count", "counts", "ranks", "seconds_per_image",
    "median_seconds", "masked", "seconds_per_masked_image", "mask_seconds_per_image",
    "peak_vram_mib", "baseline_vram_mib", "provider", "failures", "yielded",
    "elapsed_seconds", "held_vram_mib", "throughput_seconds_per_image",
}  # fmt: skip
_STAGE_KEYS = {"stage_seconds_per_image", "stage_median_seconds", "stage_images"}


class _CostlyStore(LocalObjectStore):
    """A bucket whose GET and PUT cost fake seconds on the batch's clock."""

    def __init__(self, root: Path, clock: _Clock, *, get: float, put: float = 0.0) -> None:
        super().__init__(root)
        self._clock = clock
        self._get_cost = get
        self._put_cost = put

    def get(self, key: str) -> bytes:
        self._clock.advance(self._get_cost)
        return super().get(key)

    def put(self, key: str, data: bytes, content_type: str) -> None:
        self._clock.advance(self._put_cost)
        super().put(key, data, content_type)


class _Pass(_Model):
    """One model pass costing fake seconds: ``costs`` in call order, the last
    repeating."""

    def __init__(self, clock: _Clock, *costs: float) -> None:
        super().__init__()
        self._clock = clock
        self._costs = costs
        self.calls = 0

    def predict_alpha(self, image: Image.Image) -> np.ndarray:
        self._clock.advance(self._costs[min(self.calls, len(self._costs) - 1)])
        self.calls += 1
        return super().predict_alpha(image)


def _costly_lane(
    db: Session,
    clock: _Clock,
    monkeypatch: pytest.MonkeyPatch,
    *,
    statement: float = 0.0,
    rollback: float = 0.0,
    checkout: float = 0.0,
) -> None:
    """The lane's database and artifact costing fake seconds on ``clock``: a
    commit 0.5 s, each SQL statement ``statement`` s, a rollback ``rollback``
    s, each pool checkout ``checkout`` s (a transaction's first query: the
    pre-ping's round trip, which no SQL event spans); each appearance
    (analysis) 0.375 s; the mask encode 0.625 s."""
    real_encode, real_appearance = mask_artifact.encode_mask_png, mask_artifact._appearance

    def encode(alpha: np.ndarray) -> bytes:
        clock.advance(0.625)
        return real_encode(alpha)

    def appearance(rgba: Image.Image) -> Any:
        clock.advance(0.375)
        return real_appearance(rgba)

    monkeypatch.setattr(mask_artifact, "encode_mask_png", encode)
    monkeypatch.setattr(mask_artifact, "_appearance", appearance)
    engine = db.get_bind()
    sa_event.listen(engine, "commit", lambda conn: clock.advance(0.5))
    if statement:
        sa_event.listen(engine, "before_cursor_execute", lambda *a: clock.advance(statement))
    if checkout:
        sa_event.listen(engine, "checkout", lambda *a: clock.advance(checkout))
    if rollback:
        real_rollback = db.rollback

        def costly_rollback() -> None:
            clock.advance(rollback)
            real_rollback()

        monkeypatch.setattr(db, "rollback", costly_rollback)


def _costly_runtime(
    db: Session,
    tmp_path: Path,
    raw: _Raw,
    clock: _Clock,
    monkeypatch: pytest.MonkeyPatch,
    events: list[dict[str, Any]],
    *,
    primary: MattingBackend | None = None,
    statement: float = 0.0,
    rollback: float = 0.0,
    checkout: float = 0.0,
    clean_encode: float = 0.0,
) -> mb.Runtime:
    """The batch with every stage costing fake seconds on its clock: the raw
    GET 1 s; a media PUT 0.25 s, a media GET 0.125 s; BiRefNet (``primary``)
    2 s; the base-keep donors 0.75 + 0.5 s; the seed's clean-copy encode
    ``clean_encode`` s; the database and the artifact as in ``_costly_lane``.
    Nothing else moves the clock."""
    _costly_lane(db, clock, monkeypatch, statement=statement, rollback=rollback, checkout=checkout)
    real_encode = mask_seed._encode

    def encode(picture: Image.Image, original: bytes) -> tuple[bytes, str]:
        clock.advance(clean_encode)
        return real_encode(picture, original)

    monkeypatch.setattr(mask_seed, "_encode", encode)
    backend = BaseKeptMattingBackend(
        primary=primary or _Pass(clock, 2.0),
        donors=[_Pass(clock, 0.75), _Pass(clock, 0.5)],
    )
    rt = _runtime(db, tmp_path, raw, clock=clock, backend=backend, events=events)
    rt.raw_store = mb.ReadOnlyStore(_CostlyStore(tmp_path / "raw", clock, get=1.0))
    rt.media_store = _CostlyStore(tmp_path / "media", clock, get=0.125, put=0.25)
    return rt


def _item_records(progress: Path) -> list[dict[str, Any]]:
    lines = [json.loads(line) for line in progress.read_text().splitlines()]
    return [r for r in lines if r["event"] == "item"]


def test_the_chunk_reports_where_each_images_seconds_go(
    db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mask_seconds mixes the model with the bucket and the database: the chunk
    splits each image's seconds by stage (mean and median) and changes
    nothing else."""
    raw = _Raw(tmp_path / "raw")
    clock = _Clock()
    events: list[dict[str, Any]] = []
    rt = _costly_runtime(
        db_session, tmp_path, raw, clock, monkeypatch, events, primary=_Pass(clock, 2, 3, 7)
    )
    items = [raw.add(1, rank=0), raw.add(2, rank=1), raw.add(3, rank=2)]
    assert mb.run_batch(items, mb.BatchConfig(chunk_size=3), rt) == 0

    # unchanged: these held before the stages were timed
    minted = _item_records(tmp_path / "progress.jsonl")
    assert [set(r) for r in minted] == [_ITEM_KEYS] * 3
    assert [r["status"] for r in minted] == ["ok", "ok", "ok"]
    # 5.5 s around the model (raw 1, seed 1, settle 3.5), plus BiRefNet's 2, 3, 7
    assert [r["seconds"] for r in minted] == [7.5, 8.5, 12.5]
    assert [r["mask_seconds"] for r in minted] == [5.5, 6.5, 10.5]
    (chunk,) = _events(events, "chunk")
    assert chunk["counts"] == {"ok": 3}
    assert chunk["seconds_per_image"] == 9.5
    assert chunk["median_seconds"] == 8.5
    assert chunk["seconds_per_masked_image"] == 9.5
    assert chunk["mask_seconds_per_image"] == 7.5
    assert chunk["provider"] == "FakeExecutionProvider"

    # new: the stages, in pipeline order, the database last
    assert set(chunk) == _CHUNK_KEYS | _STAGE_KEYS
    means = {
        "fetch": 1.0,  # the raw GET (and its decode)
        "seed": 0.0,  # its own work costs nothing here (PUTs, SQL and commit are not its)
        "readback": 0.125,  # the lane re-reads the copy it hashes
        "inference": 4.0,  # BiRefNet: (2 + 3 + 7) / 3
        "base_keep": 1.25,  # the two IS-Net donors
        "analysis": 0.75,  # the appearances, unmasked and masked
        "encode": 0.625,  # the mask PNG
        "upload": 0.75,  # three PUTs: the original, its clean copy, the mask
        "db": 1.0,  # the seed's commit and the mask row's
    }
    assert chunk["stage_seconds_per_image"] == means
    assert list(chunk["stage_seconds_per_image"]) == list(means)
    assert chunk["stage_median_seconds"] == {**means, "inference": 3.0}
    assert list(chunk["stage_median_seconds"]) == list(means)
    assert chunk["stage_images"] == dict.fromkeys(means, 3)
    assert list(chunk["stage_images"]) == list(means)
    # every image ran every stage here, so the means add up to the mean image
    assert sum(means.values()) == chunk["seconds_per_image"]


def _captured(monkeypatch: pytest.MonkeyPatch) -> list[StageTimings]:
    """Every image's stage timings, in the order the batch made them."""
    made: list[StageTimings] = []

    class _Kept(StageTimings):
        def __init__(self, clock: Callable[[], float]) -> None:
            super().__init__(clock)
            made.append(self)

    monkeypatch.setattr(mb, "StageTimings", _Kept)
    return made


class _FullOn(_Pass):
    """BiRefNet answering the whole frame on its ``frame``-th call (from 0):
    the mask is held (full_frame), its PNG stored under held-masks/."""

    def __init__(self, clock: _Clock, cost: float, *, frame: int) -> None:
        super().__init__(clock, cost)
        self._frame = frame

    def predict_alpha(self, image: Image.Image) -> np.ndarray:
        alpha = super().predict_alpha(image)
        return np.full_like(alpha, 255) if self.calls - 1 == self._frame else alpha


def test_no_database_second_escapes_the_stages(
    db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every SQL statement costs time too (the batch reaches the media
    database over a tunnel): each one lands in a stage, whichever way an
    image settles: masked, held, its judgment raising, or ineligible."""
    raw = _Raw(tmp_path / "raw")
    clock = _Clock()
    events: list[dict[str, Any]] = []
    rt = _costly_runtime(
        db_session,
        tmp_path,
        raw,
        clock,
        monkeypatch,
        events,
        primary=_FullOn(clock, 2.0, frame=1),
        statement=0.0625,
        rollback=0.375,
        checkout=0.03125,
        clean_encode=0.125,
    )
    real_judgment = mask_job.judged_origins
    judged: list[str] = []

    def judgment(db: Session, sha256: str) -> Any:
        judged.append(sha256)
        if len(judged) == 3:
            raise ValueError("an origin that will not parse")
        return real_judgment(db, sha256)

    monkeypatch.setattr(mask_job, "judged_origins", judgment)
    made = _captured(monkeypatch)
    items = [raw.add(1, rank=0), raw.add(2, rank=1), raw.add(3, rank=2), raw.add(4, url=PICTURE)]
    assert mb.run_batch(items, mb.BatchConfig(), rt) == 1  # done, with a failure
    records = _item_records(tmp_path / "progress.jsonl")
    assert [(r["status"], r["reason"]) for r in records] == [
        ("ok", None),
        ("held", "full_frame"),
        ("failed", "ValueError"),
        ("ineligible", "user_photo"),
    ]
    assert any(p.parent.name == "held-masks" for p in (tmp_path / "media").rglob("*.png"))
    # every second of every image is in exactly one stage
    assert [round(sum(t.seconds.values()), 3) for t in made] == [r["seconds"] for r in records]
    # every PUT is an upload (the seed's two, the mask's held or not), nothing else is
    assert [t.seconds["upload"] for t in made] == [0.75, 0.75, 0.5, 0.5]
    # the seed keeps its own work alone (its clean copy's encode): none of its SQL
    assert [t.seconds["seed"] for t in made] == [0.125] * 4
    # the failed judgment's rollback and row (statements, commit) are the database's
    assert made[2].seconds["db"] > 0.875
    assert set(made[2].seconds) == {"fetch", "seed", "upload", "db"}
    (chunk,) = _events(events, "chunk")
    stages = chunk["stage_seconds_per_image"]
    assert stages["upload"] == 0.625
    assert stages["seed"] == 0.125
    assert stages["db"] > 1.0  # the statements and checkouts, beyond the two commits
    # each mean is over the images that ran its stage, and says how many did
    assert chunk["stage_images"] == {
        "fetch": 4, "seed": 4, "readback": 2, "inference": 2, "base_keep": 2,
        "analysis": 2, "encode": 2, "upload": 4, "db": 4,
    }  # fmt: skip
    # so the means add up to the mean image only weighted by those counts
    total = sum(mean * chunk["stage_images"][name] for name, mean in stages.items())
    assert total == pytest.approx(chunk["seconds_per_image"] * len(records), abs=0.01)
    assert sum(stages.values()) > chunk["seconds_per_image"] + 1


def test_the_seed_keeps_its_own_work_and_its_sql_is_the_databases(
    db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whichever stage runs a statement, a commit or a rollback, its seconds
    are the database's: the seed keeps its clean copy's encode alone."""
    raw = _Raw(tmp_path / "raw")
    clock = _Clock()
    events: list[dict[str, Any]] = []
    rt = _costly_runtime(
        db_session,
        tmp_path,
        raw,
        clock,
        monkeypatch,
        events,
        statement=0.0625,
        checkout=0.03125,
        clean_encode=0.125,
    )
    made = _captured(monkeypatch)
    assert mb.run_batch([raw.add(1)], mb.BatchConfig(), rt) == 0
    (record,) = _item_records(tmp_path / "progress.jsonl")
    assert record["status"] == "ok"
    (timings,) = made
    assert timings.seconds == {
        # 34 statements (the lookup, the seed's 21, the lane's 12), two commits
        # (the seed's, the mask row's) and two checkouts (the lookup's, the lane's)
        "db": 34 * 0.0625 + 2 * 0.5 + 2 * 0.03125,
        "fetch": 1.0,
        "upload": 0.75,
        "seed": 0.125,
        "readback": 0.125,
        "base_keep": 1.25,
        "inference": 2.0,
        "encode": 0.625,
        "analysis": 0.75,
    }
    assert round(sum(timings.seconds.values()), 3) == record["seconds"]  # every second, once


def test_the_batch_listens_to_its_database_only_while_it_runs(
    db_session: Session, tmp_path: Path
) -> None:
    """Done or refused, the batch leaves its session and engine with the
    listeners it found."""
    engine = db_session.get_bind()

    def listeners() -> dict[str, int]:
        on_engine = ("before_execute", "after_execute", "before_cursor_execute")
        on_session = ("before_commit", "after_rollback", "after_transaction_end")
        return {
            **{n: len(getattr(engine.dispatch, n)) for n in on_engine},
            "after_cursor_execute": len(engine.dispatch.after_cursor_execute),
            "rollback": len(engine.dispatch.rollback),
            "handle_error": len(engine.dialect.dispatch.handle_error),
            **{n: len(getattr(db_session.dispatch, n)) for n in on_session},
        }

    class _NoCuda(_Model):
        def load(self) -> None:
            raise RuntimeError("onnxruntime session loaded without CUDAExecutionProvider")

    during: list[dict[str, int]] = []

    class _Watching(_Model):
        def predict_alpha(self, image: Image.Image) -> np.ndarray:
            during.append(listeners())
            return super().predict_alpha(image)

    before = listeners()
    raw = _Raw(tmp_path / "raw")
    rt = _runtime(db_session, tmp_path, raw, backend=_Watching())
    assert mb.run_batch([raw.add(1)], mb.BatchConfig(), rt) == 0
    assert during and all(n == before[name] + 1 for name, n in during[0].items())
    assert listeners() == before
    refused = _runtime(db_session, tmp_path, raw, backend=_NoCuda(), progress=tmp_path / "2.jsonl")
    assert mb.run_batch([raw.add(2)], mb.BatchConfig(), refused) == 2
    assert listeners() == before


def test_a_stage_that_never_ran_is_omitted_not_zero(
    db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _Raw(tmp_path / "raw")
    clock = _Clock()
    events: list[dict[str, Any]] = []
    rt = _costly_runtime(db_session, tmp_path, raw, clock, monkeypatch, events)
    # an ineligible photo is judged, never read back nor run through the model
    assert mb.run_batch([raw.add(4, url=PICTURE)], mb.BatchConfig(), rt) == 0
    (chunk,) = _events(events, "chunk")
    assert chunk["counts"] == {"ineligible": 1}
    # the seed ran (its own work free here); db: its commit and the row's
    staged = {"fetch": 1.0, "seed": 0.0, "upload": 0.5, "db": 1.0}
    assert chunk["stage_seconds_per_image"] == staged
    assert chunk["stage_median_seconds"] == staged
    assert chunk["stage_images"] == dict.fromkeys(staged, 1)

    # nothing minted (already done, by the media database): no stage at all
    events.clear()
    rt.progress = mb.Progress(tmp_path / "other.jsonl")
    assert mb.run_batch([raw.add(4, url=PICTURE)], mb.BatchConfig(), rt) == 0
    (again,) = _events(events, "chunk")
    assert again["counts"] == {"already": 1}
    assert again["stage_seconds_per_image"] == {}
    assert again["stage_median_seconds"] == {}
    assert again["stage_images"] == {}


def test_a_failed_images_seconds_are_staged_up_to_its_failure(
    db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An outage's wait shows in its stage (a raw GET that times out after
    4 s); an image the model fails on keeps the stages it went through, and
    the rollback after it is a database second."""
    raw = _Raw(tmp_path / "raw")
    clock = _Clock()
    timeout_key = raw.add(1, rank=0).storage_key

    class _TimingOut(_CostlyStore):
        def get(self, key: str) -> bytes:
            if key == timeout_key:
                clock.advance(4.0)
                raise TimeoutError("read timed out")
            return super().get(key)

    class _Crash(_Pass):
        def predict_alpha(self, image: Image.Image) -> np.ndarray:
            super().predict_alpha(image)
            raise RuntimeError("CUDA error")

    events: list[dict[str, Any]] = []
    rt = _costly_runtime(
        db_session,
        tmp_path,
        raw,
        clock,
        monkeypatch,
        events,
        primary=_Crash(clock, 2.0),
        rollback=0.25,
    )
    rt.raw_store = mb.ReadOnlyStore(_TimingOut(tmp_path / "raw", clock, get=1.0))
    items = [raw.add(1, rank=0), raw.add(2, rank=1)]
    assert mb.run_batch(items, mb.BatchConfig(), rt) == 1  # done, with failures
    (chunk,) = _events(events, "chunk")
    assert chunk["counts"] == {"interrupted": 1, "failed": 1}
    assert [r["seconds"] for r in _item_records(tmp_path / "progress.jsonl")] == [4.0, 6.5]
    assert chunk["stage_seconds_per_image"] == {
        "fetch": 2.5,  # (4 + 1) / 2
        "seed": 0.0,
        "readback": 0.125,
        "inference": 2.0,  # up to the crash
        "base_keep": 1.25,  # the donors ran first
        "analysis": 0.375,  # the unmasked appearance only
        "upload": 0.5,  # the seed's two PUTs
        "db": 0.625,  # (0 + the seed's commit, the rollback and the failed row's commit) / 2
    }
    assert chunk["stage_images"] == {
        "fetch": 2, "seed": 1, "readback": 1, "inference": 1, "base_keep": 1, "analysis": 1,
        "upload": 1, "db": 2,
    }  # fmt: skip


@pytest.mark.parametrize("where", ["db", "seed", "settle"])
def test_the_rollback_after_a_failure_is_a_database_second(
    db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str
) -> None:
    """Over the tunnel a rollback is a round trip like a query: the one after
    a lookup, a seed or a lane that raised lands in "db"."""
    raw = _Raw(tmp_path / "raw")
    clock = _Clock()
    events: list[dict[str, Any]] = []
    rt = _costly_runtime(db_session, tmp_path, raw, clock, monkeypatch, events, rollback=0.375)

    def down(*a: object, **k: object) -> None:
        raise ConnectionError("server closed the connection")

    monkeypatch.setattr(
        mb,
        {"db": "original_settled", "seed": "seed_clean_image", "settle": "settle_clean_copy"}[
            where
        ],
        down,
    )
    made = _captured(monkeypatch)
    assert mb.run_batch([raw.add(1)], mb.BatchConfig(), rt) == 1
    (record,) = _item_records(tmp_path / "progress.jsonl")
    assert record["reason"] == f"{where}:ConnectionError"
    (timings,) = made
    # the rollback (the lookup costing nothing here), after a lane's the seed's commit too
    assert timings.seconds["db"] == 0.375 + (0.5 if where == "settle" else 0.0)
    assert sum(timings.seconds.values()) == record["seconds"]


def test_the_batch_row_read_again_after_a_rollback_is_a_database_second(
    db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rollback expires the batch's row: the next image's record reads its
    id back from the database, a round trip like any query."""
    from sqlalchemy.exc import OperationalError

    raw = _Raw(tmp_path / "raw")
    clock = _Clock()
    events: list[dict[str, Any]] = []
    rt = _costly_runtime(
        db_session, tmp_path, raw, clock, monkeypatch, events, statement=0.0625, checkout=0.03125
    )
    real_write = mask_job._write_row
    calls: list[str] = []

    def write_once_down(*a: Any, **k: Any) -> None:
        # the mask row's INSERT, with the lane's queries open in the transaction
        calls.append("write")
        if len(calls) == 1:
            raise OperationalError("INSERT", {}, Exception("server closed the connection"))
        real_write(*a, **k)

    monkeypatch.setattr(mask_job, "_write_row", write_once_down)
    made = _captured(monkeypatch)
    assert mb.run_batch([raw.add(1, rank=0), raw.add(2, rank=1)], mb.BatchConfig(), rt) == 1
    records = _item_records(tmp_path / "progress.jsonl")
    assert [r["status"] for r in records] == ["interrupted", "ok"]
    assert [round(sum(t.seconds.values()), 3) for t in made] == [r["seconds"] for r in records]
    # the read again is the database's, with the checkout of the transaction
    # it opens: 35 statements (the batch row, the lookup, the seed's 21, the
    # lane's 12), two commits (the seed's, the mask row's) and two checkouts
    # (this read's, the lane's); the raw GET alone is the fetch, and the seed
    # keeps none of it
    assert made[1].seconds["db"] == 35 * 0.0625 + 2 * 0.5 + 2 * 0.03125
    assert made[1].seconds["fetch"] == 1.0
    assert made[1].seconds["seed"] == 0.0


def _recaptured(raw: _Raw, n: int) -> mb.WorkItem:
    """Picture ``n`` again in other bytes (a re-save with a text chunk)."""
    from PIL import PngImagePlugin

    meta = PngImagePlugin.PngInfo()
    meta.add_text("Comment", "captured again")
    buf = io.BytesIO()
    Image.open(io.BytesIO(_photo(n))).save(buf, format="PNG", pnginfo=meta)
    again = buf.getvalue()
    sha = hashlib.sha256(again).hexdigest()
    key = f"raw-img/sha256/{sha[:2]}/{sha}.png"
    raw.store.put(key, again, "image/png")
    return mb.WorkItem(blob_sha256=sha, storage_key=key, url=ITEM.format(n=n), product_id="p2")


def test_the_lookup_after_a_copy_settled_by_another_original_is_a_database_second(
    db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lane answers None for a copy another original settled first; the
    batch then asks the database for that copy's status: those statements
    land in "db" like every other."""
    raw = _Raw(tmp_path / "raw")
    clock = _Clock()
    events: list[dict[str, Any]] = []
    rt = _costly_runtime(db_session, tmp_path, raw, clock, monkeypatch, events, statement=0.0625)
    made = _captured(monkeypatch)
    assert mb.run_batch([raw.add(1), _recaptured(raw, 1)], mb.BatchConfig(), rt) == 0
    records = _item_records(tmp_path / "progress.jsonl")
    assert [(r["status"], r.get("mask_status")) for r in records] == [
        ("ok", None),
        ("already", "ok"),
    ]
    assert [round(sum(t.seconds.values()), 3) for t in made] == [r["seconds"] for r in records]
    # the database holds 18 statements (the lookup, the seed's 15, the lane's
    # candidate query, the late lookup) and the seed's commit; the seed keeps
    # none; the original's PUT is the upload (its picture needs no clean copy)
    assert made[1].seconds == {
        "db": 18 * 0.0625 + 0.5,
        "fetch": 1.0,
        "upload": 0.25,
        "seed": 0.0,
    }


def test_the_lanes_bucket_stages_each_put_as_an_upload_and_each_get_as_a_readback(
    tmp_path: Path,
) -> None:
    """The media bucket as one image's lane sees it: a PUT is an upload and a
    GET a readback, inside whichever stage made it (the seed's included),
    and one that raises keeps its seconds and its error."""
    clock = _Clock()
    timings = StageTimings(clock.monotonic)
    media = mb._Witness(_CostlyStore(tmp_path / "ok", clock, get=0.125, put=0.25), timings)
    with timings.stage("seed"):
        clock.advance(1.0)
        media.put("k", b"bytes", "image/png")
    assert media.get("k") == b"bytes"
    assert timings.seconds == {"upload": 0.25, "seed": 1.0, "readback": 0.125}

    class _Down(LocalObjectStore):
        def get(self, key: str) -> bytes:
            clock.advance(2.0)
            raise TimeoutError("read timed out")

        def put(self, key: str, data: bytes, content_type: str) -> None:
            clock.advance(4.0)
            raise ConnectionError("connection reset")

    down = mb._Witness(_Down(tmp_path / "down"), timings)
    with pytest.raises(ConnectionError):
        down.put("k", b"bytes", "image/png")
    assert isinstance(down.error, ConnectionError)
    with pytest.raises(TimeoutError):
        down.get("k")
    assert isinstance(down.error, TimeoutError)
    assert timings.seconds == {"upload": 4.25, "seed": 1.0, "readback": 2.125}


@pytest.mark.parametrize(
    ("answer", "status", "staged"),
    [
        ("mask", "ok", {"readback": 0.125, "analysis": 0.75, "inference": 2.0, "encode": 0.625}),
        ("full", "held", {"readback": 0.125, "analysis": 0.75, "inference": 2.0, "encode": 0.625}),
        ("crash", "failed", {"readback": 0.125, "analysis": 0.375, "inference": 2.0}),
        ("judgment", "failed", {}),
    ],
)
def test_settle_clean_copy_stages_its_own_seconds(
    db_session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    answer: str,
    status: str,
    staged: dict[str, float],
) -> None:
    """On its own, over a plain bucket (no batch around it), the lane still
    puts each second in a stage: the copy read back, the model, the mask's
    PUT (held or not), its rows, and the rollback after a failure."""
    clock = _Clock()
    store = _CostlyStore(tmp_path / "media", clock, get=0.125, put=0.25)
    obj = _seed(db_session, store, 7)
    batch = mask_job.open_mask_batch(db_session, backend_name="stub-model")
    _costly_lane(db_session, clock, monkeypatch, rollback=0.375, checkout=0.03125)

    class _Primary(_Pass):
        def predict_alpha(self, image: Image.Image) -> np.ndarray:
            alpha = super().predict_alpha(image)
            if answer == "crash":
                raise RuntimeError("CUDA error")
            return np.full_like(alpha, 255) if answer == "full" else alpha

    def judgment(db: Session, sha256: str) -> Any:
        raise ValueError("an origin that will not parse")

    if answer == "judgment":
        monkeypatch.setattr(mask_job, "judged_origins", judgment)
    timings = StageTimings(clock.monotonic)
    start = clock.monotonic()
    report = mask_job.settle_clean_copy(
        db_session, store, obj.sha256, backend=_Primary(clock, 2.0), batch=batch, timings=timings
    )
    assert report is not None and report.status == status
    stored = status in ("ok", "held")
    # the row's commit and the checkout of the lane's transaction; after a
    # failure its rollback, and the checkout of the transaction the row opens
    db = 0.5 + 0.03125 if stored else 0.375 + 0.5 + 2 * 0.03125
    assert timings.seconds == {**staged, **({"upload": 0.25} if stored else {}), "db": db}
    assert sum(timings.seconds.values()) == clock.monotonic() - start


def test_timing_a_settle_changes_no_result(db_session: Session, tmp_path: Path) -> None:
    """The same clean copy seeded and settled with and without the stage
    timings and the database's listeners: the same report, the same row,
    the same mask bytes."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    other = sessionmaker(bind=engine, expire_on_commit=False)()
    clock = _Clock()
    sa_event.listen(engine, "before_cursor_execute", lambda *a: clock.advance(0.0625))
    start = clock.monotonic()
    timed_run = StageTimings(clock.monotonic)
    outcomes = []
    try:
        for db, root, timings in (
            (db_session, tmp_path / "plain", None),
            (other, tmp_path / "timed", timed_run),
        ):
            sql = None if timings is None else DatabaseSeconds(db)
            try:
                with nullcontext() if sql is None else sql.charging(timed_run):
                    store = LocalObjectStore(root)
                    obj = _seed(db, store, 7)
                    batch = mask_job.open_mask_batch(db, backend_name="stub-model")
                    report = mask_job.settle_clean_copy(
                        db, store, obj.sha256, backend=_Model(), batch=batch, timings=timings
                    )
            finally:
                if sql is not None:
                    sql.close()
            elapsed = clock.monotonic() - start  # only the timed run's statements cost
            row = db.execute(select(ImageMask)).scalar_one()
            columns = {
                c.key: getattr(row, c.key)
                for c in ImageMask.__table__.columns
                if c.key not in ("id", "batch_id", "created_at", "updated_at")
            }
            files = {
                p.relative_to(root).as_posix(): p.read_bytes()
                for p in sorted(root.rglob("*"))
                if p.is_file()
            }
            outcomes.append((report, columns, files))
        assert outcomes[0] == outcomes[1]
        assert outcomes[0][0] is not None and outcomes[0][0].status == "ok"
        assert any(k.endswith(".png") for k in outcomes[0][2])  # the mask, among the files
        # the listeners did time it: every statement, the seed's (no stage) too
        assert sum(timed_run.seconds.values()) == elapsed > 0
    finally:
        other.close()
        engine.dispose()


# --------------------------------------------------------------------------
# MASK-PERF-2: the pipeline's options, the throughput, the recent writes
# --------------------------------------------------------------------------


def test_the_pipeline_is_on_by_default_on_the_command_line() -> None:
    args = mb._parser().parse_args(["run", "--worklist", "w"])
    assert (args.writers, args.prefetch) == (mb.DEFAULT_WRITERS, mb.DEFAULT_PREFETCH)
    assert args.writers == 4  # the live chunk tunes it with --writers
    assert args.prefetch == 2  # each original read ahead is decoded once more: memory
    assert mb.BatchConfig().writers == 0  # a library caller runs one image at a time


def test_main_passes_the_pipeline_options_and_keeps_writes_for_every_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def run_batch(items: list[mb.WorkItem], config: mb.BatchConfig, rt: mb.Runtime) -> int:
        seen.update(config=config, store=rt.media_store)
        return 0

    monkeypatch.setattr(mb, "run_batch", run_batch)
    wl = _worklist_file(tmp_path, [])
    argv = ["--sandbox", str(tmp_path / "sb"), "run", "--worklist", str(wl), "--raw-dir"]
    argv += [str(tmp_path / "raw"), "--device", "cpu", "--no-residential-check"]
    assert mb.main([*argv, "--writers", "3", "--prefetch", "2"]) == 0
    assert (seen["config"].writers, seen["config"].prefetch) == (3, 2)
    assert seen["store"].keep == 4 + 3 * 3  # each worker's original, copy and mask
    assert mb.main([*argv, "--writers", "0"]) == 0
    assert seen["store"].keep == 4


def test_main_runs_the_pipeline_on_a_pool_that_never_lends_past_what_it_keeps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A connection a QueuePool lends past its size is closed when it comes
    back, and any close kills the live run's kubectl port-forward: the
    command line runs the pipeline on the same database through a pool that
    keeps a connection for every worker and the batch, and lends no more."""
    seen: list[Session] = []

    def run_batch(items: list[mb.WorkItem], config: mb.BatchConfig, rt: mb.Runtime) -> int:
        seen.append(rt.open_db())
        return 0

    monkeypatch.setattr(mb, "run_batch", run_batch)
    wl = _worklist_file(tmp_path, [])
    sandbox = tmp_path / "sb"
    argv = ["--sandbox", str(sandbox), "run", "--worklist", str(wl), "--raw-dir"]
    argv += [str(tmp_path / "raw"), "--device", "cpu", "--no-residential-check"]
    assert mb.main([*argv, "--writers", "9"]) == 0
    assert mb.main([*argv, "--writers", "0"]) == 0
    piped, alone = (db.get_bind().engine for db in seen)
    assert isinstance(piped.pool, QueuePool)
    assert (piped.pool.size(), piped.pool._max_overflow) == (10, 0)
    assert piped.url.database == alone.url.database == str(sandbox / "masks.db")
    assert seen[0].autoflush and not seen[0].expire_on_commit  # the sandbox's session, as was
    assert alone.pool._max_overflow == 10  # one image at a time: the session as opened


def test_a_sized_pool_never_recycles_whatever_the_opened_one_does(tmp_path: Path) -> None:
    """A recycle closes a pooled connection, which the pipelined run must not
    see; the opened pool's own recycle (app.db's 1800 s) is not carried over."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'm.db'}", pool_size=10, max_overflow=20, pool_recycle=1800
    )
    sized = mb._sized(Session(engine), 4)
    pool = sized.get_bind().engine.pool
    assert engine.pool._recycle == 1800
    assert pool._recycle == -1
    sized.close()
    sized.get_bind().engine.dispose()
    engine.dispose()


def test_a_sized_session_keeps_its_database_and_session_settings(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'm.db'}", pool_size=10, max_overflow=20)
    db = Session(engine, autoflush=False, expire_on_commit=False)
    sized = mb._sized(db, 4)
    pool = sized.get_bind().engine.pool
    assert isinstance(pool, QueuePool)
    assert (pool.size(), pool._max_overflow) == (10, 0)  # never below what the pool kept
    assert pool._recycle == -1  # a recycle closes its connection: pre-ping replaces a dead one
    assert (sized.autoflush, sized.expire_on_commit) == (False, False)
    assert sized.get_bind().engine.url == engine.url
    sized.close()
    sized.get_bind().engine.dispose()
    one = Session(create_engine("sqlite://", poolclass=StaticPool))
    assert mb._sized(one, 4) is one  # not a QueuePool: _pipeline_refusal judges it
    assert mb._sized(db, 0) is db
    engine.dispose()


def test_the_chunk_reports_its_throughput(db_session: Session, tmp_path: Path) -> None:
    """Wall seconds of the chunk over its images: what a pipelined chunk
    gains shows here, not in the per-image seconds."""
    raw = _Raw(tmp_path / "raw")
    clock = _Clock()

    class _Timed(_Model):
        def predict_alpha(self, image: Image.Image) -> np.ndarray:
            clock.advance(3)
            return super().predict_alpha(image)

    events: list[dict[str, Any]] = []
    rt = _runtime(db_session, tmp_path, raw, clock=clock, backend=_Timed(), events=events)
    assert mb.run_batch([raw.add(1), raw.add(2)], mb.BatchConfig(chunk_size=2), rt) == 0
    (chunk,) = _events(events, "chunk")
    assert chunk["elapsed_seconds"] == 6.0
    assert chunk["throughput_seconds_per_image"] == 3.0


def test_the_chunks_throughput_leaves_out_its_already_images(
    db_session: Session, tmp_path: Path
) -> None:
    """An 'already' image is one lookup: the throughput is the chunk's wall
    seconds over the images it minted, as ``seconds_per_image`` is their
    mean; a chunk that minted none has none."""
    raw = _Raw(tmp_path / "raw")
    clock = _Clock()

    class _Timed(_Model):
        def predict_alpha(self, image: Image.Image) -> np.ndarray:
            clock.advance(3)
            return super().predict_alpha(image)

    first = raw.add(1)
    events: list[dict[str, Any]] = []
    rt = _runtime(db_session, tmp_path, raw, clock=clock, backend=_Timed(), events=events)
    assert mb.run_batch([first, raw.add(2), first], mb.BatchConfig(chunk_size=3), rt) == 0
    (chunk,) = _events(events, "chunk")
    assert chunk["counts"] == {"ok": 2, "already": 1}
    assert (chunk["elapsed_seconds"], chunk["throughput_seconds_per_image"]) == (6.0, 3.0)
    events.clear()
    again = tmp_path / "again.jsonl"  # a fresh record: every image is found settled
    rt = _runtime(db_session, tmp_path, raw, clock=clock, progress=again, events=events)
    assert mb.run_batch([first], mb.BatchConfig(chunk_size=1), rt) == 0
    (chunk,) = _events(events, "chunk")
    assert chunk["counts"] == {"already": 1}
    assert chunk["throughput_seconds_per_image"] is None


def test_the_buckets_keep_a_connection_for_every_worker_and_read_ahead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """botocore keeps 10 connections and closes any it lent past them when it
    comes back: the batch's clients keep one for every PUT a worker has in
    flight and every original read ahead."""
    for name, value in (
        ("RAW_S3_ENDPOINT_URL", "http://127.0.0.1:9"),
        ("RAW_S3_BUCKET", "raw"),
        ("RAW_S3_ACCESS_KEY", "k"),
        ("RAW_S3_SECRET_KEY", "s"),
    ):
        monkeypatch.setenv(name, value)
    seen: list[tuple[int, int]] = []

    def run_batch(items: list[mb.WorkItem], config: mb.BatchConfig, rt: mb.Runtime) -> int:
        raw, media = rt.raw_store, rt.media_store
        assert isinstance(raw, mb.ReadOnlyStore) and isinstance(media, mb.RecentWrites)
        pools = (raw._store, media._store)
        seen.append(tuple(s._client.meta.config.max_pool_connections for s in pools))  # type: ignore[arg-type]
        return 0

    monkeypatch.setattr(mb, "run_batch", run_batch)
    wl = _worklist_file(tmp_path, [])
    argv = ["run", "--worklist", str(wl), "--device", "cpu", "--no-residential-check"]
    assert mb.main([*argv, "--writers", "12", "--prefetch", "6"]) == 0
    assert mb.main([*argv, "--writers", "0"]) == 0
    # an image's original and copy PUT at once (MASK-PERF-3), never fewer than botocore's own 10
    assert seen == [(2 * 12 + 6,) * 2, (10, 10)]
    assert mb.bucket_connections(16, 4) == 36


def test_the_batchs_bucket_clients_bound_a_stalled_request_and_retry_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MASK-PERF-3: a request that stalls (a connection the home router
    dropped, a hung PUT) is cut and retried, not waited on for botocore's
    60 s twice over; idle connections are kept alive between chunks. The
    API's own client keeps botocore's defaults."""
    from app.s3 import s3_client

    for name, value in (
        ("RAW_S3_ENDPOINT_URL", "http://127.0.0.1:9"),
        ("RAW_S3_BUCKET", "raw"),
        ("RAW_S3_ACCESS_KEY", "k"),
        ("RAW_S3_SECRET_KEY", "s"),
    ):
        monkeypatch.setenv(name, value)
    args = mb._parser().parse_args(["run", "--worklist", "w"])
    raw = mb._raw_store(args, 12)
    media = mb._media_store(12)
    for store in (raw._store, media):  # type: ignore[attr-defined]
        config = store._client.meta.config
        assert config.max_pool_connections == 12
        assert (config.connect_timeout, config.read_timeout) == (10, 30)
        assert config.retries == {"mode": "standard", "total_max_attempts": 4}
        assert config.tcp_keepalive is True
    api = s3_client().meta.config
    assert (api.connect_timeout, api.read_timeout, api.max_pool_connections) == (60, 60, 10)
    assert not api.tcp_keepalive


def test_recent_writes_are_safe_from_many_workers() -> None:
    """Workers evicting at once must not evict one copy twice (a KeyError
    in a PUT that succeeded) nor grow past the bound."""
    import threading

    class _Memory:
        def __init__(self) -> None:
            self.objects: dict[str, bytes] = {}

        def put(self, key: str, data: bytes, content_type: str) -> None:
            self.objects[key] = data

        def get(self, key: str) -> bytes:
            return self.objects[key]

    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # switch threads as often as the interpreter can
    try:
        store = mb.RecentWrites(_Memory(), keep=4)  # type: ignore[arg-type]
        errors: list[BaseException] = []

        def work(n: int) -> None:
            try:
                for i in range(3000):
                    store.put(f"k{n}-{i}", b"x", "image/png")
            except BaseException as exc:  # the failure itself
                errors.append(exc)

        workers = [threading.Thread(target=work, args=(n,)) for n in range(8)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
    finally:
        sys.setswitchinterval(interval)
    assert errors == []
    assert len(store._recent) == 4


# --------------------------------------------------------------------------
# MM-CAPS-1: the explicit limits on an original (a decompression-bomb gate)
# --------------------------------------------------------------------------


def _png_claiming(width: int, height: int) -> bytes:
    """A PNG whose header claims width x height and holds no pixels: a decompression bomb to a header read."""
    import struct
    import zlib

    def chunk(kind: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(b""))
        + chunk(b"IEND", b"")
    )


def test_decode_stops_a_gzip_bomb_and_an_oversize_original(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.workers.image_limits import ImageLimitError

    monkeypatch.setenv("MM_MAX_IMAGE_BYTES", "100000")
    ok = _photo(1)
    assert mb._decode(gzip.compress(ok), "gzip") == ok
    assert mb._decode(ok, "identity") == ok
    with pytest.raises(ImageLimitError) as bomb:
        mb._decode(gzip.compress(b"\x00" * 5_000_000), "gzip")  # ~5 KB packed, 5 MB expanded
    assert bomb.value.code == "gzip_too_large"
    with pytest.raises(ImageLimitError) as big:
        mb._decode(b"\x00" * 100_001, "identity")
    assert big.value.code == "too_large_bytes"


def test_a_bomb_original_is_one_images_refusal_and_the_rest_carry_on(
    db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MM_MAX_IMAGE_BYTES", "100000")
    monkeypatch.setenv("MM_MAX_IMAGE_PIXELS", "2400")  # the 60 x 40 test photos are exactly 2400
    raw = _Raw(tmp_path / "raw")
    good = raw.add(1, rank=0)
    zipbomb = gzip.compress(b"\x00" * 5_000_000)
    zsha = hashlib.sha256(zipbomb).hexdigest()
    zkey = f"raw-img/sha256/{zsha[:2]}/{zsha}.png"
    raw.store.put(zkey, zipbomb, "image/png")
    bomb_zip = mb.WorkItem(
        blob_sha256=zsha,
        storage_key=zkey,
        url=ITEM.format(n=2),
        product_id="p2",
        rank=0,
        codec="gzip",
    )
    wide = _png_claiming(3000, 3000)
    wsha = hashlib.sha256(wide).hexdigest()
    wkey = f"raw-img/sha256/{wsha[:2]}/{wsha}.png"
    raw.store.put(wkey, wide, "image/png")
    bomb_px = mb.WorkItem(
        blob_sha256=wsha, storage_key=wkey, url=ITEM.format(n=3), product_id="p3", rank=0
    )
    events: list[dict[str, Any]] = []
    rt = _runtime(db_session, tmp_path, raw, events=events)
    assert mb.run_batch([bomb_zip, bomb_px, good], mb.BatchConfig(chunk_size=10), rt) == 1
    (chunk,) = _events(events, "chunk")
    assert chunk["counts"]["ok"] == 1
    reasons = {f["blob_sha256"]: f["reason"] for f in chunk["failures"]}
    assert reasons == {zsha: "refused:gzip_too_large", wsha: "refused:too_many_pixels"}
    # an image's OWN fault: failed (a plain rerun does not retry it), never interrupted; nothing was seeded for either
    assert chunk["counts"]["failed"] == 2 and chunk["counts"].get("interrupted", 0) == 0
    assert mask_job.original_settled(db_session, zsha) is None
    assert mask_job.original_settled(db_session, wsha) is None


def test_seed_clean_image_refuses_a_bomb_before_it_stores_or_decodes_anything(
    db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.workers.image_limits import ImageLimitError

    monkeypatch.setenv("MM_MAX_IMAGE_PIXELS", "2400")
    store = LocalObjectStore(tmp_path / "media")
    puts: list[str] = []
    real_put = store.put
    monkeypatch.setattr(
        store, "put", lambda key, data, ctype: (puts.append(key), real_put(key, data, ctype))[1]
    )
    with pytest.raises(ImageLimitError) as err:
        seed_clean_image(
            db_session,
            store,
            _png_claiming(3000, 3000),
            origins=[ImageOrigin(url=ITEM.format(n=9), source="mfc")],
            figure_id="f",
        )
    assert err.value.code == "too_many_pixels"
    assert puts == []
    assert mask_seed.make_clean_copy(_photo(4))[2:] == (
        40,
        60,
    )  # an ordinary photo still goes through
    with pytest.raises(ImageLimitError):
        mask_seed.make_clean_copy(_png_claiming(3000, 3000))
    with pytest.raises(ImageLimitError):
        mask_seed.pixel_digest(_png_claiming(3000, 3000))


# --------------------------------------------------------------------------
# SHOW1: a concurrent first seed (the cluster display-copy lane)
# --------------------------------------------------------------------------


def test_a_seed_that_loses_a_concurrent_first_seed_is_retried_and_lands(
    db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The display-copy lane may insert the same picture's first rows a moment
    earlier: the batch's seed rolls back and seeds again, finding them."""
    from sqlalchemy.exc import IntegrityError

    real = mb.seed_clean_image
    calls: list[int] = []

    def raced(*a: Any, **k: Any) -> Any:
        calls.append(1)
        if len(calls) == 1:
            raise IntegrityError(
                "INSERT", {}, Exception("duplicate key value violates unique constraint")
            )
        return real(*a, **k)

    monkeypatch.setattr(mb, "seed_clean_image", raced)
    raw = _Raw(tmp_path / "raw")
    item = raw.add(1)
    assert mb.run_batch([item], mb.BatchConfig(), _runtime(db_session, tmp_path, raw)) == 0
    record = mb.Progress(tmp_path / "progress.jsonl").load()[item.blob_sha256]
    assert record["status"] == "ok" and len(calls) == 2


def test_a_seed_that_keeps_losing_the_race_is_interrupted_never_failed(
    db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An IntegrityError is a race, not the image's fault: a plain rerun takes it."""
    from sqlalchemy.exc import IntegrityError

    def always(*a: Any, **k: Any) -> Any:
        raise IntegrityError(
            "INSERT", {}, Exception("duplicate key value violates unique constraint")
        )

    monkeypatch.setattr(mb, "seed_clean_image", always)
    raw = _Raw(tmp_path / "raw")
    item = raw.add(1)
    assert mb.run_batch([item], mb.BatchConfig(), _runtime(db_session, tmp_path, raw)) == 1
    record = mb.Progress(tmp_path / "progress.jsonl").load()[item.blob_sha256]
    assert (record["status"], record["reason"]) == ("interrupted", "seed:IntegrityError")


# --------------------------------------------------------------------------
# SHOW1 step 3: the daily store work list
# --------------------------------------------------------------------------


def test_the_stores_worklist_covers_every_site_since_an_instant_newest_first() -> None:
    sql = mb.stores_worklist_sql("2026-10-23T21:21Z")
    assert sql.startswith("BEGIN TRANSACTION READ ONLY;") and sql.endswith("ROLLBACK;\n")
    assert "s.site = 'mfc'" not in sql  # every site: the catalog list is MFC's
    assert "pi.first_seen_at >= '2026-10-23T21:21:00+00:00'::timestamptz" in sql
    assert "s.derive_permitted IN ('silent', 'granted')" in sql  # showable by default
    assert "'rank', 8" in sql  # ahead of every rank-9 list, whatever the file order
    assert "JOIN raw.url u ON u.id = c.url_id" in sql  # never an originless row (F3)
    assert "pi.state <> 'suppressed' AND pi.suppressed_at IS NULL" in sql
    assert "ORDER BY pi.first_seen_at DESC" in sql


def test_the_stores_worklist_takes_a_site_list_a_scope_and_a_rank() -> None:
    sql = mb.stores_worklist_sql(
        "2026-10-23T21:21:00+00:00", scope="all", sites="amiami,hlj,hobby-genki", rank=9
    )
    assert "derive_permitted" not in sql
    assert "AND s.site IN ('amiami', 'hlj', 'hobby-genki')" in sql
    assert "'rank', 9" in sql


@pytest.mark.parametrize(
    ("kwargs", "why"),
    [
        ({"since": "yesterday"}, "since"),
        ({"since": "2026-10-23T21:21"}, "time zone"),
        ({"since": "2026-10-23T21:21Z", "scope": "some"}, "scope"),
        ({"since": "2026-10-23T21:21Z", "sites": "amiami,x'; DROP"}, "site"),
        ({"since": "2026-10-23T21:21Z", "sites": ""}, "site"),
        ({"since": "2026-10-23T21:21Z", "rank": 10}, "rank"),
    ],
)
def test_the_stores_worklist_refuses_bad_input(kwargs: dict[str, Any], why: str) -> None:
    with pytest.raises(ValueError, match=why):
        mb.stores_worklist_sql(**kwargs)


def test_the_mfc_lists_sql_is_unchanged_by_the_split() -> None:
    """The catalog and gallery lists keep their exact text (the live batch's
    lists were built with it)."""
    assert mb._IMAGE_JOINS.startswith("JOIN source s ON s.id = pi.source_id\n")
    assert "WHERE s.site = 'mfc'\n  AND pi.state <> 'suppressed'" in mb._IMAGE_JOINS
    assert "s.site" not in mb._ANY_SITE_JOINS
    assert mb._ANY_SITE_JOINS.endswith("AND b.visibility = 'visible'")


def test_main_prints_the_stores_worklist(capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["sql", "stores-worklist", "--since", "2026-10-23T21:21Z", "--sites", "amiami"]
    assert mb.main(argv) == 0
    printed = capsys.readouterr().out
    assert "s.site IN ('amiami')" in printed and "'rank', 8" in printed
    assert mb.main(["sql", "stores-worklist", "--since", "nope"]) == 2


# --------------------------------------------------------------------------
# MASK5: why the circuit opened (exit 3), on its own line
# --------------------------------------------------------------------------


def test_the_circuit_open_line_names_the_outage_its_connection_and_the_last_success(
    db_session: Session, tmp_path: Path
) -> None:
    """Ross 11-03: the batch exits 3 on an outage several times a day and
    nothing recorded why. The circuit_open line, written before the exit,
    carries the streak's first outage (its exception's class and message,
    the connection it broke on, the stage and the image), every outage of
    the streak by connection and class, and the last image that got
    through: its chunk, when, and which."""
    raw = _Raw(tmp_path / "raw")
    bucket = _Bucket(tmp_path / "raw")
    raw.store = bucket
    items = [raw.add(n) for n in range(1, 6)]
    bucket.down = {i.storage_key for i in items[1:]}
    events: list[dict[str, Any]] = []
    rt = _runtime(db_session, tmp_path, raw, events=events)
    config = mb.BatchConfig(chunk_size=2, max_consecutive_failures=3)
    assert mb.run_batch(items, config, rt) == 3
    (line,) = _events(events, "circuit_open")
    assert line == {
        "event": "circuit_open",
        "consecutive_failures": 3,
        "interrupted": 3,
        "cause_exception": "ConnectionError",
        "cause_message": "connection dropped",
        "cause_connection": "raw_bucket",
        "cause_stage": "fetch",
        "cause_blob_sha256": items[1].blob_sha256,
        "cause_at": "2026-10-14T15:30:00+00:00",
        "streak_causes": {"raw_bucket:ConnectionError": 3},
        "last_success_chunk": 1,
        "last_success_blob_sha256": items[0].blob_sha256,
        "last_success_at": "2026-10-14T15:30:00+00:00",
    }
    assert events[-1] is line  # the run's last line: it exits 3 right after
    # the progress records are as ever: the cause rides on the line alone
    progress = mb.Progress(tmp_path / "progress.jsonl").load()
    assert all("cause" not in record for record in progress.values())


def test_a_database_outage_is_named_the_database_and_the_first_cause_is_kept(
    db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The streak's first outage is its cause (the database dropped), the
    others are counted by connection and class; no image got through."""
    from sqlalchemy.exc import OperationalError

    raw = _Raw(tmp_path / "raw")
    bucket = _Bucket(tmp_path / "raw")
    raw.store = bucket
    items = [raw.add(n) for n in range(1, 4)]
    bucket.down = {i.storage_key for i in items[1:]}
    real = mb.original_settled

    def tunnel_gone(db: Session, sha: str, **kw: Any) -> str | None:
        if sha == items[0].blob_sha256:
            raise OperationalError(
                "SELECT", {}, Exception("server closed the connection unexpectedly")
            )
        return real(db, sha, **kw)

    monkeypatch.setattr(mb, "original_settled", tunnel_gone)
    events: list[dict[str, Any]] = []
    rt = _runtime(db_session, tmp_path, raw, events=events)
    assert mb.run_batch(items, mb.BatchConfig(max_consecutive_failures=3), rt) == 3
    (line,) = _events(events, "circuit_open")
    assert (line["cause_exception"], line["cause_connection"], line["cause_stage"]) == (
        "OperationalError",
        "database",
        "db",
    )
    assert "server closed the connection unexpectedly" in line["cause_message"]
    assert line["streak_causes"] == {
        "database:OperationalError": 1,
        "raw_bucket:ConnectionError": 2,
    }
    assert (line["last_success_chunk"], line["last_success_at"]) == (None, None)
    assert line["last_success_blob_sha256"] is None


def test_a_media_bucket_outage_the_lane_caught_is_named_the_media_bucket(
    db_session: Session, tmp_path: Path
) -> None:
    """The lane keeps only the class; the bucket's own error, which the
    driver saw, gives the message and the connection."""
    raw = _Raw(tmp_path / "raw")
    media = _Bucket5xx(tmp_path / "media")
    media.reads_too = True  # the lane reads the seeded copy back: 503
    events: list[dict[str, Any]] = []
    rt = _runtime(db_session, tmp_path, raw, events=events)
    rt.media_store = media
    items = [raw.add(n) for n in range(2)]
    assert mb.run_batch(items, mb.BatchConfig(max_consecutive_failures=2), rt) == 3
    (line,) = _events(events, "circuit_open")
    assert (line["cause_exception"], line["cause_connection"], line["cause_stage"]) == (
        "ClientError",
        "media_bucket",
        "settle",
    )
    assert "ServiceUnavailable" in line["cause_message"]
    assert line["streak_causes"] == {"media_bucket:ClientError": 2}


def test_a_put_that_failed_in_the_seed_is_named_the_media_bucket(
    db_session: Session, tmp_path: Path
) -> None:
    raw = _Raw(tmp_path / "raw")
    media = _Bucket5xx(tmp_path / "media")
    media.down = True  # every PUT: the seed's original first
    events: list[dict[str, Any]] = []
    rt = _runtime(db_session, tmp_path, raw, events=events)
    rt.media_store = media
    items = [raw.add(n) for n in range(2)]
    assert mb.run_batch(items, mb.BatchConfig(max_consecutive_failures=2), rt) == 3
    (line,) = _events(events, "circuit_open")
    assert (line["cause_exception"], line["cause_connection"], line["cause_stage"]) == (
        "ClientError",
        "media_bucket",
        "seed",
    )


def test_a_database_blip_the_lane_caught_is_named_by_its_class_alone(
    db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lane recorded only the class name and no bucket error was seen:
    the class still names the connection; there is no message to give."""
    from sqlalchemy.exc import OperationalError

    def blip(*a: object, **k: object) -> None:
        raise OperationalError("INSERT", {}, Exception("server closed the connection"))

    monkeypatch.setattr(mask_job, "register_public_object", blip)
    raw = _Raw(tmp_path / "raw")
    events: list[dict[str, Any]] = []
    rt = _runtime(db_session, tmp_path, raw, events=events)
    items = [raw.add(n) for n in range(2)]
    assert mb.run_batch(items, mb.BatchConfig(max_consecutive_failures=2), rt) == 3
    (line,) = _events(events, "circuit_open")
    assert (line["cause_exception"], line["cause_connection"], line["cause_stage"]) == (
        "OperationalError",
        "database",
        "settle",
    )
    assert line["cause_message"] is None


@pytest.mark.parametrize(
    ("stage", "exc", "connection"),
    [
        ("db", ConnectionResetError("reset"), "database"),
        ("fetch", TimeoutError("read timed out"), "raw_bucket"),
        ("seed", TimeoutError("write timed out"), "media_bucket"),
        ("settle", ConnectionError("gone"), "media_bucket"),
        ("seed", None, "unknown"),
    ],
)
def test_an_outage_is_named_by_its_class_first_then_by_its_stage(
    stage: str, exc: BaseException | None, connection: str
) -> None:
    assert mb._connection(stage, exc, None) == connection


def test_a_lane_reason_names_the_database_by_its_class_or_nothing() -> None:
    assert mb._connection("settle", None, "InterfaceError") == "database"
    assert mb._connection("settle", None, "ReadTimeoutError") == "unknown"


def test_the_causes_message_is_one_bounded_line() -> None:
    long = "connection to server at 10.0.0.1, port 5432 failed:\n" + "x" * 2000
    cause = mb._outage_cause(
        "db", ConnectionError(long), None, now=dt.datetime(2026, 11, 3, tzinfo=dt.UTC)
    )
    assert "\n" not in cause["message"]
    assert len(cause["message"]) == mb.CAUSE_MESSAGE_MAX
    assert cause["message"].startswith("connection to server at 10.0.0.1, port 5432 failed: x")


def test_main_writes_why_the_circuit_opened_before_it_exits_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """On the run's own lines: app.circuit_open names the cause, and job.end
    (exit 3) carries its scalars too."""
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(1, rank=0), raw.add(2, rank=0)]
    bucket = _Bucket(tmp_path / "raw")
    bucket.down = {i.storage_key for i in items}
    wl = _worklist_file(tmp_path, items)
    monkeypatch.setattr(mb, "build_backend", lambda args: _Model())
    monkeypatch.setattr(mb, "_raw_store", lambda args, connections=10: mb.ReadOnlyStore(bucket))
    argv = ["--sandbox", str(tmp_path / "sandbox"), "run", "--worklist", str(wl)]
    argv += ["--progress", str(tmp_path / "p.jsonl"), "--device", "cpu", "--no-residential-check"]
    argv += ["--busy-windows", "", "--max-consecutive-failures", "2", "--writers", "0"]
    assert mb.main(argv) == 3
    out = _stdout_events(capsys)
    line = next(e for e in out if e["event"] == "circuit_open")
    assert (line["cause_exception"], line["cause_connection"]) == ("ConnectionError", "raw_bucket")
    assert line["cause_message"] == "connection dropped"
    end = out[-1]
    assert (end["event"], end["exit"]) == ("job.end", 3)
    assert (end["cause_exception"], end["cause_connection"], end["cause_stage"]) == (
        "ConnectionError",
        "raw_bucket",
        "fetch",
    )


# --------------------------------------------------------------------------
# MASK5: --cpu-workers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "why"),
    [("-1", "must be 0 or more"), ("17", "must be 16 or less"), ("many", "not a whole number")],
)
def test_a_bad_cpu_workers_is_refused_by_the_parser(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], value: str, why: str
) -> None:
    wl = _worklist_file(tmp_path, [])
    argv = ["run", "--worklist", str(wl), "--no-residential-check", "--dry-run"]
    with pytest.raises(SystemExit) as exc:
        mb.main([*argv, "--cpu-workers", value])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "argument --cpu-workers" in err and why in err


def test_main_runs_the_cpu_stages_on_the_workers_it_is_given_or_sized_to_the_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made: list[int] = []

    def run_batch(items: list[mb.WorkItem], config: mb.BatchConfig, rt: mb.Runtime) -> int:
        stages = rt.cpu_stages(config.cpu_workers)
        made.append(stages.workers)
        stages.close()
        return 0

    monkeypatch.setattr(mb, "run_batch", run_batch)
    monkeypatch.setattr(mb.os, "sched_getaffinity", lambda _pid: set(range(24)))
    wl = _worklist_file(tmp_path, [])
    argv = ["--sandbox", str(tmp_path / "sb"), "run", "--worklist", str(wl), "--device", "cpu"]
    argv += ["--no-residential-check", "--raw-dir", str(tmp_path / "raw")]
    assert mb.main([*argv, "--writers", "16", "--prefetch", "2"]) == 0  # the host's: 16
    assert mb.main([*argv, "--writers", "2", "--prefetch", "1"]) == 0  # no more than in flight
    assert mb.main([*argv, "--cpu-workers", "3"]) == 0
    assert mb.main([*argv, "--cpu-workers", "0"]) == 0  # on the workers' threads, as before
    assert mb.main([*argv, "--writers", "0"]) == 0  # one image at a time: its own thread
    assert made == [16, 3, 3, 0, 0]
