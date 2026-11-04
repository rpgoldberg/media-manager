"""MASK-PERF-2: the mask batch pipelined, so the I/O overlaps the GPU.

With ``writers`` above 0 the images of a chunk run on that many workers, each
with a database session of its own, and the next ``prefetch`` originals are
read, decoded and hash-checked ahead; the GPU stays ONE stream, on the
driver's own thread. Each image's work runs in the same order as before;
images of one picture (the ones whose seed or settle could see each other's
rows) wait for each other in work-list order, and the batch row is written in
that order too. So the pipeline writes what one image at a time writes: the
same rows (save the surrogate ids concurrent inserts draw in another order),
the same bucket bytes, the same records. A pause or a stop lets no further
image onto the GPU; the images it held are not recorded and run again.

SQLite on a file here (a pool of real connections, as on Postgres), local
directories, fake models; the clock is the real one where seconds are
measured.
"""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import io
import json
import queue
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from app.cli import mask_batch as mb
from app.models import Base, FigureGallery, ImageMask, ImageVersion, MaskBatch, PublicObject
from app.models import Image as ImageModel
from app.object_store import LocalObjectStore, derived_key
from app.workers import cpu_stages as cs
from app.workers import mask_job, mask_seed
from app.workers import mask_pipeline as mp
from app.workers.mask_policy import ImageOrigin
from app.workers.mask_seed import picture_keys, seed_clean_image, seed_footprint
from app.workers.matting import MattingBackend, StubMattingBackend
from app.workers.stage_timing import StageTimings
from PIL import Image, PngImagePlugin
from sqlalchemy import Engine, create_engine, select
from sqlalchemy import event as sa_event
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool, QueuePool, StaticPool

ITEM = "https://static.myfigurecollection.net/upload/items/2/{n}-a1b2.jpg"
PICTURE = "https://static.myfigurecollection.net/upload/pictures/2024/01/02/{n}.jpeg"
H = 200
NOON = dt.datetime(2026, 10, 14, 15, 30, tzinfo=dt.UTC)

#: The chunk event's fields that measure time: they differ run to run.
_CHUNK_TIMING = {
    "seconds_per_image",
    "median_seconds",
    "seconds_per_masked_image",
    "mask_seconds_per_image",
    "stage_seconds_per_image",
    "stage_median_seconds",
    "stage_images",
    "elapsed_seconds",
    "throughput_seconds_per_image",
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _photo(width: int, n: int, note: str | None = None) -> bytes:
    arr = np.full((H, width, 3), (250, 250, 248), dtype=np.uint8)
    arr[20:160, int(width * 0.4) : int(width * 0.6)] = (31 + 10 * n % 200, 30, 90)
    arr[0, 0] = (n % 251, 250, 250)
    info = None
    if note:
        info = PngImagePlugin.PngInfo()
        info.add_text("Comment", note)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


class _Raw:
    """A raw bucket on disk, counting its reads (and able to slow them)."""

    def __init__(self, root: Path, cost: float = 0.0) -> None:
        self.store = _Counted(root, get_cost=cost)

    def add(
        self,
        width: int,
        n: int,
        *,
        codec: str = "identity",
        url: str = ITEM,
        pixels_of: int | None = None,
        corrupt: bool = False,
    ) -> mb.WorkItem:
        data = _photo(width, pixels_of or n, note="again" if pixels_of else None)
        sha = hashlib.sha256(data).hexdigest()
        key = f"raw-img/sha256/{sha[:2]}/{sha}.png"
        stored = gzip.compress(data, mtime=0) if codec == "gzip" else data
        LocalObjectStore.put(self.store, key, _photo(width, 99) if corrupt else stored, "image/png")
        return mb.WorkItem(
            blob_sha256=sha,
            storage_key=key,
            url=url.format(n=n),
            product_id=f"00000000-0000-0000-0000-{n:012d}",
            head_id=None,
            rank=0,
            codec=codec,
        )


class _Counted(LocalObjectStore):
    """A local bucket that counts its GETs and PUTs (thread-safe), each
    costing real seconds when asked, and can be told to slow a PUT."""

    def __init__(self, root: Path, *, get_cost: float = 0.0, put_cost: float = 0.0) -> None:
        super().__init__(root)
        self.get_cost, self.put_cost = get_cost, put_cost
        self.lock = threading.Lock()
        self.gets: list[str] = []
        self.puts: list[str] = []
        self.slow: dict[str, float] = {}  # a GET's key, or a PUT's bytes' sha: its delay
        self.fail_once: set[str] = set()  # a GET's key: it times out, once
        self.get_walls: list[float] = []  # each GET's own wall, as this machine made it

    def get(self, key: str) -> bytes:
        started = time.monotonic()
        with self.lock:
            self.gets.append(key)
            failing = key in self.fail_once
            self.fail_once.discard(key)
        delay = self.get_cost + self.slow.get(key, 0.0)
        if delay:
            time.sleep(delay)
        if failing:
            raise TimeoutError("read timed out")
        data = super().get(key)
        with self.lock:
            self.get_walls.append(time.monotonic() - started)
        return data

    def put(self, key: str, data: bytes, content_type: str) -> None:
        delay = self.put_cost + self.slow.get(hashlib.sha256(data).hexdigest(), 0.0)
        if delay:
            time.sleep(delay)
        super().put(key, data, content_type)
        with self.lock:
            self.puts.append(key)


class _Shapes(MattingBackend):
    """A mask model by the photo's width: 101 a full frame (held), 102
    nothing (empty), 103 a crash (failed), any other the figure (ok). It
    notes each call's thread and how many ran at once."""

    name = "shapes"
    publishes_masks = True

    def __init__(self, cost: float = 0.0, gate: Callable[[int], None] | None = None) -> None:
        self.cost, self.gate = cost, gate
        self.lock = threading.Lock()
        self.calls: list[tuple[int, int]] = []
        self.walls: list[float] = []  # each call's own wall, as this machine made it
        self.running = 0
        self.most = 0

    @property
    def active_provider(self) -> str | None:
        return "FakeExecutionProvider"

    def matte(self, image: Image.Image) -> Image.Image:  # pragma: no cover - never asked
        raise AssertionError("the batch asks for the alpha alone")

    def predict_alpha(self, image: Image.Image) -> np.ndarray:
        width = image.width
        with self.lock:
            self.running += 1
            self.most = max(self.most, self.running)
            self.calls.append((width, threading.get_ident()))
        try:
            if self.gate is not None:
                self.gate(width)
            started = time.monotonic()
            if self.cost:
                time.sleep(self.cost)
            with self.lock:
                self.walls.append(time.monotonic() - started)
            if width == 101:
                return np.full((H, width), 255, dtype=np.uint8)
            if width == 102:
                return np.zeros((H, width), dtype=np.uint8)
            if width == 103:
                raise RuntimeError("CUDA error: an illegal memory access")
            alpha = np.zeros((H, width), dtype=np.uint8)
            alpha[20:160, int(width * 0.4) : int(width * 0.6)] = 255
            return alpha
        finally:
            with self.lock:
                self.running -= 1


class _Sampler:
    peak_mib = 3210

    def __enter__(self) -> _Sampler:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _engine(path: Path, **pool: Any) -> Engine:
    """A SQLite file with a pool of real connections (foreign keys on)."""
    engine = create_engine(
        f"sqlite:///{path}", connect_args={"check_same_thread": False, "timeout": 30}, **pool
    )

    @sa_event.listens_for(engine, "connect")
    def _fk(dbapi_conn: Any, _record: Any) -> None:
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    return engine


@pytest.fixture()
def engines(tmp_path: Path) -> Iterator[Callable[[str], Engine]]:
    made: list[Engine] = []

    def make(name: str) -> Engine:
        engine = _engine(tmp_path / f"{name}.db")
        made.append(engine)
        return engine

    yield make
    for engine in made:
        engine.dispose()


def _runtime(
    main: Session,
    root: Path,
    raw: _Raw,
    backend: MattingBackend,
    events: list[dict[str, Any]],
    *,
    media: LocalObjectStore | None = None,
    progress: mb.Progress | None = None,
    stop: Callable[[], bool] | None = None,
    gates: mb.Gates | None = None,
    sleep: Callable[[float], None] = time.sleep,
    cpu: cs.CpuStages | None = None,
) -> mb.Runtime:
    stages = cpu or cs.INLINE
    return mb.Runtime(
        gates=gates or mb.Gates(windows=(), gpu=None, residential=None, now=lambda: NOON),
        open_db=lambda: main,
        media_store=media if media is not None else LocalObjectStore(root / "media"),
        raw_store=mb.ReadOnlyStore(raw.store),
        make_backend=lambda: backend,
        progress=progress or mb.Progress(root / "progress.jsonl"),
        vram=lambda: _Sampler(),
        sleep=sleep,
        monotonic=time.monotonic,
        emit=events.append,
        stop_requested=stop or (lambda: False),
        cpu_stages=lambda _workers: stages,
    )


def _mixed(raw: _Raw) -> list[mb.WorkItem]:
    """Every path an image can take, two captures of one picture adjacent."""
    return [
        raw.add(100, 1),  # ok
        raw.add(100, 2, pixels_of=1),  # the same picture, other bytes: already
        raw.add(101, 3),  # held: a full frame
        raw.add(102, 4),  # empty
        raw.add(103, 5),  # the model crashes: failed
        raw.add(107, 6, url=PICTURE),  # ineligible
        raw.add(108, 7, corrupt=True),  # raw_hash_mismatch
        raw.add(109, 8, codec="gzip"),  # ok, gzip
        raw.add(110, 9, codec="zstd"),  # unsupported_codec
        raw.add(111, 10),  # the raw read times out once: interrupted
        raw.add(112, 11),  # its mask row's INSERT is dropped once: interrupted
        raw.add(113, 12),  # ok: the batch row read again after that rollback
    ]


def _faults(items: list[mb.WorkItem], raw: _Raw, engine: Engine) -> None:
    """Arm ``_mixed``'s two outages for one run: item 10's raw read times
    out, item 11's image_masks INSERT loses the connection, each once."""
    raw.store.fail_once = {items[9].storage_key}
    dropped: list[bool] = []

    @sa_event.listens_for(engine, "before_cursor_execute")
    def _insert(conn: Any, cursor: Any, statement: str, params: Any, *rest: Any) -> None:
        if statement.startswith("INSERT INTO image_masks") and 112 in params and not dropped:
            dropped.append(True)
            raise OperationalError(statement, params, Exception("server closed the connection"))


def _canonical(engine: Engine) -> dict[str, list[str]]:
    """Every row of every table, the surrogate ids replaced by natural keys
    (an image by its sha, a version by its image and number), timestamps
    dropped: what concurrent inserts can change is only the ids they draw."""
    with Session(engine) as db:
        images = dict(db.execute(select(ImageModel.id, ImageModel.sha256)).all())
        versions = {
            vid: f"{images[iid]}#{no}"
            for vid, iid, no in db.execute(
                select(ImageVersion.id, ImageVersion.image_id, ImageVersion.version_no)
            )
        }
        out: dict[str, list[str]] = {}
        for table in Base.metadata.sorted_tables:
            rows = []
            for mapping in db.execute(select(table)).mappings():
                row = {k: v for k, v in mapping.items() if not k.endswith("_at")}
                if table.name in ("images", "figure_galleries", "image_masks"):
                    row.pop("id")
                if table.name == "image_versions":
                    row["id"] = versions[row["id"]]
                if row.get("image_id") is not None:
                    row["image_id"] = images[row["image_id"]]
                if row.get("image_version_id") is not None:
                    row["image_version_id"] = versions[row["image_version_id"]]
                rows.append(json.dumps(row, sort_keys=True, default=str))
            out[table.name] = sorted(rows)
    return out


def _settled(tables: dict[str, list[str]]) -> dict[str, list[str]]:
    """The tables without the batch rows and each mask row's batch: a rerun
    opens batches of its own."""
    out = {name: rows for name, rows in tables.items() if name != "mask_batches"}
    masks = [json.loads(r) for r in out["image_masks"]]
    out["image_masks"] = sorted(
        json.dumps({k: v for k, v in m.items() if k != "batch_id"}, sort_keys=True) for m in masks
    )
    return out


def _media(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _untimed(line: dict[str, Any]) -> dict[str, Any]:
    drop = _CHUNK_TIMING if line["event"] == "chunk" else {"seconds", "mask_seconds"}
    return {k: v for k, v in line.items() if k not in drop}


def _run(
    tmp_path: Path,
    name: str,
    items: list[mb.WorkItem],
    raw: _Raw,
    config: mb.BatchConfig,
    *,
    backend: MattingBackend | None = None,
    engine: Engine | None = None,
    faults: bool = False,
    cpu: cs.CpuStages | None = None,
) -> dict[str, Any]:
    """One batch on a fresh database and media bucket; everything it left."""
    root = tmp_path / name
    root.mkdir()
    engine = engine or _engine(root / "media.db")
    if faults:
        _faults(items, raw, engine)
    batch_row: list[tuple[str, Any]] = []  # its UPDATEs, in order: their statements and values

    @sa_event.listens_for(engine, "before_cursor_execute")
    def _batch_row(conn: Any, cursor: Any, statement: str, params: Any, *rest: Any) -> None:
        if statement.startswith("UPDATE mask_batches") and "finished_at" not in statement:
            batch_row.append((statement, params))

    main = Session(engine, expire_on_commit=False)  # autoflush on, as the sandbox's
    events: list[dict[str, Any]] = []
    try:
        rt = _runtime(main, root, raw, backend or _Shapes(), events, cpu=cpu)
        code = mb.run_batch(items, config, rt)
    finally:
        main.close()
    outcome = {
        "exit": code,
        "tables": _canonical(engine),
        "media": _media(root / "media"),
        "progress": [_untimed(line) for line in _lines(root / "progress.jsonl")],
        "events": [_untimed(e) for e in events],
        "batch_row": batch_row,
    }
    engine.dispose()
    return outcome


def _unordered(outcome: dict[str, Any]) -> dict[str, Any]:
    """``outcome`` with what the order the images END in may change made
    order-free (MASK-PERF-3: each image is recorded, and writes its chunk's
    batch row, as it ends; each chunk reports as its last image ends): the
    records and events sorted, each chunk's ranks and
    failures sorted, each batch row's cursor checked to be the copy of one of
    its own images and dropped, and its UPDATEs by the running total they
    wrote (one image each: no update lost)."""

    def chunked(line: dict[str, Any]) -> dict[str, Any]:
        if line["event"] != "chunk":
            return line
        failures = sorted(line["failures"], key=lambda f: f["blob_sha256"])
        return {**line, "ranks": sorted(line["ranks"]), "failures": failures}

    def key(line: dict[str, Any]) -> tuple[str, str]:
        return (line["event"], json.dumps(line, sort_keys=True, default=str))

    progress = [chunked(line) for line in outcome["progress"]]
    tables = dict(outcome["tables"])
    masks = [json.loads(r) for r in tables["image_masks"]]
    batches = []
    for row in (json.loads(r) for r in tables["mask_batches"]):
        mine = {m["source_sha256"] for m in masks if m["batch_id"] == row["id"]}
        assert row.pop("cursor_sha256") in mine | ({None} if not mine else set())
        batches.append(json.dumps(row, sort_keys=True))
    tables["mask_batches"] = sorted(batches)
    return {
        **outcome,
        "tables": tables,
        "progress": sorted(progress, key=key),
        "events": sorted((chunked(e) for e in outcome["events"]), key=key),
        "batch_row": sorted(
            (params[-1], sum(json.loads(params[1]).values()))
            for _statement, params in outcome["batch_row"]
        ),
    }


# --------------------------------------------------------------------------
# the pipeline writes what one image at a time writes
# --------------------------------------------------------------------------


def test_the_pipeline_writes_what_one_image_at_a_time_writes(tmp_path: Path) -> None:
    raw = _Raw(tmp_path / "raw")
    items = _mixed(raw)
    alone = _run(tmp_path, "alone", items, raw, mb.BatchConfig(chunk_size=4), faults=True)
    records = [r for r in alone["progress"] if r["event"] == "item"]
    assert [r["status"] for r in records] == [
        *("ok", "already", "held", "empty", "failed", "ineligible", "failed", "ok"),
        *("failed", "interrupted", "interrupted", "ok"),
    ]
    assert [r["reason"] for r in records[9:11]] == ["fetch:TimeoutError", "settle:OperationalError"]
    for attempt in range(3):  # three runs: three interleavings
        config = mb.BatchConfig(chunk_size=4, writers=4)
        piped = _run(tmp_path, f"piped{attempt}", items, raw, config, faults=True)
        assert _unordered(piped) == _unordered(alone)


def test_a_rerun_after_the_pipeline_is_idempotent_and_reads_no_original_again(
    tmp_path: Path,
) -> None:
    """Every image of a settled run is 'already' on a rerun with a fresh
    record (idempotent per image sha); its prefetched read is not staged."""
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(104 + n, n) for n in range(1, 6)]
    root = tmp_path / "run"
    root.mkdir()
    engine = _engine(root / "media.db")
    main = Session(engine, expire_on_commit=False)
    config = mb.BatchConfig(chunk_size=5, writers=3)
    events: list[dict[str, Any]] = []
    assert mb.run_batch(items, config, _runtime(main, root, raw, _Shapes(), events)) == 0
    again: list[dict[str, Any]] = []
    rt = _runtime(main, root, raw, _Shapes(), again, progress=mb.Progress(root / "fresh.jsonl"))
    assert mb.run_batch(items, config, rt) == 0
    chunk = next(e for e in again if e["event"] == "chunk")
    assert chunk["counts"] == {"already": 5}
    assert "fetch" not in chunk["stage_seconds_per_image"]
    main.close()
    engine.dispose()


# --------------------------------------------------------------------------
# the overlap, the one GPU stream, the bounds
# --------------------------------------------------------------------------


def test_the_next_images_io_overlaps_the_model(tmp_path: Path) -> None:
    """While the first image is on the GPU, another image's seed stores its
    original in the media bucket (one image at a time, nothing else runs)."""
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(103 + n, n) for n in range(1, 5)]
    root = tmp_path / "run"
    media = _Counted(root / "media")
    media.slow[items[3].blob_sha256] = 0.5  # the last image's original: still uploading
    overlapped = threading.Event()
    first: list[int] = []

    def the_first_waits_for_another_upload(width: int) -> None:
        if first:
            return
        first.append(width)
        before = len(media.puts)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and len(media.puts) == before:
            time.sleep(0.01)
        if len(media.puts) > before:
            overlapped.set()

    engine = _engine(tmp_path / "media.db")
    main = Session(engine, expire_on_commit=False)
    backend = _Shapes(gate=the_first_waits_for_another_upload)
    rt = _runtime(main, root, raw, backend, [], media=media)
    assert mb.run_batch(items, mb.BatchConfig(chunk_size=4, writers=4), rt) == 0
    assert overlapped.is_set()
    main.close()
    engine.dispose()


class _Overlapping(_Counted):
    """A media bucket counting the PUTs in flight at once (and the most),
    where an original's PUT lasts until another PUT starts (5 s at most)."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.in_flight = 0
        self.most = 0
        self.started = 0

    def put(self, key: str, data: bytes, content_type: str) -> None:
        with self.lock:
            self.in_flight += 1
            self.most = max(self.most, self.in_flight)
            self.started += 1
            mine = self.started
        try:
            deadline = time.monotonic() + 5
            while not key.startswith("derived/") and time.monotonic() < deadline:
                with self.lock:
                    if self.started > mine:
                        break
                time.sleep(0.01)
            super().put(key, data, content_type)
        finally:
            with self.lock:
                self.in_flight -= 1


def test_an_images_uploads_overlap_each_other_and_its_sql_and_land_before_its_commits(
    tmp_path: Path,
) -> None:
    """MASK-PERF-3: pipelined, an image's PUTs run beside its own work (its
    original's and its clean copy's beside each other and its seed's SQL),
    and every one of them has landed before any commit of its rows: one
    writer, so whatever is in flight at a commit is that image's."""
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(100 + n * 8, n) for n in range(1, 4)]
    root = tmp_path / "run"
    media = _Overlapping(root / "media")
    engine = _engine(tmp_path / "media.db")
    driver = threading.get_ident()
    at_commit: list[int] = []

    @sa_event.listens_for(engine, "commit")
    def _commit(conn: Any) -> None:
        if threading.get_ident() != driver:
            at_commit.append(media.in_flight)

    main = Session(engine, expire_on_commit=False)
    events: list[dict[str, Any]] = []
    rt = _runtime(main, root, raw, _Shapes(), events, media=media)
    assert mb.run_batch(items, mb.BatchConfig(chunk_size=3, writers=1), rt) == 0
    main.close()
    engine.dispose()
    assert media.most == 2  # an image's original and clean copy at once
    assert len(at_commit) == 6 and set(at_commit) == {0}  # its seed's and its mask's
    (chunk,) = (e for e in events if e["event"] == "chunk")
    assert chunk["counts"] == {"ok": 3}


class _CopyDropped(_Counted):
    """The bucket's connection drops on every clean copy's PUT."""

    def put(self, key: str, data: bytes, content_type: str) -> None:
        if key.startswith("derived/") and content_type != "image/png":
            raise ConnectionError("connection reset by peer")
        super().put(key, data, content_type)


def test_a_failed_upload_fails_its_image_before_any_of_its_rows_commit(tmp_path: Path) -> None:
    """The bucket drops the clean copy's PUT: the image's seed commits
    nothing (an outage: interrupted, retried by a plain rerun), whatever ran
    beside the PUT meanwhile."""
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(100, 1)]
    root = tmp_path / "run"
    media = _CopyDropped(root / "media")  # the original lands, the copy does not
    engine = _engine(tmp_path / "media.db")
    main = Session(engine, expire_on_commit=False)
    rt = _runtime(main, root, raw, _Shapes(), [], media=media)
    assert mb.run_batch(items, mb.BatchConfig(chunk_size=1, writers=1), rt) == 1
    (record,) = (r for r in _lines(root / "progress.jsonl") if r["event"] == "item")
    assert (record["status"], record["reason"]) == ("interrupted", "seed:ConnectionError")
    with Session(engine) as db:
        assert db.execute(select(ImageModel)).first() is None
    main.close()
    engine.dispose()


def test_the_gpu_is_one_stream_on_the_drivers_thread(tmp_path: Path) -> None:
    raw = _Raw(tmp_path / "raw", cost=0.01)
    items = [raw.add(100 + n % 3 * 8, n) for n in range(1, 13)]
    backend = _Shapes(cost=0.02)
    _run(tmp_path, "run", items, raw, mb.BatchConfig(chunk_size=6, writers=4), backend=backend)
    assert len(backend.calls) == 12
    assert backend.most == 1
    assert {thread for _w, thread in backend.calls} == {threading.get_ident()}


def test_images_in_flight_and_read_ahead_are_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the GPU held up, no more than ``writers`` images are on workers
    and no more than ``prefetch`` originals are read past them."""
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(100 + n % 5 * 8, n) for n in range(1, 15)]
    reads_while_held: list[int] = []

    def first_holds_the_gpu(width: int) -> None:
        if not reads_while_held:
            time.sleep(0.5)
            reads_while_held.append(len(raw.store.gets))

    lock = threading.Lock()
    minting = [0, 0]  # now, most
    real_mint = mb._mint

    def counted_mint(*args: Any, **kwargs: Any) -> dict[str, Any]:
        with lock:
            minting[0] += 1
            minting[1] = max(minting[1], minting[0])
        try:
            return real_mint(*args, **kwargs)
        finally:
            with lock:
                minting[0] -= 1

    monkeypatch.setattr(mb, "_mint", counted_mint)
    backend = _Shapes(gate=first_holds_the_gpu)
    config = mb.BatchConfig(chunk_size=14, writers=2, prefetch=3)
    _run(tmp_path, "run", items, raw, config, backend=backend)
    assert reads_while_held == [2 + 3]
    assert minting[1] == 2
    assert len(raw.store.gets) == 14


@pytest.mark.parametrize("slots", [None, 5])
def test_no_more_copies_than_the_decode_slots_are_decoded_for_the_model_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, slots: int | None
) -> None:
    """An image's decoded copy is held from its decode, through its wait for
    the GPU, to its mask: whatever the writers, only ``decode_slots`` copies
    are decoded at once (default three: the one on the GPU and two waiting),
    so the memory stays bounded (MASK-PERF-2 at 4000x3000: ~0.55 GB each)."""
    lock = threading.Lock()
    decoded = [0, 0]  # now, most
    real = mask_job.build_mask_artifact

    def counted(*args: Any, **kwargs: Any) -> Any:
        with lock:
            decoded[0] += 1
            decoded[1] = max(decoded[1], decoded[0])
        try:
            return real(*args, **kwargs)
        finally:
            with lock:
                decoded[0] -= 1

    monkeypatch.setattr(mask_job, "build_mask_artifact", counted)
    first: list[bool] = []

    def first_holds_the_gpu(width: int) -> None:
        if not first:
            first.append(True)
            deadline = time.monotonic() + 5  # every other image gets ready for the GPU meanwhile
            while decoded[1] < (slots or 3) and time.monotonic() < deadline:
                time.sleep(0.01)
            time.sleep(0.1)  # and one more would decode now, were a slot free

    raw = _Raw(tmp_path / "raw")
    items = [raw.add(100 + n % 2 * 8, n) for n in range(1, 13)]
    engine = _engine(tmp_path / "media.db", pool_size=9)
    config = mb.BatchConfig(chunk_size=12, writers=8)
    if slots is not None:
        config.decode_slots = slots
    backend = _Shapes(gate=first_holds_the_gpu)
    outcome = _run(tmp_path, "run", items, raw, config, backend=backend, engine=engine)
    assert outcome["exit"] == 0
    assert [r["status"] for r in outcome["progress"] if r["event"] == "item"] == ["ok"] * 12
    assert decoded[1] == (slots or 3)
    assert mb.DECODED_AT_ONCE == mb.BatchConfig().decode_slots == 3


def test_a_big_original_takes_the_decode_slots_of_its_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MASK-PERF-3's memory guard: an original over ``DECODE_UNIT_PIXELS``
    takes a slot per unit of its size (up to every slot), so while a big
    copy waits for the GPU no other copy is decoded beside it."""
    monkeypatch.setattr(mb, "DECODE_UNIT_PIXELS", 100 * H)  # a 100-px-wide photo: one unit
    lock = threading.Lock()
    decoded = [0]
    beside_the_big_one: list[int] = []
    real = mask_job.build_mask_artifact

    def counted(*args: Any, **kwargs: Any) -> Any:
        with lock:
            decoded[0] += 1
        try:
            return real(*args, **kwargs)
        finally:
            with lock:
                decoded[0] -= 1

    monkeypatch.setattr(mask_job, "build_mask_artifact", counted)

    def the_big_one_holds_the_gpu(width: int) -> None:
        if width == 300:
            time.sleep(0.5)  # every other image is ready for the GPU meanwhile
            beside_the_big_one.append(decoded[0])

    raw = _Raw(tmp_path / "raw")
    items = [raw.add(300, 1), *(raw.add(100 + n % 2 * 8, n) for n in range(2, 8))]
    config = mb.BatchConfig(chunk_size=7, writers=6, decode_slots=3)
    backend = _Shapes(gate=the_big_one_holds_the_gpu)
    engine = _engine(tmp_path / "media.db", pool_size=7)
    outcome = _run(tmp_path, "run", items, raw, config, backend=backend, engine=engine)
    assert [r["status"] for r in outcome["progress"] if r["event"] == "item"] == ["ok"] * 7
    assert beside_the_big_one == [1]  # three units of three: it alone


def test_decode_units_are_the_originals_size_in_units_up_to_every_slot() -> None:
    unit = mb.DECODE_UNIT_PIXELS
    assert unit == 4000 * 3000  # the size the 0.55 GB peak was measured at
    assert mb._decode_units(None, 4) == 1  # unknown (it fails before its decode)
    assert mb._decode_units(unit, 4) == 1
    assert mb._decode_units(unit + 1, 4) == 2
    assert mb._decode_units(64_000_000, 8) == 6  # the largest original the limits let in
    assert mb._decode_units(64_000_000, 3) == 3  # alone, every slot its own


def test_a_model_call_that_raises_still_gives_its_decode_slot_back(tmp_path: Path) -> None:
    """More images crash the model than there are decode slots: each gives its
    slot back as it fails, so the chunk finishes with every image failed (a
    slot kept would leave the next image waiting for the GPU for good)."""
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(103, n) for n in range(1, mb.DECODED_AT_ONCE * 2 + 1)]
    engine = _engine(tmp_path / "media.db", pool_size=5)
    config = mb.BatchConfig(chunk_size=len(items), writers=4)
    done: list[dict[str, Any]] = []
    worker = threading.Thread(
        target=lambda: done.append(_run(tmp_path, "run", items, raw, config, engine=engine)),
        daemon=True,  # a deadlock must fail this test, not hold the suite
    )
    worker.start()
    worker.join(30)
    assert not worker.is_alive(), "a failed model call kept its decode slot: the chunk hangs"
    (outcome,) = done
    statuses = [r["status"] for r in outcome["progress"] if r["event"] == "item"]
    assert statuses == ["failed"] * len(items)


def test_decode_slots_go_to_the_earliest_image_waiting() -> None:
    slots = mp.Slots(1)
    order: list[int] = []

    def take(index: int) -> None:
        slots.acquire(index)
        order.append(index)
        slots.release()

    slots.acquire(0)
    threads = [threading.Thread(target=take, args=(i,)) for i in (5, 3, 4)]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + 5
    while slots.waiting() < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    slots.release()
    for thread in threads:
        thread.join(5)
    assert order == [3, 4, 5]
    two = mp.Slots(2)
    two.acquire(7)
    two.acquire(9)  # a second slot: no wait
    assert two.waiting() == 0


def test_a_slot_of_several_units_waits_for_them_and_holds_back_later_images() -> None:
    """The earliest image waiting is served first, however many units it
    takes: a big one is not passed over for good by small ones behind it."""
    slots = mp.Slots(3)
    slots.acquire(0, 2)
    order: list[int] = []

    def take(index: int, units: int) -> None:
        slots.acquire(index, units)
        order.append(index)

    big = threading.Thread(target=take, args=(1, 3))
    big.start()
    deadline = time.monotonic() + 5
    while slots.waiting() < 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    small = threading.Thread(target=take, args=(2, 1))  # one unit is free: still it waits
    small.start()
    while slots.waiting() < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.05)
    assert order == []
    slots.release(2)
    big.join(5)
    assert order == [1]
    slots.release(3)
    small.join(5)
    assert order == [1, 2]


# --------------------------------------------------------------------------
# records, related pictures, the batch row
# --------------------------------------------------------------------------


class _Checked(mb.Progress):
    """A progress file that, as each image's record is written, reads the
    image's settled row from the database on a connection of its own."""

    def __init__(self, path: Path, engine: Engine) -> None:
        super().__init__(path)
        self.engine = engine
        self.seen: list[tuple[str, str | None]] = []

    def append(self, record: dict[str, Any]) -> None:
        if record.get("event") == "item" and record.get("clean_sha256"):
            with Session(self.engine) as db:
                status = db.execute(
                    select(ImageMask.status).where(
                        ImageMask.source_sha256 == record["clean_sha256"]
                    )
                ).scalar_one_or_none()
            self.seen.append((record["status"], status))
        super().append(record)


def test_a_record_is_written_only_after_its_writes_commit(tmp_path: Path) -> None:
    raw = _Raw(tmp_path / "raw")
    items = _mixed(raw)
    root = tmp_path / "run"
    engine = _engine(tmp_path / "media.db")
    main = Session(engine, expire_on_commit=False)
    progress = _Checked(root / "progress.jsonl", engine)
    rt = _runtime(main, root, raw, _Shapes(), [], progress=progress)
    mb.run_batch(items, mb.BatchConfig(chunk_size=6, writers=4), rt)
    by_record = {"already": "ok"}  # its picture's copy, settled by the first capture
    assert progress.seen
    for status, row in progress.seen:
        assert row == by_record.get(status, status)
    main.close()
    engine.dispose()


def test_two_captures_of_one_picture_never_overlap(tmp_path: Path) -> None:
    """The second capture's work starts only once the first's has ended,
    however slow the first: it finds the copy settled ('already')."""
    raw = _Raw(tmp_path / "raw")
    first, second = raw.add(100, 1), raw.add(100, 2, pixels_of=1)
    media = _Counted(tmp_path / "run" / "media")
    media.slow[first.blob_sha256] = 0.4  # its original's PUT
    root = tmp_path / "run"
    engine = _engine(tmp_path / "media.db")
    main = Session(engine, expire_on_commit=False)
    rt = _runtime(main, root, raw, _Shapes(), [], media=media)
    assert mb.run_batch([first, second], mb.BatchConfig(chunk_size=2, writers=2), rt) == 0
    records = [r for r in _lines(root / "progress.jsonl") if r["event"] == "item"]
    assert [r["status"] for r in records] == ["ok", "already"]
    assert records[1]["mask_status"] == "ok"
    second_original = f"{second.blob_sha256[:2]}/{second.blob_sha256[2:4]}/{second.blob_sha256}"
    # the first's original, copy and mask, and only then the second's original
    assert len(media.puts) == 4
    assert media.puts[-1].startswith(second_original)
    assert not any(k.startswith(second_original) for k in media.puts[:3])
    main.close()
    engine.dispose()


def _stored(raw: _Raw, data: bytes, n: int, url: str = ITEM) -> mb.WorkItem:
    """``data`` in the raw bucket, as a work item of product ``n`` at ``url``."""
    sha = hashlib.sha256(data).hexdigest()
    key = f"raw-img/sha256/{sha[:2]}/{sha}.png"
    LocalObjectStore.put(raw.store, key, data, "image/png")
    return mb.WorkItem(
        blob_sha256=sha,
        storage_key=key,
        url=url.format(n=n),
        product_id=f"00000000-0000-0000-0000-{n:012d}",
        head_id=None,
        rank=0,
    )


def _slow_first(tmp_path: Path, raw: _Raw, items: list[mb.WorkItem]) -> tuple[_Counted, Path]:
    """``items`` pipelined two at a time, the first image's original PUT
    slow (0.4 s): its media bucket and its run's directory."""
    root = tmp_path / "run"
    media = _Counted(root / "media")
    media.slow[items[0].blob_sha256] = 0.4
    engine = _engine(tmp_path / "media.db")
    main = Session(engine, expire_on_commit=False)
    rt = _runtime(main, root, raw, _Shapes(), [], media=media)
    mb.run_batch(items, mb.BatchConfig(chunk_size=len(items), writers=2), rt)
    main.close()
    engine.dispose()
    return media, root


def _original_key(item: mb.WorkItem) -> str:
    return f"{item.blob_sha256[:2]}/{item.blob_sha256[2:4]}/{item.blob_sha256}"


def test_a_capture_of_a_copys_own_pixels_waits_for_the_image_minting_that_copy(
    tmp_path: Path,
) -> None:
    """Our JPEG copy captured again (the pixels it decodes to, other bytes)
    is a capture of that copy's picture: its claim meets the claim of the
    image still minting the copy by the copy's own pixel digest, so it waits
    for it, however slow, and finds the copy settled ('already')."""
    raw = _Raw(tmp_path / "raw")
    first = raw.add(100, 1)
    original = LocalObjectStore.get(raw.store, first.storage_key)
    clean = mask_seed._encode(mask_seed._picture(original), original)[0]
    with Image.open(io.BytesIO(clean)) as copy:
        buf = io.BytesIO()
        copy.convert("RGB").save(buf, format="PNG")  # its own pixels, other bytes
    second = _stored(raw, buf.getvalue(), 2)
    assert picture_keys(buf.getvalue()).digest == picture_keys(original).own_digest
    media, root = _slow_first(tmp_path, raw, [first, second])
    records = [r for r in _lines(root / "progress.jsonl") if r["event"] == "item"]
    assert [r["status"] for r in records] == ["ok", "already"]
    # the first's original, copy and mask, and only then the second's original
    assert len(media.puts) == 4
    assert media.puts[-1].startswith(_original_key(second))


def test_images_of_one_gallery_link_wait_for_each_other(tmp_path: Path) -> None:
    """Two pictures of one product under one URL: the seed links the product
    to the first image only, as one image at a time does, because the second
    waits for the first, however slow, before its first query."""
    raw = _Raw(tmp_path / "raw")
    first = raw.add(100, 1, url=PICTURE)
    second = _stored(raw, _photo(108, 2), 1, url=PICTURE)  # another picture
    assert (first.product_id, first.url) == (second.product_id, second.url)
    media, root = _slow_first(tmp_path, raw, [first, second])
    statuses = [r["status"] for r in _lines(root / "progress.jsonl") if r["event"] == "item"]
    assert statuses == ["ineligible", "ineligible"]  # a PICTURE url: no mask, no upload
    assert len(media.puts) == 4  # each its original and copy
    assert not any(k.startswith(_original_key(second)) for k in media.puts[:2])
    engine = _engine(tmp_path / "media.db")
    with Session(engine) as db:
        (linked,) = db.execute(select(FigureGallery.image_id)).scalars()
        assert db.get(ImageModel, linked).sha256 == first.blob_sha256
    engine.dispose()


def test_the_batch_row_is_read_again_after_an_images_write_rolls_back(tmp_path: Path) -> None:
    """One image writes the batch row, the next one's write rolls back, the
    one after reads the row again before its own: the counts and cursor are
    one image at a time's (the rolled-back image's copy of the row is
    stale)."""
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(100, 1), raw.add(112, 2), raw.add(113, 3)]

    def dropping(name: str) -> Engine:
        engine = _engine(tmp_path / f"{name}.db")
        dropped: list[bool] = []

        @sa_event.listens_for(engine, "before_cursor_execute")
        def _insert(conn: Any, cursor: Any, statement: str, params: Any, *rest: Any) -> None:
            if statement.startswith("INSERT INTO image_masks") and 112 in params and not dropped:
                dropped.append(True)
                raise OperationalError(statement, params, Exception("server closed the connection"))

        return engine

    alone = _run(tmp_path, "alone", items, raw, mb.BatchConfig(), engine=dropping("alone"))
    config = mb.BatchConfig(writers=2)
    piped = _run(tmp_path, "piped", items, raw, config, engine=dropping("piped"))
    statuses = [r["status"] for r in alone["progress"] if r["event"] == "item"]
    assert statuses == ["ok", "interrupted", "ok"]
    (row,) = (json.loads(r) for r in alone["tables"]["mask_batches"])
    assert row["counts"]["ok"] == 2
    assert _unordered(piped) == _unordered(alone)


def test_no_more_than_writers_images_are_in_flight_and_each_is_recorded_as_it_ends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Images that end early (here an unsupported codec: no row to write)
    are recorded as they end and make way for the next while the image
    ahead of them is held up (its copy's readback): never more than
    ``writers`` in flight, every one of them unrecorded, so an outage opens
    the circuit after at most writers - 1 images more than one at a time
    would have tried, and a power cut loses at most ``writers`` records."""
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(100, 1), *(raw.add(110 + n, n, codec="zstd") for n in range(2, 9))]
    lock = threading.Lock()
    minting = [0, 0]  # now, most
    real_mint = mb._mint

    def counted_mint(*args: Any, **kwargs: Any) -> dict[str, Any]:
        with lock:
            minting[0] += 1
            minting[1] = max(minting[1], minting[0])
        try:
            return real_mint(*args, **kwargs)
        finally:
            with lock:
                minting[0] -= 1

    monkeypatch.setattr(mb, "_mint", counted_mint)
    records, _events, engine = _slowed(
        tmp_path, items, raw, items[0], mb.BatchConfig(chunk_size=8, writers=2), code=1
    )
    engine.dispose()
    assert [r["status"] for r in records] == ["failed"] * 7 + ["ok"]
    assert minting[1] == 2


class _HeldBack(_Counted):
    """A media bucket whose GET of ``key`` (an image's clean copy read
    back: a bucket call's tail, a 10 s upload live; a read, so no SQLite
    write lock is held meanwhile) lasts until ``until`` holds of the
    progress file's lines, 10 s at most."""

    def __init__(self, root: Path, key: str, until: Callable[[list[dict[str, Any]]], bool]) -> None:
        super().__init__(root)
        self.key, self.until, self.progress = key, until, root.parent / "progress.jsonl"

    def get(self, key: str) -> bytes:
        deadline = time.monotonic() + 10
        while key == self.key and time.monotonic() < deadline:
            if self.progress.exists() and self.until(_lines(self.progress)):
                break
            time.sleep(0.01)
        return super().get(key)


def _items_but_one(items: list[mb.WorkItem]) -> Callable[[list[dict[str, Any]]], bool]:
    return lambda lines: sum(line["event"] == "item" for line in lines) >= len(items) - 1


def _slowed(
    tmp_path: Path,
    items: list[mb.WorkItem],
    raw: _Raw,
    slow: mb.WorkItem,
    config: mb.BatchConfig,
    code: int = 0,
    until: Callable[[list[dict[str, Any]]], bool] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Engine]:
    """``items`` run with ``slow`` held up at its copy's readback until
    ``until`` (by default, until every other image is recorded): its
    records, its events and its database."""
    root = tmp_path / "run"
    original = LocalObjectStore.get(raw.store, slow.storage_key)
    clean, mime = mask_seed._encode(mask_seed._picture(original), original)
    key = derived_key(hashlib.sha256(clean).hexdigest(), mime)
    media = _HeldBack(root / "media", key, until or _items_but_one(items))
    engine = _engine(tmp_path / "media.db")
    main = Session(engine, expire_on_commit=False)
    events: list[dict[str, Any]] = []
    rt = _runtime(main, root, raw, _Shapes(), events, media=media)
    assert mb.run_batch(items, config, rt) == code
    main.close()
    records = [r for r in _lines(root / "progress.jsonl") if r["event"] == "item"]
    return records, events, engine


def test_a_slow_image_holds_up_no_other_image(tmp_path: Path) -> None:
    """MASK-PERF-3: an image whose upload is slow keeps its own worker only:
    every other image ends, is recorded as it ends and makes way for the
    next, so nothing waits for the slowest image in flight."""
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(100 + n % 3 * 8, n) for n in range(1, 7)]
    records, _events, engine = _slowed(
        tmp_path, items, raw, items[0], mb.BatchConfig(chunk_size=6, writers=2)
    )
    assert [r["status"] for r in records] == ["ok"] * 6
    assert records[-1]["blob_sha256"] == items[0].blob_sha256  # the slow one ends last
    engine.dispose()


def test_the_next_chunk_starts_before_the_slowest_image_of_the_one_before_ends(
    tmp_path: Path,
) -> None:
    """No chunk barrier: a chunk is its batch row, its gate and its report;
    the next one's images start while the one before still has one in
    flight. Each chunk counts its own images and reports once its last one
    ended, so one held up by a slow image reports after the next: a stuck
    image keeps no later chunk's batch row open nor its VRAM sampler running."""
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(100 + n % 3 * 8, n) for n in range(1, 7)]
    chunk_reported = lambda lines: any(line["event"] == "chunk" for line in lines)  # noqa: E731
    config = mb.BatchConfig(chunk_size=3, writers=3)
    records, events, engine = _slowed(tmp_path, items, raw, items[2], config, until=chunk_reported)
    order = [r["blob_sha256"] for r in records]
    assert order[-1] == items[2].blob_sha256  # the second chunk ended first
    assert {r["chunk"] for r in records[:-1] if r["blob_sha256"] in order[3:]} == {2}
    chunks = [e for e in events if e["event"] == "chunk"]
    assert [(c["chunk"], c["count"]) for c in chunks] == [(2, 3), (1, 3)]
    with Session(engine) as db:
        rows = db.execute(select(MaskBatch).order_by(MaskBatch.id)).scalars().all()
        assert [(sum(b.counts.values()), b.status) for b in rows] == [(3, "done"), (3, "done")]
    engine.dispose()


def test_the_gate_before_each_chunk_reads_the_gpu_with_no_model_call_running(
    tmp_path: Path,
) -> None:
    """Rolling, the gate before the next chunk is read while the chunk before
    still has images in flight: on the driver's thread, so no model call
    runs from its settle to its last reading (the readings are another
    consumer's alone, as when the chunk before had ended)."""
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(100 + n % 3 * 8, n) for n in range(1, 7)]
    backend = _Shapes(cost=0.05)
    running: list[int] = []

    def settle(seconds: float) -> None:
        running.append(backend.running)

    gpu = mb.GpuGate(query=lambda: f"{backend.running}, 12000, 300, 12282", sleep=settle)
    gates = mb.Gates(windows=(), gpu=gpu, residential=None, now=lambda: NOON)
    root = tmp_path / "run"
    engine = _engine(tmp_path / "media.db")
    main = Session(engine, expire_on_commit=False)
    events: list[dict[str, Any]] = []
    rt = _runtime(main, root, raw, backend, events, gates=gates)
    assert mb.run_batch(items, mb.BatchConfig(chunk_size=2, writers=3), rt) == 0
    chunks = [e for e in events if e["event"] == "chunk"]
    assert [c["count"] for c in chunks] == [2, 2, 2]
    assert running  # the settles and samples of the gates while the model was held
    assert set(running) == {0}
    assert "paused" not in [e["event"] for e in events]
    main.close()
    engine.dispose()


def test_a_gate_that_fails_at_a_chunk_boundary_holds_the_pipeline_until_it_passes(
    tmp_path: Path,
) -> None:
    """Another consumer takes the GPU while a chunk is in flight: the gate
    before the next chunk fails, no image of it starts, the images not past
    the GPU are let go, the model is released while the batch waits, and
    once the gate passes the held images run first: every image once."""
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(100 + n % 3 * 8, n) for n in range(1, 7)]
    busy = {"on": False}

    def query() -> str:
        return "1, 1000, 11500, 12282" if busy["on"] else "0, 2800, 9500, 12282"

    measured = threading.Event()

    class _Measured(mb.GpuGate):
        def measure_own(self) -> None:
            super().measure_own()
            measured.set()

    games: list[bool] = []

    def the_game_starts(width: int) -> None:
        if measured.is_set() and not games:  # once our model is measured, a game starts, once
            games.append(True)
            busy["on"] = True

    def poll(seconds: float) -> None:
        busy["on"] = False  # the game ends while the batch waits

    class _Counting(_Shapes):
        released = 0

        def release(self) -> None:
            self.released += 1

    backend = _Counting(gate=the_game_starts)
    gpu = _Measured(query=query, sleep=lambda seconds: None)
    gates = mb.Gates(windows=(), gpu=gpu, residential=None, now=lambda: NOON)
    root = tmp_path / "run"
    engine = _engine(tmp_path / "media.db")
    main = Session(engine, expire_on_commit=False)
    events: list[dict[str, Any]] = []
    rt = _runtime(main, root, raw, backend, events, gates=gates, sleep=poll)
    assert mb.run_batch(items, mb.BatchConfig(chunk_size=2, writers=2), rt) == 0
    records = [r for r in _lines(root / "progress.jsonl") if r["event"] == "item"]
    assert sorted(r["blob_sha256"] for r in records) == sorted(i.blob_sha256 for i in items)
    assert {r["status"] for r in records} == {"ok"}
    paused = [e for e in events if e["event"] == "paused"]
    assert [e["gate"] for e in paused] == ["gpu"]
    assert backend.released == 1
    assert sum(e["count"] for e in events if e["event"] == "chunk") == 6
    main.close()
    engine.dispose()


def test_a_put_that_failed_beside_the_seed_is_the_seeds_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A PUT made before the seed broke off is what broke it, as when it ran
    in turn: the bucket's outage (interrupted), not the error after it."""

    def seed(db: Session, store: Any, data: bytes, **_: Any) -> Any:
        store.put("aa/bb/original.png", data, "image/png")  # beside: it fails later
        raise ValueError("a row the seed could not write")

    monkeypatch.setattr(mb, "seed_clean_image", seed)
    raw = _Raw(tmp_path / "raw")
    root = tmp_path / "run"
    engine = _engine(tmp_path / "media.db")
    main = Session(engine, expire_on_commit=False)
    rt = _runtime(main, root, raw, _Shapes(), [], media=_Outage(root / "media", healthy=0))
    assert mb.run_batch([raw.add(100, 1)], mb.BatchConfig(chunk_size=1, writers=1), rt) == 1
    (record,) = (r for r in _lines(root / "progress.jsonl") if r["event"] == "item")
    assert (record["status"], record["reason"]) == ("interrupted", "seed:ConnectionError")
    main.close()
    engine.dispose()


def test_each_workers_sql_is_charged_to_its_own_images_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every statement a worker runs -- its claim's, its seed's, its
    judgment's, not only those inside an explicit "db" stage -- is its
    image's "db" seconds."""
    by_thread: dict[str, StageTimings] = {}

    class Capturing(StageTimings):
        def __init__(self, clock: Callable[[], float]) -> None:
            super().__init__(clock)
            by_thread[threading.current_thread().name] = self

    monkeypatch.setattr(mb, "StageTimings", Capturing)
    engine = _engine(tmp_path / "media.db")
    lock = threading.Lock()
    statements: dict[str, int] = {}
    cost = 0.01

    @sa_event.listens_for(engine, "after_cursor_execute")
    def _slow(*_: Any) -> None:
        name = threading.current_thread().name
        if name.startswith("mask-image-"):
            with lock:
                statements[name] = statements.get(name, 0) + 1
            time.sleep(cost)

    raw = _Raw(tmp_path / "raw")
    items = [raw.add(100 + n * 8, n) for n in range(1, 5)]
    main = Session(engine, expire_on_commit=False)
    rt = _runtime(main, tmp_path / "run", raw, _Shapes(), [])
    assert mb.run_batch(items, mb.BatchConfig(chunk_size=4, writers=2), rt) == 0
    main.close()
    engine.dispose()
    assert sorted(statements) == sorted(by_thread) == [f"mask-image-{i}" for i in range(4)]
    for name, timings in by_thread.items():
        assert statements[name] >= 15  # the claim's, the seed's, the judgment's, the rows'
        assert timings.seconds["db"] >= statements[name] * cost


def test_the_batch_rows_counts_are_one_image_at_a_times(tmp_path: Path) -> None:
    """Its cursor is the copy of the last of its images to commit (the order
    they end in): one of its own images, as ``_unordered`` checks."""
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(100 + n % 4, n) for n in range(1, 11)]  # ok, held, empty, crash, ...
    alone = _run(tmp_path, "alone", items, raw, mb.BatchConfig(chunk_size=10))
    engine = _engine(tmp_path / "piped.db", pool_size=7)  # a connection per worker and the batch
    config = mb.BatchConfig(chunk_size=10, writers=6)
    piped = _run(tmp_path, "piped", items, raw, config, engine=engine)
    assert _unordered(piped)["tables"] == _unordered(alone)["tables"]
    (row,) = (json.loads(r) for r in piped["tables"]["mask_batches"])
    assert sum(row["counts"].values()) == 10


# --------------------------------------------------------------------------
# a pause, a stop, an outage
# --------------------------------------------------------------------------


def test_a_stop_lets_no_further_image_onto_the_gpu_and_a_rerun_finishes(
    tmp_path: Path,
) -> None:
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(100 + n % 3 * 8, n) for n in range(1, 9)]
    raw.store.slow[items[3].storage_key] = 0.5  # still reading when the stop comes
    stop = threading.Event()
    backend = _Shapes(gate=lambda width: stop.set())
    root = tmp_path / "run"
    engine = _engine(tmp_path / "media.db")
    main = Session(engine, expire_on_commit=False)
    events: list[dict[str, Any]] = []
    config = mb.BatchConfig(chunk_size=8, writers=4)
    rt = _runtime(main, root, raw, backend, events, stop=stop.is_set)
    assert mb.run_batch(items, config, rt) == 130
    assert len(backend.calls) == 1  # the image already on the GPU, no other
    records = [r for r in _lines(root / "progress.jsonl") if r["event"] == "item"]
    assert len(records) == 1
    assert events[-1] == {"event": "stopped", "counts": {"ok": 1}, "pending": 7}
    rerun: list[dict[str, Any]] = []
    rt = _runtime(main, root, raw, _Shapes(), rerun)
    assert mb.run_batch(items, config, rt) == 0
    assert rerun[0]["pending"] == 7
    main.close()
    alone = _run(tmp_path, "alone", items, raw, mb.BatchConfig(chunk_size=8))
    assert _settled(_canonical(engine)) == _settled(alone["tables"])
    assert _media(root / "media") == alone["media"]
    engine.dispose()


def test_a_busy_window_mid_chunk_yields_before_the_next_gpu_image(tmp_path: Path) -> None:
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(100 + n % 3 * 8, n) for n in range(1, 7)]
    wall = {"minute": 30}
    minutes: list[int] = []

    def opens_the_window(width: int) -> None:
        minutes.append(wall["minute"])
        wall["minute"] = 5  # a scraper roll window begins

    def sleep(seconds: float) -> None:
        wall["minute"] = 30  # the window passed while the batch waited

    gates = mb.Gates(
        windows=mb.parse_windows("03-26"),
        gpu=None,
        residential=None,
        now=lambda: NOON.replace(minute=wall["minute"]),
    )
    root = tmp_path / "run"
    engine = _engine(tmp_path / "media.db")
    main = Session(engine, expire_on_commit=False)
    events: list[dict[str, Any]] = []
    backend = _Shapes(gate=opens_the_window)
    rt = _runtime(main, root, raw, backend, events, gates=gates, sleep=sleep)
    assert mb.run_batch(items, mb.BatchConfig(chunk_size=6, writers=3), rt) == 0
    assert set(minutes) == {30}  # no image on the GPU inside the window
    chunks = [e for e in events if e["event"] == "chunk"]
    assert chunks[0]["yielded"] is True
    assert chunks[0]["count"] == 1
    assert sum(c["count"] for c in chunks) == 6
    assert [e["event"] for e in events].count("paused") == len(chunks) - 1
    main.close()
    engine.dispose()


class _Outage(_Counted):
    """The media bucket's connection drops after ``healthy`` PUTs."""

    def __init__(self, root: Path, healthy: int) -> None:
        super().__init__(root)
        self.healthy = healthy

    def put(self, key: str, data: bytes, content_type: str) -> None:
        with self.lock:
            down = len(self.puts) >= self.healthy
        if down:
            raise ConnectionError("connection reset by peer")
        super().put(key, data, content_type)


def test_a_bucket_outage_opens_the_circuit_and_a_plain_rerun_finishes(tmp_path: Path) -> None:
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(100 + n % 3 * 8, n) for n in range(1, 13)]
    root = tmp_path / "run"
    engine = _engine(tmp_path / "media.db")
    main = Session(engine, expire_on_commit=False)
    events: list[dict[str, Any]] = []
    config = mb.BatchConfig(chunk_size=12, writers=4, max_consecutive_failures=3)
    media = _Outage(root / "media", healthy=6)
    assert (
        mb.run_batch(items, config, _runtime(main, root, raw, _Shapes(), events, media=media)) == 3
    )
    circuit = next(e for e in events if e["event"] == "circuit_open")
    assert circuit["consecutive_failures"] >= 3
    records = [r for r in _lines(root / "progress.jsonl") if r["event"] == "item"]
    assert {r["status"] for r in records} <= {"ok", "interrupted"}
    assert len(records) < 12
    media.healthy = 10**6
    assert mb.run_batch(items, config, _runtime(main, root, raw, _Shapes(), [], media=media)) == 0
    main.close()
    alone = _run(tmp_path, "alone", items, raw, mb.BatchConfig(chunk_size=12))
    assert _settled(_canonical(engine)) == _settled(alone["tables"])
    assert _media(root / "media") == alone["media"]
    engine.dispose()


def test_a_dropped_database_connection_on_the_workers_is_an_outage(tmp_path: Path) -> None:
    """The port-forward drops: every image's query fails on its own
    connection (OperationalError), each image is interrupted, the circuit
    opens (exit 3) and a plain rerun retries them."""
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(100 + n % 3 * 8, n) for n in range(1, 9)]
    root = tmp_path / "run"
    engine = _engine(tmp_path / "media.db")
    driver = threading.get_ident()
    down = threading.Event()

    @sa_event.listens_for(engine, "before_cursor_execute")
    def _dropped(*args: Any) -> None:
        if down.is_set() and threading.get_ident() != driver:
            raise OperationalError("SELECT", {}, Exception("server closed the connection"))

    main = Session(engine, expire_on_commit=False)
    events: list[dict[str, Any]] = []
    config = mb.BatchConfig(chunk_size=8, writers=4, max_consecutive_failures=3)
    down.set()
    assert mb.run_batch(items, config, _runtime(main, root, raw, _Shapes(), events)) == 3
    records = [r for r in _lines(root / "progress.jsonl") if r["event"] == "item"]
    assert {r["status"] for r in records} == {"interrupted"}
    assert {r["reason"] for r in records} == {"db:OperationalError"}
    down.clear()
    assert mb.run_batch(items, config, _runtime(main, root, raw, _Shapes(), [])) == 0
    main.close()
    engine.dispose()


def test_an_unexpected_error_on_a_worker_stops_the_batch_without_hanging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(100 + n % 3 * 8, n) for n in range(1, 7)]
    real_mint = mb._mint

    def mint(db: Session, item: mb.WorkItem, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if item is items[2]:
            raise LookupError("a bug")
        return real_mint(db, item, *args, **kwargs)

    monkeypatch.setattr(mb, "_mint", mint)
    with pytest.raises(LookupError):
        _run(tmp_path, "run", items, raw, mb.BatchConfig(chunk_size=6, writers=3))
    assert [t.name for t in threading.enumerate() if t.name.startswith("mask-")] == []


class _Unwritable(mb.Progress):
    def append(self, record: dict[str, Any]) -> None:
        if record.get("event") == "item":
            raise OSError(28, "No space left on device")
        super().append(record)


def test_a_record_that_cannot_be_written_stops_the_batch_and_lets_every_worker_go(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first record fails to write while another image waits for the
    GPU: the driver refuses that image, lets every worker go and raises.
    The order is forced: image 1 holds the GPU until image 0 is done and
    image 2 has asked for the GPU after it."""
    posted: list[str] = []
    seen = threading.Condition()

    class _Watched(queue.Queue[Any]):
        def put(self, item: Any, block: bool = True, timeout: float | None = None) -> None:
            super().put(item, block, timeout)
            with seen:
                posted.append(type(item).__name__)
                seen.notify_all()

    def until(check: Callable[[], bool]) -> None:
        with seen:
            assert seen.wait_for(check, timeout=5)

    monkeypatch.setattr(mb, "queue", SimpleNamespace(Queue=_Watched))
    rgba = Image.new("RGBA", (100, H))
    on_gpu = threading.Event()

    def model(width: int) -> None:
        on_gpu.set()
        until(lambda: posted[-2:] == ["_Done", "GpuJob"])

    def mint(db: Session, item: mb.WorkItem, rt: mb.Runtime, gpu: MattingBackend, *a: Any) -> Any:
        n = int(item.product_id[-1])
        if n == 1:
            gpu.predict_alpha(rgba)
        else:
            assert on_gpu.wait(timeout=5)
            if n == 2:
                until(lambda: "_Done" in posted)
                gpu.predict_alpha(rgba)  # refused: Held
        return {"event": "item", "status": "ok", "blob_sha256": item.blob_sha256, "rank": 0}

    monkeypatch.setattr(mb, "_mint", mint)
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(100, n) for n in range(3)]
    backend = _Shapes(gate=model)
    root = tmp_path / "run"
    engine = _engine(tmp_path / "media.db")
    main = Session(engine, expire_on_commit=False)
    progress = _Unwritable(root / "progress.jsonl")
    rt = _runtime(main, root, raw, backend, [], progress=progress)
    with pytest.raises(OSError):
        mb.run_batch(items, mb.BatchConfig(chunk_size=3, writers=3), rt)
    assert len(backend.calls) == 1  # image 2 never got the GPU
    assert [t.name for t in threading.enumerate() if t.name.startswith("mask-")] == []
    main.close()
    engine.dispose()


def test_a_read_ahead_that_breaks_stops_the_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(100 + n % 3 * 8, n) for n in range(1, 5)]
    real = mb._read_ahead

    def read_ahead(item: mb.WorkItem, rt: mb.Runtime, *cpu: Any) -> Any:
        if item is items[1]:
            raise MemoryError
        return real(item, rt, *cpu)

    monkeypatch.setattr(mb, "_read_ahead", read_ahead)
    with pytest.raises(MemoryError):
        _run(tmp_path, "run", items, raw, mb.BatchConfig(chunk_size=4, writers=2))
    assert [t.name for t in threading.enumerate() if t.name.startswith("mask-")] == []


# --------------------------------------------------------------------------
# refusals: a pipeline needs a connection per worker
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pool",
    [
        {"poolclass": StaticPool},  # one connection for every session
        {"poolclass": QueuePool, "pool_size": 2, "max_overflow": 0},  # fewer than writers + 1
        # enough only with its overflow, whose connections are closed when they come back
        {"poolclass": QueuePool, "pool_size": 4, "max_overflow": 20},
    ],
)
def test_the_pipeline_is_refused_without_a_connection_per_worker(
    tmp_path: Path, pool: dict[str, Any]
) -> None:
    raw = _Raw(tmp_path / "raw")
    engine = _engine(tmp_path / "media.db", **pool)
    main = Session(engine, expire_on_commit=False)
    events: list[dict[str, Any]] = []
    rt = _runtime(main, tmp_path / "run", raw, _Shapes(), events)
    assert mb.run_batch([raw.add(100, 1)], mb.BatchConfig(writers=4), rt) == 2
    assert events[-1]["event"] == "refused"
    assert "--writers 0" in events[-1]["detail"]
    assert main.execute(select(MaskBatch)).first() is None
    main.close()
    engine.dispose()


def test_a_pool_that_opens_a_connection_per_checkout_runs_the_pipeline(tmp_path: Path) -> None:
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(100 + n % 3 * 8, n) for n in range(1, 5)]
    engine = _engine(tmp_path / "media.db", poolclass=NullPool)
    outcome = _run(tmp_path, "run", items, raw, mb.BatchConfig(writers=4), engine=engine)
    assert outcome["exit"] == 0


def test_a_chunk_at_the_command_lines_defaults_closes_no_pooled_connection(
    tmp_path: Path,
) -> None:
    """The live run reaches its database through a kubectl port-forward,
    which any closed connection kills: a chunk at the command line's
    defaults, on a pool like the media db's (10 kept, 20 more lent past
    them and closed when they come back), closes none of its connections."""
    raw = _Raw(tmp_path / "raw", cost=0.01)
    items = [raw.add(100 + n % 3 * 8, n) for n in range(1, 25)]
    engine = _engine(tmp_path / "media.db", poolclass=QueuePool, pool_size=10, max_overflow=20)
    closed: list[str] = []
    for name in ("close", "close_detached"):
        sa_event.listen(engine, name, lambda *_, name=name: closed.append(name))
    args = mb._parser().parse_args(["run", "--worklist", "w"])
    config = mb.BatchConfig(
        chunk_size=args.chunk_size, writers=args.writers, prefetch=args.prefetch
    )
    root = tmp_path / "run"
    main = Session(engine, expire_on_commit=False)
    rt = _runtime(main, root, raw, _Shapes(cost=0.03), [])
    assert mb.run_batch(items, config, rt) == 0
    main.close()
    assert closed == []
    records = [r for r in _lines(root / "progress.jsonl") if r["event"] == "item"]
    assert len(records) == 24
    engine.dispose()


# --------------------------------------------------------------------------
# the seconds: each stage the image's own wall time
# --------------------------------------------------------------------------


def test_each_stage_is_the_images_own_wall_and_the_chunk_tells_the_throughput(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[StageTimings] = []

    class Capturing(StageTimings):
        def __init__(self, clock: Callable[[], float]) -> None:
            super().__init__(clock)
            captured.append(self)

    monkeypatch.setattr(mb, "StageTimings", Capturing)
    raw = _Raw(tmp_path / "raw", cost=0.05)
    items = [raw.add(100 + n % 2 * 8, n) for n in range(1, 7)]
    root = tmp_path / "run"
    engine = _engine(tmp_path / "media.db")
    main = Session(engine, expire_on_commit=False)
    events: list[dict[str, Any]] = []
    shapes = _Shapes(cost=0.08)
    rt = _runtime(main, root, raw, shapes, events)
    assert mb.run_batch(items, mb.BatchConfig(chunk_size=6, writers=3), rt) == 0
    records = {r["blob_sha256"]: r for r in _lines(root / "progress.jsonl") if "status" in r}
    assert len(captured) == 6
    # what this machine made of the run's own calls: a loaded one stretches a 0.05 s sleep
    # to 0.25 s and more, so a stage is held to its own call's wall, not to a fixed window
    slowest_get, slowest_model = max(raw.store.get_walls), max(shapes.walls)
    assert len(raw.store.get_walls) == len(shapes.walls) == 6
    for timings in captured:
        seconds = timings.seconds
        # each its own: six images queue six model calls (about 3x one call's wall for the
        # last in line), which no image's fetch or inference holds (that is gpu_wait's): a
        # stage is its own call and a bounded overhead (its decode, its bookkeeping)
        assert 0.05 <= seconds["fetch"] < slowest_get + 0.1
        assert 0.08 <= seconds["inference"] < slowest_model + 0.1
        assert seconds["gpu_wait"] >= 0.0
        assert {"claim_keys", "claim_footprint", "claim_register"} <= set(seconds)
        assert {"seed", "readback", "upload", "db", "turn_wait"} <= set(seconds)
        assert "claim" not in seconds  # told apart (MASKRATE)
    for record in records.values():
        assert record["seconds"] >= 0.05 + 0.08
    staged = sorted(round(sum(t.seconds.values()), 2) for t in captured)
    walls = sorted(r["seconds"] for r in records.values())
    # the stages account for the image's wall, each second once: never above it (one clock, so on any
    # load), and short of it only by the bookkeeping between a stage and the clock, where a loaded
    # machine can deschedule a thread for 0.1 s and more whatever the wall: 0.1 s and a fifth of it
    for got, wall in zip(staged, walls, strict=True):
        assert got <= wall + 0.02
        assert got >= wall - (0.1 + 0.2 * wall)
    chunk = next(e for e in events if e["event"] == "chunk")
    # from the unrounded elapsed seconds: within a rounding of the reported ones
    assert chunk["throughput_seconds_per_image"] == pytest.approx(
        chunk["elapsed_seconds"] / 6, abs=0.001
    )
    assert chunk["throughput_seconds_per_image"] < chunk["seconds_per_image"]  # overlapped
    main.close()
    engine.dispose()


def test_the_claim_tells_its_keys_its_footprint_and_its_wait_for_earlier_images_apart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MASKRATE: the claim is three stages. ``claim_keys``: the picture keys
    its read-ahead computed; ``claim_footprint``: its footprint read but for
    the read's SQL ("db"): the pool's checkout, the ORM; ``claim_register``:
    its wait to register behind every earlier image, which waits for THEIR
    footprint reads. An earlier image's slow footprint is a later image's
    claim_register, never its claim_footprint, and the chunk line tells the
    three apart."""
    by_thread: dict[str, StageTimings] = {}

    class Capturing(StageTimings):
        def __init__(self, clock: Callable[[], float]) -> None:
            super().__init__(clock)
            by_thread[threading.current_thread().name] = self

    monkeypatch.setattr(mb, "StageTimings", Capturing)
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(100, n) for n in range(1, 4)]
    real = mb.seed_neighbourhood

    def footprint(db: Session, *, sha: str, digest: str | None) -> mask_seed.Neighbourhood:
        if sha == items[0].blob_sha256:
            time.sleep(0.6)  # the first image's read, slow
        return real(db, sha=sha, digest=digest)

    monkeypatch.setattr(mb, "seed_neighbourhood", footprint)
    root = tmp_path / "run"
    engine = _engine(tmp_path / "media.db")
    main = Session(engine, expire_on_commit=False)
    events: list[dict[str, Any]] = []
    rt = _runtime(main, root, raw, _Shapes(cost=0.01), events)
    assert mb.run_batch(items, mb.BatchConfig(chunk_size=3, writers=3), rt) == 0
    first, *later = (by_thread[f"mask-image-{n}"].seconds for n in range(3))
    assert first["claim_footprint"] >= 0.6
    for seconds in later:
        assert seconds["claim_register"] >= 0.3  # behind the first image's read
        assert seconds["claim_footprint"] < 0.3
    for seconds in (first, *later):
        assert seconds["claim_keys"] > 0  # the read-ahead's keys, added
        assert "claim" not in seconds
    chunk = next(e for e in events if e["event"] == "chunk")
    assert {"claim_keys", "claim_footprint", "claim_register"} <= set(
        chunk["stage_seconds_per_image"]
    )
    assert chunk["stage_seconds_per_image"]["claim_register"] >= 0.2
    main.close()
    engine.dispose()


def test_an_images_connection_checkout_is_its_db_never_its_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MASKRATE: the footprint read is a worker's first query of its image,
    so the pool's checkout (its pre-ping a round trip over the tunnel, no
    statement event around it) lands there. It is the image's "db", as
    unpipelined, never its claim_footprint."""
    by_thread: dict[str, StageTimings] = {}

    class Capturing(StageTimings):
        def __init__(self, clock: Callable[[], float]) -> None:
            super().__init__(clock)
            by_thread[threading.current_thread().name] = self

    def slow_checkout(*_: Any) -> None:
        if threading.current_thread().name.startswith("mask-image-"):
            time.sleep(0.3)

    monkeypatch.setattr(mb, "StageTimings", Capturing)
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(100, n) for n in range(1, 3)]
    root = tmp_path / "run"
    engine = _engine(tmp_path / "media.db")
    main = Session(engine, expire_on_commit=False)
    events: list[dict[str, Any]] = []
    rt = _runtime(main, root, raw, _Shapes(cost=0.01), events)
    sa_event.listen(QueuePool, "checkout", slow_checkout)  # the run's own sized pool too
    try:
        assert mb.run_batch(items, mb.BatchConfig(chunk_size=2, writers=2), rt) == 0
    finally:
        sa_event.remove(QueuePool, "checkout", slow_checkout)
    for n in range(2):
        seconds = by_thread[f"mask-image-{n}"].seconds
        assert seconds["claim_footprint"] < 0.25
        assert seconds["db"] >= 0.3
    main.close()
    engine.dispose()


# --------------------------------------------------------------------------
# a picture's keys and its neighbourhood in the database
# --------------------------------------------------------------------------


def test_picture_keys_are_the_keys_the_seed_registers(db_session: Session, tmp_path: Path) -> None:
    original = _photo(100, 1)
    keys = picture_keys(original)
    store = LocalObjectStore(tmp_path / "media")
    copy = seed_clean_image(
        db_session, store, original, origins=[ImageOrigin(url=ITEM.format(n=1))], figure_id="f1"
    )
    assert (keys.digest, keys.clean_sha256, keys.own_digest) == (
        copy.pixel_sha256,
        copy.sha256,
        copy.own_pixel_sha256,
    )


def test_the_footprint_reaches_every_capture_and_copy_of_a_picture(
    db_session: Session, tmp_path: Path
) -> None:
    store = LocalObjectStore(tmp_path / "media")
    one, again, other = _photo(100, 1), _photo(100, 1, note="again"), _photo(100, 7)
    for n, data in ((1, one), (2, again), (3, other)):
        seed_clean_image(
            db_session,
            store,
            data,
            origins=[ImageOrigin(url=ITEM.format(n=n))],
            figure_id=f"f{n}",
        )
    copy = db_session.execute(
        select(PublicObject.sha256).where(PublicObject.pixel_sha256 == picture_keys(one).digest)
    ).scalar_one()
    sha = {d: hashlib.sha256(d).hexdigest() for d in (one, again, other)}
    # the second capture, by its own sha alone, reaches the first and the copy
    found = seed_footprint(db_session, shas={sha[again]}, digests=set())
    assert {sha[one], sha[again], copy} <= found
    assert sha[other] not in found
    # a new picture reaches nothing but its own keys
    assert seed_footprint(db_session, shas={"f" * 64}, digests={"e" * 64}) == {"f" * 64, "e" * 64}


# --------------------------------------------------------------------------
# the pipeline's parts
# --------------------------------------------------------------------------


def test_claims_are_registered_in_order_and_join_related_images() -> None:
    claims = mp.Claims()
    since = claims.snapshot()
    order: list[int] = []

    def register(index: int, keys: set[str]) -> None:
        claims.register(index, keys, since)
        order.append(index)

    late = threading.Thread(target=register, args=(1, {"b"}))
    late.start()
    time.sleep(0.05)
    assert order == []  # 1 waits for 0
    first, waits = claims.register(0, {"a", "x"}, since)
    late.join(timeout=5)
    assert order == [1] and waits == []
    third, waits = claims.register(2, {"x"}, since)
    assert waits == [first]
    assert third.keys == {"a", "x"}
    claims.finish(first)
    assert first.done.is_set()
    claims.skip(3)
    fifth, waits = claims.register(4, {"a"}, since)
    assert waits == [third]
    assert fifth.keys == {"a", "x"}


def test_a_claim_finished_after_the_snapshot_still_lends_its_keys() -> None:
    """Its writes may be half visible to a footprint read meanwhile: its
    keys join (no wait), so later related images still wait for this one."""
    claims = mp.Claims()
    since = claims.snapshot()
    first, _ = claims.register(0, {"a", "b"}, since)
    claims.finish(first)
    second, waits = claims.register(1, {"b"}, since)
    assert waits == [] and second.keys == {"a", "b"}
    later = claims.snapshot()
    third, waits = claims.register(2, {"a"}, later)
    assert waits == [second] and third.keys == {"a", "b"}


def test_the_turn_is_one_image_at_a_time_in_the_order_they_get_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """From its first write after its judgment to its end an image holds the
    turn: no two images write at once, and an image behind a slow one in the
    work list writes first."""
    log: list[tuple[str, int]] = []
    real_turn, real_close = mb._ImageLane.turn, mb._ImageLane.close

    def turn(lane: Any) -> None:
        taken = lane._turned
        real_turn(lane)
        if not taken:
            log.append(("in", lane._index))

    def close(lane: Any) -> None:
        if lane._turned:
            log.append(("out", lane._index))
        real_close(lane)

    monkeypatch.setattr(mb._ImageLane, "turn", turn)
    monkeypatch.setattr(mb._ImageLane, "close", close)
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(100 + n % 3 * 8, n) for n in range(1, 6)]
    _records, _events, engine = _slowed(
        tmp_path, items, raw, items[0], mb.BatchConfig(chunk_size=5, writers=3)
    )
    engine.dispose()
    assert [step for step, _index in log] == ["in", "out"] * 5  # never two at once
    assert [index for _step, index in log[::2]][-1] == 0  # the slow first image writes last


def test_a_gpu_job_runs_on_the_serving_thread_and_returns_its_answer_or_error() -> None:
    jobs = [mp.GpuJob(threading.get_ident), mp.GpuJob(lambda: 1 / 0), mp.GpuJob(lambda: "x")]
    answers: list[Any] = [None, None, None]

    def wait(index: int) -> None:
        try:
            answers[index] = jobs[index].wait()
        except BaseException as exc:  # the test looks at it
            answers[index] = exc

    callers = [threading.Thread(target=wait, args=(i,)) for i in range(3)]
    for caller in callers:
        caller.start()
    jobs[0].run()
    jobs[1].run()
    jobs[2].refuse()
    for caller in callers:
        caller.join(timeout=5)
    assert answers[0] == threading.get_ident()  # run here, on the serving thread
    assert isinstance(answers[1], ZeroDivisionError)
    assert isinstance(answers[2], mp.Held)


def test_a_gpu_job_the_serving_thread_dies_in_lets_its_caller_go() -> None:
    def interrupted() -> None:
        raise KeyboardInterrupt

    job = mp.GpuJob(interrupted)
    with pytest.raises(KeyboardInterrupt):
        job.run()
    with pytest.raises(mp.Held):
        job.wait()


def test_held_is_no_exception_a_lane_could_catch() -> None:
    assert not issubclass(mp.Held, Exception)


def test_the_gpu_proxy_is_the_model_by_name_and_waits_as_gpu_wait() -> None:
    served: list[Callable[[], Any]] = []

    def submit(call: Callable[[], Any]) -> Any:
        served.append(call)
        return call()

    model = _Shapes()
    proxy = mp.on_the_gpu(model, submit)
    assert (proxy.name, proxy.publishes_masks, proxy.active_provider) == (
        "shapes",
        True,
        "FakeExecutionProvider",
    )
    rgba = Image.new("RGBA", (100, H))
    timings = StageTimings(time.monotonic)
    assert proxy.predict_alpha_timed(rgba, timings).shape == (H, 100)
    assert set(timings.seconds) == {"gpu_wait", "inference"}
    assert proxy.predict_alpha(rgba).shape == (H, 100)
    assert len(served) == 2
    stub = mp.on_the_gpu(StubMattingBackend(), submit)
    assert stub.matte(rgba).size == (100, H)
    assert stub.name == StubMattingBackend.name and not stub.publishes_masks
    assert len(served) == 3


# --- LG-U11: each chunk is an item (fc-shared server/job: one trace per item) ---------------


@pytest.mark.parametrize("writers", [0, 2])
def test_each_chunk_is_an_item_its_images_and_its_line_in_its_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, writers: int
) -> None:
    """A run is one root trace; each chunk is an ITEM, a new root trace linked
    to the run: its images are minted in it (on the driver, or on the
    pipeline's worker threads) and its chunk line is written in it. The lines
    around the chunks (start, done) stay in the run's trace."""
    from app.telemetry import tracing as tr
    from app.telemetry.log import Logger
    from opentelemetry import trace

    def current() -> str:
        return f"{trace.get_current_span().get_span_context().trace_id:032x}"

    root, raw = tmp_path / "root", _Raw(tmp_path / "raw")
    items = [raw.add(120 + n, n) for n in range(4)]
    minted: list[str] = []
    real_mint = mb._mint

    def mint(*args: Any, **kwargs: Any) -> Any:
        minted.append(current())
        return real_mint(*args, **kwargs)

    monkeypatch.setattr(mb, "_mint", mint)
    engine = _engine(tmp_path / "media.db")
    main = Session(engine, expire_on_commit=False)
    events: list[tuple[str, str]] = []
    lines: list[dict[str, Any]] = []
    log = Logger(service="mask-batch", env={}, sink=lambda s: lines.append(json.loads(s)))
    tr.start_tracing("mask-batch", env={})

    def body(run: tr.JobRun) -> int:
        rt = _runtime(main, root, raw, _Shapes(), [])
        rt.emit = lambda e: events.append((e["event"], current()))
        rt.open_item = lambda n: run.open_item(name="item.chunk", attributes={"fc.chunk": n})
        return mb.run_batch(items, mb.BatchConfig(chunk_size=2, writers=writers), rt)

    try:
        assert tr.run_job("mask-batch", body, logger=log) == 0
    finally:
        main.close()
        engine.dispose()
    run_id = lines[0]["trace_id"]
    chunks = [t for e, t in events if e == "chunk"]
    assert len(chunks) == 2 and len({run_id, *chunks}) == 3
    assert sorted(minted) == sorted([chunks[0]] * 2 + [chunks[1]] * 2)
    assert {t for e, t in events if e in ("start", "done")} == {run_id}


def test_a_chunk_left_open_by_a_failure_still_ends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An item the run opened is ended (failed) when the batch stops on a fault."""
    ended: list[bool] = []

    class Item:
        def run(self, fn: Callable[[], Any]) -> Any:
            return fn()

        def end(self, error: bool = False) -> None:
            ended.append(error)

    def broken(*_: Any, **__: Any) -> Any:
        raise RuntimeError("the bucket is gone")

    monkeypatch.setattr(mb, "_mint", broken)
    root, raw = tmp_path / "root", _Raw(tmp_path / "raw")
    engine = _engine(tmp_path / "media.db")
    main = Session(engine, expire_on_commit=False)
    rt = _runtime(main, root, raw, _Shapes(), [])
    rt.open_item = lambda n: Item()
    try:
        with pytest.raises(RuntimeError):
            mb.run_batch([raw.add(130, 1)], mb.BatchConfig(chunk_size=1), rt)
    finally:
        main.close()
        engine.dispose()
    assert ended == [True]


# --------------------------------------------------------------------------
# MASK5: the CPU stages on worker processes, and the keys read ahead (SHOULD-1)
# --------------------------------------------------------------------------


class _Recording(cs.CpuStages):
    """The CPU stages inline, each call noted: its function and its pixels.
    A pool's stand-in: with workers, the mask is judged on them."""

    workers = 1

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.calls: list[tuple[str, int | None]] = []

    def run(self, fn: Callable[..., Any], *args: Any, pixels: int | None = None) -> Any:
        with self.lock:
            self.calls.append((fn.__name__, pixels))
        return fn(*args)


def _encodes(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every clean-copy encode, by the thread that made it."""
    made: list[str] = []
    real = mask_seed._encode

    def encode(picture: Image.Image, original: bytes) -> tuple[bytes, str]:
        made.append(threading.current_thread().name)
        return real(picture, original)

    monkeypatch.setattr(mask_seed, "_encode", encode)
    return made


def _statuses(outcome: dict[str, Any]) -> list[str]:
    return [r["status"] for r in outcome["progress"] if r["event"] == "item"]


def test_a_new_picture_is_encoded_once_and_a_picture_with_a_copy_never(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SHOULD-1 (the MASKRATE challenge): the read-ahead computes the
    picture's pixel digest alone. The clean copy is encoded once, in the
    image's claim, only when no copy of its picture is registered, and the
    seed mints the copy from that encode; a picture with a copy is never
    encoded."""
    made = _encodes(monkeypatch)
    raw = _Raw(tmp_path / "raw")
    engine = _engine(tmp_path / "media.db")
    config = mb.BatchConfig(chunk_size=1, writers=2)
    first = _run(tmp_path, "first", [raw.add(100, 1)], raw, config, engine=engine)
    assert _statuses(first) == ["ok"]
    assert made == ["mask-image-0"]  # in its claim, on its worker: never read ahead
    made.clear()
    again = raw.add(100, 2, pixels_of=1)  # the same picture, other bytes: its copy is known
    second = _run(tmp_path, "again", [again], raw, config, engine=engine)
    assert _statuses(second) == ["already"]
    assert made == []


def test_the_pipelines_cpu_stages_are_calls_on_its_cpu_stages(tmp_path: Path) -> None:
    """Read ahead: the picture's digest. In the claim: what the seed will
    need and the database does not have (the clean copy of a new picture,
    the perceptual hash of a new original), nothing when it has both. After
    the model: the mask's judgment and its encode. Each sized by its
    picture's pixels."""
    stages = _Recording()
    raw = _Raw(tmp_path / "raw")
    engine = _engine(tmp_path / "media.db")
    config = mb.BatchConfig(chunk_size=4, writers=2)
    items = [
        raw.add(100, 1),  # new: ok
        raw.add(100, 2, pixels_of=1),  # a copy known, the original not: already
        raw.add(102, 3),  # new: empty
        raw.add(107, 4, url=PICTURE),  # new: ineligible, no model
    ]
    outcome = _run(tmp_path, "run", items, raw, config, cpu=stages, engine=engine)
    assert sorted(_statuses(outcome)) == ["already", "empty", "ineligible", "ok"]  # as they end
    assert sorted(stages.calls) == sorted(
        [
            *(("picture_digest", w * H) for w in (100, 100, 102, 107)),
            *(("seed_inputs", w * H) for w in (100, 100, 102, 107)),
            *(("judge_mask", w * H) for w in (100, 102)),
        ]
    )
    stages.calls.clear()
    rerun = raw.add(100, 1)  # the first original again, its progress gone: both known
    outcome = _run(tmp_path, "rerun", [rerun], raw, config, cpu=stages, engine=engine)
    assert _statuses(outcome) == ["already"]
    assert stages.calls == [("picture_digest", 100 * H)]


def test_without_cpu_workers_the_mask_is_judged_where_its_copy_was_decoded(
    tmp_path: Path,
) -> None:
    """``--cpu-workers 0``, pipelined: the judgment stays on the image's
    thread, on the copy decoded for the model, as before MASK5; it is never
    handed to the inline stages, which would decode the copy a second time."""

    class _Inline(_Recording):
        workers = 0

    stages = _Inline()
    raw = _Raw(tmp_path / "raw")
    items = [raw.add(100, 1), raw.add(102, 2)]
    outcome = _run(tmp_path, "run", items, raw, mb.BatchConfig(chunk_size=2, writers=2), cpu=stages)
    assert sorted(_statuses(outcome)) == ["empty", "ok"]
    assert sorted(name for name, _ in stages.calls) == [
        "picture_digest",
        "picture_digest",
        "seed_inputs",
        "seed_inputs",
    ]


def test_the_pipeline_on_cpu_worker_processes_writes_what_one_image_at_a_time_writes(
    tmp_path: Path,
) -> None:
    raw = _Raw(tmp_path / "raw")
    items = _mixed(raw)
    alone = _run(tmp_path, "alone", items, raw, mb.BatchConfig(chunk_size=4), faults=True)
    stages = cs.ProcessStages(2)
    try:
        config = mb.BatchConfig(chunk_size=4, writers=4)
        piped = _run(tmp_path, "piped", items, raw, config, faults=True, cpu=stages)
    finally:
        stages.close()
    assert stages.replaced == 0
    assert _unordered(piped) == _unordered(alone)


def test_the_mask_encode_in_a_worker_process_is_its_own_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[StageTimings] = []

    class Capturing(StageTimings):
        def __init__(self, clock: Callable[[], float]) -> None:
            super().__init__(clock)
            captured.append(self)

    monkeypatch.setattr(mb, "StageTimings", Capturing)
    raw = _Raw(tmp_path / "raw")
    stages = cs.ProcessStages(1)
    try:
        config = mb.BatchConfig(chunk_size=1, writers=1)
        outcome = _run(tmp_path, "run", [raw.add(100, 1)], raw, config, cpu=stages)
    finally:
        stages.close()
    assert _statuses(outcome) == ["ok"]
    (seconds,) = (t.seconds for t in captured)
    assert seconds["encode"] > 0 and seconds["analysis"] > 0 and seconds["claim_keys"] > 0


def test_the_picture_digest_is_the_pixel_digest_or_none(tmp_path: Path) -> None:
    one = _photo(100, 1)
    assert mask_seed.picture_digest(one) == mask_seed.pixel_digest(one)
    assert mask_seed.picture_digest(b"not a picture") is None


def test_the_neighbourhood_tells_whether_the_original_and_a_copy_of_its_picture_are_known(
    db_session: Session, tmp_path: Path
) -> None:
    store = LocalObjectStore(tmp_path / "media")
    one, again = _photo(100, 1), _photo(100, 1, note="again")
    sha = {d: hashlib.sha256(d).hexdigest() for d in (one, again)}
    digest = mask_seed.pixel_digest(one)
    near = mask_seed.seed_neighbourhood(db_session, sha=sha[one], digest=digest)
    assert (near.image_known, near.copy_known, near.keys) == (False, False, {sha[one], digest})
    seed_clean_image(
        db_session, store, one, origins=[ImageOrigin(url=ITEM.format(n=1))], figure_id="f1"
    )
    near = mask_seed.seed_neighbourhood(db_session, sha=sha[one], digest=digest)
    assert (near.image_known, near.copy_known) == (True, True)
    assert near.keys == seed_footprint(db_session, shas={sha[one]}, digests={digest})
    near = mask_seed.seed_neighbourhood(db_session, sha=sha[again], digest=digest)
    assert (near.image_known, near.copy_known) == (False, True)
    assert sha[one] in near.keys
    # a picture that will not decode: its byte sha alone
    near = mask_seed.seed_neighbourhood(db_session, sha="f" * 64, digest=None)
    assert (near.image_known, near.copy_known, near.keys) == (False, False, {"f" * 64})


def test_an_original_whose_bytes_are_a_mask_has_no_copy_of_its_picture_yet(
    db_session: Session, tmp_path: Path
) -> None:
    """Only a SANITIZED object behind its sha or digests is a copy the seed
    links to (``_copies_showing``); a mask with its bytes is not."""
    store = LocalObjectStore(tmp_path / "media")
    data = _photo(100, 5)
    sha = hashlib.sha256(data).hexdigest()
    mask_seed.register_public_object(
        db_session, sha256=sha, kind="mask", mime="image/png", size=len(data),
        width=100, height=H, storage_key=f"masks/{sha}.png",
    )  # fmt: skip
    db_session.commit()
    near = mask_seed.seed_neighbourhood(db_session, sha=sha, digest=mask_seed.pixel_digest(data))
    assert (near.image_known, near.copy_known) == (False, False)
    assert sha in near.keys
    del store


def test_a_seed_from_inputs_computed_ahead_writes_what_a_seed_from_the_bytes_writes(
    tmp_path: Path,
) -> None:
    """The claim's inputs (MASK5) are the seed's own values: seeded from
    them or from the bytes alone, the same rows and the same objects."""
    pictures = [_photo(100, 1), _photo(100, 1, note="again"), _photo(104, 2)]
    outcome: dict[str, Any] = {}
    for name in ("bytes", "inputs"):
        engine = _engine(tmp_path / f"{name}.db")
        store = LocalObjectStore(tmp_path / name)
        with Session(engine, expire_on_commit=False) as db:
            for n, data in enumerate(pictures):
                inputs = None
                if name == "inputs":
                    digest = mask_seed.picture_digest(data)
                    assert digest is not None
                    near = mask_seed.seed_neighbourhood(
                        db, sha=hashlib.sha256(data).hexdigest(), digest=digest
                    )
                    inputs = mask_seed.seed_inputs(
                        data, digest, clean=not near.copy_known, phash=not near.image_known
                    )
                    assert (inputs.clean is None, inputs.phash is None) == (
                        n == 1,  # its picture's copy is the first's
                        False,
                    )
                seed_clean_image(
                    db, store, data, origins=[ImageOrigin(url=ITEM.format(n=n))],
                    figure_id=f"f{n}", inputs=inputs,
                )  # fmt: skip
        outcome[name] = (_canonical(engine), _media(tmp_path / name))
        engine.dispose()
    assert outcome["inputs"] == outcome["bytes"]


def test_a_seed_whose_inputs_lack_what_it_needs_computes_it(tmp_path: Path) -> None:
    """A copy the claim saw that is gone by the seed (never, in practice:
    copies are not deleted), or an original the claim saw recorded: the seed
    computes what its inputs lack from the bytes, the same values."""
    data = _photo(100, 9)
    rows = {}
    for name, inputs in (
        ("bytes", None),
        ("bare", mask_seed.SeedInputs(digest=mask_seed.pixel_digest(data))),
    ):
        engine = _engine(tmp_path / f"{name}.db")
        with Session(engine, expire_on_commit=False) as db:
            seed_clean_image(
                db, LocalObjectStore(tmp_path / name), data,
                origins=[ImageOrigin(url=ITEM.format(n=9))], figure_id="f9", inputs=inputs,
            )  # fmt: skip
        rows[name] = (_canonical(engine), _media(tmp_path / name))
        engine.dispose()
    assert rows["bare"] == rows["bytes"]
