"""MASK-BATCH-1: mint the cut-out masks in gated, resumable chunks on a GPU box.

    python -m app.cli.mask_batch [--sandbox DIR] run --worklist FILE [--worklist FILE ...]
        [--progress FILE] [--chunk-size N] [--limit N] [--retry-failed] [--dry-run]
        [--backend birefnet-basekept|birefnet] [--device cuda|auto|cpu] [--cpu-threads N]
        [--gpu-index N] [--gpu-util-max PCT] [--gpu-free-min-mib MIB] [--gpu-samples N]
        [--busy-windows SPEC] [--residential-ip IP|CIDR | --no-residential-check]
        [--ip-echo-url URL] [--residential-ttl S] [--poll-seconds S]
        [--max-wait-seconds S] [--max-consecutive-failures N] [--raw-dir DIR]
        [--writers N] [--prefetch N] [--decode-slots N] [--cpu-workers N]
    python -m app.cli.mask_batch sql ross-heads --user UUID
    python -m app.cli.mask_batch sql ross-worklist --heads FILE
    python -m app.cli.mask_batch sql catalog-worklist [--shard I/N]
    python -m app.cli.mask_batch sql official-gallery-worklist [--shard I/N]
    python -m app.cli.mask_batch sql stores-worklist --since INSTANT [--scope showable|all]
        [--sites a,b,...] [--rank N]

THE WORK. A work list is JSON lines, one per ORIGINAL the spine captured
(``blob_sha256``, its ``storage_key`` in the raw bucket, ``codec``, the origin
``url`` and ``source``, its ``source_class`` (GAL3: kept on the gallery link,
app.workers.mask_policy), ``product_id``, ``head_id``, ``rank``): Ross's
holdings ranked 0 (owned), 1 (ordered), 2 (wished), the rest of the catalog 9.
The ``sql`` commands print the read-only queries that produce it (run by the
operator against the coordinator and spine databases; psql output is fine,
non-JSON lines are skipped). Work lists are merged in rank order, each original
once at its lowest rank.

PER ORIGINAL, in one chunk: skipped when a clean copy behind it already has a
mask result (``original_settled``: idempotent per image sha, its bytes are not
even read); else its bytes are read from the raw bucket (READ ONLY: the store
refuses a write) and checked against its sha; the clean copy is seeded
(``seed_clean_image``: original private, copy public) and its mask settled
(``settle_clean_copy``) in the media database and bucket. A failure is that
image's alone and is named (``fetch:<class>``, ``raw_hash_mismatch``,
``unsupported_codec:<codec>``, ``seed:<class>``, ``settle:<class>``, or the
lane's own reason). An OUTAGE-class failure (``is_outage``: a dropped
connection, a timeout, the database's OperationalError, the bucket's 5xx) is
recorded ``interrupted``: a plain rerun retries it (its failed media-db row
too), and ``--max-consecutive-failures`` of them in a row stop the batch. An
image's OWN fault (its codec, its hash, NoSuchKey, an undecodable picture,
the model's error) stays ``failed``: neither counted toward that streak nor
resetting it, and retried only with ``--retry-failed``.

THE GATE, before EVERY chunk (Ross 2026-10-12): not inside a scraper roll
window (``--busy-windows``, minutes of the hour, default :03-:26, :41-:43,
:50-:56; a chunk that runs into one stops before its next image); the GPU idle
(utilisation < 5 %) OR at least 9 GiB of it free, on every one of
``--gpu-samples`` readings -- while the model is held, no other consumer
instead: the card's memory not grown more than 512 MiB since the model was
measured and its utilisation not at or above 5 % on every reading, taken 2 s
after the chunk (``GpuGate``); and the box on its residential address (a VPN
window moves it; ``--residential-ip`` takes an address or the ISP's block).
While the gate fails the batch waits, polling every
``--poll-seconds``, with the model released (a game gets the GPU back; it is
loaded again, ~4 s, when the gate passes).

THE RECORD. Every image's result is appended to ``--progress`` (JSON lines,
flushed per line: a power cut loses at most the images in flight, one or,
pipelined, up to ``--writers``; each reruns idempotently), and a run
skips what it lists (``--retry-failed`` re-drives its failures). A run is one
root span in the estate's log lines on stdout (LG-U11, app.telemetry):
job.start, one ``app.<event>`` line per event, job.end with the counts and
``exit``; each chunk is an ITEM (fc-shared's job model: a new root trace
linked to the run) in which its images are minted and its line written, and
every line carries run_id. Per chunk one ``app.chunk`` line: count, statuses, seconds per image (and the model's
share, and per MASKED image -- ok or held -- apart from the cheap ineligible ones), peak GPU memory used (whole card, sampled), the failures,
and where the seconds went: each stage's mean and median per image over the
images that ran it (``stage_seconds_per_image``, ``stage_median_seconds``;
app.workers.stage_timing: fetch, claim_keys, claim_footprint, claim_register,
seed, readback, gpu_wait, inference,
base_keep, analysis, encode, upload, turn_wait, db; a stage no image ran is
absent) and how many
did (``stage_images``): each mean is over those images only, so the means
add up to the seconds per image only when every image ran every stage. ``--dry-run``
reads the work list and the progress file, checks the gate once and prints the
plan: no database, no bucket, no model, nothing written.

THE PIPELINE (MASK-PERF-2, rolling since MASK-PERF-3). The images run
``--writers`` at a time (default 4), each on a worker with a database
session of its own, and the next ``--prefetch`` originals (default 2) are
read, decoded and hash-checked ahead; the model runs on the driver's thread
alone, one image at a time, so one image's I/O (its read, its seed, its
judgment, its uploads and rows) overlaps another's model run, and an image's
original and clean copy upload beside its seed's SQL (both land before its
rows commit). The pipeline ROLLS across chunks: a chunk is its batch row,
its gate and its report, not a barrier, so the next chunk's images start
while the one before still has images in flight (its gate is read on the
driver's thread, no model call running meanwhile), and a chunk reports once
its last image ended (one held up by a slow image after the next). Each
image's own work runs in the same order as before, and what it writes is
what one image at a time writes: images of one picture (their originals,
captures, copies and gallery link meet: app.workers.mask_pipeline) wait for
each other in work-list order, and from its first write after its judgment
to its commit an image holds the TURN, so the batch rows (counts, and the
cursor: the last image to commit) and a mask two images share are written
one image at a time, in the order the images get there. Each image is
recorded as it ends, once its writes committed: a slow image keeps its own
worker and no other. A stop, a scraper window, a failing gate or the
circuit lets no further image start or onto the GPU: the images past the
GPU finish, the others are let go unrecorded and come back first. Each
image's stages are its own wall time, the pipeline's waits among them
(``claim_register``, ``gpu_wait``, ``turn_wait``); the chunk's
``throughput_seconds_per_image`` is its wall time over the images it minted
(an 'already' one, a lookup, is left out, as in the seconds per image). At
most ``--decode-slots`` copies (default three) are decoded for the model at
once, whatever the writers, and an original over 4000x3000 takes a slot per
4000x3000 begun (the memory bound). The database pool keeps a connection
for every worker and the batch and lends none past them, and each bucket
client keeps one for every PUT the workers have in flight (two each) and
every original read ahead: no connection is closed while the run lasts (a
close kills the kubectl port-forward the live run reaches its database
through). ``--writers 0`` runs one image at a time on the driver's own
session, chunk after chunk.

THE CPU STAGES (MASK5). Pipelined, each image's CPU work -- its picture's
digest read ahead, its clean copy's encode and its original's perceptual
hash in its claim (only when the database has neither: SHOULD-1), its
mask's judgment and encode after the model -- runs on ``--cpu-workers``
processes (app.workers.cpu_stages: forked from a clean fork server, the
decoded pixels in them at once bounded by size), the same code on the same
bytes; ``--cpu-workers 0``, and one image at a time, run it on the image's
own thread. The default is the host's CPUs less two, at most ``--writers``
+ ``--prefetch`` and 16. A worker the pool replaced (one that died) is an
``app.cpu_pool_replaced`` line.

Exit status: 0 done, 1 done with failures, 2 refused, 3 too many consecutive
failures, 4 waited longer than ``--max-wait-seconds``, 130 stopped (SIGINT or
SIGTERM: the image in flight is finished first; pipelined, the images past
the GPU are, the others are let go unrecorded and come back first). Before
an exit 3 the ``app.circuit_open`` line says why (MASK5): the streak's first
outage (``cause_exception``, ``cause_message``, ``cause_connection``:
database, raw_bucket or media_bucket, ``cause_stage``, ``cause_blob_sha256``,
``cause_at``), its outages by connection and class (``streak_causes``), and
the last image that got through (``last_success_chunk``,
``last_success_blob_sha256``, ``last_success_at``); job.end repeats its
scalars.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import functools
import hashlib
import io
import ipaddress
import itertools
import json
import math
import os
import queue
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import Any

import httpx
from PIL import Image as PILImage
from sqlalchemy import Engine, create_engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.orm.attributes import set_committed_value
from sqlalchemy.pool import QueuePool, SingletonThreadPool, StaticPool

from .. import telemetry
from ..config import get_settings
from ..models import MaskBatch
from ..object_store import ObjectStore, S3ObjectStore
from ..telemetry.events import EventLines
from ..telemetry.tracing import NULL_ITEM, Item
from ..workers.cpu_stages import (
    INLINE,
    MAX_CPU_WORKERS,
    CpuStages,
    ProcessStages,
    default_cpu_workers,
)
from ..workers.image_limits import ImageLimitError
from ..workers.mask_job import (
    finish_mask_batch,
    open_mask_batch,
    original_settled,
    settle_clean_copy,
)
from ..workers.mask_pipeline import Claim, Claims, GpuJob, Held, Slots, on_the_gpu
from ..workers.mask_policy import ImageOrigin
from ..workers.mask_seed import (
    Neighbourhood,
    SeedInputs,
    picture_digest,
    seed_clean_image,
    seed_footprint,
    seed_inputs,
    seed_neighbourhood,
    seed_with_retry,
)
from ..workers.matting import BiRefNetMattingBackend, IsNetMattingBackend, MattingBackend
from ..workers.raw_source import (
    RawStoreConfigError,
    ReadOnlyStore,
    is_outage,
    is_outage_reason,
    raw_store_from_env,
)
from ..workers.raw_source import decode_stored as _decode
from ..workers.stage_timing import STAGES, DatabaseSeconds, StageTimings, timed

#: The scraper's roll windows (minutes of the hour): no chunk runs in them.
DEFAULT_BUSY_WINDOWS = "03-26,41-43,50-56"
#: Ross's gate: the GPU is idle below this utilisation (percent)...
DEFAULT_GPU_UTIL_MAX = 5
#: ...or has at least this much memory free (MiB; 9 GiB, "the comfortable mark").
DEFAULT_GPU_FREE_MIN_MIB = 9 * 1024
#: While the model is held: the seconds the gate waits after a chunk before
#: its readings (the batch's own compute has left them by then)...
DEFAULT_GPU_SETTLE_SECONDS = 2.0
#: ...and the growth of the card's used memory (MiB) that means another consumer.
DEFAULT_GPU_GROWTH_MAX_MIB = 512
#: The most of the card's growth that counts as the batch's own model (MiB).
#: Measured on the RTX 4070 Ti: ~9,470 MiB (9.27-9.48 GiB) with BiRefNet plus its
#: two IS-Net donors loaded. The caps are far higher (BiRefNet 8 GiB + two IS-Net
#: 2 GiB each = 12 GiB), so this is an empirical bound (8 GiB + 2 GiB + 512 MiB,
#: ~1.3 GB over the measurement), not the sum of the caps; growth beyond it counts
#: as another consumer's.
OWN_MAX_MIB = (
    (BiRefNetMattingBackend.GPU_MEM_LIMIT + IsNetMattingBackend.GPU_MEM_LIMIT) >> 20
) + 512
DEFAULT_IP_ECHO_URL = "https://api.ipify.org"
#: MASK-PERF-2: the images a chunk runs at once, each on a worker with its own
#: database session (0: one image at a time, the driver's own session); the
#: live chunk tunes it with ``--writers``. Simulated at the live costs: 5.5 s
#: an image one at a time, 2.75 s with 2 writers, 1.78 s with 4, 1.21 s with 8,
#: 1.05 s with 12. MASK-PERF-3, rolling, on a simulator calibrated to the live
#: chunks (1,363 images an hour at 8 writers before it, 1,319 measured live):
#: 2,877 at 8, about 5,500 at 16 with 100-image chunks (3 or 4 decode slots, 2
#: or 4 read ahead alike), 6,426 at 23 (the turn, ~6 round trips an image, is
#: then the queue). The command
#: line's pool keeps writers + 1 connections and lends none past them
#: (``_sized``), so the most writers keep the run at 24 connections to the
#: production database...
DEFAULT_WRITERS = 4
MAX_WRITERS = 23
#: ...and the originals read, decoded and hash-checked ahead of them: two
#: keep the next worker's ready, and each one read ahead is decoded once more
#: (4000x3000: 2.6 GB peak at 4 writers with 2, 3.3 GB with 4, same speed).
DEFAULT_PREFETCH = 2
MAX_PREFETCH = 16
#: The images whose copies are decoded for the model at once, whatever the
#: writers: by default the one on the GPU and two waiting
#: (app.workers.mask_pipeline.Slots); ``--decode-slots`` sizes it to the memory
#: budget. Each holds ~0.55 GB at its peak at 4000x3000...
DECODED_AT_ONCE = 3
MAX_DECODE_SLOTS = 8
#: ...so an original over that size takes a slot per unit of it (MASK-PERF-3:
#: the memory guard; the largest the image limits let in, 64 MP, takes six),
#: up to every slot: it is then decoded alone.
DECODE_UNIT_PIXELS = 4000 * 3000

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_MAX_RANK = 99
#: product_image.source_class's vocabulary (the spine's CHECK, fc-aggregation 0027).
SOURCE_CLASSES = ("manufacturer_press", "retailer_studio", "user_photo", "unknown")
#: The recorded statuses a run does not count as done.
_REDO = ("failed", "interrupted")
#: The statuses of an image that got a mask (published, or held for review).
_MASKED = ("ok", "held")
#: The statuses of an image the model ran on (its arena is full afterwards).
_RAN_MODEL = ("ok", "held", "empty")


# --------------------------------------------------------------------------
# the work list
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkItem:
    """One original the spine captured, as the work-list SQL emits it."""

    blob_sha256: str
    storage_key: str
    url: str
    product_id: str
    source: str = "mfc"
    head_id: str | None = None
    rank: int = 9
    codec: str = "identity"
    #: The spine's product_image.source_class (GAL3: ``manufacturer_press``
    #: = MFC's official gallery list); None when the capture lane gave none.
    source_class: str | None = None

    def as_row(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _field(row: dict[str, Any], key: str, where: str, *, required: bool = True) -> str | None:
    value = row.get(key)
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where}: {key} must be a non-empty string")
    return value


def _item(row: dict[str, Any], where: str) -> WorkItem:
    sha = row.get("blob_sha256")
    if not isinstance(sha, str) or not _SHA256.match(sha):
        raise ValueError(f"{where}: blob_sha256 must be 64 lowercase hex characters")
    rank = row.get("rank", 9)
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank <= _MAX_RANK:
        raise ValueError(f"{where}: rank must be an integer 0-{_MAX_RANK}")
    codec = row.get("codec", "identity")
    if not isinstance(codec, str):
        raise ValueError(f"{where}: codec must be a string")
    source_class = row.get("source_class")
    if source_class is not None and source_class not in SOURCE_CLASSES:
        raise ValueError(f"{where}: source_class must be one of {', '.join(SOURCE_CLASSES)}")
    return WorkItem(
        blob_sha256=sha,
        storage_key=str(_field(row, "storage_key", where)),
        url=str(_field(row, "url", where)),
        product_id=str(_field(row, "product_id", where)),
        source=str(_field(row, "source", where)),
        head_id=_field(row, "head_id", where, required=False),
        rank=rank,
        codec=codec,
        source_class=source_class,
    )


def _parse(
    entries: Iterable[tuple[str, int, str]], *, every_row: bool = False
) -> tuple[list[WorkItem], int]:
    best: dict[str, tuple[int, WorkItem]] = {}
    rows: list[WorkItem] = []
    skipped = 0
    for order, (label, n, line) in enumerate(entries):
        text = line.strip()
        if not text:
            continue
        if not text.startswith("{"):
            skipped += 1  # psql headers, footers, the runner's banner
            continue
        where = f"{label} line {n}"
        try:
            row = json.loads(text)
        except json.JSONDecodeError:
            raise ValueError(f"{where}: not a JSON object") from None
        if not isinstance(row, dict):  # pragma: no cover - "{" always decodes to a dict
            raise ValueError(f"{where}: not a JSON object")
        item = _item(row, where)
        rows.append(item)
        held = best.get(item.blob_sha256)
        if held is None or item.rank < held[1].rank:
            best[item.blob_sha256] = (order, item)
    if every_row:
        return rows, skipped
    ordered = sorted(best.values(), key=lambda kept: (kept[1].rank, kept[0]))
    return [item for _, item in ordered], skipped


def parse_worklist(lines: Iterable[str], label: str = "work list") -> tuple[list[WorkItem], int]:
    """``(items, skipped)``: the originals in rank order (file order within a
    rank), each once at its lowest rank; ``skipped`` counts the non-blank
    lines that are not JSON objects. A JSON row that is not a valid item is
    refused (ValueError naming its line)."""
    return _parse((label, n, line) for n, line in enumerate(lines, 1))


def read_worklists(paths: Sequence[Path], *, every_row: bool = False) -> tuple[list[WorkItem], int]:
    """As ``parse_worklist``, over the files in order; ``every_row`` keeps
    every valid row in file order instead (one per depiction: ``masks
    official-gallery`` judges each link, not each original once)."""

    def entries() -> Iterable[tuple[str, int, str]]:
        for path in paths:
            for n, line in enumerate(Path(path).read_text().splitlines(), 1):
                yield str(path), n, line

    return _parse(entries(), every_row=every_row)


# --------------------------------------------------------------------------
# the gates
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GateResult:
    ok: bool
    gate: str
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class GpuReading:
    util: int
    free_mib: int
    used_mib: int
    total_mib: int


def nvidia_smi_query(index: int) -> Callable[[], str]:
    """A reader of one GPU's utilisation and memory, as nvidia-smi prints them."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.free,memory.used,memory.total",
        "--format=csv,noheader,nounits",
        "-i",
        str(index),
    ]

    def query() -> str:
        done = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=15)
        return str(done.stdout).strip()

    return query


class GpuGate:
    """Ross's rule (2026-10-12): the GPU is free for a chunk when its
    utilisation is below ``util_max`` percent OR at least ``free_min_mib`` of
    its memory is free, on each of ``samples`` readings ``interval`` seconds
    apart (one failing reading fails the gate). An unreadable GPU fails it
    (closed); with ``absent_ok`` (``--device auto``) a box with no nvidia-smi
    at all passes: the model then runs on the CPU, and there is no GPU to
    protect.

    That rule holds until the model is loaded. WHILE it is held, the model's
    own memory (9,453-9,481 MiB measured: more than the 9 GiB mark) would fail
    it on every check, and crediting that memory as free would pass every
    check, a game included. So while held the gate looks for ANOTHER consumer
    instead, after waiting ``settle`` seconds (the chunk has ended: the batch
    is idle, so its own compute is not in the readings): it pauses when the
    card's used memory has grown more than ``growth_max_mib`` since the model
    was measured, or when the utilisation is at or above ``util_max`` on EVERY
    one of the samples. Under WSL nvidia-smi cannot attribute memory to a
    process, so the model is measured: the memory in use after the first image
    it runs (its arena is full by then: both models take fixed 1024 px inputs)
    less the reading before it was loaded (``mark_loading``), once per load;
    ``forget_own`` on release. A consumer that starts while the model loads is
    measured as ours, so no more than ``OWN_MAX_MIB`` is taken as ours: the
    rest counts as growth. Until the model has run (a chunk of ineligible
    photos only), or when the measurement fails, Ross's rule applies, after
    the settle."""

    def __init__(
        self,
        *,
        query: Callable[[], str],
        util_max: int = DEFAULT_GPU_UTIL_MAX,
        free_min_mib: int = DEFAULT_GPU_FREE_MIN_MIB,
        samples: int = 3,
        interval: float = 0.5,
        settle: float = DEFAULT_GPU_SETTLE_SECONDS,
        growth_max_mib: int = DEFAULT_GPU_GROWTH_MAX_MIB,
        sleep: Callable[[float], None] = time.sleep,
        absent_ok: bool = False,
    ) -> None:
        self._query = query
        self.absent_ok = absent_ok
        self.util_max = util_max
        self.free_min_mib = free_min_mib
        self.samples = max(1, samples)
        self.interval = interval
        self.settle = settle
        self.growth_max_mib = growth_max_mib
        self._sleep = sleep
        self.last: GpuReading | None = None
        self.own_mib = 0
        self._held = False
        self._baseline_mib: int | None = None
        self._reference_mib: int | None = None

    def mark_loading(self) -> None:
        """The model is about to load: the last reading is the card without it."""
        self._held = True
        self._baseline_mib = self.last.used_mib if self.last is not None else None
        self._reference_mib = None
        self.own_mib = 0

    def measure_own(self) -> None:
        """The model has run once: what it added since ``mark_loading`` is
        ours (up to ``OWN_MAX_MIB``), and the card as it is now, the reference
        another consumer is measured against."""
        if self._baseline_mib is None:
            return
        try:
            reading = self.read()
        except Exception:
            return  # not measured: Ross's rule stays in force
        self.own_mib = max(0, reading.used_mib - self._baseline_mib)
        self._reference_mib = self._baseline_mib + min(self.own_mib, OWN_MAX_MIB)

    def forget_own(self) -> None:
        self.own_mib = 0
        self._held = False
        self._baseline_mib = None
        self._reference_mib = None

    def read(self) -> GpuReading:
        parts = [p.strip() for p in self._query().split(",")]
        if len(parts) != 4:
            raise ValueError(f"unexpected nvidia-smi output: {parts!r}")
        util, free, used, total = (int(p) for p in parts)
        return GpuReading(util=util, free_mib=free, used_mib=used, total_mib=total)

    def check(self) -> GateResult:
        if self._held:
            self._sleep(self.settle)
        detail = ""
        busy = 0
        for n in range(self.samples):
            if n:
                self._sleep(self.interval)
            try:
                reading = self.read()
            except FileNotFoundError:
                if self.absent_ok:
                    return GateResult(True, "gpu", "no GPU on this box (device auto: CPU)")
                return GateResult(False, "gpu", "GPU unreadable: no nvidia-smi")
            except Exception as exc:
                return GateResult(False, "gpu", f"GPU unreadable: {type(exc).__name__}")
            self.last = reading
            if self._reference_mib is None:
                detail = (
                    f"util {reading.util}% (passes below {self.util_max}%), "
                    f"free {reading.free_mib} MiB (passes at {self.free_min_mib} MiB)"
                )
                if not (reading.util < self.util_max or reading.free_mib >= self.free_min_mib):
                    return GateResult(False, "gpu", detail)
                continue
            # memory another process freed lowers the reference: what it takes next shows
            self._reference_mib = min(self._reference_mib, reading.used_mib)
            grown = reading.used_mib - self._reference_mib
            detail = (
                f"util {reading.util}%, used {reading.used_mib} MiB, {grown:+d} MiB since "
                f"our model ({self.own_mib} MiB) was measured "
                f"(another consumer above +{self.growth_max_mib} MiB)"
            )
            if grown > self.growth_max_mib:
                return GateResult(False, "gpu", detail)
            busy += reading.util >= self.util_max
        if self._reference_mib is not None and busy == self.samples:
            return GateResult(
                False,
                "gpu",
                f"{detail}; util at or above {self.util_max}% on every one of "
                f"{self.samples} readings with our model idle: another consumer",
            )
        return GateResult(True, "gpu", detail)


def parse_windows(spec: str) -> tuple[tuple[int, int], ...]:
    """``"03-26,41-43"`` -> ``((3, 26), (41, 43))``: inclusive minute ranges."""
    windows: list[tuple[int, int]] = []
    for part in (p.strip() for p in spec.split(",")):
        if not part:
            continue
        m = re.fullmatch(r"(\d{1,2})-(\d{1,2})", part)
        if not m or not 0 <= int(m[1]) <= int(m[2]) <= 59:
            raise ValueError(f"bad busy window {part!r}: want MM-MM within 00-59, start <= end")
        windows.append((int(m[1]), int(m[2])))
    return tuple(windows)


def busy_window(minute: int, windows: Sequence[tuple[int, int]]) -> tuple[int, int] | None:
    for start, end in windows:
        if start <= minute <= end:
            return (start, end)
    return None


def public_ip(url: str) -> Callable[[], str]:
    def fetch() -> str:
        response = httpx.get(url, timeout=10.0)
        response.raise_for_status()
        return str(response.text)

    return fetch


class ResidentialCheck:
    """The box is on its residential exit when its public address is the
    expected one, or inside the expected block (``198.51.100.0/24``: a DHCP
    renewal stays in the ISP's block, a VPN window leaves it). A pass is
    trusted for ``ttl`` seconds; a failure is never cached."""

    def __init__(
        self,
        *,
        expected_ip: str,
        fetch: Callable[[], str],
        ttl: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        try:
            self.expected = ipaddress.ip_network(expected_ip.strip(), strict=False)
        except ValueError:
            raise ValueError(
                f"the residential address {expected_ip!r} is not an IP or a CIDR"
            ) from None
        self._fetch = fetch
        self.ttl = ttl
        self._clock = clock
        self._ok_until: float | None = None

    def check(self) -> GateResult:
        now = self._clock()
        if self._ok_until is not None and now < self._ok_until:
            return GateResult(True, "residential", "residential address (checked recently)")
        try:
            seen = self._fetch().strip()
        except Exception as exc:
            return GateResult(
                False, "residential", f"public address unreadable: {type(exc).__name__}"
            )
        try:
            home = ipaddress.ip_address(seen) in self.expected
        except ValueError:
            home = False
        if not home:
            return GateResult(
                False, "residential", f"public address {seen} is not the residential exit (VPN?)"
            )
        self._ok_until = now + self.ttl
        return GateResult(True, "residential", "residential address")


@dataclass
class Gates:
    windows: tuple[tuple[int, int], ...]
    gpu: GpuGate | None
    residential: ResidentialCheck | None
    now: Callable[[], dt.datetime]

    def busy(self) -> tuple[int, int] | None:
        return busy_window(self.now().minute, self.windows)

    def check(self) -> GateResult:
        """Window first (no reading needed), then the GPU, then the address."""
        window = self.busy()
        if window is not None:
            return GateResult(
                False,
                "window",
                f"scraper roll window :{window[0]:02d}-:{window[1]:02d} "
                f"(now :{self.now().minute:02d})",
            )
        for gate in (self.gpu, self.residential):
            if gate is not None:
                result = gate.check()
                if not result.ok:
                    return result
        return GateResult(True, "all", "quiet window, GPU free, residential address")


# --------------------------------------------------------------------------
# the record, the VRAM sampler, the raw bucket
# --------------------------------------------------------------------------


class Progress:
    """The batch's own record: JSON lines, one per image (and per chunk),
    each flushed to disk as it is written."""

    def __init__(self, path: Path | None) -> None:
        self.path = path

    def load(self) -> dict[str, dict[str, Any]]:
        """Per original sha, its last recorded result. A torn line (the power
        went out mid-write) is ignored."""
        if self.path is None or not self.path.exists():
            return {}
        found: dict[str, dict[str, Any]] = {}
        for line in self.path.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and record.get("event") == "item":
                sha = record.get("blob_sha256")
                if isinstance(sha, str):
                    found[sha] = record
        return found

    def append(self, record: dict[str, Any]) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+b") as fh:
            fh.seek(0, os.SEEK_END)
            if fh.tell():
                fh.seek(-1, os.SEEK_END)
                if fh.read(1) != b"\n":
                    fh.write(b"\n")  # close a torn line so it cannot swallow this one
            fh.write((json.dumps(record, sort_keys=True, default=str) + "\n").encode())
            fh.flush()
            os.fsync(fh.fileno())


def vram_command(index: int) -> list[str]:
    return [
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
        "-i",
        str(index),
        "-lms",
        "200",
    ]


class VramSampler:
    """The peak of a GPU's memory in use while a chunk runs (whole card, MiB:
    under WSL nvidia-smi cannot attribute memory to a process). No command,
    or one that cannot start, samples nothing (``peak_mib`` None)."""

    def __init__(self, command: Sequence[str] | None, *, grace: float = 0.0) -> None:
        self._command = list(command) if command else None
        self._grace = grace
        self._proc: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self.peak_mib: int | None = None

    def _read(self, stream: Iterable[str]) -> None:
        for line in stream:
            try:
                value = int(line.strip())
            except ValueError:
                continue
            self.peak_mib = value if self.peak_mib is None else max(self.peak_mib, value)

    def __enter__(self) -> VramSampler:
        if self._command is None:
            return self
        try:
            self._proc = subprocess.Popen(
                self._command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
            )
        except OSError:
            return self
        assert self._proc.stdout is not None
        self._thread = threading.Thread(target=self._read, args=(self._proc.stdout,), daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._proc is None:
            return
        try:
            self._proc.wait(timeout=self._grace)
        except subprocess.TimeoutExpired:
            self._proc.terminate()
            self._proc.wait(timeout=10)
        assert self._thread is not None  # started with the process
        self._thread.join(timeout=10)


class RecentWrites:
    """The media bucket with its last few writes kept in memory: the lane
    re-reads the clean copy it has just stored (to hash it before the mask),
    and over the WAN that read is a GET of the whole image per image. Every
    write still goes to the bucket, and is kept only once the PUT returned.
    A kept copy is served only when the bucket's ETag (a HEAD, no body) is
    the md5 of the kept bytes, so the lane's hash check still covers what the
    bucket holds: a PUT the bucket stored wrong, a store with no ETag, an ETag
    that is no md5 (multipart) or a HEAD that fails all read the bucket.
    Pipelined (MASK-PERF-2) its workers share it: ``keep`` covers each
    one's last original, copy and mask, and the kept copies are locked."""

    def __init__(self, store: ObjectStore, *, keep: int = 4) -> None:
        self._store = store
        self.keep = keep
        self._recent: dict[str, bytes] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> bytes:
        with self._lock:
            kept = self._recent.get(key)
        if kept is not None and self._stored_as(key, kept):
            return kept
        return self._store.get(key)

    def _stored_as(self, key: str, data: bytes) -> bool:
        etag = getattr(self._store, "etag", None)
        if etag is None:
            return False
        try:
            stored = etag(key)
        except Exception:
            return False  # the GET that follows succeeds or fails on its own
        return bool(stored == hashlib.md5(data, usedforsecurity=False).hexdigest())

    def put(self, key: str, data: bytes, content_type: str) -> None:
        self._store.put(key, data, content_type)
        with self._lock:
            self._recent.pop(key, None)
            self._recent[key] = data
            while len(self._recent) > self.keep:
                del self._recent[next(iter(self._recent))]


# --------------------------------------------------------------------------
# the driver
# --------------------------------------------------------------------------


@dataclass
class BatchConfig:
    """``writers`` above 0 runs each chunk's images that many at a time, each
    on a worker with a database session of its own, ``prefetch`` originals
    read ahead (MASK-PERF-2); 0 runs one image at a time on the driver's own
    session, as before (the command line's default is the pipeline)."""

    chunk_size: int = 20
    limit: int | None = None
    retry_failed: bool = False
    dry_run: bool = False
    poll_seconds: float = 60.0
    max_wait_seconds: float | None = None
    max_consecutive_failures: int = 5
    writers: int = 0
    prefetch: int = 4
    decode_slots: int = DECODED_AT_ONCE
    #: MASK5: the CPU worker processes of a pipelined run (0: its CPU stages
    #: on its workers' threads, as before; Runtime.cpu_stages makes them)
    cpu_workers: int = 0


@dataclass
class Runtime:
    gates: Gates
    open_db: Callable[[], Session]
    media_store: ObjectStore | None
    raw_store: ObjectStore | None
    make_backend: Callable[[], MattingBackend]
    progress: Progress
    vram: Callable[[], Any]
    sleep: Callable[[float], None]
    monotonic: Callable[[], float]
    emit: Callable[[dict[str, Any]], None]
    stop_requested: Callable[[], bool]
    #: Each chunk as an item of the run (LG-U11, fc-shared's job model: its own
    #: trace, linked to the run); outside a run, no item.
    open_item: Callable[[int], Item] = lambda _number: NULL_ITEM
    #: MASK5: a pipelined run's CPU stages on ``cpu_workers`` processes (the
    #: command line's: app.workers.cpu_stages.ProcessStages); by default, and
    #: for 0, on the workers' own threads.
    cpu_stages: Callable[[int], CpuStages] = lambda _workers: INLINE


class _Stopped(Exception):
    pass


class _GaveUp(Exception):
    def __init__(self, result: GateResult) -> None:
        self.result = result


@dataclass
class _Chunk:
    number: int
    batch_id: int | None
    started: float
    baseline_vram_mib: int | None
    provider: str | None
    records: list[dict[str, Any]] = field(default_factory=list)
    #: per record, its seconds by stage (app.workers.stage_timing)
    stages: list[dict[str, float]] = field(default_factory=list)
    yielded: bool = False
    #: The chunk's item: its images and its line run in it (Runtime.open_item).
    item: Item = field(default=NULL_ITEM, repr=False)

    def report(self, peak_vram_mib: int | None, elapsed: float) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for r in self.records:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        minted = [r for r in self.records if r["status"] != "already" and "seconds" in r]
        seconds = [r["seconds"] for r in minted]
        mask_seconds = [r["mask_seconds"] for r in minted if r.get("mask_seconds") is not None]
        # an ineligible photo costs next to nothing (no model): the real rate
        masked = [r["seconds"] for r in minted if r["status"] in _MASKED]
        # each stage over the minted images that ran it: one that never ran is absent
        by_stage: dict[str, list[float]] = {name: [] for name in STAGES}
        for r, stages in zip(self.records, self.stages, strict=True):
            if r["status"] != "already" and "seconds" in r:
                for name, took in stages.items():
                    by_stage.setdefault(name, []).append(took)
        ran = {name: per_image for name, per_image in by_stage.items() if per_image}
        return {
            "event": "chunk",
            "chunk": self.number,
            "batch_id": self.batch_id,
            "count": len(self.records),
            "counts": counts,
            "ranks": [r["rank"] for r in self.records],
            "seconds_per_image": round(statistics.fmean(seconds), 3) if seconds else None,
            "median_seconds": round(statistics.median(seconds), 3) if seconds else None,
            "masked": len(masked),
            "seconds_per_masked_image": round(statistics.fmean(masked), 3) if masked else None,
            "mask_seconds_per_image": (
                round(statistics.fmean(mask_seconds), 3) if mask_seconds else None
            ),
            "stage_seconds_per_image": {n: round(statistics.fmean(t), 3) for n, t in ran.items()},
            "stage_median_seconds": {n: round(statistics.median(t), 3) for n, t in ran.items()},
            "stage_images": {n: len(t) for n, t in ran.items()},
            "peak_vram_mib": peak_vram_mib,
            "baseline_vram_mib": self.baseline_vram_mib,
            "provider": self.provider,
            "failures": [
                {"blob_sha256": r["blob_sha256"], "reason": r["reason"]}
                for r in self.records
                if r["status"] in _REDO
            ],
            "yielded": self.yielded,
            "elapsed_seconds": round(elapsed, 3),
            # the chunk's wall seconds over the images it minted (an 'already'
            # one is a lookup, as in seconds_per_image): what overlapping them gains
            "throughput_seconds_per_image": round(elapsed / len(minted), 3) if minted else None,
        }


class _Witness:
    """The media bucket as one image's lane sees it, keeping the last error
    it raised (the lane records only its class: a 503 and a 403 are both
    ``ClientError`` there). With ``timings``, each PUT is an "upload" and
    each GET a "readback", inside whichever stage made it (the seed's too).
    ``beside`` an executor (MASK-PERF-3, pipelined: the seed's original and
    clean copy) a PUT runs on it, beside whatever the image does next, and
    ``flush`` waits for them -- before its session commits
    (``_WorkerSession``) and before any GET -- so its objects still land
    before its rows; that wait is its "upload"."""

    def __init__(self, store: ObjectStore, timings: StageTimings | None = None) -> None:
        self._store = store
        self._timings = timings
        self.error: Exception | None = None
        self._uploads: ThreadPoolExecutor | None = None
        self._pending: list[Future[None]] = []

    def get(self, key: str) -> bytes:
        self.flush()
        try:
            with timed(self._timings, "readback"):
                return self._store.get(key)
        except Exception as exc:
            self.error = exc
            raise

    def put(self, key: str, data: bytes, content_type: str) -> None:
        if self._uploads is not None:
            self._pending.append(self._uploads.submit(self._store.put, key, data, content_type))
            return
        try:
            with timed(self._timings, "upload"):
                self._store.put(key, data, content_type)
        except Exception as exc:
            self.error = exc
            raise

    @contextlib.contextmanager
    def beside(self, uploads: ThreadPoolExecutor | None) -> Iterator[None]:
        """Meanwhile each PUT runs on ``uploads`` (None: in turn, as ever)."""
        self._uploads = uploads
        try:
            yield
        finally:
            self._uploads = None

    def flush(self) -> None:
        """Wait for every PUT in flight; the first of them that failed raises."""
        pending, self._pending = self._pending, []
        if not pending:
            return
        failed: BaseException | None = None
        with timed(self._timings, "upload"):
            for future in pending:
                exc = future.exception()
                failed = failed or exc
        if isinstance(failed, Exception):
            self.error = failed
        if failed is not None:
            raise failed


class _WorkerSession(Session):
    """A pipeline worker's session: the PUTs its image has in flight
    (``info["witness"]``, a ``_Witness``) land before any commit."""

    def commit(self) -> None:
        witness = self.info.get("witness")
        if witness is not None:
            witness.flush()
        super().commit()


def _mint(
    db: Session,
    item: WorkItem,
    rt: Runtime,
    backend: MattingBackend,
    batch: Any,
    retry_failed: bool,
    timings: StageTimings | None = None,
    lane: _ImageLane | None = None,
) -> dict[str, Any]:
    """One original, start to finish; the result as its progress record.
    ``timings``, when given, gets the seconds of each stage (MASK-PERF-1);
    its SQL's seconds are the batch's to time (DatabaseSeconds, run_batch).
    ``lane``, pipelined (MASK-PERF-2): the image's claim before its first
    query, its original read ahead, the hold checked before its seed and its
    turn before its batch-row writes; its seconds read ahead count as its."""
    assert rt.raw_store is not None and rt.media_store is not None
    start = rt.monotonic() - (lane.ahead if lane is not None else 0.0)
    with timed(timings, "db"):  # read again from the database once a rollback expired it
        batch_id = getattr(batch, "id", None)
    record: dict[str, Any] = {
        "event": "item",
        "blob_sha256": item.blob_sha256,
        "product_id": item.product_id,
        "head_id": item.head_id,
        "rank": item.rank,
        "batch_id": batch_id,
        "clean_sha256": None,
        "mask_sha256": None,
        "reason": None,
    }

    def failed(
        reason: str, outage: bool = False, where: str = "", exc: BaseException | None = None
    ) -> dict[str, Any]:
        # an outage's casualty is 'interrupted': it counts toward the circuit and
        # a plain rerun retries it; an image's own fault stays 'failed'
        status = "interrupted" if outage else "failed"
        record.update(status=status, reason=reason, seconds=round(rt.monotonic() - start, 3))
        if outage:  # MASK5: why, for the circuit's line (_Tally takes it off the record)
            cause = _outage_cause(where, exc, reason, now=rt.gates.now())
            record["cause"] = {**cause, "blob_sha256": item.blob_sha256}
        return record

    def raised(where: str, exc: Exception) -> dict[str, Any]:
        if isinstance(exc, ImageLimitError):  # the image's own fault, named by the limit it broke
            return failed(f"refused:{exc.code}")
        return failed(f"{where}:{type(exc).__name__}", is_outage(exc), where, exc)

    def rolled_back(where: str, exc: Exception) -> dict[str, Any]:
        with timed(timings, "db"):  # a round trip like any query
            db.rollback()
        return raised(where, exc)

    media = _Witness(rt.media_store, timings)
    db.info["witness"] = media
    try:
        if lane is not None:
            lane.claim(db)  # its footprint read, and the related images it waits for
        with timed(timings, "db"):  # the image's first query: the pool's checkout too
            settled = original_settled(db, item.blob_sha256, retry_failed=retry_failed)
    except Exception as exc:
        return rolled_back("db", exc)
    if settled is not None:
        record.update(status="already", mask_status=settled)
        return record
    if item.codec not in ("identity", "gzip"):
        return failed(f"unsupported_codec:{item.codec}")
    try:
        if lane is None:
            with timed(timings, "fetch"):
                data = _decode(rt.raw_store.get(item.storage_key), item.codec)
        else:
            start -= lane.fetch_seconds  # read ahead, before the image had a worker
            data = lane.fetched()
    except Exception as exc:
        return raised("fetch", exc)
    if hashlib.sha256(data).hexdigest() != item.blob_sha256:
        return failed("raw_hash_mismatch")
    if lane is not None:
        lane.before_seed()  # held: nothing of it written, a rerun takes it
    # pipelined, what its claim computed of the bytes ahead (MASK5: SeedInputs)
    seed = seed_clean_image
    if lane is not None and lane.inputs is not None:
        seed = functools.partial(seed_clean_image, inputs=lane.inputs)
    try:
        # pipelined, its original's and its copy's PUTs run beside its SQL
        with timed(timings, "seed"), media.beside(lane.uploads if lane is not None else None):
            # SHOW1: retried once when a concurrent first seed of the same
            # picture (the cluster display-copy lane) wins the unique key
            clean = seed_with_retry(
                db,
                media,
                data,
                origins=[
                    ImageOrigin(url=item.url, source=item.source, source_class=item.source_class)
                ],
                figure_id=item.product_id,
                seed=seed,
            )
        clean_sha = clean.sha256
    except Exception as exc:
        cause = exc
        try:
            media.flush()  # a PUT made before the error failed first: it is the cause
        except Exception as put_failed:
            cause = put_failed
        if isinstance(cause, IntegrityError):
            # still racing after the retry: never the image's fault, a rerun takes it
            with timed(timings, "db"):
                db.rollback()
            return failed("seed:IntegrityError", True, "seed", cause)
        return rolled_back("seed", cause)
    record["clean_sha256"] = clean_sha
    mask_start = rt.monotonic()
    try:
        report = settle_clean_copy(
            db,
            media,
            clean_sha,
            backend=backend,
            batch=batch,
            retry_failed=retry_failed,
            timings=timings,
            turn=lane.turn if lane is not None else None,
            slot=lane.decoded if lane is not None else None,
            # its mask judged on a CPU worker (MASK5); without workers, on this thread as ever
            cpu=lane.cpu if lane is not None and lane.cpu.workers else None,
        )
    except Exception as exc:
        return rolled_back("settle", exc)
    record["mask_seconds"] = round(rt.monotonic() - mask_start, 3)
    if report is None:
        # another original's copy of the same picture was settled first
        record.update(status="already", mask_status=original_settled(db, item.blob_sha256))
    elif report.status == "failed":
        # the lane caught it and kept the class name; the bucket's error, if any, says more
        witnessed = media.error is not None and is_outage(media.error)
        outage = is_outage_reason(report.reason) or witnessed
        failed(str(report.reason), outage, "settle", media.error if witnessed else None)
    else:
        record.update(status=report.status, reason=report.reason, mask_sha256=report.mask_sha256)
    record["seconds"] = round(rt.monotonic() - start, 3)
    return record


# --------------------------------------------------------------------------
# the pipeline (MASK-PERF-2)
# --------------------------------------------------------------------------


@dataclass
class _Fetched:
    """An original read ahead: its bytes or the error reading them, its
    picture's pixel digest (None when it never reaches the seed), and the
    seconds each took on the read-ahead thread."""

    data: bytes | None = None
    error: Exception | None = None
    digest: str | None = None
    fetch_seconds: float = 0.0
    keys_seconds: float = 0.0
    #: its width x height from its header (None: it would not open)
    pixels: int | None = None


def _read_ahead(item: WorkItem, rt: Runtime, cpu: CpuStages = INLINE) -> _Fetched:
    """``item``'s original from the raw bucket, decoded and checked against
    its sha, and its picture's pixel digest (on ``cpu``), before the image
    has a worker: no more of its keys (MASK5, SHOULD-1: its clean copy is
    encoded in its claim, only when the picture has no copy yet). An
    original in a codec the lane refuses is never read."""
    assert rt.raw_store is not None
    if item.codec not in ("identity", "gzip"):
        return _Fetched()
    start = rt.monotonic()
    try:
        data = _decode(rt.raw_store.get(item.storage_key), item.codec)
    except Exception as exc:
        return _Fetched(error=exc, fetch_seconds=rt.monotonic() - start)
    fetched = rt.monotonic()
    digest = pixels = None
    if hashlib.sha256(data).hexdigest() == item.blob_sha256:
        with contextlib.suppress(Exception), PILImage.open(io.BytesIO(data)) as header:
            pixels = header.width * header.height
        # a picture that will not decode (None) fails the seed before it writes a row
        digest = cpu.run(picture_digest, data, pixels=pixels)
    return _Fetched(
        data=data,
        digest=digest,
        fetch_seconds=fetched - start,
        keys_seconds=rt.monotonic() - fetched,
        pixels=pixels,
    )


def _decode_units(pixels: int | None, slots: int) -> int:
    """The decode slots an original of ``pixels`` takes: one per
    ``DECODE_UNIT_PIXELS`` begun, up to every slot (it is then decoded
    alone); one when its size is unknown (it fails before its decode)."""
    if pixels is None:
        return 1
    return max(1, min(slots, -(-pixels // DECODE_UNIT_PIXELS)))


class _BatchRow:
    """The chunk's batch row as the last image to write it left it (its
    counts and cursor), lent to the next image's own copy of the row at its
    turn: each image's UPDATE is the one one image at a time makes. Unknown
    after an image whose last write rolled back: the next one reads the row
    again, as one image at a time does after a rollback -- at its turn,
    before any write of its own is pending, so an autoflushing session
    still makes one UPDATE of both (never the cursor flushed alone first)."""

    def __init__(self, batch: MaskBatch) -> None:
        self._state: tuple[dict[str, int], str | None] | None = (
            dict(batch.counts),
            batch.cursor_sha256,
        )

    def lend(self, db: Session, batch: MaskBatch) -> None:
        if self._state is None:
            db.refresh(batch, ["counts", "cursor_sha256"])
            return
        counts, cursor = self._state
        set_committed_value(batch, "counts", dict(counts))
        set_committed_value(batch, "cursor_sha256", cursor)

    def keep(self, batch: MaskBatch) -> None:
        state = inspect(batch)
        loaded = state.dict
        if state.modified or "counts" not in loaded or "cursor_sha256" not in loaded:
            self._state = None
        else:
            self._state = (dict(loaded["counts"]), loaded["cursor_sha256"])


@dataclass
class _Done:
    """An image off its worker: its record, or None when it was held."""

    index: int
    record: dict[str, Any] | None
    stages: dict[str, float]
    error: BaseException | None = None


@dataclass(eq=False)
class _Flow:
    """What every image in flight shares, whichever chunk it is of (the
    pipeline rolls across chunk boundaries, MASK-PERF-3): the queue to the
    driver's thread (a GPU job or an image done; at most one of either per
    worker, so never more than ``writers``), the claims, the hold, the
    decode slots, and the TURN: the lock an image holds from its first write
    after its judgment (a mask registered, or its row and its chunk's batch
    row) to its commit, so the batch rows and a mask two images share are
    written by one image at a time, in the order the images get there."""

    events: queue.Queue[GpuJob | _Done]
    #: the decode slots, ``size`` units
    size: int
    slots: Slots = field(init=False)
    claims: Claims = field(default_factory=Claims)
    hold: threading.Event = field(default_factory=threading.Event)
    turn: threading.Lock = field(default_factory=threading.Lock)
    #: the seeds' PUTs, beside their SQL: two an image (its original and copy)
    uploads: ThreadPoolExecutor | None = None
    #: MASK5: where the images' CPU stages run (a pool of worker processes)
    cpu: CpuStages = INLINE

    def __post_init__(self) -> None:
        self.slots = Slots(self.size)

    def submit(self, call: Callable[[], Any]) -> Any:
        """Run ``call`` (a model call) on the driver's thread, and wait."""
        job = GpuJob(call)
        self.events.put(job)
        return job.wait()


@dataclass(eq=False)
class _ChunkRun:
    """One chunk in flight: its record and batch row (the row as the last
    image to write it left it), its VRAM sampler, and its images started
    and ended. ``full``: no further image of it starts."""

    flow: _Flow
    chunk: _Chunk
    batch: MaskBatch
    sampler: Any
    row: _BatchRow = field(init=False)
    started: int = 0
    ended: int = 0
    full: bool = False

    def __post_init__(self) -> None:
        self.row = _BatchRow(self.batch)

    @property
    def complete(self) -> bool:
        return self.full and self.ended == self.started


class _ImageLane:
    """One image's place in the pipeline, as ``_mint`` meets it."""

    def __init__(
        self,
        run: _ChunkRun,
        index: int,
        item: WorkItem,
        fetched: _Fetched,
        db: Session,
        batch: MaskBatch,
        timings: StageTimings,
    ) -> None:
        self._run, self._index, self._item = run, index, item
        self._fetched, self._db, self._batch, self._timings = fetched, db, batch, timings
        self._claim: Claim | None = None
        self._registered = False
        self._turned = False
        timings.add("claim_keys", fetched.keys_seconds)  # its picture's digest, read ahead
        self.ahead = fetched.keys_seconds
        self.fetch_seconds = fetched.fetch_seconds
        self.uploads = run.flow.uploads
        self.cpu = run.flow.cpu
        #: what its claim computed of its bytes for its seed (None: it will not decode)
        self.inputs: SeedInputs | None = None

    def claim(self, db: Session) -> None:
        """Before the image's first query: read its footprint by its
        original's sha and its picture's digest (``claim_footprint``, its
        SQL "db"); when the picture has no copy yet, encode the copy its
        seed will mint (``claim_keys``, on the CPU stages: MASK5, SHOULD-1)
        and read the footprint of that copy's keys too; register it in
        work-list order (``claim_register``: the wait for every earlier
        image to have read its own) and wait for every earlier image it
        meets (``turn_wait``)."""
        self.before_seed()  # held before it began: not even a query
        with self._timings.stage("claim_footprint"):
            since = self._run.flow.claims.snapshot()
            try:
                # the image's first query: the pool's checkout (its pre-ping a
                # round trip no statement event spans) is its "db", as unpipelined
                with timed(self._timings, "db"):
                    db.connection()
                near = seed_neighbourhood(
                    db, sha=self._item.blob_sha256, digest=self._fetched.digest
                )
                footprint = near.keys
                self.inputs = self._seed_inputs(near)
                clean = self.inputs.clean if self.inputs is not None else None
                if clean is not None:  # the copy it will mint: whatever its keys reach
                    footprint |= seed_footprint(db, shas={clean.sha256}, digests={clean.own_digest})
            except BaseException:
                self._registered = True
                self._run.flow.claims.skip(self._index)
                raise
            # the seed reads its gallery link by the product and the URL
            footprint.add(f"gallery {self._item.product_id} {self._item.url or ''}")
        with self._timings.stage("claim_register"):
            self._claim, waits = self._run.flow.claims.register(self._index, footprint, since)
            self._registered = True
        with self._timings.stage("turn_wait"):
            for earlier in waits:
                earlier.done.wait()

    def _seed_inputs(self, near: Neighbourhood) -> SeedInputs | None:
        """What the seed will need of the bytes and the database does not
        have: the clean copy when no copy of the picture is registered, the
        perceptual hash when the original is not recorded (nothing to
        compute when both are); None when the picture will not decode."""
        digest, data = self._fetched.digest, self._fetched.data
        if digest is None or data is None:
            return None
        if near.copy_known and near.image_known:
            return SeedInputs(digest)
        with self._timings.stage("claim_keys"):
            return self.cpu.run(
                seed_inputs,
                data,
                digest,
                not near.copy_known,
                not near.image_known,
                pixels=self._fetched.pixels,
            )

    def fetched(self) -> bytes:
        self._timings.add("fetch", self._fetched.fetch_seconds)
        if self._fetched.error is not None:
            raise self._fetched.error
        assert self._fetched.data is not None
        return self._fetched.data

    def before_seed(self) -> None:
        if self._run.flow.hold.is_set():
            raise Held

    @contextlib.contextmanager
    def decoded(self) -> Iterator[None]:
        """Its copy decoded for the model, a slot held meanwhile: the wait
        for one is a wait for the GPU."""
        units = _decode_units(self._fetched.pixels, self._run.flow.size)
        with self._timings.stage("gpu_wait"):
            self._run.flow.slots.acquire(self._index, units)
        try:
            yield
        finally:
            self._run.flow.slots.release(units)

    def turn(self) -> None:
        """Before the image's first write after its judgment: the turn, held
        to its end (``close``); its chunk's row as the last image to write it
        left it."""
        if self._turned:
            return
        with self._timings.stage("turn_wait"):
            self._run.flow.turn.acquire()
        self._turned = True
        self._run.row.lend(self._db, self._batch)

    def close(self) -> None:
        """The image ended (recorded, held or broken): whoever waits goes on."""
        if not self._registered:
            self._run.flow.claims.skip(self._index)
        if self._turned:
            self._run.row.keep(self._batch)
            self._run.flow.turn.release()
        if self._claim is not None:
            self._run.flow.claims.finish(self._claim)


class _Pipeline:
    """The work, ``writers`` images at a time (each on a worker thread with
    a session of its own, ``_mint`` as ever), ``prefetch`` originals read
    ahead, the model on the driver's thread alone (MASK-PERF-2), ROLLING
    across chunks (MASK-PERF-3): a chunk is its batch row, its gate and its
    report, never a barrier. Each image is recorded as it ends, once its
    writes committed, so a slow image keeps its own worker and no other. A
    stop, a scraper window, a failing gate or the circuit HOLDS the pipeline:
    no further image starts or gets onto the GPU; the images already past
    the GPU finish, the others are let go unrecorded (``Held``) and come back
    first. At most ``writers`` images are in flight, every one unrecorded,
    so an outage opens the circuit after at most ``writers`` - 1 images more
    than one at a time would have tried, and a power cut loses at most
    ``writers`` records (each image reruns idempotently)."""

    def __init__(
        self,
        rt: Runtime,
        config: BatchConfig,
        backend: MattingBackend,
        sql: DatabaseSeconds,
        sessions: sessionmaker[Session],
        interrupted: set[str],
        cpu: CpuStages = INLINE,
    ) -> None:
        self.rt, self.config, self.backend, self.sql = rt, config, backend, sql
        self.sessions, self.interrupted, self.cpu = sessions, interrupted, cpu

    def roll(
        self,
        remaining: deque[WorkItem],
        open_chunk: Callable[[], tuple[_Chunk, MaskBatch]],
        close_chunk: Callable[[_Chunk, MaskBatch, int | None], None],
        take: Callable[[_Chunk, dict[str, Any], dict[str, float]], bool],
    ) -> None:
        """Run ``remaining`` (each item popped as its image starts) until it
        is empty or the pipeline is held; ``take`` each record (True: hold,
        the circuit is open). A chunk opens (``open_chunk``) as its first
        image starts -- after the first, once the gate passed
        (``rt.gates.check``, on this thread, so no model call runs meanwhile
        and the GPU readings are another consumer's alone) -- and closes
        (``close_chunk``) once its last image ended: one held up by a slow
        image after the next. The held images go back to the front of
        ``remaining``, in work-list order."""
        writers, prefetch, size = self.config.writers, self.config.prefetch, self.config.chunk_size
        flow = _Flow(
            events=queue.Queue(maxsize=2 * writers), size=self.config.decode_slots, cpu=self.cpu
        )
        flow.uploads = ThreadPoolExecutor(max_workers=2 * writers, thread_name_prefix="mask-upload")
        gpu = on_the_gpu(self.backend, flow.submit)
        pool = ThreadPoolExecutor(max_workers=prefetch, thread_name_prefix="mask-read-ahead")
        ahead: dict[str, Future[_Fetched]] = {}
        workers: list[threading.Thread] = []
        runs: list[_ChunkRun] = []  # the chunks open
        lanes: dict[int, tuple[_ChunkRun, WorkItem]] = {}
        held: list[tuple[int, WorkItem]] = []
        failure: BaseException | None = None
        current: _ChunkRun | None = None
        started = in_flight = 0

        def holding() -> bool:
            if not flow.hold.is_set():
                if self.rt.stop_requested():
                    flow.hold.set()
                elif self.rt.gates.busy() is not None:
                    for run in runs:
                        run.chunk.yielded = True
                    flow.hold.set()
            return flow.hold.is_set()

        def read_ahead() -> None:
            for item in itertools.islice(remaining, prefetch):
                if item.blob_sha256 not in ahead:
                    ahead[item.blob_sha256] = pool.submit(_read_ahead, item, self.rt, self.cpu)

        def close_ready() -> None:
            for run in [r for r in runs if r.complete]:
                runs.remove(run)
                run.sampler.__exit__(None, None, None)
                close_chunk(run.chunk, run.batch, run.sampler.peak_mib)

        def next_run() -> _ChunkRun | None:
            """The chunk the next image is of; None when the gate held it."""
            nonlocal current
            if current is not None:
                if not current.full:
                    return current
                if not self.rt.gates.check().ok:
                    flow.hold.set()
                    return None
            chunk, batch = open_chunk()
            sampler = self.rt.vram()
            sampler.__enter__()
            current = _ChunkRun(flow=flow, chunk=chunk, batch=batch, sampler=sampler)
            runs.append(current)
            return current

        try:
            while True:
                while in_flight < writers and remaining and not holding():
                    run = next_run()
                    if run is None:
                        break
                    item = remaining.popleft()
                    fetched = ahead.pop(item.blob_sha256, None)
                    if fetched is None:
                        fetched = pool.submit(_read_ahead, item, self.rt, self.cpu)
                    lanes[started] = (run, item)
                    worker = threading.Thread(
                        # the image's work in its chunk's item (LG-U11): its lines in that trace
                        target=run.chunk.item.run,
                        args=(functools.partial(self._work, run, gpu, started, item, fetched),),
                        name=f"mask-image-{started}",
                    )
                    workers.append(worker)
                    worker.start()
                    run.started += 1
                    run.full = run.started == size or not remaining
                    started += 1
                    in_flight += 1
                if not flow.hold.is_set():
                    read_ahead()
                if not in_flight:
                    break
                event = flow.events.get()
                if isinstance(event, GpuJob):
                    if holding():
                        event.refuse()
                    else:
                        event.run()
                    continue
                in_flight -= 1
                run, item = lanes.pop(event.index)
                run.ended += 1
                if event.error is not None:
                    failure = failure or event.error
                    flow.hold.set()
                elif event.record is None:
                    held.append((event.index, item))
                elif take(run.chunk, event.record, event.stages):
                    flow.hold.set()
                close_ready()
            if failure is None:
                for run in runs:  # every image ended: each chunk still open closes
                    run.full = True
                close_ready()
        finally:
            flow.hold.set()
            while in_flight:  # the driver broke off: let every worker go
                event = flow.events.get()
                if isinstance(event, GpuJob):
                    event.refuse()
                else:
                    in_flight -= 1
            pool.shutdown(wait=True, cancel_futures=True)
            for worker in workers:
                worker.join()
            flow.uploads.shutdown(wait=True)
            for run in runs:  # not closed: the batch broke off
                run.sampler.__exit__(None, None, None)
        if failure is not None:
            raise failure
        remaining.extendleft(item for _index, item in sorted(held, reverse=True))

    def _work(
        self,
        run: _ChunkRun,
        gpu: MattingBackend,
        index: int,
        item: WorkItem,
        ahead: Future[_Fetched],
    ) -> None:
        db = self.sessions()
        timings = StageTimings(self.rt.monotonic)
        lane: _ImageLane | None = None
        record: dict[str, Any] | None = None
        error: BaseException | None = None
        try:
            fetched = ahead.result()
            batch = db.merge(run.batch, load=False)  # the row, without reading it
            lane = _ImageLane(run, index, item, fetched, db, batch, timings)
            redrive = self.config.retry_failed or item.blob_sha256 in self.interrupted
            with self.sql.charging(timings):
                record = _mint(db, item, self.rt, gpu, batch, redrive, timings, lane)
        except Held:
            with contextlib.suppress(Exception):
                db.rollback()  # its judgment's reads: its seed, if any, is committed
        except BaseException as exc:  # a fault of the batch's own: it stops
            error = exc
        finally:
            if lane is not None:
                lane.close()
            else:
                run.flow.claims.skip(index)
            with contextlib.suppress(Exception):
                db.close()
            run.flow.events.put(_Done(index, record, timings.seconds, error))


def _pipeline_refusal(db: Session, writers: int) -> str | None:
    """Why the pipeline cannot run on ``db``'s pool, or None: every worker
    holds a connection of its own while its image runs, and the batch its own.
    A QueuePool's overflow does not count: a connection it lends past its size
    is closed when it comes back, and any close kills the kubectl
    port-forward the live run reaches its database through."""
    pool = db.get_bind().engine.pool
    needed = writers + 1
    if isinstance(pool, StaticPool | SingletonThreadPool):
        kept = 1
    elif isinstance(pool, QueuePool):
        kept = pool.size()
    else:
        return None  # a connection per checkout
    if kept >= needed:
        return None
    return (
        f"--writers {writers} needs {needed} database connections at once (one per "
        f"worker, one for the batch), the pool keeps {kept}: fewer writers, "
        "or --writers 0 for one image at a time"
    )


def _sized(db: Session, writers: int) -> Session:
    """``db`` for a run of ``writers`` (the command line's): a session like it
    on the same database, through a pool that keeps a connection for every
    worker and one for the batch and lends none past them (no overflow) and
    never recycles one (a recycle closes it; the pre-ping still replaces a
    dead one), so no connection is closed while the run lasts
    (``_pipeline_refusal``). One image at a time, or a pool other than a
    QueuePool: ``db`` itself."""
    engine = db.get_bind().engine
    pool = engine.pool
    if not writers or not isinstance(pool, QueuePool):
        return db
    sized = create_engine(
        engine.url,
        pool_size=max(pool.size(), writers + 1),
        max_overflow=0,
        pool_pre_ping=True,  # as app.db's engine
        pool_recycle=-1,
    )
    db.close()
    return Session(sized, autoflush=db.autoflush, expire_on_commit=db.expire_on_commit)


#: The most characters of an outage's message the circuit's line carries.
CAUSE_MESSAGE_MAX = 300
#: The database driver's modules: an outage raised there is the database's.
_DATABASE_MODULES = ("sqlalchemy", "psycopg2", "psycopg")
#: The class names the lane records of a database outage it caught.
_DATABASE_REASONS = frozenset({"OperationalError", "InterfaceError", "IntegrityError"})


def _connection(stage: str, exc: BaseException | None, reason: str | None) -> str:
    """Which connection an outage broke on: ``database`` (the media
    database, through the operator's tunnel), ``raw_bucket`` (the spine's
    raw bucket the originals are read from) or ``media_bucket``. By the
    exception's class first (the database driver's), then by the stage it
    broke in: the original's read is the raw bucket's, a seed's or a
    settle's other outage the media bucket's (their PUTs and GETs). With no
    exception (the lane kept only a class name), by that name, else
    ``unknown``. The batch reaches no spine API: its raw bucket is the
    spine's."""
    if exc is None:
        return "database" if reason in _DATABASE_REASONS else "unknown"
    if stage == "db" or type(exc).__module__.split(".")[0] in _DATABASE_MODULES:
        return "database"
    return "raw_bucket" if stage == "fetch" else "media_bucket"


def _outage_cause(
    stage: str, exc: BaseException | None, reason: str | None, *, now: dt.datetime
) -> dict[str, Any]:
    """One outage, as the circuit's line names it (MASK5): the exception's
    class (the lane's recorded class name when no exception was seen) and
    message (one line, at most ``CAUSE_MESSAGE_MAX`` characters; the log
    line redacts it again), the connection, the stage and when."""
    message = None if exc is None else " ".join(str(exc).split())[:CAUSE_MESSAGE_MAX]
    return {
        "exception": type(exc).__name__ if exc is not None else reason,
        "message": message,
        "connection": _connection(stage, exc, reason),
        "stage": stage,
        "at": now.isoformat(),
    }


class _Tally:
    """The batch's running count across chunks: the outage streak (its first
    outage, every outage of it by connection and class, and the last image
    that got through before it: MASK5) and whether the model's own memory
    is measured since it was loaded."""

    def __init__(self, max_consecutive_failures: int) -> None:
        self.max = max_consecutive_failures
        self.streak = (
            0  # outage casualties in a row (an image's own fault neither counts nor resets)
        )
        self.measured = False
        self.cause: dict[str, Any] = {}  # the streak's first outage
        self.causes: dict[str, int] = {}  # the streak's outages, by connection:class
        self.last_success: dict[str, Any] = {}  # its chunk, image and when

    def take(
        self,
        rt: Runtime,
        chunk: _Chunk,
        gpu: GpuGate | None,
        record: dict[str, Any],
        stages: dict[str, float],
    ) -> bool:
        """One image's record, in work-list order: True once the circuit opens."""
        cause = record.pop("cause", None)  # the circuit's line carries it, not the record
        record["chunk"] = chunk.number
        if gpu is not None and not self.measured and record["status"] in _RAN_MODEL:
            gpu.measure_own()
            self.measured = True
        rt.progress.append(record)
        chunk.records.append(record)
        chunk.stages.append(stages)
        if record["status"] == "interrupted":
            cause = cause or {}
            if not self.streak:
                self.cause, self.causes = cause, {}
            self.streak += 1
            key = f"{cause.get('connection', 'unknown')}:{cause.get('exception')}"
            self.causes[key] = self.causes.get(key, 0) + 1
        elif record["status"] != "failed":
            self.streak = 0
            self.last_success = {
                "chunk": chunk.number,
                "blob_sha256": record["blob_sha256"],
                "at": rt.gates.now().isoformat(),
            }
        return self.streak >= self.max

    def circuit(self) -> dict[str, Any]:
        """Why the circuit opened (MASK5, Ross 11-03): the streak's first
        outage, its outages by connection and class, the last success."""
        cause, last = self.cause, self.last_success
        return {
            **{f"cause_{key}": cause.get(key) for key in _CAUSE_KEYS},
            "streak_causes": dict(self.causes),
            "last_success_chunk": last.get("chunk"),
            "last_success_blob_sha256": last.get("blob_sha256"),
            "last_success_at": last.get("at"),
        }


_CAUSE_KEYS = ("exception", "message", "connection", "stage", "blob_sha256", "at")


def _one_at_a_time(
    items: list[WorkItem],
    rt: Runtime,
    config: BatchConfig,
    backend: MattingBackend,
    db: Session,
    batch: MaskBatch,
    sql: DatabaseSeconds,
    interrupted: set[str],
    chunk: _Chunk,
    take: Callable[[dict[str, Any], dict[str, float]], bool],
) -> list[WorkItem]:
    """``items`` one at a time on the driver's own session (``writers`` 0).
    Returns the items not done: a stop or a scraper window stops before an
    image, the circuit after one."""
    for n, item in enumerate(items):
        if rt.stop_requested():
            return items[n:]
        if rt.gates.busy() is not None:
            chunk.yielded = True
            return items[n:]
        # an outage's casualties are re-driven without --retry-failed
        redrive = config.retry_failed or item.blob_sha256 in interrupted
        timings = StageTimings(rt.monotonic)
        with sql.charging(timings):
            record = _mint(db, item, rt, backend, batch, redrive, timings)
        if take(record, timings.seconds):
            return items[n + 1 :]
    return []


def _wait_for_gate(rt: Runtime, config: BatchConfig, backend: MattingBackend | None) -> bool:
    """Return once the gate passes -- True when the model was released on the
    way -- meanwhile poll and say so once per gate. Any pause releases the
    model: a game gets the GPU back, and a scraper window (up to 24 minutes,
    during which no GPU reading is taken) does not hold ~9.5 GB idle."""
    paused: str | None = None
    released = False
    waited = 0.0
    while True:
        if rt.stop_requested():
            raise _Stopped
        result = rt.gates.check()
        if result.ok:
            if paused is not None:
                rt.emit({"event": "resumed", "waited_seconds": waited})
            return released
        if result.gate != paused:
            paused = result.gate
            rt.emit({"event": "paused", "gate": result.gate, "detail": result.detail})
            if backend is not None and not released:
                backend.release()
                released = True
                if rt.gates.gpu is not None:
                    rt.gates.gpu.forget_own()
        if config.max_wait_seconds is not None and waited >= config.max_wait_seconds:
            raise _GaveUp(result)
        rt.sleep(config.poll_seconds)
        waited += config.poll_seconds


def _plan(pending: list[WorkItem], config: BatchConfig, rt: Runtime) -> int:
    result = rt.gates.check()
    rt.emit(
        {
            "event": "plan",
            "pending": len(pending),
            "chunks": math.ceil(len(pending) / config.chunk_size),
            "ranks": {
                str(r): sum(1 for i in pending if i.rank == r)
                for r in sorted({i.rank for i in pending})
            },
            "gate": result.as_dict(),
            "first_chunk": [
                {"blob_sha256": i.blob_sha256, "product_id": i.product_id, "rank": i.rank}
                for i in pending[: config.chunk_size]
            ],
        }
    )
    return 0


def run_batch(items: Sequence[WorkItem], config: BatchConfig, rt: Runtime) -> int:
    """Mint ``items`` chunk by chunk under the gate, lowest rank first (a
    stable sort: work-list order within a rank). Returns the exit status
    (module docstring)."""
    if config.chunk_size < 1:
        rt.emit({"event": "refused", "detail": f"chunk size {config.chunk_size} is below 1"})
        return 2
    items = sorted(items, key=lambda i: i.rank)
    recorded = rt.progress.load()
    done = {sha for sha, r in recorded.items() if r.get("status") not in _REDO}
    failed_before = {sha for sha, r in recorded.items() if r.get("status") == "failed"}
    interrupted = {sha for sha, r in recorded.items() if r.get("status") == "interrupted"}
    pending = [
        i
        for i in items
        if i.blob_sha256 not in done and (config.retry_failed or i.blob_sha256 not in failed_before)
    ]
    if config.limit is not None:
        pending = pending[: config.limit]
    rt.emit(
        {
            "event": "start",
            "items": len(items),
            "done": len(done & {i.blob_sha256 for i in items}),
            "failed_before": len(failed_before & {i.blob_sha256 for i in items}),
            "interrupted_before": len(interrupted & {i.blob_sha256 for i in items}),
            "pending": len(pending),
            "chunk_size": config.chunk_size,
            "dry_run": config.dry_run,
        }
    )
    if config.dry_run:
        return _plan(pending, config, rt)

    backend = rt.make_backend()
    if not backend.publishes_masks:
        rt.emit({"event": "refused", "detail": f"backend {backend.name!r} is not a mask model"})
        return 2
    db = rt.open_db()
    sessions: sessionmaker[Session] | None = None
    if config.writers:
        refusal = _pipeline_refusal(db, config.writers)
        if refusal is not None:
            rt.emit({"event": "refused", "detail": refusal})
            return 2
        # each worker's session as the batch's own is, on its engine
        sessions = sessionmaker(
            bind=db.get_bind(),
            class_=_WorkerSession,
            autoflush=db.autoflush,
            expire_on_commit=db.expire_on_commit,
        )
    totals: dict[str, int] = {}
    tally = _Tally(config.max_consecutive_failures)
    remaining = deque(pending)
    chunk_no = 0
    loaded = False
    gpu = rt.gates.gpu
    # each image's statements, commits and rollbacks as its "db"
    sql = DatabaseSeconds(db, sessions)
    cpu: CpuStages = INLINE
    pipeline: _Pipeline | None = None
    if sessions is not None:
        cpu = rt.cpu_stages(config.cpu_workers)  # MASK5: its CPU stages' workers
        pipeline = _Pipeline(rt, config, backend, sql, sessions, interrupted, cpu)

    open_items: dict[int, Item] = {}  # the chunks' items not yet ended

    def open_chunk() -> tuple[_Chunk, MaskBatch]:
        nonlocal chunk_no
        chunk_no += 1
        chunk = _Chunk(
            number=chunk_no,
            batch_id=None,
            started=rt.monotonic(),
            baseline_vram_mib=gpu.last.used_mib if gpu is not None and gpu.last else None,
            provider=backend.active_provider,
            item=rt.open_item(chunk_no),
        )
        open_items[chunk_no] = chunk.item
        batch = chunk.item.run(lambda: open_mask_batch(db, backend_name=backend.name))
        chunk.batch_id = batch.id
        return chunk, batch

    def close_chunk(chunk: _Chunk, batch: MaskBatch, peak_vram_mib: int | None) -> None:
        def close() -> None:
            finish_mask_batch(db, batch)
            report = chunk.report(peak_vram_mib, rt.monotonic() - chunk.started)
            report["held_vram_mib"] = gpu.own_mib if gpu is not None else None
            rt.emit(report)
            rt.progress.append(report)
            for status, n in report["counts"].items():
                totals[status] = totals.get(status, 0) + n

        chunk.item.run(close)
        open_items.pop(chunk.number, None)
        chunk.item.end()

    def take(chunk: _Chunk, record: dict[str, Any], stages: dict[str, float]) -> bool:
        return tally.take(rt, chunk, gpu, record, stages)

    try:
        while remaining:
            if _wait_for_gate(rt, config, backend if loaded else None):
                loaded = False
            if not loaded:
                if gpu is not None:
                    gpu.mark_loading()
                    tally.measured = False
                try:
                    backend.load()
                except Exception as exc:
                    rt.emit({"event": "refused", "detail": str(exc)})
                    return 2
                loaded = True
            if pipeline is None:
                chunk, batch = open_chunk()

                def sequential(chunk: _Chunk = chunk, batch: MaskBatch = batch) -> int | None:
                    todo = [
                        remaining.popleft() for _ in range(min(config.chunk_size, len(remaining)))
                    ]
                    with rt.vram() as sampler:
                        left = _one_at_a_time(
                            todo,
                            rt,
                            config,
                            backend,
                            db,
                            batch,
                            sql,
                            interrupted,
                            chunk,
                            functools.partial(take, chunk),
                        )
                    remaining.extendleft(reversed(left))
                    peak: int | None = sampler.peak_mib
                    return peak

                close_chunk(chunk, batch, chunk.item.run(sequential))
            else:
                pipeline.roll(remaining, open_chunk, close_chunk, take)
            if tally.streak >= config.max_consecutive_failures:
                rt.emit(
                    {
                        "event": "circuit_open",
                        "consecutive_failures": tally.streak,
                        "interrupted": tally.streak,
                        **tally.circuit(),  # MASK5: why, before the exit
                    }
                )
                return 3
            if rt.stop_requested():
                raise _Stopped
    except _Stopped:
        rt.emit({"event": "stopped", "counts": totals, "pending": len(remaining)})
        return 130
    except _GaveUp as gave_up:
        rt.emit({"event": "gave_up", **gave_up.result.as_dict()})
        return 4
    finally:
        sql.close()
        cpu.close()
        for item in open_items.values():  # a chunk a fault left open
            item.end(error=True)
    rt.emit({"event": "done", "counts": totals, "pending": len(remaining)})
    return 1 if totals.get("failed") or totals.get("interrupted") else 0


# --------------------------------------------------------------------------
# the work-list SQL (run by the operator, read only)
# --------------------------------------------------------------------------

_SQL_HEAD = """BEGIN TRANSACTION READ ONLY;
\\pset tuples_only on
\\pset footer off
"""
_SQL_TAIL = "ROLLBACK;\n"

_IMAGE_COLUMNS = """json_build_object(
    'blob_sha256', encode(pi.blob_sha256, 'hex'),
    'storage_key', b.storage_key,
    'codec', b.codec,
    'url', u.url,
    'source', s.site,
    'product_id', pi.product_id,
    'head_id', {head},
    'rank', {rank},
    'position', pi.position,
    'source_class', pi.source_class)"""

_IMAGE_JOINS = """JOIN source s ON s.id = pi.source_id
JOIN raw.blob b ON b.sha256 = pi.blob_sha256
JOIN raw.capture c ON c.id = pi.origin_capture_id AND c.fetched_at = pi.origin_captured_at
JOIN raw.url u ON u.id = c.url_id
WHERE s.site = 'mfc'
  AND pi.state <> 'suppressed' AND pi.suppressed_at IS NULL
  AND b.visibility = 'visible'"""
#: SHOW1: the same joins and liveness, every site (the store work list).
_ANY_SITE_JOINS = _IMAGE_JOINS.replace("WHERE s.site = 'mfc'\n  AND ", "WHERE ")
_SITE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,49}")


def _uuid(value: object, what: str) -> str:
    if not isinstance(value, str) or not _UUID.match(value):
        raise ValueError(f"{what} must be a uuid, got {value!r}")
    return value.lower()


def ross_heads_sql(user_id: str) -> str:
    """Step 1 of Ross's work list, on the COORDINATOR database: the head of
    every live copy of one user, ranked by the copy's status (owned 0,
    ordered 1, wished 2, anything else 3), one JSON line each."""
    user = _uuid(user_id, "the user")
    return (
        _SQL_HEAD
        + f"""-- MASK-BATCH-1: the heads behind one user's live copies (coordinator db).
SELECT json_build_object('head_id', head_id, 'rank', rank) FROM (
  SELECT h.payload::jsonb->>'head_id' AS head_id,
         min(CASE s.payload::jsonb->>'status'
               WHEN 'owned' THEN 0 WHEN 'ordered' THEN 1 WHEN 'wished' THEN 2 ELSE 3 END) AS rank
  FROM facet_state h
  JOIN facet_state s
    ON s.user_id = h.user_id AND s.op = 'upsert'
   AND s.facet_key = 'occ/' || split_part(h.facet_key, '/', 2) || '/status'
  WHERE h.user_id = '{user}'::uuid AND h.op = 'upsert' AND h.facet_key LIKE 'occ/%/head'
  GROUP BY 1
) heads
WHERE head_id IS NOT NULL
ORDER BY rank, head_id;
"""
        + _SQL_TAIL
    )


def ross_worklist_sql(head_lines: Iterable[str]) -> str:
    """Step 2, on the SPINE database: every live MFC item image of those heads
    (and of the products merged into them), the heads' ranks carried."""
    heads: dict[str, int] = {}
    for line in head_lines:
        text = line.strip()
        if not text.startswith("{"):
            continue
        row = json.loads(text)
        head = _uuid(row.get("head_id"), "a head")
        rank = row.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank <= 9:
            raise ValueError(f"a head's rank must be an integer 0-9, got {rank!r}")
        heads[head] = min(rank, heads.get(head, rank))
    if not heads:
        raise ValueError("no heads in the input")
    values = ",\n  ".join(f"('{h}'::uuid, {r})" for h, r in sorted(heads.items()))
    columns = _IMAGE_COLUMNS.format(head="m.head_id", rank="m.rank")
    return (
        _SQL_HEAD
        + f"""-- MASK-BATCH-1: Ross's holdings, their MFC item images (spine db).
WITH RECURSIVE heads(head_id, rank) AS (VALUES
  {values}
),
members(product_id, head_id, rank) AS (
  SELECT head_id, head_id, rank FROM heads
  UNION
  SELECT p.id, m.head_id, m.rank FROM product p JOIN members m ON p.redirect_to_product_id = m.product_id
)
SELECT {columns}
FROM members m
JOIN product_image pi ON pi.product_id = m.product_id
{_IMAGE_JOINS}
ORDER BY m.rank, m.head_id, pi.position NULLS LAST, pi.blob_sha256;
"""
        + _SQL_TAIL
    )


def _shard_clause(shard: str | None) -> str:
    if shard is None:
        return ""
    m = re.fullmatch(r"(\d+)/(\d+)", shard)
    if not m or not 0 <= int(m[1]) < int(m[2]):
        raise ValueError(f"bad shard {shard!r}: want I/N with 0 <= I < N")
    return f"\n  AND get_byte(pi.blob_sha256, 0) % {int(m[2])} = {int(m[1])}"


def catalog_worklist_sql(shard: str | None = None) -> str:
    """The whole catalog's live MFC item images (spine db), rank 9; ``shard``
    ``I/N`` keeps the originals whose sha's first byte is I modulo N, so the
    list can be fetched in pieces."""
    where = _shard_clause(shard)
    columns = _IMAGE_COLUMNS.format(head="NULL", rank="9")
    return (
        _SQL_HEAD
        + f"""-- MASK-BATCH-1: the catalog's MFC item images (spine db).
SELECT {columns}
FROM product_image pi
{_IMAGE_JOINS}{where}
ORDER BY pi.product_id, pi.position NULLS LAST, pi.blob_sha256;
"""
        + _SQL_TAIL
    )


def stores_worklist_sql(
    since: str, *, scope: str = "showable", sites: str | None = None, rank: int = 8
) -> str:
    """SHOW1 step 3: every site's live images first seen since ``since`` (an
    ISO instant with its zone), newest first -- ONE cumulative list rebuilt
    daily (a row already done is skipped in memory, a duplicate kept once).
    ``scope`` ``showable`` keeps the sources the spine lets derive
    (derive_permitted silent or granted), ``all`` every source; ``sites`` (a
    comma list) keeps only those sites -- pass MASK_STORE_GRANTS' stores: a
    capture from any other store, seeded behind a picture's published mask,
    takes that mask offline (mask eligibility is pooled). ``rank`` 8 puts the
    list ahead of every rank-9 list whatever the file order; Ross's heads
    (0-3) still go first. The join on the capture's URL keeps out a row with
    no origin (F3: an originless seed takes masks offline too)."""
    try:
        instant = dt.datetime.fromisoformat(since.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"bad --since {since!r}: want an ISO instant") from exc
    if instant.tzinfo is None:
        raise ValueError(f"--since {since!r} needs a time zone (Z or +00:00)")
    if scope not in ("showable", "all"):
        raise ValueError(f"bad scope {scope!r}: showable or all")
    if isinstance(rank, bool) or not 0 <= rank <= 9:
        raise ValueError(f"a rank must be an integer 0-9, got {rank!r}")
    where = f"\n  AND pi.first_seen_at >= '{instant.astimezone(dt.UTC).isoformat()}'::timestamptz"
    if scope == "showable":
        where += "\n  AND s.derive_permitted IN ('silent', 'granted')"
    if sites is not None:
        names = [n.strip() for n in sites.split(",") if n.strip()]
        bad = [n for n in names if not _SITE.fullmatch(n)]
        if not names or bad:
            raise ValueError(f"bad site list {sites!r}: want site names, comma separated")
        where += "\n  AND s.site IN (" + ", ".join(f"'{n}'" for n in names) + ")"
    columns = _IMAGE_COLUMNS.format(head="NULL", rank=str(rank))
    return (
        _SQL_HEAD
        + f"""-- SHOW1: every site's images first seen since {instant.isoformat()} (spine db).
SELECT {columns}
FROM product_image pi
{_ANY_SITE_JOINS}{where}
ORDER BY pi.first_seen_at DESC, pi.product_id, pi.position NULLS LAST, pi.blob_sha256;
"""
        + _SQL_TAIL
    )


def official_gallery_worklist_sql(shard: str | None = None) -> str:
    """GAL3: the live MFC images the gallery-capture lane took from MFC's
    official gallery list (spine db: source_class ``manufacturer_press``),
    rank 9, sharded as ``catalog_worklist_sql``. Input to ``masks
    official-gallery`` and then to ``mask_batch run``."""
    where = _shard_clause(shard)
    columns = _IMAGE_COLUMNS.format(head="NULL", rank="9")
    return (
        _SQL_HEAD
        + f"""-- GAL3: MFC's official gallery list, as the gallery-capture lane recorded it (spine db).
SELECT {columns}
FROM product_image pi
{_IMAGE_JOINS}
  AND pi.source_class = 'manufacturer_press'{where}
ORDER BY pi.product_id, pi.position NULLS LAST, pi.blob_sha256;
"""
        + _SQL_TAIL
    )


# --------------------------------------------------------------------------
# the command line
# --------------------------------------------------------------------------


class Refused(Exception):
    """A request refused before anything was touched (exit status 2)."""


def build_backend(args: argparse.Namespace) -> MattingBackend:
    from .masks import make_backend

    backend = make_backend(
        args.backend, None, None, None, device=args.device, cpu_threads=args.cpu_threads
    )
    assert backend is not None
    return backend


#: The batch's bucket clients (MASK-PERF-3): a connection that cannot be
#: made, or a request that makes no progress -- a PUT's body sent in chunks,
#: each within the connect timeout (urllib3 2), its answer within the read
#: timeout -- is cut and retried with backoff (botocore's standard mode,
#: four attempts in all), not waited on for botocore's 60 s; TCP keepalive
#: keeps a pooled connection open through a pause between chunks. Live, a
#: TCP and TLS connect to the bucket takes 0.5 s and a PUT 0.15-0.4 s.
BUCKET_CONFIG: dict[str, Any] = {
    "connect_timeout": 10,
    "read_timeout": 30,
    "retries": {"mode": "standard", "total_max_attempts": 4},
    "tcp_keepalive": True,
}


def bucket_connections(writers: int, prefetch: int) -> int:
    """The connections each bucket client keeps: one for each PUT in flight
    (an image's original and copy at once, MASK-PERF-3) and each original
    read ahead. botocore keeps 10 and closes any it lent past them when it
    comes back: a new TCP and TLS connect, 0.5 s live, per PUT past them."""
    return max(10, 2 * writers + prefetch)


def _media_store(connections: int) -> ObjectStore:
    """The media bucket (S3_*), its client keeping ``connections`` (``_run``)."""
    from ..s3 import s3_client

    client = s3_client(max_pool_connections=connections, **BUCKET_CONFIG)
    return S3ObjectStore(client=client, bucket=get_settings().s3_bucket)


def _raw_store(args: argparse.Namespace, connections: int = 10) -> ObjectStore:
    """The raw bucket, read only; its client keeps ``connections`` (``_run``)."""
    if args.raw_dir:
        from ..object_store import LocalObjectStore

        return ReadOnlyStore(LocalObjectStore(args.raw_dir))
    try:
        return raw_store_from_env(connections, BUCKET_CONFIG)
    except RawStoreConfigError as exc:
        raise Refused(f"{exc} (or --raw-dir)") from exc


SERVICE = "mask-batch"
#: The events whose scalars and counts job.end carries.
_SUMMARIZED = ("plan", "start", "done", "stopped", "gave_up", "circuit_open")


def _run(
    args: argparse.Namespace, emit: Callable[[dict[str, Any]], None], run: telemetry.JobRun
) -> int:
    try:
        items, skipped = read_worklists(args.worklist)
        windows = parse_windows(args.busy_windows)
    except (OSError, ValueError) as exc:
        raise Refused(str(exc)) from exc
    residential_ip = args.residential_ip or os.environ.get("MASK_BATCH_RESIDENTIAL_IP")
    if not args.no_residential_check and not residential_ip:
        raise Refused(
            "the residential check needs --residential-ip (or MASK_BATCH_RESIDENTIAL_IP), "
            "or --no-residential-check for a local run"
        )
    on_gpu = args.device != "cpu"
    gates = Gates(
        windows=windows,
        gpu=(
            GpuGate(
                query=nvidia_smi_query(args.gpu_index),
                util_max=args.gpu_util_max,
                free_min_mib=args.gpu_free_min_mib,
                samples=args.gpu_samples,
                absent_ok=args.device == "auto",
            )
            if on_gpu
            else None
        ),
        residential=(
            None
            if args.no_residential_check
            else ResidentialCheck(
                expected_ip=str(residential_ip),
                fetch=public_ip(args.ip_echo_url),
                ttl=args.residential_ttl,
            )
        ),
        now=lambda: dt.datetime.now(dt.UTC),
    )
    config = BatchConfig(
        chunk_size=args.chunk_size,
        limit=args.limit,
        retry_failed=args.retry_failed,
        dry_run=args.dry_run,
        poll_seconds=args.poll_seconds,
        max_wait_seconds=args.max_wait_seconds,
        max_consecutive_failures=args.max_consecutive_failures,
        writers=args.writers,
        prefetch=args.prefetch,
        decode_slots=args.decode_slots,
        cpu_workers=(
            default_cpu_workers(args.writers, args.prefetch)
            if args.cpu_workers is None
            else args.cpu_workers
        ),
    )
    stop = threading.Event()
    previous: dict[int, Any] = {}
    if not args.dry_run:

        def on_signal(signum: int, frame: FrameType | None) -> None:
            stop.set()

        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, on_signal)

    media_store: ObjectStore | None = None
    raw_store: ObjectStore | None = None
    sessions: list[Session] = []
    engines: list[Engine] = []  # the ones made for this run
    if not args.dry_run:
        from .masks import open_stores

        connections = bucket_connections(args.writers, args.prefetch)
        raw_store = _raw_store(args, connections)
        opened, bucket = open_stores(args.sandbox)
        if args.sandbox is None:
            bucket = _media_store(connections)
        db = _sized(opened, args.writers)
        if db is not opened:
            engines.append(db.get_bind().engine)
        # each worker's original, copy and mask kept until it reads its copy back
        media_store = RecentWrites(bucket, keep=4 + 3 * args.writers)
        sessions.append(db)
    rt = Runtime(
        gates=gates,
        open_db=lambda: sessions[0],
        media_store=media_store,
        raw_store=raw_store,
        make_backend=lambda: build_backend(args),
        progress=Progress(Path(args.progress)),
        vram=lambda: VramSampler(vram_command(args.gpu_index) if on_gpu else None),
        sleep=time.sleep,
        monotonic=time.monotonic,
        emit=emit,
        stop_requested=stop.is_set,
        open_item=lambda n: run.open_item(name="item.chunk", attributes={"fc.chunk": n}),
        cpu_stages=lambda workers: _cpu_stages(workers, emit),
    )
    if skipped:
        emit({"event": "worklist", "items": len(items), "skipped_lines": skipped})
    try:
        return run_batch(items, config, rt)
    finally:
        for db in sessions:
            db.close()
        for engine in engines:
            engine.dispose()
        for number, handler in previous.items():
            signal.signal(number, handler)


#: What the CPU workers' fork server imports: this module, the main module
#: each worker runs again (app.workers.cpu_stages), and with it the stages'.
CPU_PRELOAD = ("app.cli.mask_batch",)


def _cpu_stages(workers: int, emit: Callable[[dict[str, Any]], None]) -> CpuStages:
    """The command line's CPU stages: ``workers`` processes (MASK5), each
    worker the pool replaces said on a line; none, the workers' threads."""
    if not workers:
        return INLINE
    return ProcessStages(
        workers,
        on_replaced=lambda why: emit({"event": "cpu_pool_replaced", "detail": why}),
        preload=CPU_PRELOAD,
    )


def _sql(args: argparse.Namespace) -> int:
    try:
        if args.query == "ross-heads":
            sql = ross_heads_sql(args.user)
        elif args.query == "ross-worklist":
            sql = ross_worklist_sql(Path(args.heads).read_text().splitlines())
        elif args.query == "official-gallery-worklist":
            sql = official_gallery_worklist_sql(args.shard)
        elif args.query == "stores-worklist":
            sql = stores_worklist_sql(
                args.since, scope=args.scope, sites=args.sites, rank=args.rank
            )
        else:
            sql = catalog_worklist_sql(args.shard)
    except (OSError, ValueError) as exc:
        raise Refused(str(exc)) from exc
    sys.stdout.write(sql)
    return 0


def _whole(minimum: int, maximum: int | None = None) -> Callable[[str], int]:
    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not a whole number: {text!r}") from None
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be {minimum} or more, got {value}")
        if maximum is not None and value > maximum:
            raise argparse.ArgumentTypeError(f"must be {maximum} or less, got {value}")
        return value

    return parse


def _seconds(
    minimum: float, maximum: float = math.inf, *, above: bool = False
) -> Callable[[str], float]:
    """Finite seconds from ``minimum`` (exclusive with ``above``) to ``maximum``."""

    def parse(text: str) -> float:
        try:
            value = float(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not a number of seconds: {text!r}") from None
        low = value > minimum if above else value >= minimum
        if not (math.isfinite(value) and low and value <= maximum):
            bound = f"above {minimum:g} s" if above else f"{minimum:g} s or more"
            if maximum < math.inf:
                bound += f" and at most {maximum:g} s"
            raise argparse.ArgumentTypeError(f"must be {bound}, got {text}")
        return value

    return parse


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m app.cli.mask_batch",
        description=(__doc__ or "").split("\n")[0],
        allow_abbrev=False,
    )
    p.add_argument("--sandbox", help="SQLite + local objects in DIR instead of DATABASE_URL/S3")
    sub = p.add_subparsers(dest="command", required=True)

    # no abbreviations: the wrapper sizes the container by the literal --device
    run = sub.add_parser(
        "run", help="mint masks for a work list (gated, chunked, resumable)", allow_abbrev=False
    )
    run.add_argument("--worklist", action="append", required=True, type=Path)
    run.add_argument("--progress", default="mask-batch-progress.jsonl")
    run.add_argument("--chunk-size", type=_whole(1), default=20)
    run.add_argument("--limit", type=_whole(0))
    run.add_argument("--retry-failed", action="store_true")
    run.add_argument(
        "--dry-run", action="store_true", help="plan only: nothing is opened or written"
    )
    run.add_argument(
        "--backend", choices=["birefnet-basekept", "birefnet"], default="birefnet-basekept"
    )
    run.add_argument("--device", choices=["cuda", "auto", "cpu"], default="cuda")
    run.add_argument("--cpu-threads", type=int)
    run.add_argument("--gpu-index", type=int, default=0)
    run.add_argument("--gpu-util-max", type=int, default=DEFAULT_GPU_UTIL_MAX)
    run.add_argument("--gpu-free-min-mib", type=_whole(1024), default=DEFAULT_GPU_FREE_MIN_MIB)
    run.add_argument("--gpu-samples", type=_whole(1), default=3)
    run.add_argument("--busy-windows", default=DEFAULT_BUSY_WINDOWS)
    run.add_argument("--residential-ip")
    run.add_argument("--no-residential-check", action="store_true")
    run.add_argument("--ip-echo-url", default=DEFAULT_IP_ECHO_URL)
    run.add_argument("--residential-ttl", type=_seconds(0.0, above=True), default=300.0)
    run.add_argument("--poll-seconds", type=_seconds(1.0, 3600.0), default=60.0)
    run.add_argument("--max-wait-seconds", type=_seconds(0.0))
    run.add_argument("--max-consecutive-failures", type=_whole(1), default=5)
    run.add_argument(
        "--writers",
        type=_whole(0, MAX_WRITERS),
        default=DEFAULT_WRITERS,
        help=(
            "images in flight at once, across chunks, each on a worker with a database "
            f"connection of its own (0: one at a time; default {DEFAULT_WRITERS}, up to "
            f"{MAX_WRITERS}: 24 connections with the batch's own)"
        ),
    )
    run.add_argument(
        "--prefetch",
        type=_whole(1, MAX_PREFETCH),
        default=DEFAULT_PREFETCH,
        help=f"originals read ahead of the workers (default {DEFAULT_PREFETCH})",
    )
    run.add_argument(
        "--decode-slots",
        type=_whole(1, MAX_DECODE_SLOTS),
        default=DECODED_AT_ONCE,
        help=(
            "copies decoded for the model at once, ~0.55 GB each at 4000x3000; a bigger "
            f"original takes one per {DECODE_UNIT_PIXELS} pixels (default {DECODED_AT_ONCE})"
        ),
    )
    run.add_argument(
        "--cpu-workers",
        type=_whole(0, MAX_CPU_WORKERS),
        help=(
            "worker processes for the pipeline's decodes, encodes and mask judgments (0: on "
            "the workers' threads; default the host's CPUs less two, at most writers + "
            f"prefetch and {MAX_CPU_WORKERS})"
        ),
    )
    run.add_argument("--raw-dir", help="read originals from this directory instead of RAW_S3_*")

    sql = sub.add_parser("sql", help="print a read-only work-list query")
    queries = sql.add_subparsers(dest="query", required=True)
    heads = queries.add_parser("ross-heads", help="coordinator db: one user's heads, ranked")
    heads.add_argument("--user", required=True)
    ross = queries.add_parser("ross-worklist", help="spine db: those heads' MFC item images")
    ross.add_argument("--heads", required=True)
    catalog = queries.add_parser("catalog-worklist", help="spine db: every MFC item image")
    catalog.add_argument("--shard")
    gallery = queries.add_parser(
        "official-gallery-worklist", help="spine db: MFC's official gallery list (GAL3)"
    )
    gallery.add_argument("--shard")
    stores = queries.add_parser(
        "stores-worklist", help="spine db: every site's images since an instant (SHOW1)"
    )
    stores.add_argument("--since", required=True, help="an ISO instant, e.g. 2026-10-23T21:21Z")
    stores.add_argument("--scope", default="showable", help="showable (default) or all")
    stores.add_argument("--sites", help="only these sites, comma separated (MASK_STORE_GRANTS')")
    stores.add_argument("--rank", type=int, default=8)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """``sql`` prints its query (the product, plain text); ``run`` is one root
    trace in the estate's lines (LG-U11): job.start, one ``app.<event>`` line
    per event (chunk, plan, paused ...), job.end with the counts and ``exit``."""
    args = _parser().parse_args(argv)
    if args.command == "sql":
        try:
            return _sql(args)
        except Refused as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 2
    log = telemetry.configure(SERVICE)
    telemetry.start_tracing(SERVICE)
    lines = EventLines(log, summarize=_SUMMARIZED)
    try:
        return telemetry.run_job(
            SERVICE,
            lambda run: _run(args, lines, run),
            logger=log,
            counts=lambda code: {**lines.summary, "exit": code},
            ok=lambda code: code == 0,
        )
    except Refused:
        return 2  # job.end says why (err)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
