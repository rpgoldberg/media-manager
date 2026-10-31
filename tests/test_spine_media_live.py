"""MM-REPORT-1, LIVE: the replay against the REAL spine.

Boots fc-aggregation's own ``src/server/main.ts`` (develop 34943676 or a
checkout of it, with its node_modules) against a spine Postgres migrated
through 0038, with the media listener bound to this host's NON-LOOPBACK
address (MEDIA_H2C_HOST, no MEDIA_H2C_DEV_ANY_HOST) and the allow list naming
``media-report``. There is no Linkerd here, so the test's channel adds the
``l5d-client-id`` header the inbound proxy would; the replay's own client
never does. The read half is the spine's real SpineRead/GetProductImages
over gRPC.

Proves: report -> GetProductImages returns the derivative and mask URLs;
withdrawn -> no mask URL, restored -> back; every rerun stores 0; a caller
with no mesh identity is UNAUTHENTICATED and the run stops (resumable).

Runs only when pointed at a spine:
    SPINE_LIVE_SRC=<fc-aggregation checkout with node_modules>
    SPINE_LIVE_PGPORT=<port of its test Postgres on 127.0.0.1>
      (fc-aggregation: sh tools/setup-testdb.sh <name> <port>)
    SPINE_LIVE_HOST=<a non-loopback IP of this host>  (default: discovered)
    SPINE_LIVE_LOG=<file for the spine's boot log>     (optional)
"""

from __future__ import annotations

import collections
import datetime as dt
import importlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import grpc
import pytest
from app.models import Base, ImageMask, PublicObject
from app.spine import replay as rp
from app.spine.media_client import RetryPolicy, SpineMediaClient, open_channel
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from spine_fakes import seed_copy

SRC = os.environ.get("SPINE_LIVE_SRC", "")
PGPORT = os.environ.get("SPINE_LIVE_PGPORT", "")
pytestmark = pytest.mark.skipif(
    not (SRC and PGPORT), reason="SPINE_LIVE_SRC and SPINE_LIVE_PGPORT name a real spine"
)

WORKER = "media-report.media.serviceaccount.identity.linkerd.app.mesh.estate"
BASE = "https://images.figurecollecting.com/d"
NOW_ISO = "2026-10-15T12:00:00Z"


def _free_port(host: str) -> int:
    with socket.socket() as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def _non_loopback() -> str:
    explicit = os.environ.get("SPINE_LIVE_HOST")
    if explicit:
        return explicit
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("10.255.255.255", 1))  # no packet is sent; picks the route's source
        return str(s.getsockname()[0])


class _Details(
    collections.namedtuple(
        "_Details",
        ("method", "timeout", "metadata", "credentials", "wait_for_ready", "compression"),
    ),
    grpc.ClientCallDetails,
):
    pass


class _AsLinkerdWould(grpc.UnaryUnaryClientInterceptor):  # type: ignore[misc]
    """What the spine's inbound proxy adds: the caller's mesh identity."""

    def intercept_unary_unary(self, continuation: Any, details: Any, request: Any) -> Any:
        metadata = [*(details.metadata or ()), ("l5d-client-id", WORKER)]
        return continuation(
            _Details(
                details.method,
                details.timeout,
                metadata,
                details.credentials,
                details.wait_for_ready,
                details.compression,
            ),
            request,
        )


@pytest.fixture(scope="module")
def spine(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    import psycopg2

    host = _non_loopback()
    assert not host.startswith("127."), f"{host} is loopback: set SPINE_LIVE_HOST"
    ports = {name: _free_port("127.0.0.1") for name in ("ingest", "read", "read_h2c")}
    media_port = _free_port(host)
    env = {
        **os.environ,
        "PGHOST": "127.0.0.1",
        "PGPORT": PGPORT,
        "PGUSER": "postgres",
        "PGPASSWORD": "x",
        "PGDATABASE": "postgres",
        "INGEST_HOST": "127.0.0.1",
        "INGEST_PORT": str(ports["ingest"]),
        "READ_PORT": str(ports["read"]),
        "INGEST_H2C_PORT": "off",
        "READ_H2C_PORT": str(ports["read_h2c"]),
        "MEDIA_H2C_PORT": str(media_port),
        "MEDIA_H2C_HOST": host,
        "MEDIA_ALLOWED_IDENTITIES": WORKER,
        "MEDIA_PUBLIC_BASE_URL": BASE,
    }
    env.pop("MEDIA_H2C_DEV_ANY_HOST", None)
    log: list[str] = []
    proc = subprocess.Popen(
        [str(Path(SRC) / "node_modules" / ".bin" / "tsx"), "src/server/main.ts"],
        cwd=SRC,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdout is not None
    out = proc.stdout
    threading.Thread(target=lambda: log.extend(iter(out.readline, "")), daemon=True).start()
    deadline = time.monotonic() + 90
    while not any("SpineIngest listening on" in line for line in log):
        assert proc.poll() is None, "".join(log)
        assert time.monotonic() < deadline, "".join(log)
        time.sleep(0.2)

    # read.v1 stubs from the spine's own contract package, for the read half
    from grpc_tools import protoc

    proto_root = Path(SRC) / "node_modules" / "@figurecollecting" / "ingest-contract" / "proto"
    gen = tmp_path_factory.mktemp("readstubs")
    rc = protoc.main(
        [
            "protoc",
            f"-I{proto_root}",
            f"--python_out={gen}",
            f"--grpc_python_out={gen}",
            str(proto_root / "read" / "v1" / "read.proto"),
        ]
    )
    assert rc == 0
    sys.path.insert(0, str(gen))
    read_pb2 = importlib.import_module("read.v1.read_pb2")
    read_grpc = importlib.import_module("read.v1.read_pb2_grpc")
    read_channel = grpc.insecure_channel(f"127.0.0.1:{ports['read_h2c']}")
    reader = read_grpc.SpineReadStub(read_channel)

    def images_of(product_id: str) -> list[dict[str, Any]]:
        answer = reader.GetProductImages(
            read_pb2.GetProductImagesRequest(product_ids=[product_id], now_iso=NOW_ISO),
            timeout=10,
        )
        products = json.loads(answer.images_json)["products"]
        return list(products[0]["images"]) if products else []

    pg = psycopg2.connect(
        host="127.0.0.1", port=int(PGPORT), user="postgres", password="x", dbname="postgres"
    )
    pg.autocommit = True
    try:
        yield {
            "url": f"http://{host}:{media_port}",
            "host": host,
            "media_port": media_port,
            "images_of": images_of,
            "pg": pg,
            "log": log,
        }
    finally:
        read_channel.close()
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        path = os.environ.get("SPINE_LIVE_LOG")
        if path:
            Path(path).write_text("".join(log))
        pg.close()
        sys.path.remove(str(gen))


def _q(pg: Any, sql: str, *args: Any) -> list[tuple[Any, ...]]:
    with pg.cursor() as cur:
        cur.execute(sql, args)
        return list(cur.fetchall()) if cur.description else []


@pytest.fixture()
def world(spine: dict[str, Any], tmp_path: Path) -> Iterator[dict[str, Any]]:
    """A media DB (SQLite) with three originals behind two clean copies, and
    the spine rows that captured them: product A depicts originals A and A2
    (one picture, two captures), product B depicts original B, whose mask is
    withdrawn in media-manager."""
    pg = spine["pg"]
    run = uuid.uuid4().hex[:8]
    site = f"mmreport-{run}"
    engine = create_engine(f"sqlite:///{tmp_path / 'media.db'}")
    Base.metadata.create_all(engine)
    db = Session(engine, expire_on_commit=False)
    a = seed_copy(
        db, f"{run}-A", extra_originals=(f"{run}-A2",), masks=(("mask-v1", "ok", f"{run}-mA"),)
    )
    b = seed_copy(db, f"{run}-B", masks=(("mask-v1", "withdrawn", f"{run}-mB"),))
    source = _q(
        pg,
        "INSERT INTO source (site, kind, derive_permitted, derive_reviewed_at) "
        "VALUES (%s, 'retailer', 'granted', now()) RETURNING id",
        site,
    )[0][0]

    def product(name: str, *origins: str) -> str:
        pid = _q(pg, "INSERT INTO product (name_en) VALUES (%s) RETURNING id", f"{site} {name}")[0][
            0
        ]
        for pos, origin in enumerate(origins):
            _q(
                pg,
                "INSERT INTO raw.blob (sha256, bytes, storage_key) VALUES (decode(%s, 'hex'), 2048, %s) "
                "ON CONFLICT (sha256) DO NOTHING",
                origin,
                f"raw-img/sha256/{origin[:2]}/{origin}.jpg",
            )
            _q(
                pg,
                "INSERT INTO product_image (product_id, source_id, blob_sha256, role, position, "
                "first_seen_at, last_seen_at) VALUES (%s, %s, decode(%s, 'hex'), 'gallery', %s, now(), now())",
                pid,
                source,
                origin,
                pos,
            )
        return str(pid)

    world = {
        "db": db,
        "a": a,
        "b": b,
        "pA": product("A", a["origin"], a["others"][0]),
        "pB": product("B", b["origin"]),
        "site": site,
    }
    try:
        yield world
    finally:
        db.close()
        engine.dispose()
        scope = "(SELECT id FROM source WHERE site = %s)"
        ours = "(SELECT id FROM product WHERE name_en LIKE %s)"
        like = f"{site} %"
        _q(pg, "SET session_replication_role = replica")
        _q(
            pg,
            f"DELETE FROM media_lease WHERE product_image_id IN (SELECT id FROM product_image WHERE source_id IN {scope})",
            site,
        )
        _q(
            pg,
            f"DELETE FROM image_mask WHERE image_id IN (SELECT id FROM image WHERE product_id IN {ours})",
            like,
        )
        _q(pg, f"DELETE FROM image WHERE product_id IN {ours}", like)
        _q(pg, f"DELETE FROM product_image WHERE source_id IN {scope}", site)
        _q(pg, "DELETE FROM product WHERE name_en LIKE %s", like)
        _q(pg, "DELETE FROM source WHERE site = %s", site)
        _q(
            pg,
            "DELETE FROM media_mask_tombstone WHERE mask_sha256 = decode(%s, 'hex')",
            b["masks"][0],
        )
        _q(pg, "SET session_replication_role = origin")


def _replay(db: Session, url: str, *, identity: bool = True) -> tuple[int, list[dict[str, Any]]]:
    lines: list[dict[str, Any]] = []

    def connect() -> SpineMediaClient:
        channel = open_channel(url)
        if identity:
            channel = grpc.intercept_channel(channel, _AsLinkerdWould())
        return SpineMediaClient(channel, policy=RetryPolicy(max_attempts=2, initial_backoff=0.05))

    code = rp.run_replay(db, rp.ReplayConfig(lease_seconds=60), connect=connect, emit=lines.append)
    return code, lines


def _sum(lines: list[dict[str, Any]], rpc: str, key: str) -> int:
    return sum(line[rpc][key] for line in lines if line["event"] == "batch")


def _set_mask(db: Session, clean: str, status: str, reason: str | None) -> None:
    row = db.execute(select(ImageMask).where(ImageMask.source_sha256 == clean)).scalar_one()
    row.status, row.reason = status, reason
    row.updated_at = dt.datetime.now(dt.UTC)
    db.commit()


def test_the_replay_lands_on_the_real_spine_and_the_read_sends_it(
    spine: dict[str, Any], world: dict[str, Any]
) -> None:
    db, a, b, pg = world["db"], world["a"], world["b"], spine["pg"]
    images_of: Callable[[str], list[dict[str, Any]]] = spine["images_of"]
    mask_a, mask_b = a["masks"][0], b["masks"][0]

    code, lines = _replay(db, spine["url"])
    assert code == rp.EXIT_DONE, lines
    batch = next(line for line in lines if line["event"] == "batch")
    assert batch["matched"] == 3 and batch["unrequested"] == 0
    assert batch["ReportDerivative"]["calls"] == 3
    assert batch["ReportDerivative"]["states"] == {"derived": 3}
    assert batch["ReportMask"]["stored"] >= 1
    assert batch["WithdrawMask"]["stored"] == 1  # B's withdrawal: a tombstone
    tombstones = _q(
        pg,
        "SELECT count(*) FROM media_mask_tombstone WHERE mask_sha256 = decode(%s, 'hex')",
        mask_b,
    )
    assert tombstones == [(1,)]
    states = _q(
        pg,
        "SELECT pi.state, count(*) FROM product_image pi JOIN source s ON s.id = pi.source_id "
        "WHERE s.site = %s GROUP BY 1",
        world["site"],
    )
    assert states == [("derived", 3)]

    shown_a = images_of(world["pA"])
    assert {i.get("url") for i in shown_a} == {f"{BASE}/{a['clean']}"}  # A and A2: one picture
    assert all(i.get("mask", {}).get("url") == f"{BASE}/{mask_a}" for i in shown_a)
    shown_b = images_of(world["pB"])
    assert [i.get("url") for i in shown_b] == [f"{BASE}/{b['clean']}"]
    assert all("url" not in (i.get("mask") or {}) for i in shown_b)  # withdrawn: no mask url
    first = next(i for i in shown_a)
    assert first.get("thumbhash") and first.get("dominantColor") == "#112233"

    code, lines = _replay(db, spine["url"])  # a rerun changes nothing
    assert code == rp.EXIT_DONE
    for rpc in ("ReportDerivative", "ReportMask", "WithdrawMask", "RestoreMask"):
        assert _sum(lines, rpc, "stored") == 0, rpc

    _set_mask(db, a["clean"], "withdrawn", "store_revoked")
    code, lines = _replay(db, spine["url"])
    assert code == rp.EXIT_DONE and _sum(lines, "WithdrawMask", "stored") >= 1
    assert all("url" not in (i.get("mask") or {}) for i in images_of(world["pA"]))
    code, lines = _replay(db, spine["url"])
    assert _sum(lines, "WithdrawMask", "stored") == 0 and _sum(lines, "ReportMask", "stored") == 0

    _set_mask(db, a["clean"], "ok", None)
    code, lines = _replay(db, spine["url"])
    assert code == rp.EXIT_DONE and _sum(lines, "RestoreMask", "stored") >= 1
    assert all(i.get("mask", {}).get("url") == f"{BASE}/{mask_a}" for i in images_of(world["pA"]))
    code, lines = _replay(db, spine["url"])
    for rpc in ("ReportDerivative", "ReportMask", "WithdrawMask", "RestoreMask"):
        assert _sum(lines, rpc, "stored") == 0, rpc
    leases = _q(
        pg,
        "SELECT count(*) FROM media_lease l JOIN product_image pi ON pi.id = l.product_image_id "
        "JOIN source s ON s.id = pi.source_id WHERE s.site = %s",
        world["site"],
    )
    assert leases == [(0,)]


def test_a_copy_of_unknown_width_lands_and_so_does_its_mask(
    spine: dict[str, Any], world: dict[str, Any]
) -> None:
    db, a = world["db"], world["a"]
    copy = db.get(PublicObject, a["clean"])
    assert copy is not None
    copy.width = 0  # the real spine refuses a DERIVED report with no width
    db.commit()
    code, lines = _replay(db, spine["url"])
    assert code == rp.EXIT_DONE, lines
    batch = next(line for line in lines if line["event"] == "batch")
    assert batch["ReportDerivative"]["refused"] == {} and batch["ReportMask"]["unregistered"] == 0
    assert batch["ReportMask"]["stored"] >= 1
    assert any("stored as unknown" in w for w in batch["warnings_sample"]), batch
    mask_url = f"{BASE}/{a['masks'][0]}"
    assert all(i.get("mask", {}).get("url") == mask_url for i in spine["images_of"](world["pA"]))


def test_originals_the_spine_has_no_live_depiction_for_are_orphans_not_failures(
    spine: dict[str, Any], world: dict[str, Any]
) -> None:
    db, pg, site = world["db"], spine["pg"], world["site"]
    run = site.removeprefix("mmreport-")
    never = seed_copy(db, f"{run}-N", masks=(("mask-v1", "ok", f"{run}-mN"),))
    stamped = seed_copy(db, f"{run}-T", masks=(("mask-v1", "ok", f"{run}-mT"),))
    # T's only depiction carries a takedown stamp: the spine never leases it
    pid = _q(pg, "INSERT INTO product (name_en) VALUES (%s) RETURNING id", f"{site} T")[0][0]
    _q(
        pg,
        "INSERT INTO raw.blob (sha256, bytes, storage_key) VALUES (decode(%s, 'hex'), 2048, %s) "
        "ON CONFLICT (sha256) DO NOTHING",
        stamped["origin"],
        f"raw-img/sha256/{stamped['origin'][:2]}/{stamped['origin']}.jpg",
    )
    _q(
        pg,
        "INSERT INTO product_image (product_id, source_id, blob_sha256, role, position, "
        "first_seen_at, last_seen_at, suppressed_at) VALUES (%s, (SELECT id FROM source "
        "WHERE site = %s), decode(%s, 'hex'), 'gallery', 0, now(), now(), now())",
        pid,
        site,
        stamped["origin"],
    )
    code, lines = _replay(db, spine["url"])
    assert code == rp.EXIT_DONE, lines
    batches = [line for line in lines if line["event"] == "batch"]
    assert sorted(o for b in batches for o in b["orphans"]) == sorted(
        [never["origin"], stamped["origin"]]
    )
    assert _sum(lines, "ReportMask", "orphan") == 2
    assert _sum(lines, "ReportMask", "unregistered") == 0
    assert all(b["ReportMask"]["refused"] == {} for b in batches)
    assert any("FAILED_PRECONDITION" in w for b in batches for w in b["warnings_sample"])
    assert _sum(lines, "ReportMask", "stored") >= 1  # A's mask still lands


def test_a_newer_copy_of_a_derived_original_never_replaces_the_one_the_spine_shows(
    spine: dict[str, Any], world: dict[str, Any]
) -> None:
    """REPLAY-REDERIV: A2 (a capture behind A's copy) is derived with that
    copy; media then mints a newer copy from A2, and A2 is depicted again on
    another product. The real spine absorbs the newer copy (its registry is
    write-once per original) and refuses its mask: ``kept_other``, exit 0,
    and the read keeps sending the held copy and its mask on both products."""
    db, a, pg, site = world["db"], world["a"], spine["pg"], world["site"]
    images_of: Callable[[str], list[dict[str, Any]]] = spine["images_of"]
    held_url, held_mask = f"{BASE}/{a['clean']}", f"{BASE}/{a['masks'][0]}"
    code, lines = _replay(db, spine["url"])
    assert code == rp.EXIT_DONE, lines

    a2 = a["others"][0]
    run = site.removeprefix("mmreport-")
    newer = seed_copy(db, f"{run}-N", minted_by=a2, masks=(("mask-v1", "ok", f"{run}-mN"),))
    ((again,),) = _q(pg, "INSERT INTO product (name_en) VALUES (%s) RETURNING id", f"{site} A2")
    _q(
        pg,
        "INSERT INTO product_image (product_id, source_id, blob_sha256, role, position, "
        "first_seen_at, last_seen_at) VALUES (%s, (SELECT id FROM source WHERE site = %s), "
        "decode(%s, 'hex'), 'gallery', 0, now(), now())",
        again,
        site,
        a2,
    )
    code, lines = _replay(db, spine["url"])
    assert code == rp.EXIT_DONE, lines
    assert _sum(lines, "ReportMask", "kept_other") == 1
    assert _sum(lines, "ReportMask", "unregistered") == 0
    assert any(
        "different bytes" in w for b in lines if b["event"] == "batch" for w in b["warnings_sample"]
    )
    held = _q(
        pg,
        "SELECT encode(derivative_sha256, 'hex') FROM image WHERE origin_blob_sha256 = decode(%s, 'hex')",
        a2,
    )
    assert held == [(a["clean"],)]  # the newer copy was never registered
    for product in (str(again), world["pA"]):
        shown = images_of(product)
        assert shown and {i.get("url") for i in shown} == {held_url}, shown
        assert all(i.get("mask", {}).get("url") == held_mask for i in shown), shown

    code, lines = _replay(db, spine["url"])  # nothing pending: an orphan, still not a failure
    assert code == rp.EXIT_DONE, lines
    assert _sum(lines, "ReportMask", "orphan") == 1 and _sum(lines, "ReportMask", "kept_other") == 0
    assert newer["clean"] not in json.dumps(images_of(str(again)))


def test_a_caller_without_a_mesh_identity_is_refused_and_the_run_stops(
    spine: dict[str, Any], world: dict[str, Any]
) -> None:
    code, lines = _replay(world["db"], spine["url"], identity=False)
    assert code == rp.EXIT_STOPPED
    assert lines[-1]["event"] == "stopped" and "UNAUTHENTICATED" in lines[-1]["why"]


@pytest.mark.parametrize("allowed", [True, False], ids=["allowlisted", "off-the-allowlist"])
def test_u12_the_spine_continues_each_items_trace(
    spine: dict[str, Any], world: dict[str, Any], monkeypatch: pytest.MonkeyPatch, allowed: bool
) -> None:
    """LG-U12, fc-shared's job model: the run is one root trace and each leased
    batch an ITEM, a new root trace linked to the run. Every SpineMedia call's
    rpc.in line, written by the spine's own withRequestLog (fc-aggregation
    LG-ADOPT-1), carries its ITEM's trace id: never the run's, and one per
    batch. Here the listener is an IP, so the test puts it on the propagation
    allowlist; in the cluster it is a *.svc.cluster.local name. Off the
    allowlist no header leaves, and the spine starts its own traces."""
    from app.telemetry import tracing as tr
    from app.telemetry.log import Logger

    monkeypatch.setenv("OTEL_PROPAGATION_ALLOWLIST", spine["host"] if allowed else "*.svc")
    tr.start_tracing("mm-report")
    ours: list[dict[str, Any]] = []
    log = Logger(service="mm-report", env={}, sink=lambda line: ours.append(json.loads(line)))
    items: list[str] = []

    def connect() -> SpineMediaClient:
        channel = grpc.intercept_channel(open_channel(spine["url"]), _AsLinkerdWould())
        return SpineMediaClient(channel, policy=RetryPolicy(max_attempts=2, initial_backoff=0.05))

    def body(run: tr.JobRun) -> int:
        def item(index: int, fn: Callable[[], Any]) -> Any:
            def one() -> Any:
                items.append(tr.current_traceparent() or "")
                return fn()

            return run.item(one, name="item.batch", attributes={"fc.batch": index})

        config = rp.ReplayConfig(lease_seconds=60, batch_size=1)
        return rp.run_replay(world["db"], config, connect=connect, emit=lambda _: None, item=item)

    seen = len(spine["log"])
    assert tr.run_job("mm-report", body, logger=log) == rp.EXIT_DONE
    run_trace = ours[0]["trace_id"]
    item_traces = {tp.split("-")[1] for tp in items}
    assert len(item_traces) == len(items) >= 2 and run_trace not in item_traces

    def media_lines() -> list[dict[str, Any]]:
        out = []
        for raw in list(spine["log"])[seen:]:
            if raw.startswith("{") and '"rpc.in"' in raw:
                line = json.loads(raw)
                if str(line.get("call", "")).startswith("media.v1.SpineMedia/"):
                    out.append(line)
        return out

    deadline = time.monotonic() + 10
    while not media_lines() and time.monotonic() < deadline:
        time.sleep(0.1)
    if not any('"rpc.in"' in raw for raw in spine["log"]):
        pytest.skip("this spine writes no rpc.in lines: it predates fc-aggregation LG-ADOPT-1")
    lines = media_lines()
    assert {x["call"] for x in lines} >= {
        "media.v1.SpineMedia/LeasePendingImages",
        "media.v1.SpineMedia/ReportDerivative",
        "media.v1.SpineMedia/ReportMask",
    }
    assert {x["peer"] for x in lines} == {WORKER}
    assert all(x["code"] == "ok" for x in lines), lines
    traces = {x["trace_id"] for x in lines}
    if allowed:
        # The spine sees each ITEM's trace, never the run's (the last batch, withdrawals
        # only when there are none, makes no call); each batch's lease is in its own trace.
        assert traces <= item_traces and len(traces) >= 2, (lines, items)
        leases = [x["trace_id"] for x in lines if x["call"].endswith("/LeasePendingImages")]
        assert len(leases) == len(set(leases)) >= 2
    else:
        assert not traces & (item_traces | {run_trace}) and len(traces) == len(lines), lines
    evidence = os.environ.get("SPINE_LIVE_U12_EVIDENCE")
    if evidence:
        Path(f"{evidence}-{'allowlisted' if allowed else 'off-the-allowlist'}.json").write_text(
            json.dumps({"run": ours, "items": items, "spine_rpc_in": lines}, indent=1)
        )


def test_the_media_listener_is_not_on_loopback(spine: dict[str, Any]) -> None:
    with socket.socket() as s, pytest.raises(ConnectionRefusedError):
        s.connect(("127.0.0.1", spine["media_port"]))
