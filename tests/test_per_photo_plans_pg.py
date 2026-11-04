"""MASKRATE: every query the WSL mask batch and the cluster display seeder
make for one photo reads the media tables by index, never whole.

The tables grow by ~8k rows an hour (the display drain), so a per-photo
query that walks one gets slower every hour: in prod (Nov 2026, ~900k rows
each) three of them cost the batch half its rate -- figure_galleries read by
image_id with no index, the candidate check's NOT IN (subquery) (Postgres
cannot anti-join it: a hashed SubPlan reads all of image_masks and images on
every call) and the footprint's OR across a join (a hash join of both whole
tables on every turn of its loop).

Run on PostgreSQL 17 (prod's major) in a disposable container, migrated to
head and filled with ROWS-scale tables (>= 300k rows in each table the
queries read), ANALYZE'd, the planner left to choose. Each statement the
code runs is captured as it reaches the driver and EXPLAINed with its own
parameters: a read of a media table must be an index probe bounded on the
index's leading column (a Seq Scan, a hashed subquery over the whole table,
or an index walked end to end fails). Hosts without docker skip.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import io
import os
import re
import shutil
import string
import subprocess
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from app.models import (
    DisplayDepiction,
    FigureGallery,
    ImageMask,
    ImageVersion,
    PublicObject,
    PublicObjectImage,
)
from app.models import Image as ImageModel
from app.object_store import LocalObjectStore
from app.publication import gallery_origins, mask_refusal
from app.spine import copies
from app.workers import display_copy as dc
from app.workers.mask import MASK_PIPELINE_VERSION
from app.workers.mask_job import (
    _candidates,
    open_mask_batch,
    original_settled,
    settle_clean_copy,
)
from app.workers.mask_policy import ImageOrigin
from app.workers.mask_seed import (
    picture_keys,
    seed_clean_image,
    seed_footprint,
    seed_neighbourhood,
    seed_with_retry,
)
from app.workers.matting import StubMattingBackend
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from sqlalchemy import Engine, Select, create_engine, event, func, or_, select, text
from sqlalchemy.orm import Session, sessionmaker

from test_migration_chain_replay import _run_alembic

pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="needs docker"),
    pytest.mark.timeout(600),  # the first test fills the tables (its fixture's setup)
]

IMAGE = "postgres:17-alpine"
#: seconds the container lives at most (the module takes a few minutes)
LIFETIME = 1800
#: a throwaway server: nothing it writes needs to survive a crash
THROWAWAY = [
    *("-c", "fsync=off", "-c", "synchronous_commit=off", "-c", "full_page_writes=off"),
    *("-c", "shared_buffers=512MB", "-c", "max_wal_size=8GB"),
]
#: rows per media table (images, image_versions, public_objects' copies,
#: figure_galleries; image_masks 19 in 20 of them; public_object_images more)
ROWS = 320_000

#: The media tables a per-photo query may only probe.
MEDIA_TABLES = (
    "images",
    "image_versions",
    "public_objects",
    "public_object_images",
    "image_masks",
    "figure_galleries",
    "display_originals",
    "display_depictions",
)

#: The display queue holds every image's original, 1 in 40 pending (1 in 80
#: of those still claimed), one depiction each (unlinked while pending).
#: Every 25th image is a second capture of the copy before it (multi-hop
#: links); 1 image in 97 soft-deleted, 1 gallery link in 89; 19 copies in 20
#: settled at this mask version (16 ok with a registered mask, 1 failed, 2
#: ineligible), the 20th not yet.
FILL = """
INSERT INTO images (sha256, pixel_sha256, mime, width, height, bytes, storage_key,
                    created_at, deleted_at)
  SELECT encode(sha256(convert_to('img:' || g, 'UTF8')), 'hex'),
         encode(sha256(convert_to('px:' || g, 'UTF8')), 'hex'),
         'image/jpeg', 4000, 3000, 2000000, 'orig/' || g, now(),
         CASE WHEN g % 97 = 0 THEN now() END
  FROM generate_series(1, $rows) g;
INSERT INTO image_versions (image_id, version_no, derived_from_version, transform_spec,
                            mime, storage_key, visibility, age_rating, created_at)
  SELECT id, 2, 1, '{}', 'image/jpeg', 'clean/' || id, 'public', 0, now() FROM images;
INSERT INTO public_objects (sha256, kind, mime, bytes, width, height, storage_key,
                            image_version_id, pixel_sha256, own_pixel_sha256, created_at)
  SELECT encode(sha256(convert_to('copy:' || v.image_id, 'UTF8')), 'hex'), 'sanitized',
         'image/jpeg', 1, 1, 1, 'copy/' || v.image_id, v.id, i.pixel_sha256,
         encode(sha256(convert_to('own:' || v.image_id, 'UTF8')), 'hex'), now()
  FROM image_versions v JOIN images i ON i.id = v.image_id;
INSERT INTO public_object_images (sha256, image_id)
  SELECT p.sha256, v.image_id FROM public_objects p
  JOIN image_versions v ON v.id = p.image_version_id;
INSERT INTO public_object_images (sha256, image_id)
  SELECT p.sha256, v.image_id + 1 FROM public_objects p
  JOIN image_versions v ON v.id = p.image_version_id
  WHERE v.image_id % 25 = 0 AND v.image_id < $rows;
INSERT INTO public_objects (sha256, kind, mime, bytes, width, height, storage_key, created_at)
  SELECT encode(sha256(convert_to('mask:' || g, 'UTF8')), 'hex'), 'mask', 'image/png',
         1, 1, 1, 'mask/' || g, now()
  FROM generate_series(1, $rows) g WHERE g % 20 BETWEEN 1 AND 16;
INSERT INTO image_masks (source_sha256, pipeline_version, status, mask_sha256,
                         created_at, updated_at)
  SELECT encode(sha256(convert_to('copy:' || g, 'UTF8')), 'hex'), '$version',
         CASE WHEN g % 20 BETWEEN 1 AND 16 THEN 'ok'
              WHEN g % 20 = 17 THEN 'failed' ELSE 'ineligible' END,
         CASE WHEN g % 20 BETWEEN 1 AND 16
              THEN encode(sha256(convert_to('mask:' || g, 'UTF8')), 'hex') END,
         now(), now()
  FROM generate_series(1, $rows) g WHERE g % 20 <> 0;
INSERT INTO figure_galleries (figure_id, source_url, image_id, position, source,
                              created_at, updated_at, deleted_at)
  SELECT 'fig-' || (id / 4), 'https://a.example/' || id || '.jpg', id, 0,
         CASE WHEN id % 3 = 0 THEN 'amiami' ELSE 'mfc' END, now(), now(),
         CASE WHEN id % 89 = 0 THEN now() END
  FROM images;
INSERT INTO display_originals (origin_sha256, storage_key, codec, first_seen_at, state,
                               attempts, claimed_until, force_seed, created_at, updated_at)
  SELECT encode(sha256(convert_to('img:' || g, 'UTF8')), 'hex'), 'raw/' || g, 'identity',
         now() - g * interval '1 second',
         CASE WHEN g % 40 = 0 THEN 'pending' WHEN g % 97 = 1 THEN 'stuck' ELSE 'seeded' END,
         0, CASE WHEN g % 80 = 0 THEN now() + interval '1 hour' END, false, now(), now()
  FROM generate_series(1, $rows) g;
INSERT INTO display_depictions (product_image_id, origin_sha256, product_id, origin_url,
                                source_site, first_seen_at, linked_at, reported_at, enqueued_at)
  SELECT md5('pi:' || g)::uuid::text, encode(sha256(convert_to('img:' || g, 'UTF8')), 'hex'),
         md5('prod:' || (g / 4))::uuid::text, 'https://a.example/' || g || '.jpg', 'mfc',
         now() - g * interval '1 second',
         CASE WHEN g % 40 <> 0 THEN now() END, CASE WHEN g % 40 <> 0 THEN now() END, now()
  FROM generate_series(1, $rows) g;
ANALYZE;
"""


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


@contextlib.contextmanager
def filled_postgres() -> Iterator[sessionmaker[Session]]:
    """A disposable postgres:17 (removed by its container id), migrated to
    head and filled (FILL). It shuts itself down after LIFETIME seconds, so
    a run killed before its teardown (pytest-timeout exits the process)
    leaves no container behind (--rm removes it once it stops)."""
    started = subprocess.run(
        ["docker", "run", "-d", "--rm", "-e", "POSTGRES_PASSWORD=postgres"]
        + ["-p", "127.0.0.1::5432", "--entrypoint", "timeout", IMAGE]
        + ["-s", "INT", str(LIFETIME), "docker-entrypoint.sh", "postgres", *THROWAWAY],
        capture_output=True,
        text=True,
    )
    if started.returncode != 0:
        pytest.skip(f"could not start postgres: {started.stderr}")
    container = started.stdout.strip()
    if log := os.environ.get("PG_TEST_CONTAINER_LOG"):  # the operator's: what to remove by id
        with open(log, "a") as out:
            out.write(f"{container}\n")
    try:
        port = (
            subprocess.run(
                ["docker", "port", container, "5432/tcp"],
                capture_output=True,
                text=True,
                check=True,
            )
            .stdout.split()[0]
            .rsplit(":", 1)[-1]
        )
        url = f"postgresql+psycopg2://postgres:postgres@127.0.0.1:{port}/postgres"
        engine = create_engine(url, pool_size=4)
        deadline = time.monotonic() + 60
        while True:
            try:
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.5)
        up = _run_alembic("upgrade", "head", database_url=url)
        assert up.returncode == 0, up.stdout + up.stderr
        with engine.begin() as conn:
            conn.exec_driver_sql("SET LOCAL session_replication_role = replica")  # no FK checks
            # no parameters: the driver passes the SQL's % (modulo) through as it is
            conn.execution_options(no_parameters=True).exec_driver_sql(
                string.Template(FILL).substitute(rows=ROWS, version=MASK_PIPELINE_VERSION)
            )
        yield sessionmaker(bind=engine, expire_on_commit=False)
        engine.dispose()
    finally:
        subprocess.run(["docker", "rm", "-f", container], capture_output=True, text=True)


@pytest.fixture(scope="module")
def pg() -> Iterator[sessionmaker[Session]]:
    with filled_postgres() as factory:
        yield factory


# --------------------------------------------------------------------------
# the plan check
# --------------------------------------------------------------------------


def _captured(engine: Engine, run: Callable[[], Any]) -> list[tuple[str, Any]]:
    """Every statement ``run`` sends through ``engine``, with its parameters."""
    statements: list[tuple[str, Any]] = []

    def record(*args: Any) -> None:
        statements.append((args[2], args[3]))  # (conn, cursor, statement, parameters, ...)

    event.listen(engine, "before_cursor_execute", record)
    try:
        run()
    finally:
        event.remove(engine, "before_cursor_execute", record)
    return statements


def _leading_column(conn: Any, index: str) -> str:
    return str(
        conn.exec_driver_sql(
            "SELECT a.attname FROM pg_index i JOIN pg_attribute a"
            " ON a.attrelid = i.indrelid AND a.attnum = i.indkey[0]"
            " WHERE i.indexrelid = %(ix)s::regclass",
            {"ix": index},
        ).scalar_one()
    )


def _whole_reads(conn: Any, node: dict[str, Any], used: set[str]) -> list[str]:
    """Each read of a MEDIA_TABLES table in the plan ``node`` that is no index
    probe bounded on the index's leading column: a Seq Scan (parallel or
    not), or an index walked whole (no Index Cond, or one on a later
    column). ``used`` collects the indexes probed, as ``table.index``."""
    found: list[str] = []
    table = node.get("Relation Name")
    if table in MEDIA_TABLES and node["Node Type"] != "ModifyTable":  # an UPDATE's target
        probes = [node] if "Index Name" in node else []
        stack = [] if probes else list(node.get("Plans", []))
        while stack:  # a Bitmap Heap Scan reads through its Bitmap Index Scans
            child = stack.pop()
            probes += [child] if child["Node Type"] == "Bitmap Index Scan" else []
            stack += child.get("Plans", [])
        bounded = bool(probes)
        for probe in probes:
            column = _leading_column(conn, probe["Index Name"])
            bounded &= bool(re.search(rf"\(\(?{column}\b", probe.get("Index Cond", "")))
            used.add(f"{table}.{probe['Index Name']}")
        if not bounded:
            found.append(f"{node['Node Type']} on {table} {node.get('Index Name', '')}".strip())
    for child in node.get("Plans", []):
        found += _whole_reads(conn, child, used)
    return found


def _walks(
    factory: sessionmaker[Session], statements: list[tuple[str, Any]], used: set[str]
) -> list[str]:
    """The whole-table reads of every SELECT, UPDATE and DELETE among
    ``statements``, each named with its statement."""
    walked: list[str] = []
    with factory() as db:
        conn = db.connection()
        for statement, parameters in statements:
            if not statement.lstrip().upper().startswith(("SELECT", "UPDATE", "DELETE")):
                continue
            explained = conn.exec_driver_sql(
                "EXPLAIN (FORMAT JSON) " + statement, parameters
            ).scalar_one()
            walked += [
                f"{w}: {' '.join(statement.split())}"
                for w in _whole_reads(conn, explained[0]["Plan"], used)
            ]
    return walked


def _png(label: str, note: str | None = None) -> bytes:
    """A small picture of its own per ``label``; ``note`` makes other bytes
    of the same pixels (a second capture)."""
    shade = int(_sha(label)[:6], 16)
    picture = Image.new("RGB", (48, 64), (shade % 256, shade // 256 % 256, shade // 65536))
    picture.putpixel((3, 5), (255 - shade % 256, 0, 0))
    out = io.BytesIO()
    info = PngInfo()
    if note is not None:
        info.add_text("Comment", note)
    picture.save(out, format="PNG", pnginfo=info)
    return out.getvalue()


class _Model(StubMattingBackend):
    """The stub's arithmetic standing in for a mask MODEL (publishes)."""

    name = "stub-model"
    publishes_masks = True


def _seed(db: Session, store: LocalObjectStore, data: bytes, n: str) -> PublicObject:
    return seed_with_retry(
        db,
        store,
        data,
        origins=[ImageOrigin(url=f"https://a.example/plan-{n}.png", source="amiami")],
        figure_id=f"plan-{n}",
        seed=seed_clean_image,
    )


# --------------------------------------------------------------------------
# the three per-photo reads prod walked whole
# --------------------------------------------------------------------------


def test_an_images_gallery_links_are_read_by_its_index(pg: sessionmaker[Session]) -> None:
    """gallery_origins: figure_galleries by image_id, for every image behind a
    copy judged (the batch's settle, the seeder's withdraw check)."""
    used: set[str] = set()
    with pg() as db:
        image_id = db.execute(select(func.min(FigureGallery.image_id))).scalar_one()
        statements = _captured(db.get_bind(), lambda: gallery_origins(db, image_id))
    assert _walks(pg, statements, used) == []
    assert used == {"figure_galleries.ix_figure_galleries_image_id"}


def test_a_photos_candidate_check_probes_its_mask_and_captures(
    pg: sessionmaker[Session],
) -> None:
    """settle_clean_copy's _candidates(only=[copy]): the settled and the
    dead-capture checks are anti-joins probing image_masks and
    public_object_images by the copy's sha, re-driven or not."""
    used: set[str] = set()
    copy = _sha("copy:20")  # not yet settled
    with pg() as db:
        engine = db.get_bind()
        statements = _captured(
            engine,
            lambda: [
                _candidates(db, after=None, page_size=1, redrive=redrive, only=(copy,))
                for redrive in ((), ("failed",), ("failed", "ineligible"))
            ],
        )
    assert _walks(pg, statements, used) == []
    assert {
        "image_masks.uq_image_mask_source_version",
        "public_object_images.public_object_images_pkey",
    } <= used


@pytest.mark.parametrize(
    "keys",
    [
        # a picture with captures and copies linked (the loop turns until its fixed point)
        lambda: {"shas": {_sha("img:25"), _sha("copy:25")}, "digests": {_sha("px:25")}},
        # a new picture: nothing behind its keys
        lambda: {"shas": {_sha("new"), _sha("new-copy")}, "digests": {_sha("new-px")}},
        # a picture that will not decode: its byte sha alone
        lambda: {"shas": {_sha("img:50")}, "digests": set()},
        lambda: {"shas": set(), "digests": {_sha("px:75")}},
    ],
    ids=["linked", "new", "no-digests", "no-shas"],
)
def test_a_photos_footprint_probes_every_table_it_reads(
    pg: sessionmaker[Session], keys: Callable[[], dict[str, set[str]]]
) -> None:
    """seed_footprint (the pipelined batch's claim): images and public_objects
    by their sha and pixel digests, then each turn of the fixed point
    through public_object_images by the copy's sha and by the image."""
    used: set[str] = set()
    with pg() as db:
        statements = _captured(db.get_bind(), lambda: seed_footprint(db, **keys()))
    assert _walks(pg, statements, used) == []


@pytest.mark.parametrize(
    "keys",
    [
        # a picture with captures and copies linked (the loop turns until its fixed point)
        lambda: {"sha": _sha("img:25"), "digest": _sha("px:25")},
        # a new picture: nothing behind its keys
        lambda: {"sha": _sha("new"), "digest": _sha("new-px")},
        # a picture that will not decode: its byte sha alone
        lambda: {"sha": _sha("img:50"), "digest": None},
    ],
    ids=["linked", "new", "no-digest"],
)
def test_a_photos_neighbourhood_probes_every_table_it_reads(
    pg: sessionmaker[Session], keys: Callable[[], dict[str, Any]]
) -> None:
    """seed_neighbourhood (MASK5: the pipelined batch's claim, by the
    original's sha and its digest): the same probes as its footprint, the
    public objects' kind read beside their sha."""
    used: set[str] = set()
    with pg() as db:
        statements = _captured(db.get_bind(), lambda: seed_neighbourhood(db, **keys()))
    assert _walks(pg, statements, used) == []
    with pg() as db:
        near = seed_neighbourhood(db, sha=_sha("img:25"), digest=_sha("px:25"))
        assert near.keys == seed_footprint(db, shas={_sha("img:25")}, digests={_sha("px:25")})


# --------------------------------------------------------------------------
# everything else one photo makes the batch and the seeder read
# --------------------------------------------------------------------------


def test_the_batch_and_the_seeders_per_photo_reads_all_probe(
    pg: sessionmaker[Session], tmp_path: Path
) -> None:
    """One new photo through the batch as _mint runs it (the claim's
    footprint, the settled check, the seed, the settle with a mask model),
    a second capture of it the same way, and the display seeder's checks
    for it (the withdraw judgment, its origins, its copies, the copy and
    appearance it reports): no statement walks a media table."""
    store = LocalObjectStore(tmp_path / "media")
    first, second = _png("audit"), _png("audit", note="captured again")
    statements: list[tuple[str, Any]] = []

    def batch_one(db: Session, data: bytes, n: str) -> str:
        keys = picture_keys(data)
        sha = hashlib.sha256(data).hexdigest()
        seed_footprint(db, shas={sha, keys.clean_sha256}, digests={keys.digest, keys.own_digest})
        original_settled(db, sha)
        copy = _seed(db, store, data, n).sha256
        backend = _Model()
        batch = open_mask_batch(db, backend_name=backend.name)
        settle_clean_copy(db, store, copy, backend=backend, batch=batch)
        original_settled(db, sha)
        return copy

    with pg() as db:
        engine = db.get_bind()
        made: list[str] = []
        statements += _captured(engine, lambda: made.append(batch_one(db, first, "1")))
        statements += _captured(engine, lambda: made.append(batch_one(db, second, "2")))
        assert made[0] == made[1]  # the second capture has no copy of its own
        origin = hashlib.sha256(second).hexdigest()
        depiction = DisplayDepiction(
            product_image_id="00000000-0000-4000-8000-0000000000aa",
            origin_sha256=origin,
            product_id="plan-3",
            origin_url="https://a.example/plan-3.png",
            source_site="mfc",
        )

        def seeder() -> None:
            dc._withdraws_a_mask(db, second, origin, [depiction], None)
            dc._seeded_origins(db, origin, [depiction])
            dc._picture_copies(db, {made[0]}, origin)
            assert copies.reported_copy(db, origin) == made[0]
            copies.measured_appearance(db, made[0])
            mask_refusal(db, made[0])

        statements += _captured(engine, seeder)
        db.rollback()
    used: set[str] = set()
    assert _walks(pg, statements, used) == []
    assert "figure_galleries.ix_figure_galleries_image_id" in used


def test_the_seeders_claim_and_its_queue_reads_probe(
    pg: sessionmaker[Session], tmp_path: Path
) -> None:
    """The display seeder's own per-photo reads of its queue: the claim of
    the newest pending original no one holds (FOR UPDATE SKIP LOCKED), its
    depictions read for the seed and again at the settle, and the settle's
    write of the original (an original whose raw object is gone here, so
    the seed itself is the batch test's)."""
    now = dt.datetime.now(dt.UTC)
    raw, media = LocalObjectStore(tmp_path / "raw"), LocalObjectStore(tmp_path / "media")
    config = dc.SeedConfig(threads=0)
    results: list[dc.SeedResult] = []

    def one_photo(db: Session) -> None:
        claimed = dc.claim_with_history(db, owner="plan", claim_seconds=60, now=now)
        assert claimed is not None
        original, _abandoned = claimed
        result = dc.seed_one(db, raw, media, original, units=dc._Units(4), config=config)
        results.append(dc._settle(db, original, result, now=now, max_attempts=3, hold_until=now))

    with pg() as db:
        statements = _captured(db.get_bind(), lambda: one_photo(db))
    assert [r.reason for r in results] == ["missing_raw"]
    used: set[str] = set()
    assert _walks(pg, statements, used) == []
    assert {
        "display_originals.ix_display_originals_state_seen",
        "display_depictions.ix_display_depictions_origin",
    } <= used


# --------------------------------------------------------------------------
# the rewrites answer exactly as the queries they replace
# --------------------------------------------------------------------------


def test_the_anti_join_keys_are_never_null(pg: sessionmaker[Session]) -> None:
    """``x NOT IN (subquery)`` and ``NOT EXISTS (... = x)`` differ only when x
    or a subquery row is NULL: every key the candidate check compares is
    NOT NULL in the migrated schema, so the two are one answer."""
    with pg() as db:
        nullable = dict(
            db.execute(
                text(
                    "SELECT table_name || '.' || column_name, is_nullable"
                    " FROM information_schema.columns WHERE (table_name, column_name) IN"
                    " (('public_objects', 'sha256'), ('image_masks', 'source_sha256'),"
                    "  ('public_object_images', 'sha256'), ('public_object_images', 'image_id'),"
                    "  ('images', 'id'), ('images', 'sha256'))"
                )
            ).all()
        )
    assert nullable == {
        "public_objects.sha256": "NO",
        "image_masks.source_sha256": "NO",
        "public_object_images.sha256": "NO",
        "public_object_images.image_id": "NO",
        "images.id": "NO",
        "images.sha256": "NO",
    }


def _candidates_before(
    db: Session,
    *,
    after: str | None,
    page_size: int,
    redrive: tuple[str, ...],
    source: str | None = None,
    only: tuple[str, ...] | None = None,
) -> list[str]:
    """mask_job._candidates as it was (b53e965): NOT IN (subquery)."""
    from app.workers.mask_job import _of_source

    settled = select(ImageMask.source_sha256).where(
        ImageMask.pipeline_version == MASK_PIPELINE_VERSION,
        ImageMask.status.not_in(redrive),
    )
    dead_capture = (
        select(PublicObjectImage.sha256)
        .join(ImageModel, ImageModel.id == PublicObjectImage.image_id)
        .where(ImageModel.deleted_at.is_not(None))
    )
    query: Select[Any] = (
        select(PublicObject.sha256)
        .join(ImageVersion, ImageVersion.id == PublicObject.image_version_id)
        .join(ImageModel, ImageModel.id == ImageVersion.image_id)
        .where(
            PublicObject.kind == "sanitized",
            ImageVersion.deleted_at.is_(None),
            ImageModel.deleted_at.is_(None),
            PublicObject.sha256.not_in(settled),
            PublicObject.sha256.not_in(dead_capture),
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


def test_the_candidates_are_the_ones_not_in_found(pg: sessionmaker[Session]) -> None:
    """Settled (ok, ineligible), re-driven (failed), unsettled, a copy with a
    soft-deleted capture behind it, a deleted version, a deleted owner; by
    page, after a cursor, by source, by sha: the same copies, in the same
    order, as the NOT IN query found."""
    # copy:N is settled ok (N % 20 in 1..16), failed (17), ineligible (18, 19),
    # unsettled (0); its owner is deleted at N % 97 == 0; copy:N-1 (N % 25 == 1)
    # has image N behind it too
    picked = tuple(_sha(f"copy:{n}") for n in range(1, 400)) + (_sha("unknown"),)
    cursor = sorted(picked)[len(picked) // 2]
    with pg() as db:
        # the NOT IN oracle hashes its subquery only when the hash fits work_mem;
        # past it, Postgres runs the subquery again for every row (hours here)
        db.execute(text("SET LOCAL work_mem = '256MB'"))
        version = db.execute(select(ImageVersion).where(ImageVersion.image_id == 120)).scalar_one()
        version.deleted_at = func.now()  # a deleted version (copy:120, unsettled)
        db.execute(
            text("UPDATE images SET deleted_at = now() WHERE id = 301")  # behind copy:300 too
        )
        db.flush()
        cases: list[dict[str, Any]] = [
            {"after": after, "page_size": size, "redrive": redrive, "source": source, "only": only}
            for after in (None, cursor)
            for size in (1, 1000)
            for redrive in ((), ("failed",), ("failed", "ineligible"))
            for source in (None, "amiami")
            for only in (picked, (_sha("copy:20"),))
        ]
        cases += [
            {"after": None, "page_size": 50, "redrive": (), "source": None, "only": None},
            {
                "after": cursor,
                "page_size": 50,
                "redrive": ("failed",),
                "source": None,
                "only": None,
            },
        ]
        for case in cases:
            now = [c.sha256 for c in _candidates(db, **case)]
            assert now == _candidates_before(db, **case), case
        # the cases are not vacuous: each kind of copy is in or out as it should be
        found = set(_candidates_before(db, after=None, page_size=1000, redrive=(), only=picked))
        assert _sha("copy:20") in found and _sha("copy:40") in found
        assert not {_sha("copy:1"), _sha("copy:17"), _sha("copy:18")} & found
        assert not {_sha("copy:120"), _sha("copy:300"), _sha("copy:97")} & found
        redriven = _candidates_before(
            db, after=None, page_size=1000, redrive=("failed",), only=picked
        )
        assert _sha("copy:17") in redriven
        db.rollback()


def _footprint_before(db: Session, *, shas: set[str], digests: set[str]) -> set[str]:
    """mask_seed.seed_footprint as it was (b53e965): one OR across the join."""
    images = set(
        db.execute(
            select(ImageModel.sha256).where(
                or_(ImageModel.sha256.in_(shas), ImageModel.pixel_sha256.in_(digests))
            )
        ).scalars()
    )
    objects = set(
        db.execute(
            select(PublicObject.sha256).where(
                or_(
                    PublicObject.sha256.in_(shas),
                    PublicObject.pixel_sha256.in_(digests),
                    PublicObject.own_pixel_sha256.in_(digests),
                )
            )
        ).scalars()
    )
    while images or objects:
        links = db.execute(
            select(PublicObjectImage.sha256, ImageModel.sha256)
            .join(ImageModel, ImageModel.id == PublicObjectImage.image_id)
            .where(or_(PublicObjectImage.sha256.in_(objects), ImageModel.sha256.in_(images)))
        ).all()
        more_objects = {obj for obj, _image in links} - objects
        more_images = {image for _obj, image in links} - images
        if not more_objects and not more_images:
            break
        objects |= more_objects
        images |= more_images
    return shas | digests | images | objects


def test_the_footprint_is_the_one_the_or_join_found(
    pg: sessionmaker[Session], tmp_path: Path
) -> None:
    """Multi-hop links (a capture behind two copies, chained), a CYCLE (two
    copies each behind both captures), a picture seeded for real (its
    original, copy, second capture and mask), unknown keys, and empty
    inputs: the same footprint as the OR join found."""
    store = LocalObjectStore(tmp_path / "media")
    first, second = _png("footprint"), _png("footprint", note="again")
    with pg() as db:
        made = [_seed(db, store, data, f"fp{n}").sha256 for n, data in enumerate((first, second))]
        # a cycle: copy:1003 behind image 1004 as well, copy:1004 behind image 1003
        db.execute(
            text(
                "INSERT INTO public_object_images (sha256, image_id) VALUES"
                " (:a, 1004), (:b, 1003)"
            ),
            {"a": _sha("copy:1003"), "b": _sha("copy:1004")},
        )
        db.flush()
        keys = picture_keys(first)
        cases = [
            {"shas": {_sha("img:25")}, "digests": set()},  # chain 25 -> 26
            {"shas": {_sha("copy:50")}, "digests": {_sha("px:51")}},
            {"shas": {_sha("img:1003")}, "digests": set()},  # the cycle, from either end
            {"shas": set(), "digests": {_sha("own:1004")}},
            {
                "shas": {hashlib.sha256(first).hexdigest(), keys.clean_sha256},
                "digests": {keys.digest, keys.own_digest},
            },
            {"shas": {hashlib.sha256(second).hexdigest()}, "digests": set()},
            {"shas": {made[0]}, "digests": set()},
            {"shas": {_sha("unknown")}, "digests": {_sha("unknown-px")}},
            {"shas": set(), "digests": set()},
            {"shas": {_sha("img:97")}, "digests": set()},  # a soft-deleted image
        ]
        for case in cases:
            assert seed_footprint(db, **case) == _footprint_before(db, **case), case
        # not vacuous: the chain, the cycle and the seeded picture reach their far ends
        assert _sha("img:26") in seed_footprint(db, shas={_sha("img:25")}, digests=set())
        cycle = seed_footprint(db, shas={_sha("img:1003")}, digests=set())
        assert {_sha("img:1004"), _sha("copy:1004"), _sha("copy:1003")} <= cycle
        seeded = seed_footprint(db, shas={hashlib.sha256(second).hexdigest()}, digests=set())
        assert {hashlib.sha256(first).hexdigest(), made[0]} <= seeded
        db.rollback()


def test_the_footprint_query_is_one_round_trip_a_turn(pg: sessionmaker[Session]) -> None:
    """Each turn of the fixed point is ONE statement (the two probes in a
    UNION ALL), so the split costs the batch no round trip over its tunnel."""
    with pg() as db:
        statements = _captured(
            db.get_bind(),
            lambda: seed_footprint(db, shas={_sha("img:25")}, digests={_sha("px:25")}),
        )
    links = [s for s, _p in statements if "public_object_images" in s]
    assert links and all("UNION ALL" in s for s in links)
    assert len(statements) == 2 + len(links)
