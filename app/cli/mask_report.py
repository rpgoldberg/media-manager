"""MM-REPORT-1: replay the media DB's clean copies and masks into the spine.

    python -m app.cli.mask_report [--dry-run] [--spine-url URL] [--batch-size N]
        [--lease-seconds S] [--deadline-seconds S] [--max-attempts N] [--restart]
        [--sandbox DIR]

Runs in the cluster as the ``mm-report`` Job (namespace ``media``, service
account ``media-report``): it reads the media DB (``DATABASE_URL``) and calls
media.v1 SpineMedia at ``SPINE_MEDIA_URL``
(``http://ingest-server.fc.svc.cluster.local:50063``) in plaintext through its
Linkerd proxy, which carries the call over mTLS under that identity. What it
sends, in what order, and how it resumes: app.spine.replay.

``--dry-run`` reads the media DB and prints every call it would make; it
opens no gRPC channel and writes nothing. ``--restart`` abandons an
unfinished run and starts over (by default an unfinished run is resumed
after its cursor). ``--sandbox DIR`` reads ``DIR/masks.db`` (SQLite) instead
of DATABASE_URL.

Output (LG-U11): the estate's log line on stdout (app.telemetry), one run =
one root span. ``job.start``; one line per replay event, ``event`` being
``app.plan``, ``app.resume``, ``app.would_send`` (dry run), ``app.batch``
(counts per RPC, stored/matched, warnings, refusals by status),
``app.stopped``, ``app.done``, the event's fields snake_case and a nested
one as one-line JSON; then ``job.end`` with the plan's sizes, the ledger's
counts and ``exit``. Each batch is an ITEM (fc-shared's job model): a new
root trace linked to the run, holding its app.batch line and an rpc.out line
per call; every SpineMedia call carries its batch's traceparent, so the
spine's rpc.in lines are in that batch's trace. Every line carries run_id.

Exit status: 0 done; 1 done, but something did not land in this run or an
earlier attempt of it (a refusal, a value not sent, a mask with no public
object, or a mask whose derivative a report of this run stored and the
spine still does not have); 2 bad arguments (nothing ran); 3 stopped,
resumable: run it again (a spine that ignores the origin filter stops it
too). A mask whose copy's originals the spine has no depiction for is an
``orphan`` (named in the batch line), and one whose copy the spine kept out
for originals it already holds another copy for is ``kept_other``: neither
is a failure.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

import grpc
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from .. import telemetry
from ..spine.media_client import (
    DEFAULT_DEADLINE_SECONDS,
    DEFAULT_LEASE_SECONDS,
    RetryPolicy,
    SpineMediaClient,
    open_channel,
    target_from_url,
)
from ..spine.replay import ReplayConfig, run_replay
from ..telemetry.events import EventLines

EXIT_USAGE = 2
SERVICE = "mm-report"


class Refused(ValueError):
    """Bad arguments: nothing ran (exit 2)."""


def _open_db(sandbox: str | None) -> Session:
    if sandbox is None:
        from .. import db as configured_db

        return configured_db.SessionLocal()
    engine = create_engine(f"sqlite:///{Path(sandbox) / 'masks.db'}")
    return Session(engine, expire_on_commit=False)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m app.cli.mask_report",
        description="Replay the media DB's clean copies and masks into the spine (media.v1).",
        allow_abbrev=False,
    )
    p.add_argument("--dry-run", action="store_true", help="print the calls; send nothing")
    p.add_argument("--spine-url", help="SpineMedia, http://host:port (default SPINE_MEDIA_URL)")
    p.add_argument("--batch-size", type=int, default=200, help="originals per lease (1-200)")
    p.add_argument("--lease-seconds", type=int, default=DEFAULT_LEASE_SECONDS)
    p.add_argument("--deadline-seconds", type=float, default=DEFAULT_DEADLINE_SECONDS)
    p.add_argument("--max-attempts", type=int, default=5, help="per call, retries included")
    p.add_argument("--restart", action="store_true", help="abandon an unfinished run")
    p.add_argument("--sandbox", help="read DIR/masks.db (SQLite) instead of DATABASE_URL")
    return p


def _run(args: argparse.Namespace, emit: EventLines, run: telemetry.JobRun) -> int:
    url = args.spine_url or os.environ.get("SPINE_MEDIA_URL", "")
    try:
        config = ReplayConfig(
            batch_size=args.batch_size,
            lease_seconds=args.lease_seconds,
            dry_run=args.dry_run,
            restart=args.restart,
        )
        policy = RetryPolicy(max_attempts=args.max_attempts)
        if not args.deadline_seconds > 0:
            raise ValueError("--deadline-seconds must be above 0")
        if not args.dry_run:
            target_from_url(url)
    except ValueError as exc:
        raise Refused(f"mask_report: {exc}") from exc

    channels: list[grpc.Channel] = []

    def connect() -> SpineMediaClient:
        channels.append(open_channel(url))
        return SpineMediaClient(channels[-1], deadline_seconds=args.deadline_seconds, policy=policy)

    db = _open_db(args.sandbox)
    try:
        return run_replay(
            db,
            config,
            connect=connect,
            emit=emit,
            # each leased batch is an item: its own trace, linked to the run (fc-shared's job model)
            item=lambda index, fn: run.item(fn, name="item.batch", attributes={"fc.batch": index}),
        )
    finally:
        db.close()
        for channel in channels:
            channel.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    log = telemetry.configure(SERVICE)
    telemetry.start_tracing(SERVICE)
    lines = EventLines(log, summarize=("plan", "stopped", "done"))
    try:
        return telemetry.run_job(
            SERVICE,
            lambda run: _run(args, lines, run),
            logger=log,
            counts=lambda code: {**lines.summary, "exit": code},
            ok=lambda code: code == 0,
        )
    except Refused:
        return EXIT_USAGE  # job.end says why (err)


if __name__ == "__main__":  # pragma: no cover - the module's entry point
    raise SystemExit(main())
