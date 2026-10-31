"""Test doubles for the media.v1 replay: a stateful in-process SpineMedia
(served over real gRPC) and a media-DB seeder.

``FakeSpine`` keeps just enough of the contract to test the replay's own
logic -- depictions leased by origin and held, a derivative registry, mask
rows keyed (derivative, pipeline_version) that a report never changes,
withdrawals by mask bytes with tombstones -- and can be told to fail an RPC
with a status first. The REAL spine is exercised by
tests/test_spine_media_live.py."""

from __future__ import annotations

import datetime as dt
import hashlib
import uuid
from collections import deque
from collections.abc import Iterator
from concurrent import futures
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import grpc
from app.models import Image, ImageMask, ImageVersion, PublicObject, PublicObjectImage
from app.spine.media_v1 import media_pb2, media_pb2_grpc
from sqlalchemy import select
from sqlalchemy.orm import Session

AT = dt.datetime(2026, 10, 14, 9, 0, tzinfo=dt.UTC)


def sha(tag: str) -> str:
    return hashlib.sha256(tag.encode()).hexdigest()


@dataclass
class FakeSpine(media_pb2_grpc.SpineMediaServicer):
    depictions: dict[str, dict[str, Any]] = field(default_factory=dict)
    held: set[str] = field(default_factory=set)
    registry: set[bytes] = field(default_factory=set)
    #: origin -> the derivative bytes first registered for it
    by_origin: dict[bytes, bytes] = field(default_factory=dict)
    masks: dict[tuple[bytes, str], dict[str, Any]] = field(default_factory=dict)
    tombstones: dict[bytes, str] = field(default_factory=dict)
    #: Per RPC, what its next calls do: a status to abort with, or None to answer.
    failures: dict[str, deque[grpc.StatusCode | None]] = field(default_factory=dict)
    calls: list[tuple[str, Any]] = field(default_factory=list)
    #: Per call, the metadata it arrived with (LG-U11: the caller's traceparent).
    metadata: list[tuple[str, dict[str, str]]] = field(default_factory=list)
    ignore_filter: bool = False

    def depict(self, origin_hex: str) -> str:
        pid = str(uuid.uuid4())
        self.depictions[pid] = {"origin": bytes.fromhex(origin_hex), "state": "captured"}
        return pid

    def fail(self, rpc: str, *codes: grpc.StatusCode | None) -> None:
        self.failures.setdefault(rpc, deque()).extend(codes)

    def sent(self, rpc: str) -> list[Any]:
        return [r for name, r in self.calls if name == rpc]

    def _enter(self, rpc: str, request: Any, context: grpc.ServicerContext) -> None:
        self.calls.append((rpc, request))
        self.metadata.append((rpc, {k: str(v) for k, v in context.invocation_metadata()}))
        queue = self.failures.get(rpc)
        code = queue.popleft() if queue else None
        if code is not None:
            context.abort(code, f"fake {code.name}")

    def LeasePendingImages(self, request: Any, context: grpc.ServicerContext) -> Any:  # noqa: N802
        self._enter("LeasePendingImages", request, context)
        wanted = set(request.origin_sha256)
        limit = min(request.max_items or 50, 200)
        out = []
        for pid, d in self.depictions.items():
            if d["state"] != "captured" or pid in self.held:
                continue
            if not self.ignore_filter and d["origin"] not in wanted:
                continue
            out.append(media_pb2.PendingImage(product_image_id=pid, origin_sha256=d["origin"]))
            if len(out) == limit:
                break
        self.held.update(i.product_image_id for i in out)
        return media_pb2.LeasePendingImagesResponse(
            images=out, lease_expires_at="2026-10-15T00:02:00.000000Z"
        )

    def ReportDerivative(self, request: Any, context: grpc.ServicerContext) -> Any:  # noqa: N802
        self._enter("ReportDerivative", request, context)
        d = self.depictions.get(request.product_image_id)
        if d is None:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "unknown product_image_id")
        d["state"] = "derived"
        self.held.discard(request.product_image_id)
        # The registry's identity is the ORIGIN's (derivative-writer.ts): other
        # bytes for an original already registered are absorbed, never stored.
        first = self.by_origin.setdefault(d["origin"], request.derivative_sha256)
        if first != request.derivative_sha256:
            return media_pb2.DerivativeAck(
                stored=False,
                image_id=first.hex()[:8],
                product_image_state="derived",
                warnings=[
                    f"a sanitized derivative at {request.pipeline_version} is already "
                    f"registered for this original with different bytes ({first.hex()}); "
                    f"the registered one stands and the reported "
                    f"{request.derivative_sha256.hex()} is not registered"
                ],
            )
        stored = request.derivative_sha256 not in self.registry
        self.registry.add(request.derivative_sha256)
        return media_pb2.DerivativeAck(
            stored=stored,
            image_id=request.derivative_sha256.hex()[:8],
            product_image_state="derived",
        )

    def ReportMask(self, request: Any, context: grpc.ServicerContext) -> Any:  # noqa: N802
        self._enter("ReportMask", request, context)
        if request.derivative_sha256 not in self.registry:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "no registered derivative")
        key = (request.derivative_sha256, request.pipeline_version)
        if key in self.masks:
            return media_pb2.MaskAck(stored=0, matched=1)
        withdrawn = request.mask_sha256 in self.tombstones or any(
            m["mask"] == request.mask_sha256 and m["status"] == "withdrawn"
            for m in self.masks.values()
        )
        self.masks[key] = {
            "mask": request.mask_sha256,
            "status": "withdrawn" if withdrawn else "published",
        }
        warnings = ["withdrawn bytes absorbed"] if withdrawn else []
        return media_pb2.MaskAck(stored=1, matched=1, warnings=warnings)

    def WithdrawMask(self, request: Any, context: grpc.ServicerContext) -> Any:  # noqa: N802
        self._enter("WithdrawMask", request, context)
        rows = [m for m in self.masks.values() if m["mask"] == request.mask_sha256]
        target = "published" if request.restore else "withdrawn"
        stored = 0
        for m in rows:
            if m["status"] != target:
                m["status"] = target
                stored += 1
        if request.restore:
            stored += 1 if self.tombstones.pop(request.mask_sha256, None) is not None else 0
        elif not rows and request.mask_sha256 not in self.tombstones:
            self.tombstones[request.mask_sha256] = request.reason
            stored += 1
        return media_pb2.MaskAck(stored=stored, matched=len(rows))

    def published(self, mask_hex: str) -> bool:
        return any(
            m["mask"] == bytes.fromhex(mask_hex) and m["status"] == "published"
            for m in self.masks.values()
        )


@contextmanager
def serve(spine: FakeSpine) -> Iterator[str]:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    media_pb2_grpc.add_SpineMediaServicer_to_server(spine, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.stop(grace=None)


GROUND = {
    "bottom_margin_frac": 0.0125,
    "top_margin_frac": 0.04,
    "box_top_frac": 0.04,
    "box_bottom_frac": 0.9875,
    "box_left_frac": 0.2,
    "box_right_frac": 0.8,
    "contact_band_center_x_frac": 0.51,
    "contact_band_width_frac": 0.3,
}


def seed_copy(
    db: Session,
    tag: str,
    *,
    masks: tuple[tuple[str, str, str], ...] = (("mask-v1", "ok", "m"),),
    extra_originals: tuple[str, ...] = (),
    deleted: bool = False,
    mime: str = "image/jpeg",
    own_grounding: dict[str, float] | None = None,
    minted_by: str | None = None,
) -> dict[str, Any]:
    """One original (``sha('orig:'+tag)``), its clean copy (``sha('clean:'+tag)``)
    and its mask rows: ``(pipeline_version, status, mask tag)``, the mask's
    bytes ``sha('mask:'+mask tag)``. ``extra_originals`` are other captures
    of the picture, linked behind the copy. ``minted_by``: the sha of an
    original already recorded (one linked behind another copy), which mints
    this copy instead of a new original."""
    if minted_by is None:
        origin = sha(f"orig:{tag}")
        image = Image(sha256=origin, mime="image/jpeg", width=100, height=200, bytes=10)
        if deleted:
            image.deleted_at = AT
        db.add(image)
    else:
        origin = minted_by
        image = db.execute(select(Image).where(Image.sha256 == minted_by)).scalar_one()
    db.flush()
    db.add(ImageVersion(image_id=image.id, version_no=1, storage_key=f"o/{origin}"))
    clean = sha(f"clean:{tag}")
    version = ImageVersion(
        image_id=image.id, version_no=2, derived_from_version=1, storage_key=f"d/{clean}"
    )
    db.add(version)
    db.flush()
    db.add(
        PublicObject(
            sha256=clean,
            kind="sanitized",
            mime=mime,
            bytes=10,
            width=100,
            height=200,
            storage_key=f"d/{clean}",
            image_version_id=version.id,
            pixel_sha256=sha(f"px:{tag}"),
            own_pixel_sha256=sha(f"own:{tag}"),
            created_at=AT,
        )
    )
    db.flush()
    db.add(PublicObjectImage(sha256=clean, image_id=image.id))
    others = []
    for extra in extra_originals:
        other = Image(sha256=sha(f"orig:{extra}"), mime="image/jpeg")
        db.add(other)
        db.flush()
        db.add(PublicObjectImage(sha256=clean, image_id=other.id))
        others.append(other.sha256)
    mask_shas = []
    for version_name, status, mask_tag in masks:
        mask_sha = sha(f"mask:{mask_tag}") if status in ("ok", "held", "withdrawn") else None
        if mask_sha is not None and db.get(PublicObject, mask_sha) is None:
            db.add(
                PublicObject(
                    sha256=mask_sha,
                    kind="mask",
                    mime="image/png",
                    bytes=5,
                    width=100,
                    height=200,
                    storage_key=f"d/{mask_sha}",
                    created_at=AT,
                )
            )
            db.flush()
        db.add(
            ImageMask(
                source_sha256=clean,
                pipeline_version=version_name,
                status=status,
                reason="store_revoked" if status == "withdrawn" else None,
                mask_sha256=mask_sha,
                plain_thumbhash="AAECAwQFBgcICQ==",
                plain_dominant_color="#112233",
                masked_thumbhash="CQgHBgUEAwIBAA==",
                masked_dominant_color="#445566",
                own_alpha_grounding=own_grounding,
                created_at=AT,
                updated_at=AT + dt.timedelta(hours=1),
                **(GROUND if status in ("ok", "held", "withdrawn") else {}),
            )
        )
        mask_shas.append(mask_sha)
    db.commit()
    return {"origin": origin, "clean": clean, "masks": mask_shas, "others": others}


class LapsingSpine(FakeSpine):
    """A FakeSpine whose holds lapse on ``clock`` (``lease_seconds`` after the
    lease), whose ReportDerivative takes ``report_seconds`` of that clock, and
    whose lease can hold its rows and THEN fail (the answer lost), as the
    contract says a failed lease attempt may: ``hold_then_fail`` is, per lease
    call, a status to fail with after holding, or None to answer."""

    def __init__(self, clock: Any, *, report_seconds: float = 0.0) -> None:
        super().__init__()
        self.clock = clock
        self.report_seconds = report_seconds
        self.until: dict[str, dt.datetime] = {}
        self.hold_then_fail: deque[grpc.StatusCode | None] = deque()

    def held_now(self, pid: str) -> bool:
        return pid in self.until and self.until[pid] > self.clock.now

    def LeasePendingImages(self, request: Any, context: grpc.ServicerContext) -> Any:  # noqa: N802
        self.calls.append(("LeasePendingImages", request))
        wanted = set(request.origin_sha256)
        out = []
        for pid, d in self.depictions.items():
            if d["state"] != "captured" or self.held_now(pid):
                continue
            if not self.ignore_filter and d["origin"] not in wanted:
                continue
            out.append(media_pb2.PendingImage(product_image_id=pid, origin_sha256=d["origin"]))
            if len(out) == min(request.max_items or 50, 200):
                break
        for image in out:
            self.until[image.product_image_id] = self.clock.now + dt.timedelta(
                seconds=request.lease_seconds
            )
        code = self.hold_then_fail.popleft() if self.hold_then_fail else None
        if code is not None:
            context.abort(code, "the rows were held, then the answer was lost")
        return media_pb2.LeasePendingImagesResponse(images=out, lease_expires_at="x")

    def ReportDerivative(self, request: Any, context: grpc.ServicerContext) -> Any:  # noqa: N802
        self.clock.now += dt.timedelta(seconds=self.report_seconds)
        return super().ReportDerivative(request, context)
