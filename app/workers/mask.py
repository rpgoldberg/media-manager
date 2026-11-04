"""MASK-1: the separate alpha mask and the grounding numbers, per clean image.

Ross 2026-09-26: background removal is a NON-DESTRUCTIVE, display-time
overlay. The clean copy (the sanitized derivative) is stored and served
byte-for-byte as it is; this module computes a SEPARATE mask -- our own work
product -- that the client composites over it (CSS ``mask-image`` in alpha
mode, or canvas ``destination-in``), so the case shows through around the
figure. A destructive cut-out stays only as Ross's fallback
(``app.workers.tasks._create_matted_derivative``).

THE MASK FILE. fc-api-contract 0.4.0 ``ImageMask``: same pixel dimensions as
its derivative, ``image/png``, "its alpha channel is the mask". It is one
channel of information carried as PNG ``LA``: luminance is constant white and
alpha is the mask, so a client compositing in alpha mode reads the mask and
one compositing in luminance mode reads ``1.0 x alpha`` -- the same mask. A
grayscale ``L`` PNG would carry no alpha and mask nothing.

THE NUMBERS. Grounding is measured on the mask's alpha (opaque = alpha above
10 of 255, the contract's rule); the appearance (ThumbHash and dominant
color) is computed twice, with the mask applied and without it, because the
coordinator must describe the image "as THIS caller is shown it" and a
``no_mask`` display restriction withholds the mask (catalog.proto DISPLAY
RESTRICTIONS). The lane never decides who sees what; it only stores both.

THE QUALITY GATE. A mask that removed (almost) nothing (``full_frame``), or
that kept the backdrop's floor (``bottom_edge_bleed``: see _kept_floor) is
HELD: computed and kept for review, never published. The floor case is the
black-dress-on-black-marble failure the base-keep pass can produce; it is
reproduced synthetically by the sample's ``madoka-black-dress`` entry. (The
July file fc-design-assets/madoka-matte-basekept.png is NOT that failure:
measured 2026-10-11 its alpha covers 34 % of the frame and its lowest row is
41 rows above the bottom edge.) The gate catches a floor kept ALONG THE
BOTTOM EDGE; a floor that stops short of the edge, or that is under 85 %
opaque on every row of the band, is not detected. A mask with nothing opaque
is EMPTY and nothing is stored. Only an ``ok`` mask is ever addressable.
"""

from __future__ import annotations

import hashlib
import io
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from PIL import Image, ImageOps

from .derivatives import compute_dominant_color, compute_thumbhash
from .grounding import (
    ALPHA_THRESHOLD,
    AlphaGrounding,
    _round_half_up,
    measure_alpha_grounding,
)
from .matting import MattingBackend, get_matting_backend
from .stage_timing import StageTimings, timed

if TYPE_CHECKING:
    from .cpu_stages import CpuStages

#: Bump when the mask algorithm or its encoding changes: a new version mints
#: new masks beside the old ones (an address cached on a phone never changes
#: meaning), and the batch job re-selects every clean image for it.
MASK_PIPELINE_VERSION = "mask-v1"

MASK_MIME = "image/png"

#: A mask whose opaque share of the frame is at least this removed (almost)
#: nothing: held for review rather than published.
FULL_FRAME_COVERAGE = 0.97

#: The floor band: the last FLOOR_BAND_FRAC of the image's rows (at least
#: FLOOR_MIN_ROWS) ending at the mask's lowest opaque row.
FLOOR_BAND_FRAC = 0.02
#: The band counts as on the bottom edge when its lowest row is within
#: FLOOR_EDGE_FRAC of the height (at least FLOOR_MIN_ROWS rows) of the edge:
#: a JPEG edge or a vignette may leave the last rows clear.
FLOOR_EDGE_FRAC = 0.01
FLOOR_MIN_ROWS = 2
#: A band row at least this opaque (as a share of the image width) is floor.
#: Measured 2026-10-11 on the seven July figures: the real masks reaching the
#: bottom edge are at most 73 % opaque on any row of the band.
FLOOR_ROW_COVERAGE = 0.85

MaskStatus = Literal["ok", "held", "empty"]


@dataclass(frozen=True)
class Appearance:
    """ThumbHash (base64) and dominant color (``#rrggbb``, lowercase, as the
    contract spells it) of an image as some caller is shown it."""

    thumbhash: str
    dominant_color: str


@dataclass(frozen=True)
class MaskOutcome:
    """One clean image's answer. ``mask_png``/``mask_sha256``/``grounding``/
    ``masked`` are present for ``ok`` and ``held`` and absent for ``empty``."""

    status: MaskStatus
    reason: str | None
    backend: str
    pipeline_version: str
    width: int
    height: int
    mask_png: bytes | None
    mask_sha256: str | None
    grounding: AlphaGrounding | None
    masked: Appearance | None
    unmasked: Appearance
    own_alpha_grounding: AlphaGrounding | None


def _require_mask_shape(alpha: np.ndarray) -> None:
    if alpha.ndim != 2 or alpha.dtype != np.uint8 or alpha.size == 0:
        raise ValueError(
            f"a mask is a non-empty (height, width) uint8 array, got {alpha.dtype} {alpha.shape}"
        )


def encode_mask_png(alpha: np.ndarray) -> bytes:
    """The mask as a PNG whose alpha channel is the mask (``LA``, luminance
    constant white). Deterministic: the same alpha always encodes to the same
    bytes, so the same mask always has the same content address."""
    _require_mask_shape(alpha)
    la = np.dstack([np.full_like(alpha, 255), alpha])
    out = io.BytesIO()
    Image.fromarray(la, mode="LA").save(out, format="PNG", compress_level=9)
    return out.getvalue()


def decode_mask_png(data: bytes) -> np.ndarray:
    """The ``(height, width)`` alpha of a mask written by encode_mask_png."""
    img = Image.open(io.BytesIO(data))
    if img.format != "PNG" or img.mode != "LA":
        raise ValueError(f"not a mask: {img.format} {img.mode}")
    return np.asarray(img)[..., 1].copy()


def composite_with_mask(clean: Image.Image, alpha: np.ndarray) -> Image.Image:
    """What the client draws: the clean copy's pixels, its alpha multiplied
    by the mask (CSS ``mask-image`` alpha mode / canvas ``destination-in``).
    An opaque clean copy simply takes the mask as its alpha."""
    rgba = clean.convert("RGBA")
    if (rgba.height, rgba.width) != alpha.shape:
        raise ValueError(
            f"mask {alpha.shape[::-1]} does not match its image {(rgba.width, rgba.height)}"
        )
    arr = np.asarray(rgba).copy()
    own = arr[..., 3].astype(np.uint32)
    arr[..., 3] = ((own * alpha.astype(np.uint32) + 127) // 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def _rows(height: int, frac: float) -> int:
    return max(FLOOR_MIN_ROWS, _round_half_up(height * frac))


def _kept_floor(opaque: np.ndarray) -> bool:
    """Whether the mask kept a floor: its lowest opaque row is on (or within
    the edge tolerance of) the bottom edge, and some row of the band ending
    there is at least FLOOR_ROW_COVERAGE opaque. Coverage, not the opaque
    extent, so a floor short of the side edges and a narrow image are judged
    alike; a figure's base reaching the edge is much narrower than its frame."""
    height = opaque.shape[0]
    last = int(np.nonzero(opaque.any(axis=1))[0][-1])
    if height - 1 - last > _rows(height, FLOOR_EDGE_FRAC):
        return False
    band = opaque[max(0, last - _rows(height, FLOOR_BAND_FRAC) + 1) : last + 1]
    return float(band.mean(axis=1).max()) >= FLOOR_ROW_COVERAGE


def assess_mask_quality(
    alpha: np.ndarray, grounding: AlphaGrounding | None
) -> tuple[MaskStatus, str | None]:
    """``("ok", None)``, ``("held", reason)`` or ``("empty", reason)``."""
    if grounding is None:
        return "empty", "no_opaque_pixels"
    opaque = alpha > ALPHA_THRESHOLD
    if float(np.mean(opaque)) >= FULL_FRAME_COVERAGE:
        return "held", "full_frame"
    if _kept_floor(opaque):
        return "held", "bottom_edge_bleed"
    return "ok", None


def _appearance(rgba: Image.Image) -> Appearance:
    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    data = buf.getvalue()
    return Appearance(
        thumbhash=compute_thumbhash(data),
        dominant_color=compute_dominant_color(data).lower(),
    )


def _oriented_rgba(clean: bytes) -> Image.Image:
    """``clean`` with its EXIF orientation applied, as RGBA: what a browser
    draws, and what every appearance and mask is measured on."""
    with Image.open(io.BytesIO(clean)) as src:
        oriented = ImageOps.exif_transpose(src) or src
        return oriented.convert("RGBA")


def _own_grounding(rgba: Image.Image) -> AlphaGrounding | None:
    own = np.asarray(rgba)[..., 3]
    return measure_alpha_grounding(own) if bool((own < 255).any()) else None


def plain_appearance(clean: bytes) -> tuple[Appearance, AlphaGrounding | None]:
    """The clean copy's own appearance WITHOUT a mask, and the grounding of its
    own alpha channel (a transparent copy only): exactly what
    ``build_mask_artifact`` measures as ``unmasked`` and
    ``own_alpha_grounding``, by the same code, with no model (SHOW1: the
    display-copy lane reports a copy before any mask exists, and the replay
    reports the mask row's plain values later, which are these)."""
    rgba = _oriented_rgba(clean)
    return _appearance(rgba), _own_grounding(rgba)


def build_mask_artifact(
    clean: bytes,
    *,
    backend: MattingBackend | None = None,
    timings: StageTimings | None = None,
    cpu: CpuStages | None = None,
) -> MaskOutcome:
    """Compute the mask, its grounding and both appearances for one clean
    image. Never alters ``clean``; never produces a cut-out. ``timings``, when
    given, gets the model's seconds and the mask encode's (MASK-PERF-1).

    The image is read with its EXIF orientation applied, because that is how
    a browser draws the clean copy (``image-orientation: from-image``), and
    the mask must line up with what is drawn.

    ``cpu`` (MASK5): where the judgment runs once the model answered -- the
    grounding, both appearances and the mask's encode (``judge_mask``, the
    same code on the same bytes): a CPU worker process of the pipelined
    batch, which decodes ``clean`` again there, this thread's copy let go
    first; the caller's thread when None.
    """
    active = backend or get_matting_backend()
    rgba = _oriented_rgba(clean)
    width, height = rgba.size
    # on this thread, the copy's own measures before the model, as ever
    plain = _plain(rgba) if cpu is None else None
    answer = (
        active.predict_alpha(rgba) if timings is None else active.predict_alpha_timed(rgba, timings)
    )
    alpha = np.asarray(answer, dtype=np.uint8)
    if alpha.shape != (height, width):
        raise ValueError(
            f"backend {active.name!r} answered {alpha.shape} for a {height}x{width} image"
        )
    if cpu is None:
        assert plain is not None
        return _judged(rgba, alpha, active.name, timings, plain)
    del rgba, answer  # the worker decodes its own: none held here meanwhile
    outcome, encode_seconds = cpu.run(judge_mask, clean, alpha, active.name, pixels=width * height)
    if timings is not None:
        timings.carve("encode", encode_seconds)
    return outcome


def judge_mask(clean: bytes, alpha: np.ndarray, backend_name: str) -> tuple[MaskOutcome, float]:
    """``build_mask_artifact``'s judgment of the model's ``alpha`` for
    ``clean``, from the bytes (MASK5: in a CPU worker process), and the
    seconds its mask's encode took."""
    timings = StageTimings(time.monotonic)
    rgba = _oriented_rgba(clean)
    outcome = _judged(rgba, alpha, backend_name, timings, _plain(rgba))
    return outcome, timings.seconds.get("encode", 0.0)


def _plain(rgba: Image.Image) -> tuple[AlphaGrounding | None, Appearance]:
    """The clean copy's own: the grounding of its alpha, its appearance."""
    return _own_grounding(rgba), _appearance(rgba)


def _judged(
    rgba: Image.Image,
    alpha: np.ndarray,
    backend_name: str,
    timings: StageTimings | None,
    plain: tuple[AlphaGrounding | None, Appearance],
) -> MaskOutcome:
    """The model's ``alpha`` for the clean copy ``rgba`` judged, beside the
    copy's own grounding and appearance (``plain``): the mask's grounding,
    quality and encode, and the appearance with it applied."""
    width, height = rgba.size
    own_alpha_grounding, unmasked = plain
    grounding = measure_alpha_grounding(alpha)
    status, reason = assess_mask_quality(alpha, grounding)

    if status == "empty":
        return MaskOutcome(
            status=status,
            reason=reason,
            backend=backend_name,
            pipeline_version=MASK_PIPELINE_VERSION,
            width=width,
            height=height,
            mask_png=None,
            mask_sha256=None,
            grounding=None,
            masked=None,
            unmasked=unmasked,
            own_alpha_grounding=own_alpha_grounding,
        )

    with timed(timings, "encode"):
        mask_png = encode_mask_png(alpha)
    return MaskOutcome(
        status=status,
        reason=reason,
        backend=backend_name,
        pipeline_version=MASK_PIPELINE_VERSION,
        width=width,
        height=height,
        mask_png=mask_png,
        mask_sha256=hashlib.sha256(mask_png).hexdigest(),
        grounding=grounding,
        masked=_appearance(composite_with_mask(rgba, alpha)),
        unmasked=unmasked,
        own_alpha_grounding=own_alpha_grounding,
    )
