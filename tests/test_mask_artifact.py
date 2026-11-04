"""Tests for the MASK-1 mask artifact (app.workers.mask) and the alpha-native
grounding it reports (app.workers.grounding.measure_alpha_grounding).

The mask is OUR work product: a separate single-channel alpha mask the client
composites over the clean derivative at display time (Ross 2026-09-26). The
clean derivative itself is never cut out. The grounding numbers follow
fc-api-contract 0.4.0 ProductImage (bottom_margin_frac, contact_band: opaque
means alpha above 10 of 255) plus the CONTRACT-1 follow-up fields the case
view measures client-side today (top margin, opaque box).

No real model runs here: the stub backend and fake BiRefNet sessions only.
"""

from __future__ import annotations

import base64
import hashlib
import io

import numpy as np
import pytest
from app.workers.cpu_stages import INLINE
from app.workers.grounding import (
    ALPHA_THRESHOLD,
    AlphaGrounding,
    compute_bottom_margin_frac,
    compute_contact_band,
    measure_alpha_grounding,
)
from app.workers.mask import (
    FULL_FRAME_COVERAGE,
    MASK_MIME,
    MASK_PIPELINE_VERSION,
    MaskOutcome,
    assess_mask_quality,
    build_mask_artifact,
    composite_with_mask,
    decode_mask_png,
    encode_mask_png,
)
from app.workers.matting import (
    BiRefNetMattingBackend,
    MattingBackend,
    StubMattingBackend,
)
from PIL import Image

# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def _golden_alpha() -> np.ndarray:
    """200 x 100 (W x H): a body over rows 10..79, cols 50..149, standing on
    a narrower base (rows 72..79, cols 80..119). Every grounding number is
    exact by construction (see TestMeasureAlphaGrounding.test_golden_values)."""
    alpha = np.zeros((100, 200), dtype=np.uint8)
    alpha[10:72, 50:150] = 255
    alpha[72:80, 80:120] = 255
    return alpha


def _encode(img: Image.Image, fmt: str) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _studio_photo(fmt: str = "PNG") -> bytes:
    """A near-white backdrop with a dark figure on a narrower base -- the
    stub backend separates it exactly (uniform backdrop)."""
    arr = np.full((100, 80, 3), (250, 250, 248), dtype=np.uint8)
    arr[20:82, 25:55] = (40, 30, 90)  # body
    arr[82:90, 30:50] = (40, 30, 90)  # base: the 8-row contact band exactly
    return _encode(Image.fromarray(arr, mode="RGB"), fmt)


class _ConstantAlphaBackend(MattingBackend):
    """predict_alpha answers one fixed array, whatever the image."""

    name = "constant"

    def __init__(self, alpha: np.ndarray) -> None:
        self._alpha = alpha

    def predict_alpha(self, image: Image.Image) -> np.ndarray:
        assert (image.height, image.width) == self._alpha.shape
        return self._alpha.copy()

    def matte(self, image: Image.Image) -> Image.Image:  # pragma: no cover - unused
        raise AssertionError("the mask lane must never ask for a cut-out")


class _FakeOrtSession:
    """Duck-types onnxruntime.InferenceSession: a logit map that is strongly
    foreground on the left half and background on the right half."""

    class _Input:
        name = "input_image"

    def get_inputs(self) -> list[_FakeOrtSession._Input]:
        return [self._Input()]

    def run(self, _names: object, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        (arr,) = feed.values()
        h, w = arr.shape[2], arr.shape[3]
        logits = np.full((1, 1, h, w), -12.0, dtype=np.float32)
        logits[..., : w // 2] = 12.0
        return [logits]


# ---------------------------------------------------------------------------
# grounding on a bare alpha mask
# ---------------------------------------------------------------------------


class TestMeasureAlphaGrounding:
    def test_golden_values(self) -> None:
        g = measure_alpha_grounding(_golden_alpha())
        assert g == AlphaGrounding(
            bottom_margin_frac=pytest.approx(20 / 100),
            top_margin_frac=pytest.approx(10 / 100),
            box_top_frac=pytest.approx(10 / 100),
            box_bottom_frac=pytest.approx(80 / 100),
            box_left_frac=pytest.approx(50 / 200),
            box_right_frac=pytest.approx(150 / 200),
            contact_band_center_x_frac=pytest.approx((80 + 119) / 2 / 200),
            contact_band_width_frac=pytest.approx((119 - 80) / 200),
        )

    def test_box_edges_convert_exactly_to_the_case_view_margins(self) -> None:
        """fc-mobile computeTopMarginFrac / computeSideMarginFracs /
        computeBottomMarginFrac read the clear margins; the box is the
        half-open opaque extent, so 1 - box edge == the far margin."""
        g = measure_alpha_grounding(_golden_alpha())
        assert g is not None
        assert g.box_top_frac == g.top_margin_frac
        assert 1 - g.box_bottom_frac == pytest.approx(g.bottom_margin_frac)
        assert g.box_left_frac == pytest.approx(50 / 200)  # computeSideMarginFracs left
        assert 1 - g.box_right_frac == pytest.approx((200 - 1 - 149) / 200)  # its right

    def test_single_pixel(self) -> None:
        alpha = np.zeros((10, 20), dtype=np.uint8)
        alpha[5, 7] = 200
        g = measure_alpha_grounding(alpha)
        assert g is not None
        assert g.top_margin_frac == pytest.approx(0.5)
        assert g.bottom_margin_frac == pytest.approx(0.4)
        assert (g.box_top_frac, g.box_bottom_frac) == (pytest.approx(0.5), pytest.approx(0.6))
        assert (g.box_left_frac, g.box_right_frac) == (pytest.approx(0.35), pytest.approx(0.4))
        assert g.contact_band_center_x_frac == pytest.approx(7 / 20)
        assert g.contact_band_width_frac == 0

    def test_full_frame(self) -> None:
        g = measure_alpha_grounding(np.full((40, 30), 255, dtype=np.uint8))
        assert g is not None
        assert (g.top_margin_frac, g.bottom_margin_frac) == (0, 0)
        assert (g.box_top_frac, g.box_bottom_frac, g.box_left_frac, g.box_right_frac) == (
            0,
            1,
            0,
            1,
        )
        assert g.contact_band_width_frac == pytest.approx(29 / 30)

    def test_alpha_at_threshold_is_not_opaque(self) -> None:
        alpha = np.full((8, 8), ALPHA_THRESHOLD, dtype=np.uint8)
        assert measure_alpha_grounding(alpha) is None

    def test_alpha_one_above_threshold_is_opaque(self) -> None:
        alpha = np.zeros((8, 8), dtype=np.uint8)
        alpha[3, 3] = ALPHA_THRESHOLD + 1
        assert measure_alpha_grounding(alpha) is not None

    def test_threshold_is_the_contract_value(self) -> None:
        assert ALPHA_THRESHOLD == 10

    def test_empty_and_degenerate_buffers_are_unmeasured(self) -> None:
        assert measure_alpha_grounding(np.zeros((5, 5), dtype=np.uint8)) is None
        assert measure_alpha_grounding(np.zeros((0, 5), dtype=np.uint8)) is None

    def test_parity_with_the_july_rgba_port(self) -> None:
        alpha = _golden_alpha()
        rgba = np.zeros((*alpha.shape, 4), dtype=np.uint8)
        rgba[..., 3] = alpha
        g = measure_alpha_grounding(alpha)
        assert g is not None
        assert g.bottom_margin_frac == compute_bottom_margin_frac(rgba)
        assert (g.contact_band_center_x_frac, g.contact_band_width_frac) == compute_contact_band(
            rgba
        )

    def test_rejects_a_non_2d_buffer(self) -> None:
        with pytest.raises(ValueError, match="height, width"):
            measure_alpha_grounding(np.zeros((4, 4, 4), dtype=np.uint8))


# ---------------------------------------------------------------------------
# the mask PNG: one channel of information, carried as alpha
# ---------------------------------------------------------------------------


class TestMaskPng:
    def test_alpha_carries_the_mask_and_luminance_is_constant_white(self) -> None:
        alpha = _golden_alpha()
        img = Image.open(io.BytesIO(encode_mask_png(alpha)))
        assert img.format == "PNG"
        assert img.mode == "LA"
        assert img.size == (200, 100)
        arr = np.asarray(img)
        assert bool((arr[..., 0] == 255).all())
        assert np.array_equal(arr[..., 1], alpha)

    def test_mime_is_png(self) -> None:
        assert MASK_MIME == "image/png"

    def test_encoding_is_deterministic(self) -> None:
        alpha = _golden_alpha()
        assert encode_mask_png(alpha) == encode_mask_png(alpha.copy())

    def test_round_trip(self) -> None:
        alpha = np.arange(256, dtype=np.uint8).reshape(16, 16)
        assert np.array_equal(decode_mask_png(encode_mask_png(alpha)), alpha)

    def test_decode_rejects_a_picture_that_is_not_a_mask(self) -> None:
        with pytest.raises(ValueError):
            decode_mask_png(_encode(Image.new("RGB", (4, 4)), "PNG"))

    @pytest.mark.parametrize(
        "bad",
        [
            np.zeros((4, 4, 1), dtype=np.uint8),
            np.zeros((4, 4), dtype=np.float32),
            np.zeros((0, 4), dtype=np.uint8),
        ],
    )
    def test_encode_rejects_anything_but_a_2d_uint8_mask(self, bad: np.ndarray) -> None:
        with pytest.raises(ValueError):
            encode_mask_png(bad)


# ---------------------------------------------------------------------------
# the client composite: CSS mask-image (alpha mode) / canvas destination-in
# ---------------------------------------------------------------------------


class TestCompositeWithMask:
    def test_opaque_clean_copy_takes_the_mask_as_its_alpha(self) -> None:
        clean = Image.new("RGB", (200, 100), (9, 99, 199))
        out = composite_with_mask(clean, _golden_alpha())
        arr = np.asarray(out)
        assert out.mode == "RGBA"
        assert bool((arr[..., :3] == (9, 99, 199)).all())
        assert np.array_equal(arr[..., 3], _golden_alpha())

    def test_multiplies_into_a_clean_copy_that_has_its_own_alpha(self) -> None:
        clean = Image.new("RGBA", (2, 1), (1, 2, 3, 128))
        alpha = np.array([[255, 128]], dtype=np.uint8)
        arr = np.asarray(composite_with_mask(clean, alpha))
        assert arr[0, 0, 3] == 128
        assert arr[0, 1, 3] == 64  # round(128 * 128 / 255)

    def test_rounds_the_product_to_nearest(self) -> None:
        clean = Image.new("RGBA", (1, 1), (1, 2, 3, 128))
        arr = np.asarray(composite_with_mask(clean, np.array([[129]], dtype=np.uint8)))
        assert arr[0, 0, 3] == 65  # 128 * 129 / 255 = 64.75

    def test_rejects_a_mask_of_other_dimensions(self) -> None:
        with pytest.raises(ValueError):
            composite_with_mask(Image.new("RGB", (5, 5)), np.zeros((5, 6), dtype=np.uint8))


# ---------------------------------------------------------------------------
# the backends: one alpha, whether asked for a mask or a cut-out
# ---------------------------------------------------------------------------


class TestPredictAlpha:
    def test_stub_alpha_is_the_alpha_of_its_cutout(self) -> None:
        img = Image.open(io.BytesIO(_studio_photo())).convert("RGBA")
        backend = StubMattingBackend()
        alpha = backend.predict_alpha(img)
        assert alpha.shape == (100, 80)
        assert alpha.dtype == np.uint8
        assert np.array_equal(alpha, np.asarray(backend.matte(img))[..., 3])

    def test_stub_cutout_keeps_transparency_its_mask_ignores(self) -> None:
        arr = np.full((20, 20, 4), (250, 250, 248, 255), dtype=np.uint8)
        arr[8:12, 8:12] = (20, 30, 40, 255)
        arr[10, 10, 3] = 40  # the subject already partly transparent here
        img = Image.fromarray(arr, mode="RGBA")
        backend = StubMattingBackend()
        assert backend.predict_alpha(img)[10, 10] == 255
        assert np.asarray(backend.matte(img))[10, 10, 3] == 40

    def test_birefnet_alpha_is_the_alpha_of_its_cutout(self) -> None:
        img = Image.new("RGBA", (64, 32), (10, 20, 30, 255))
        backend = BiRefNetMattingBackend(session=_FakeOrtSession())
        alpha = backend.predict_alpha(img)
        assert alpha.shape == (32, 64)
        assert alpha[16, 5] == 255
        assert alpha[16, 60] == 0
        assert np.array_equal(alpha, np.asarray(backend.matte(img))[..., 3])

    def test_predict_alpha_ignores_the_images_own_alpha(self) -> None:
        """The mask is the model's answer alone; the composite multiplies in
        the clean copy's own alpha at display time."""
        img = Image.new("RGBA", (64, 32), (10, 20, 30, 0))
        backend = BiRefNetMattingBackend(session=_FakeOrtSession())
        assert backend.predict_alpha(img)[16, 5] == 255

    def test_a_backend_that_only_mattes_still_yields_an_alpha(self) -> None:
        class _HalfCut(MattingBackend):
            def matte(self, image: Image.Image) -> Image.Image:
                arr = np.asarray(image.convert("RGBA")).copy()
                arr[:, arr.shape[1] // 2 :, 3] = 0
                return Image.fromarray(arr, mode="RGBA")

        alpha = _HalfCut().predict_alpha(Image.new("RGB", (10, 4), (1, 1, 1)))
        assert alpha.tolist() == [[255] * 5 + [0] * 5] * 4

    def test_backends_name_themselves_for_provenance(self) -> None:
        assert StubMattingBackend.name == "stub"
        assert BiRefNetMattingBackend.name == "birefnet"


# ---------------------------------------------------------------------------
# the quality gate
# ---------------------------------------------------------------------------


class TestAssessMaskQuality:
    def test_nothing_opaque_is_empty(self) -> None:
        alpha = np.zeros((10, 10), dtype=np.uint8)
        assert assess_mask_quality(alpha, measure_alpha_grounding(alpha)) == (
            "empty",
            "no_opaque_pixels",
        )

    def test_a_figure_with_clear_surroundings_is_ok(self) -> None:
        alpha = _golden_alpha()
        assert assess_mask_quality(alpha, measure_alpha_grounding(alpha)) == ("ok", None)

    def test_coverage_at_the_full_frame_line_is_held(self) -> None:
        alpha = np.zeros((10, 100), dtype=np.uint8)
        alpha[:, : int(FULL_FRAME_COVERAGE * 100)] = 255
        alpha[0, 0] = 255
        g = measure_alpha_grounding(alpha)
        assert assess_mask_quality(alpha, g) == ("held", "full_frame")

    def test_coverage_just_under_the_full_frame_line_is_not_held_for_it(self) -> None:
        alpha = np.zeros((100, 100), dtype=np.uint8)
        alpha[: int(FULL_FRAME_COVERAGE * 100), :] = 255
        alpha[0, 0] = 0  # one pixel under the line
        # three clear rows below it (the edge tolerance is two): no floor either
        g = measure_alpha_grounding(alpha)
        assert assess_mask_quality(alpha, g) == ("ok", None)

    def test_a_base_band_across_the_whole_bottom_edge_is_held(self) -> None:
        """The black-dress failure shape: the backdrop's floor is kept, so the
        bottom rows are opaque edge to edge on the image's last row."""
        alpha = _golden_alpha()
        alpha[90:100, :] = 255
        g = measure_alpha_grounding(alpha)
        assert assess_mask_quality(alpha, g) == ("held", "bottom_edge_bleed")

    def test_a_wide_base_that_stops_short_of_the_last_row_is_ok(self) -> None:
        alpha = _golden_alpha()
        alpha[85:95, :] = 255  # edge to edge, but five clear rows below it
        g = measure_alpha_grounding(alpha)
        assert assess_mask_quality(alpha, g) == ("ok", None)


def _standing(h: int, w: int) -> np.ndarray:
    """A narrow figure in the middle of the frame, clear of every edge."""
    alpha = np.zeros((h, w), dtype=np.uint8)
    alpha[h // 10 : h * 7 // 10, w * 2 // 5 : w * 3 // 5] = 255
    return alpha


_ALL = slice(None)


def _with_floor(
    alpha: np.ndarray, *, rows: int, clear_below: int = 0, cols: slice = _ALL
) -> np.ndarray:
    out = alpha.copy()
    h = out.shape[0]
    out[h - clear_below - rows : h - clear_below, cols] = 255
    return out


def _gate(alpha: np.ndarray) -> tuple[str, str | None]:
    return assess_mask_quality(alpha, measure_alpha_grounding(alpha))


class TestAKeptFloorIsHeld:
    """A FLOOR is an opaque band along the bottom of the frame. The gate looks
    at the last 2 % of the image's rows (at least 2) ending at the mask's
    lowest opaque row; when that row is within 1 % of the height (at least 2
    rows) of the bottom edge and ANY row of the band is at least 85 % opaque,
    the floor was kept: held, never published. Measured 2026-10-11: the real
    July masks that reach the bottom edge (miku-deepsea, ryuuko, spike) are
    at most 73 % opaque on any row there."""

    def test_floor_rows_with_a_clear_jpeg_edge_below_them_are_held(self) -> None:
        # h=400: the edge tolerance is 4 rows
        alpha = _with_floor(_standing(400, 300), rows=30, clear_below=4)
        assert _gate(alpha) == ("held", "bottom_edge_bleed")

    def test_a_band_clear_of_the_edge_by_more_than_the_tolerance_is_no_floor(self) -> None:
        alpha = _with_floor(_standing(400, 300), rows=30, clear_below=5)
        assert _gate(alpha) == ("ok", None)

    @pytest.mark.parametrize(("start", "stop"), [(0, 170), (15, 185), (5, 195), (3, 197)])
    def test_a_floor_short_of_the_side_edges_is_held_down_to_85_percent(
        self, start: int, stop: int
    ) -> None:
        alpha = _with_floor(_standing(400, 200), rows=30, cols=slice(start, stop))
        assert _gate(alpha) == ("held", "bottom_edge_bleed")

    def test_a_band_under_85_percent_opaque_is_no_floor(self) -> None:
        alpha = _with_floor(_standing(400, 200), rows=30, cols=slice(0, 169))
        assert _gate(alpha) == ("ok", None)

    def test_a_thin_full_width_strip_on_the_edge_is_held(self) -> None:
        alpha = _with_floor(_standing(400, 200), rows=2)
        assert _gate(alpha) == ("held", "bottom_edge_bleed")

    def test_a_wide_row_anywhere_in_the_band_is_a_floor(self) -> None:
        # h=400, band = 8 rows: rows 5..8 from the bottom are 95 % opaque,
        # the last 4 only 60 %
        alpha = _with_floor(_standing(400, 200), rows=4, clear_below=4, cols=slice(0, 190))
        alpha = _with_floor(alpha, rows=4, cols=slice(0, 120))
        assert _gate(alpha) == ("held", "bottom_edge_bleed")

    def test_a_wide_row_above_the_band_is_not_weighed(self) -> None:
        # the lowest 8 rows are 60 % opaque; the 95 % rows sit just above them
        alpha = _with_floor(_standing(400, 200), rows=4, clear_below=8, cols=slice(0, 190))
        alpha = _with_floor(alpha, rows=8, cols=slice(0, 120))
        assert _gate(alpha) == ("ok", None)

    def test_a_tiny_image_with_a_full_width_floor_is_held(self) -> None:
        alpha = np.zeros((40, 40), dtype=np.uint8)
        alpha[5:30, 15:25] = 255
        alpha[36:, :] = 255
        assert _gate(alpha) == ("held", "bottom_edge_bleed")

    @pytest.mark.parametrize(("clear_below", "status"), [(2, "held"), (3, "ok")])
    def test_a_small_image_tolerates_two_clear_rows_at_least(
        self, clear_below: int, status: str
    ) -> None:
        alpha = _with_floor(_standing(40, 40), rows=4, clear_below=clear_below)
        assert _gate(alpha)[0] == status

    def test_a_small_image_weighs_two_rows_at_least(self) -> None:
        # h=40: the last row is 60 % opaque, the one above it 95 %
        alpha = _with_floor(_standing(40, 40), rows=1, clear_below=1, cols=slice(0, 38))
        alpha = _with_floor(alpha, rows=1, cols=slice(0, 24))
        assert _gate(alpha) == ("held", "bottom_edge_bleed")

    def test_the_band_of_a_three_row_image_stops_at_its_top_edge(self) -> None:
        alpha = np.zeros((3, 10), dtype=np.uint8)
        alpha[0, :] = 255  # within two rows of the bottom edge, band rows 0..0
        assert _gate(alpha) == ("held", "bottom_edge_bleed")

    def test_a_base_reaching_the_bottom_edge_is_not_a_floor(self) -> None:
        # the July miku-deepsea shape: its base reaches the last row and is
        # 73 % of the width at its widest there
        alpha = _with_floor(_standing(400, 600), rows=8, cols=slice(80, 518))
        assert _gate(alpha) == ("ok", None)


# ---------------------------------------------------------------------------
# the artifact
# ---------------------------------------------------------------------------


class TestBuildMaskArtifact:
    def test_produces_a_separate_mask_with_contract_grounding(self) -> None:
        clean = _studio_photo("PNG")
        out = build_mask_artifact(clean, backend=StubMattingBackend())
        assert isinstance(out, MaskOutcome)
        assert out.status == "ok"
        assert out.reason is None
        assert out.backend == "stub"
        assert out.pipeline_version == MASK_PIPELINE_VERSION
        assert out.mask_png is not None
        assert out.mask_sha256 == hashlib.sha256(out.mask_png).hexdigest()
        assert (out.width, out.height) == (80, 100)
        mask = decode_mask_png(out.mask_png)
        assert mask.shape == (100, 80)
        g = out.grounding
        assert g is not None
        assert g.top_margin_frac == pytest.approx(20 / 100)
        assert g.bottom_margin_frac == pytest.approx((100 - 1 - 89) / 100)
        assert (g.box_left_frac, g.box_right_frac) == (
            pytest.approx(25 / 80),
            pytest.approx(55 / 80),
        )
        assert g.contact_band_center_x_frac == pytest.approx((30 + 49) / 2 / 80)
        assert g.contact_band_width_frac == pytest.approx((49 - 30) / 80)

    def test_the_clean_copy_is_never_altered(self) -> None:
        clean = _studio_photo("JPEG")
        before = bytes(clean)
        build_mask_artifact(clean, backend=StubMattingBackend())
        assert clean == before

    def test_masked_and_unmasked_appearance_differ(self) -> None:
        out = build_mask_artifact(_studio_photo("PNG"), backend=StubMattingBackend())
        assert out.masked is not None
        assert out.masked.dominant_color == "#281e5a"  # the figure, lowercase
        assert out.unmasked.dominant_color == "#fafaf8"  # the backdrop dominates
        # ThumbHash header bit 23 = has-alpha: set only with the mask applied.
        masked_hdr = base64.b64decode(out.masked.thumbhash)
        plain_hdr = base64.b64decode(out.unmasked.thumbhash)
        assert masked_hdr[2] & 0x80
        assert not plain_hdr[2] & 0x80

    def test_an_opaque_clean_copy_has_no_own_alpha_grounding(self) -> None:
        out = build_mask_artifact(_studio_photo("JPEG"), backend=StubMattingBackend())
        assert out.own_alpha_grounding is None

    def test_a_clean_copy_with_its_own_alpha_is_measured_on_it(self) -> None:
        rgba = np.zeros((20, 10, 4), dtype=np.uint8)
        rgba[5:15, 2:8] = (200, 10, 10, 255)
        clean = _encode(Image.fromarray(rgba, mode="RGBA"), "PNG")
        out = build_mask_artifact(
            clean, backend=_ConstantAlphaBackend(np.full((20, 10), 255, np.uint8))
        )
        assert out.own_alpha_grounding is not None
        assert out.own_alpha_grounding.top_margin_frac == pytest.approx(5 / 20)

    def test_an_alpha_of_254_is_the_clean_copys_own_alpha(self) -> None:
        rgba = np.full((20, 10, 4), 200, dtype=np.uint8)
        rgba[..., 3] = 255
        rgba[19, 9, 3] = 254  # one pixel translucent, by one step
        clean = _encode(Image.fromarray(rgba, mode="RGBA"), "PNG")
        out = build_mask_artifact(
            clean, backend=_ConstantAlphaBackend(np.full((20, 10), 255, np.uint8))
        )
        assert out.own_alpha_grounding is not None
        assert out.own_alpha_grounding.top_margin_frac == 0

    def test_an_empty_answer_stores_no_mask(self) -> None:
        out = build_mask_artifact(
            _studio_photo("PNG"), backend=_ConstantAlphaBackend(np.zeros((100, 80), np.uint8))
        )
        assert (out.status, out.reason) == ("empty", "no_opaque_pixels")
        assert out.mask_png is None
        assert out.mask_sha256 is None
        assert out.grounding is None
        assert out.masked is None

    def test_a_full_frame_answer_is_held_but_kept_for_review(self) -> None:
        out = build_mask_artifact(
            _studio_photo("PNG"),
            backend=_ConstantAlphaBackend(np.full((100, 80), 255, np.uint8)),
        )
        assert (out.status, out.reason) == ("held", "full_frame")
        assert out.mask_png is not None
        assert out.grounding is not None

    def test_same_bytes_same_mask_address(self) -> None:
        clean = _studio_photo("PNG")
        a = build_mask_artifact(clean, backend=StubMattingBackend())
        b = build_mask_artifact(clean, backend=StubMattingBackend())
        assert a.mask_sha256 == b.mask_sha256

    def test_reads_exif_orientation_like_the_clean_copy_is_shown(self) -> None:
        """A browser draws the clean copy with its EXIF orientation applied,
        so the mask must be computed on (and sized like) the oriented image."""
        arr = np.full((10, 30, 3), 250, dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        exif = Image.Exif()
        exif[0x0112] = 6  # rotate 90 CW on display -> 10 wide x 30 tall
        buf = io.BytesIO()
        img.save(buf, format="JPEG", exif=exif.tobytes())
        out = build_mask_artifact(
            buf.getvalue(), backend=_ConstantAlphaBackend(np.full((30, 10), 255, np.uint8))
        )
        assert (out.width, out.height) == (10, 30)

    def test_a_backend_answer_of_the_wrong_size_is_refused(self) -> None:
        class _WrongSize(MattingBackend):
            name = "wrong"

            def matte(self, image: Image.Image) -> Image.Image:  # pragma: no cover - unused
                raise AssertionError

            def predict_alpha(self, image: Image.Image) -> np.ndarray:
                return np.zeros((image.height + 1, image.width), dtype=np.uint8)

        with pytest.raises(ValueError, match="wrong"):
            build_mask_artifact(_studio_photo("PNG"), backend=_WrongSize())


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
def test_a_mask_judged_on_the_cpu_stages_is_the_one_judged_in_place(fmt: str) -> None:
    """MASK5: handed to the CPU stages after the model (the batch's worker
    processes; inline here), the judgment answers what it answers in place."""
    clean = _studio_photo(fmt)
    in_place = build_mask_artifact(clean, backend=StubMattingBackend())
    assert build_mask_artifact(clean, backend=StubMattingBackend(), cpu=INLINE) == in_place
