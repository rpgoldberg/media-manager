"""Pluggable matting (background-removal) pipeline.

MODEL RULE: this module never downloads a real model or imports
onnxruntime at module scope. StubMattingBackend is a lightweight
luminance/edge threshold against a near-uniform backdrop -- good enough to
exercise the pipeline end-to-end in dev/tests without a GPU. The real
backend (BiRefNetMattingBackend, running BiRefNet via onnxruntime-gpu) is
implemented below, but its ONNX-runtime import and model-session load are
LAZY -- they only happen inside BiRefNetMattingBackend._get_session(), the
first time inference actually runs. Tests exercise the full
preprocess/run/postprocess/apply-alpha pipeline by injecting a fake
session (ctor `session=`) or overriding `_run()`, so onnxruntime and real
model weights are never needed in application code or tests. The weights
themselves + GPU runtime are baked into the worker Docker image at deploy
time (see Dockerfile.worker), keyed by Settings.matting_model_path / env
MATTING_MODEL_PATH.

CRITICAL: nothing in this module (or its callers) may call
``.convert("RGB")`` on the working image -- that silently drops the alpha
channel that the grounding scalars (app.workers.grounding) and the serve
route depend on. Always work in RGBA.
"""

from __future__ import annotations

import io
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from PIL import Image

from ..hashing import detect_mime

if TYPE_CHECKING:
    from ..config import Settings
    from .stage_timing import StageTimings

logger = logging.getLogger(__name__)

#: Where the ONNX sessions run (``Settings.matting_device``, env MATTING_DEVICE):
#: ``cuda`` requires the CUDA execution provider and refuses a session without
#: it; ``cpu`` runs on the CPU provider alone, lean (no CPU arena, no memory
#: pattern); ``auto`` asks for CUDA and falls back to that lean CPU session.
MATTING_DEVICES = ("cuda", "auto", "cpu")

#: The CUDA execution provider options every GPU session shares; each model
#: adds its own arena cap (``gpu_mem_limit``, its share of the total budget).
#: ``kSameAsRequested`` grows an arena by the size asked for, not by doubling
#: (onnxruntime's default ``kNextPowerOfTwo``), so an arena stops at what the
#: model needs instead of rounding up to the next power of two.
CUDA_PROVIDER_OPTIONS: dict[str, Any] = {
    "device_id": 0,
    "arena_extend_strategy": "kSameAsRequested",
}

#: The env var holding the TOTAL GPU memory budget (MiB) for every session the
#: batch loads together; each session's ``gpu_mem_limit`` is its share.
GPU_MEM_BUDGET_ENV = "MASK_GPU_MEM_BUDGET_MB"
#: The default total (MiB): the 12.3 GB WSL card less ~3.5 GB for the other GPU
#: users (the 11-05 run reached 11.2 GB with caps summing to 12 GiB).
DEFAULT_GPU_MEM_BUDGET_MB = 8800
#: The smallest total (MiB) that loads a model: an IS-Net session runs within
#: a 1 GiB arena (MASK-BATCH-1), and its share of the budget is 2/12.
MIN_GPU_MEM_BUDGET_MB = 6144
#: The sessions the budget is split across, as ``GPU_MEM_LIMIT`` weights: one
#: BiRefNet (8) and its two IS-Net donors (2 + 2).
_BUDGET_WEIGHTS_TOTAL = (8 << 30) + 2 * (2 << 30)


def gpu_mem_budget_mb() -> int:
    """The total GPU budget in MiB from ``MASK_GPU_MEM_BUDGET_MB`` (default
    ``DEFAULT_GPU_MEM_BUDGET_MB``); ValueError if it is not a whole number, is
    not positive, or is below ``MIN_GPU_MEM_BUDGET_MB``."""
    raw = os.environ.get(GPU_MEM_BUDGET_ENV)
    if raw is None or raw.strip() == "":
        return DEFAULT_GPU_MEM_BUDGET_MB
    try:
        value = int(raw.strip())
    except ValueError:
        raise ValueError(
            f"{GPU_MEM_BUDGET_ENV}={raw!r} is not a whole number of MiB "
            f"(minimum {MIN_GPU_MEM_BUDGET_MB})"
        ) from None
    if value <= 0 or value < MIN_GPU_MEM_BUDGET_MB:
        raise ValueError(
            f"{GPU_MEM_BUDGET_ENV}={value} is too small to load a model: "
            f"minimum {MIN_GPU_MEM_BUDGET_MB} MiB"
        )
    return value


def gpu_mem_limit(weight: int) -> int:
    """The ``gpu_mem_limit`` (bytes) of a session whose model weighs ``weight``
    (its class ``GPU_MEM_LIMIT``): its proportional share of the budget."""
    return (gpu_mem_budget_mb() << 20) * weight // _BUDGET_WEIGHTS_TOTAL


class MattingBackend(ABC):
    """Interface for producing an RGBA cutout from a source image."""

    #: Recorded beside every mask as its provenance (which backend made it).
    name: ClassVar[str] = "custom"
    #: Whether an APPLIED mask batch may store and publish this backend's
    #: masks (app.workers.mask_job). Opt-in: only a real matting model sets
    #: it. The stub's backdrop-distance heuristic keeps a dark floor under a
    #: dark figure and still passes the quality gate, so it is dry run only.
    publishes_masks: ClassVar[bool] = False

    @abstractmethod
    def matte(self, image: Image.Image) -> Image.Image:
        """Given an RGBA PIL Image, return a new RGBA Image with the alpha
        channel replaced (or refined) to cut out the background."""
        raise NotImplementedError

    def predict_alpha(self, image: Image.Image) -> np.ndarray:
        """The backend's foreground answer alone, as a ``(height, width)``
        uint8 alpha -- the separate mask MASK-1 stores (Ross 2026-09-26: the
        clean copy is never cut out; the client composites this over it).

        This default reads the alpha of ``matte()`` for backends that only
        implement the cut-out; the shipped backends override it so the mask
        is the model's answer without the image's own alpha folded in.
        """
        cutout = self.matte(image.convert("RGBA"))
        return np.asarray(cutout.convert("RGBA"))[..., 3].copy()

    def predict_alpha_timed(self, image: Image.Image, timings: StageTimings) -> np.ndarray:
        """``predict_alpha``, its seconds in ``timings`` (MASK-PERF-1): one
        "inference" stage here; a backend of several passes times each."""
        with timings.stage("inference"):
            return self.predict_alpha(image)

    def load(self) -> None:
        """Open every model session now, so a broken runtime is refused before
        any image is judged. Nothing to open for a backend without a model."""
        return None

    def release(self) -> None:
        """Drop every model session this backend opened (frees the GPU memory
        it held); the next inference opens them again."""
        return None

    @property
    def active_provider(self) -> str | None:
        """The execution provider the (primary) model session runs on, once
        open; None before, or for a backend without a model."""
        return None


class StubMattingBackend(MattingBackend):
    """Default backend: samples a near-uniform backdrop color from the
    image's own corners and thresholds pixel distance from it to transparent
    vs. opaque. This is NOT real subject segmentation -- it's a stand-in
    that produces a real RGBA cutout so the rest of the pipeline (grounding
    scalars, watermark, serve route) can be built and tested against real
    alpha data without a GPU model. Swap via get_matting_backend() /
    Settings.matting_backend once a real backend is deployed.
    """

    #: Euclidean RGB distance from the sampled backdrop color below which a
    #: pixel is considered part of the backdrop (made transparent).
    BACKDROP_DISTANCE_THRESHOLD = 30.0
    #: Corner patch size (pixels) used to sample the backdrop color.
    CORNER_SAMPLE = 5

    name: ClassVar[str] = "stub"

    def predict_alpha(self, image: Image.Image) -> np.ndarray:
        arr = np.asarray(image.convert("RGBA"))
        h, w = arr.shape[0], arr.shape[1]
        rgb = arr[..., :3].astype(np.float32)

        n = max(1, min(self.CORNER_SAMPLE, h, w))
        corners = np.concatenate(
            [
                rgb[:n, :n].reshape(-1, 3),
                rgb[:n, -n:].reshape(-1, 3),
                rgb[-n:, :n].reshape(-1, 3),
                rgb[-n:, -n:].reshape(-1, 3),
            ]
        )
        backdrop = np.median(corners, axis=0)

        dist = np.sqrt(((rgb - backdrop) ** 2).sum(axis=-1))
        return np.where(dist < self.BACKDROP_DISTANCE_THRESHOLD, 0, 255).astype(np.uint8)

    def matte(self, image: Image.Image) -> Image.Image:
        rgba = image.convert("RGBA")
        arr = np.asarray(rgba)
        computed_alpha = self.predict_alpha(rgba)

        # Never make an already-transparent pixel opaque -- only ever cut
        # further into the existing alpha channel.
        existing_alpha = arr[..., 3]
        combined_alpha = np.minimum(computed_alpha, existing_alpha)

        out = np.dstack([arr[..., :3], combined_alpha]).astype(np.uint8)
        return Image.fromarray(out, mode="RGBA")


class BiRefNetMattingBackend(MattingBackend):
    """Real subject-segmentation backend: BiRefNet, run via
    onnxruntime-gpu against ONNX weights baked into the worker Docker
    image (see Dockerfile.worker) at deploy time: rembg's fp32
    BiRefNet-general export, the weights the mask lane was measured on.

    Model spec (see onnx-community/BiRefNet-ONNX preprocessor_config.json
    on Hugging Face, MIT licensed export of ZhengPeng7/BiRefNet):
      - input: single RGB image, resized to 1024x1024, rescaled to [0, 1],
        then normalized per-channel with ImageNet mean/std, NCHW float32.
      - output: a single-channel (1, 1, 1024, 1024) logit map -- a sigmoid
        must be applied to get a [0, 1] foreground probability (it is NOT
        already a probability). This backend additionally min-max
        rescales the sigmoid output per-image (the contrast-stretch used
        by the reference rembg BiRefNet session), which is a no-op when
        the mask already spans [0, 1] but keeps a saturated/low-contrast
        raw output usable.

    DEPLOY CONCERN, kept out of application code and tests: the
    onnxruntime import and the ONNX session load both happen lazily,
    inside ``_get_session()``, on first inference -- never at module
    import time and never in the constructor. Tests exercise the real
    preprocess/run/postprocess/apply-alpha pipeline by injecting a fake
    session (``session=`` ctor arg) or by overriding ``_run()`` in a
    subclass, so neither onnxruntime nor real model weights are ever
    required outside the deploy image.

    A configured-but-broken backend (missing model path, unreadable
    weights, or a session that loads without the CUDA execution provider
    -- e.g. a CUDA/cuDNN version mismatch against the pin in
    Dockerfile.worker) raises ``RuntimeError`` clearly rather than
    silently degrading to a wrong or CPU-slow result.
    """

    #: BiRefNet's fixed square input resolution.
    INPUT_SIZE = 1024

    name: ClassVar[str] = "birefnet"
    publishes_masks: ClassVar[bool] = True
    #: The CUDA arena cap (bytes). Measured on the RTX 4070 Ti (12 GB,
    #: MASK-BATCH-1): unbounded, BiRefNet's arena grows to the whole card and
    #: the base-keep pass spills into shared memory (~5.5 s per image instead
    #: of ~0.7 s); a 6 or 7 GiB cap fails (one decoder tensor is 784 MiB);
    #: 8 GiB runs, with the same masks as unbounded. Now the model's WEIGHT in
    #: the split of ``MASK_GPU_MEM_BUDGET_MB`` (its limit is 8/12 of it).
    GPU_MEM_LIMIT: ClassVar[int] = 8 << 30
    #: ImageNet normalization stats used by the model's own preprocessor.
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        *,
        model_path: str | None = None,
        session: Any | None = None,
        device: str = "cuda",
        cpu_threads: int | None = None,
    ) -> None:
        """``model_path`` is the deploy-time ONNX weights path
        (Settings.matting_model_path / env MATTING_MODEL_PATH), only
        needed on the real inference path. ``session`` is a test seam: any
        object duck-typing onnxruntime's ``InferenceSession`` (a
        ``get_inputs()`` returning objects with a ``.name``, and a
        ``run(output_names, input_feed)``) can be injected directly,
        bypassing onnxruntime and the model path entirely. ``device`` is one
        of ``MATTING_DEVICES``; ``cpu_threads`` sets a CPU session's
        intra-op threads (None: ONNX Runtime's default).
        """
        if device not in MATTING_DEVICES:
            raise ValueError(f"unknown matting device {device!r}: one of {MATTING_DEVICES}")
        self._model_path = model_path
        self._session = session
        self.device = device
        self.cpu_threads = cpu_threads

    def cuda_provider_options(self) -> dict[str, Any]:
        """This model's CUDA execution provider options."""
        return {**CUDA_PROVIDER_OPTIONS, "gpu_mem_limit": gpu_mem_limit(self.GPU_MEM_LIMIT)}

    def load(self) -> None:
        self._get_session()

    def release(self) -> None:
        # Only a session this backend opened itself: an injected one has no
        # weights to reopen it from.
        if self._model_path:
            self._session = None

    @property
    def active_provider(self) -> str | None:
        get_providers = getattr(self._session, "get_providers", None)
        if not callable(get_providers):
            return None
        providers = get_providers()
        return str(providers[0]) if providers else None

    def _open(self, factory: Callable[[], Any]) -> Any:
        try:
            return factory()
        except Exception as exc:
            raise RuntimeError(
                f"{type(self).__name__} failed to load ONNX weights from "
                f"{self._model_path!r}: {exc}"
            ) from exc

    def _cpu_session(self, ort: Any) -> Any:
        """The lean CPU session: no CPU arena and no memory pattern (the
        IMG-SERVER-1 bench: BiRefNet peaks at 7.1 GB this way, 13 GB with
        ONNX Runtime's defaults)."""
        options = ort.SessionOptions()
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        if self.cpu_threads:
            options.intra_op_num_threads = self.cpu_threads
        return self._open(
            lambda: ort.InferenceSession(
                self._model_path, sess_options=options, providers=["CPUExecutionProvider"]
            )
        )

    def _get_session(self) -> Any:
        """Lazily resolve the inference session. Only imports onnxruntime
        and loads weights the first time real inference is needed -- never
        at construction, so building a (possibly unconfigured)
        BiRefNetMattingBackend is always cheap and side-effect free."""
        if self._session is not None:
            return self._session

        if not self._model_path:
            raise RuntimeError(
                f"{type(self).__name__} has no ONNX weights configured: "
                "set Settings.matting_model_path (env MATTING_MODEL_PATH) "
                "to the BiRefNet weights baked into the worker image (see "
                "Dockerfile.worker; matting_isnet_*_path for IS-Net), or "
                "pass session= explicitly."
            )

        # Import onnxruntime lazily -- never at module import time -- so
        # application code and tests never need the GPU-only dependency.
        import onnxruntime as ort  # type: ignore[import-not-found]

        if self.device == "cpu":
            self._session = self._cpu_session(ort)
            return self._session

        providers = [
            ("CUDAExecutionProvider", self.cuda_provider_options()),
            "CPUExecutionProvider",
        ]

        def cuda_session() -> Any:
            return ort.InferenceSession(self._model_path, providers=providers)

        session: Any | None
        if self.device == "cuda":
            session = self._open(cuda_session)
        else:
            try:
                session = cuda_session()
            except Exception:
                session = None

        if session is None or "CUDAExecutionProvider" not in session.get_providers():
            if self.device == "cuda":
                # Never silently degrade to CPU: onnxruntime falls back
                # quietly on a CUDA/cuDNN version mismatch (same hazard class
                # as ultralytics' cu12/cu13 wheel collisions), which would
                # make matting orders of magnitude too slow without any error
                # -- refuse instead so the deploy-time pin gets fixed.
                raise RuntimeError(
                    "BiRefNetMattingBackend: onnxruntime session loaded "
                    "without CUDAExecutionProvider (silent CPU fallback -- "
                    "check the CUDA/cuDNN pin in Dockerfile.worker matches "
                    "the installed onnxruntime-gpu build)."
                )
            # device "auto": the fallback is explicit, lean, and said out loud.
            logger.warning(
                "matting_cpu_fallback",
                extra={"backend": type(self).__name__, "model_path": self._model_path},
            )
            session = self._cpu_session(ort)

        self._session = session
        return self._session

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """RGBA source -> normalized NCHW float32 model input tensor. Reads
        the RGB channels directly out of the RGBA array (never
        ``.convert("RGB")`` on a PIL Image -- see module docstring)."""
        rgba = image if image.mode == "RGBA" else image.convert("RGBA")
        rgb_arr = np.ascontiguousarray(np.asarray(rgba)[..., :3])
        resized = Image.fromarray(rgb_arr, mode="RGB").resize(
            (self.INPUT_SIZE, self.INPUT_SIZE), Image.BILINEAR
        )
        arr = np.asarray(resized).astype(np.float32) / 255.0
        mean = np.array(self.MEAN, dtype=np.float32)
        std = np.array(self.STD, dtype=np.float32)
        arr = (arr - mean) / std
        chw = arr.transpose(2, 0, 1)  # HWC -> CHW
        return np.expand_dims(chw, axis=0).astype(np.float32)  # NCHW

    def _run(self, input_array: np.ndarray) -> np.ndarray:
        """Run the model. Isolated as its own method so tests can override
        it (subclass) with a fake returning a synthetic logit array of the
        same shape a real onnxruntime session would produce, with no
        onnxruntime import and no real weights involved."""
        session = self._get_session()
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_array})
        return np.asarray(outputs[0])  # (1, 1, H, W) raw logits

    def _postprocess(self, raw_output: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """Raw (1, 1, H, W) logits -> a uint8 alpha mask resized to the
        source image's own (width, height)."""
        logits = raw_output[:, 0, :, :]
        probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid: logits -> [0, 1]
        pred = np.squeeze(probs, axis=0)  # (H, W)

        lo, hi = float(pred.min()), float(pred.max())
        if hi > lo:
            pred = (pred - lo) / (hi - lo)

        # round(), not a bare astype(uint8) truncation -- truncating would
        # systematically bias every pixel's alpha down by up to a full
        # level (e.g. a near-saturated 254.99 truncating to 254 instead of
        # the correct 255).
        mask_u8 = np.clip(np.round(pred * 255.0), 0, 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_u8, mode="L").resize(size, Image.BILINEAR)
        return np.asarray(mask_img)

    def predict_alpha(self, image: Image.Image) -> np.ndarray:
        rgba = image if image.mode == "RGBA" else image.convert("RGBA")
        raw_output = self._run(self._preprocess(rgba))
        return self._postprocess(raw_output, (rgba.width, rgba.height))

    def matte(self, image: Image.Image) -> Image.Image:
        rgba = image if image.mode == "RGBA" else image.convert("RGBA")
        arr = np.asarray(rgba)
        computed_alpha = self.predict_alpha(rgba)

        # Never make an already-transparent pixel opaque -- only ever cut
        # further into the existing alpha channel (same rule as
        # StubMattingBackend).
        existing_alpha = arr[..., 3]
        combined_alpha = np.minimum(computed_alpha, existing_alpha)

        out = np.dstack([arr[..., :3], combined_alpha]).astype(np.uint8)
        return Image.fromarray(out, mode="RGBA")


class IsNetMattingBackend(BiRefNetMattingBackend):
    """IS-Net (DIS) via onnxruntime: the ``isnet-general-use`` and
    ``isnet-anime`` exports the July base-keep pass used as its base donors
    (app.workers.basekept). Same lazy, CUDA-required session as BiRefNet;
    only the tensor conventions differ, and they follow the exports' own
    reference pipeline (rembg's DIS sessions): LANCZOS to 1024, scaled by
    the image's own maximum, ImageNet mean subtracted with unit std; the
    output is min-max normalised with NO sigmoid and truncated to 8 bits,
    then resized back with LANCZOS.
    """

    STD = (1.0, 1.0, 1.0)

    name: ClassVar[str] = "isnet"
    #: Measured (MASK-BATCH-1): an IS-Net session runs within a 1 GiB arena;
    #: 2 GiB leaves room, and adds nothing to the measured peak. Its weight in
    #: the budget split (each session is 2/12 of ``MASK_GPU_MEM_BUDGET_MB``).
    GPU_MEM_LIMIT: ClassVar[int] = 2 << 30

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        rgba = image if image.mode == "RGBA" else image.convert("RGBA")
        rgb_arr = np.ascontiguousarray(np.asarray(rgba)[..., :3])
        resized = Image.fromarray(rgb_arr, mode="RGB").resize(
            (self.INPUT_SIZE, self.INPUT_SIZE), Image.LANCZOS
        )
        arr = np.asarray(resized).astype(np.float32)
        scale = np.float32(max(float(arr.max()), 1e-6))
        mean = np.array(self.MEAN, dtype=np.float32)
        std = np.array(self.STD, dtype=np.float32)
        normalized = (arr / scale - mean) / std
        return np.expand_dims(normalized.transpose(2, 0, 1), axis=0).astype(np.float32)

    def _postprocess(self, raw_output: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        pred = np.squeeze(raw_output[:, 0, :, :], axis=0)
        lo, hi = float(pred.min()), float(pred.max())
        pred = (pred - lo) / max(hi - lo, 1e-12)
        mask = Image.fromarray((pred * 255).astype(np.uint8), mode="L")
        return np.asarray(mask.resize(size, Image.LANCZOS))


_BACKENDS: dict[str, type[MattingBackend]] = {
    "stub": StubMattingBackend,
    "birefnet": BiRefNetMattingBackend,
}


def get_matting_backend(name: str | None = None) -> MattingBackend:
    """Factory for the configured matting backend. Falls back to the stub
    for an unset or unrecognized name -- the pipeline should never hard-fail
    a whole ingest run because of a matting backend misconfiguration."""
    settings: Settings | None = None
    if name is None:
        from ..config import get_settings

        settings = get_settings()
        name = getattr(settings, "matting_backend", "stub")

    key = (name or "stub").lower()
    if key in ("birefnet", "birefnet_basekept"):
        if settings is None:
            from ..config import get_settings

            settings = get_settings()
        placement: dict[str, Any] = {
            "device": settings.matting_device,
            "cpu_threads": settings.matting_cpu_threads,
        }
        if key == "birefnet":
            return BiRefNetMattingBackend(model_path=settings.matting_model_path, **placement)
        from .basekept import BaseKeptMattingBackend

        return BaseKeptMattingBackend(
            primary=BiRefNetMattingBackend(model_path=settings.matting_model_path, **placement),
            donors=[
                IsNetMattingBackend(model_path=settings.matting_isnet_general_path, **placement),
                IsNetMattingBackend(model_path=settings.matting_isnet_anime_path, **placement),
            ],
        )

    backend_cls = _BACKENDS.get(key, StubMattingBackend)
    return backend_cls()


def apply_matting(data: bytes, *, backend: MattingBackend | None = None) -> tuple[bytes, str]:
    """Run the matting pipeline on raw source image bytes.

    Detects the real source mime (never trusts a caller assumption), opens
    it, converts to RGBA (never RGB -- see module docstring), runs the
    backend, and re-encodes as PNG (the alpha-capable, lossless format the
    rest of the pipeline expects). Returns ``(rgba_png_bytes, mime)``.
    """
    detect_mime(data)  # sniff the real source mime; never assume one
    img = Image.open(io.BytesIO(data))
    rgba = img.convert("RGBA")

    active_backend = backend or get_matting_backend()
    matted = active_backend.matte(rgba)
    if matted.mode != "RGBA":
        matted = matted.convert("RGBA")

    out = io.BytesIO()
    matted.save(out, format="PNG")
    return out.getvalue(), "image/png"
