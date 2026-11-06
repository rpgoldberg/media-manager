"""MASK-BATCH-1: where the ONNX sessions run (``MATTING_DEVICE``).

``cuda`` (the default) keeps MASK-1's rule: a session that loaded without the
CUDA execution provider is refused, never silently slow. ``cpu`` is an
explicit CPU run with ONNX Runtime's CPU arena and memory pattern off (the
IMG-SERVER-1 bench: 7.1-7.6 GB peak instead of 13-16 GB). ``auto`` asks for
CUDA and, when the session comes up without it, rebuilds it as that lean CPU
session and says so -- the clean fallback.

No real onnxruntime here: a fake module is put in ``sys.modules`` for the
lazy import inside ``_get_session``.
"""

from __future__ import annotations

import logging
import sys
import types
from collections.abc import Generator
from typing import Any

import pytest
from app.cli import masks as masks_cli
from app.config import Settings
from app.workers import matting
from app.workers.basekept import BaseKeptMattingBackend
from app.workers.matting import (
    CUDA_PROVIDER_OPTIONS,
    BiRefNetMattingBackend,
    IsNetMattingBackend,
    get_matting_backend,
)


class _Options:
    def __init__(self) -> None:
        self.enable_cpu_mem_arena = True
        self.enable_mem_pattern = True
        self.intra_op_num_threads = 0
        self.inter_op_num_threads = 0


class _FakeOrt(types.ModuleType):
    """``cuda_loads``: whether a session asked for CUDA comes up with it."""

    def __init__(self, *, cuda_loads: bool) -> None:
        super().__init__("onnxruntime")
        self.cuda_loads = cuda_loads
        self.sessions: list[_Session] = []
        self.SessionOptions = _Options
        ort = self

        class _Session:
            def __init__(self, path: str, sess_options: Any = None, providers: Any = None) -> None:
                self.path = path
                self.sess_options = sess_options
                self.requested = list(providers or [])
                ort.sessions.append(self)

            def get_providers(self) -> list[str]:
                names = [p[0] if isinstance(p, tuple) else p for p in self.requested]
                if not ort.cuda_loads:
                    names = [n for n in names if n != "CUDAExecutionProvider"]
                return names

        self.InferenceSession = _Session


@pytest.fixture()
def fake_ort(monkeypatch: pytest.MonkeyPatch) -> Generator[_FakeOrt, None, None]:
    ort = _FakeOrt(cuda_loads=True)
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    yield ort


def _names(session: Any) -> list[str]:
    return [p[0] if isinstance(p, tuple) else p for p in session.requested]


def test_cuda_is_the_default_and_asks_for_cuda_first_with_the_pinned_options(
    fake_ort: _FakeOrt,
) -> None:
    backend = BiRefNetMattingBackend(model_path="/m.onnx")
    backend._get_session()
    (session,) = fake_ort.sessions
    assert _names(session) == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert session.requested[0] == ("CUDAExecutionProvider", backend.cuda_provider_options())
    # GPU1-FIX: the limit is BiRefNet's share of the default 8800 MiB budget
    # (8/12), no longer a fixed 8 GiB
    assert backend.cuda_provider_options() == {
        **CUDA_PROVIDER_OPTIONS,
        "gpu_mem_limit": (8800 << 20) * 8 // 12,
    }
    assert CUDA_PROVIDER_OPTIONS["device_id"] == 0
    assert CUDA_PROVIDER_OPTIONS["arena_extend_strategy"] == "kSameAsRequested"
    assert backend.active_provider == "CUDAExecutionProvider"


def test_each_model_has_its_own_gpu_arena_cap() -> None:
    """Measured on the RTX 4070 Ti (MASK-BATCH-1 bench): unbounded, BiRefNet's
    arena grows to the whole 12 GB and the base-keep pass spills into shared
    memory at ~5.5 s/img; capped, it runs at ~0.7 s/img. BiRefNet fails with a
    6 or 7 GiB cap (one decoder tensor is 784 MiB); IS-Net runs within 1 GiB."""
    assert BiRefNetMattingBackend.GPU_MEM_LIMIT == 8 << 30
    assert IsNetMattingBackend.GPU_MEM_LIMIT == 2 << 30
    assert (
        IsNetMattingBackend(model_path="/i.onnx").cuda_provider_options()["gpu_mem_limit"]
        == (8800 << 20) * 2 // 12
    )


# GPU1-FIX: one total budget, split across the sessions in proportion to the caps.


def test_the_default_budget_leaves_the_card_headroom(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MASK_GPU_MEM_BUDGET_MB", raising=False)
    assert matting.gpu_mem_budget_mb() == 8800
    birefnet = matting.gpu_mem_limit(BiRefNetMattingBackend.GPU_MEM_LIMIT)
    isnet = matting.gpu_mem_limit(IsNetMattingBackend.GPU_MEM_LIMIT)
    assert (birefnet >> 20, isnet >> 20) == (5866, 1466)
    assert birefnet + 2 * isnet <= 8800 << 20
    assert abs(birefnet - 4 * isnet) <= 4  # proportional to the 8 GiB : 2 GiB caps (bytes rounded)


def test_a_custom_budget_is_split_in_proportion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MASK_GPU_MEM_BUDGET_MB", "12288")
    assert matting.gpu_mem_limit(BiRefNetMattingBackend.GPU_MEM_LIMIT) == 8 << 30
    assert matting.gpu_mem_limit(IsNetMattingBackend.GPU_MEM_LIMIT) == 2 << 30
    options = IsNetMattingBackend(model_path="/i.onnx").cuda_provider_options()
    assert options["gpu_mem_limit"] == 2 << 30
    assert options["arena_extend_strategy"] == "kSameAsRequested"


@pytest.mark.parametrize("raw", ["abc", "8.5", "-1", "0", "6143", "1"])
def test_an_invalid_or_too_small_budget_is_refused(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv("MASK_GPU_MEM_BUDGET_MB", raw)
    with pytest.raises(ValueError, match="MASK_GPU_MEM_BUDGET_MB.*6144"):
        matting.gpu_mem_budget_mb()


def test_the_minimum_budget_loads_and_blank_is_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MASK_GPU_MEM_BUDGET_MB", "6144")
    assert matting.gpu_mem_budget_mb() == 6144
    assert matting.gpu_mem_limit(IsNetMattingBackend.GPU_MEM_LIMIT) == 1 << 30
    monkeypatch.setenv("MASK_GPU_MEM_BUDGET_MB", "  ")
    assert matting.gpu_mem_budget_mb() == 8800


def test_cuda_refuses_a_session_without_cuda(fake_ort: _FakeOrt) -> None:
    fake_ort.cuda_loads = False
    backend = BiRefNetMattingBackend(model_path="/m.onnx", device="cuda")
    with pytest.raises(RuntimeError, match="without CUDAExecutionProvider"):
        backend._get_session()
    assert backend.active_provider is None


def test_cpu_is_one_lean_cpu_only_session(fake_ort: _FakeOrt) -> None:
    backend = BiRefNetMattingBackend(model_path="/m.onnx", device="cpu", cpu_threads=3)
    backend._get_session()
    (session,) = fake_ort.sessions
    assert _names(session) == ["CPUExecutionProvider"]
    assert session.sess_options.enable_cpu_mem_arena is False
    assert session.sess_options.enable_mem_pattern is False
    assert session.sess_options.intra_op_num_threads == 3
    assert backend.active_provider == "CPUExecutionProvider"


def test_cpu_without_a_thread_count_leaves_onnxruntimes_default(fake_ort: _FakeOrt) -> None:
    BiRefNetMattingBackend(model_path="/m.onnx", device="cpu")._get_session()
    assert fake_ort.sessions[0].sess_options.intra_op_num_threads == 0


def test_auto_with_cuda_keeps_the_cuda_session(fake_ort: _FakeOrt) -> None:
    backend = BiRefNetMattingBackend(model_path="/m.onnx", device="auto")
    first = backend._get_session()
    assert backend._get_session() is first
    assert len(fake_ort.sessions) == 1
    assert backend.active_provider == "CUDAExecutionProvider"


def test_auto_without_cuda_falls_back_to_the_lean_cpu_session_and_says_so(
    fake_ort: _FakeOrt, caplog: pytest.LogCaptureFixture
) -> None:
    fake_ort.cuda_loads = False
    backend = BiRefNetMattingBackend(model_path="/m.onnx", device="auto", cpu_threads=2)
    with caplog.at_level(logging.WARNING, logger="app.workers.matting"):
        session = backend._get_session()
    assert len(fake_ort.sessions) == 2
    assert session is fake_ort.sessions[1]
    assert _names(session) == ["CPUExecutionProvider"]
    assert session.sess_options.enable_cpu_mem_arena is False
    assert session.sess_options.intra_op_num_threads == 2
    assert backend.active_provider == "CPUExecutionProvider"
    assert any(r.getMessage() == "matting_cpu_fallback" for r in caplog.records)


def test_auto_falls_back_when_the_cuda_session_cannot_even_be_built(
    fake_ort: _FakeOrt, monkeypatch: pytest.MonkeyPatch
) -> None:
    built = fake_ort.InferenceSession

    def flaky(path: str, sess_options: Any = None, providers: Any = None) -> Any:
        if providers and isinstance(providers[0], tuple):
            raise RuntimeError("libcudnn.so.9: cannot open shared object file")
        return built(path, sess_options=sess_options, providers=providers)

    monkeypatch.setattr(fake_ort, "InferenceSession", flaky)
    backend = BiRefNetMattingBackend(model_path="/m.onnx", device="auto")
    backend._get_session()
    assert backend.active_provider == "CPUExecutionProvider"
    with pytest.raises(RuntimeError, match="failed to load ONNX weights"):
        BiRefNetMattingBackend(model_path="/m.onnx", device="cuda")._get_session()


def test_load_opens_every_session_up_front(fake_ort: _FakeOrt) -> None:
    primary = BiRefNetMattingBackend(model_path="/b.onnx")
    donors = [IsNetMattingBackend(model_path="/g.onnx"), IsNetMattingBackend(model_path="/a.onnx")]
    BaseKeptMattingBackend(primary=primary, donors=donors).load()
    assert [s.path for s in fake_ort.sessions] == ["/b.onnx", "/g.onnx", "/a.onnx"]
    matting.StubMattingBackend().load()  # nothing to open


def test_an_unknown_device_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="device"):
        BiRefNetMattingBackend(model_path="/m.onnx", device="tpu")


def test_an_injected_session_reports_no_provider_until_it_says_one() -> None:
    class _S:
        def get_providers(self) -> list[str]:
            return ["CPUExecutionProvider"]

    assert BiRefNetMattingBackend(session=_S()).active_provider == "CPUExecutionProvider"
    assert BiRefNetMattingBackend(session=object()).active_provider is None


def test_release_drops_the_session_so_the_next_call_loads_again(fake_ort: _FakeOrt) -> None:
    backend = BiRefNetMattingBackend(model_path="/m.onnx")
    backend._get_session()
    backend.release()
    assert backend.active_provider is None
    backend._get_session()
    assert len(fake_ort.sessions) == 2


def test_release_keeps_an_injected_session() -> None:
    class _S:
        def get_providers(self) -> list[str]:
            return ["CUDAExecutionProvider"]

    backend = BiRefNetMattingBackend(session=_S())
    backend.release()
    assert backend.active_provider == "CUDAExecutionProvider"


def test_base_keep_releases_and_reports_every_session(fake_ort: _FakeOrt) -> None:
    primary = BiRefNetMattingBackend(model_path="/b.onnx", device="cpu")
    donor = IsNetMattingBackend(model_path="/i.onnx", device="cpu")
    backend = BaseKeptMattingBackend(primary=primary, donors=[donor])
    assert backend.active_provider is None
    primary._get_session()
    donor._get_session()
    assert backend.active_provider == "CPUExecutionProvider"
    backend.release()
    assert primary.active_provider is None and donor.active_provider is None


def test_the_stub_has_no_provider_and_release_is_harmless() -> None:
    stub = matting.StubMattingBackend()
    stub.release()
    assert stub.active_provider is None


def test_settings_device_reaches_every_configured_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        matting_backend="birefnet_basekept",
        matting_device="cpu",
        matting_cpu_threads=4,
        matting_model_path="/b.onnx",
        matting_isnet_general_path="/g.onnx",
        matting_isnet_anime_path="/a.onnx",
    )
    monkeypatch.setattr("app.config.get_settings", lambda: settings)
    backend = get_matting_backend()
    assert isinstance(backend, BaseKeptMattingBackend)
    for b in (backend.primary, *backend.donors):
        assert isinstance(b, BiRefNetMattingBackend)
        assert (b.device, b.cpu_threads) == ("cpu", 4)
    plain = get_matting_backend("birefnet")
    assert isinstance(plain, BiRefNetMattingBackend)
    assert (plain.device, plain.cpu_threads) == ("cpu", 4)


def test_the_default_device_setting_is_cuda() -> None:
    assert Settings().matting_device == "cuda"


def test_the_mask_cli_passes_the_device_to_every_session() -> None:
    backend = masks_cli.make_backend(
        "birefnet-basekept", "/b.onnx", "/g.onnx", "/a.onnx", device="auto", cpu_threads=2
    )
    assert isinstance(backend, BaseKeptMattingBackend)
    for b in (backend.primary, *backend.donors):
        assert isinstance(b, BiRefNetMattingBackend)
        assert (b.device, b.cpu_threads) == ("auto", 2)
    plain = masks_cli.make_backend("birefnet", "/b.onnx", None, None, device="cpu")
    assert isinstance(plain, BiRefNetMattingBackend)
    assert plain.device == "cpu"
