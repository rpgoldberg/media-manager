"""MM-CAPS-1: the explicit limits every original passes before anything decodes it.

An original reaches the clean-copy step from the raw bucket as stored bytes (maybe gzip). Until this module the
only guards between those bytes and a full decode were Pillow's DEFAULT pixel guard (which only WARNS between its
bound and twice it) and nothing at all on the gzip step: ``gzip.decompress`` of a bomb allocates whatever it expands
to. The gallery download (MFC-GALLERY-2) puts hundreds of thousands of files through this path, so the bound is
stated, not assumed:

* ``max_bytes``          the stored original, before anything opens it. 32 MiB: the scraper's own capture cap
                         (imageBytes.ts DEFAULT_MAX_IMAGE_BYTES), so nothing it stored is refused for size.
* ``max_pixels``         width x height read from the HEADER (Pillow's open reads the header only; no pixel is
                         decoded to learn the size). 64 megapixels.
* ``max_decoded_bytes``  the memory the clean-copy step would need: at least the RGBA working copy
                         (4 bytes a pixel) or the mode's own depth if deeper. 256 MiB.
* ``gunzip_capped``      the gzip step stops at its output cap instead of expanding a bomb.

A refusal is an :class:`ImageLimitError` with a stable ``code`` (the batch records it as ``refused:<code>``: the
image's own fault, never an outage). An image these limits do not concern, including one Pillow cannot read at all,
is left to fail where it always did. Each limit can be set by environment (``MM_MAX_IMAGE_BYTES``,
``MM_MAX_IMAGE_PIXELS``, ``MM_MAX_DECODED_BYTES``); a value that is not a positive integer keeps the default.

NOT here: the rest of the image track's I8 contract (a format allow-list, verify-then-reopen). This is the
decompression-bomb gate that has to stand before any bulk run.
"""

from __future__ import annotations

import gzip
import io
import os
import warnings
import zlib
from collections.abc import Mapping
from dataclasses import dataclass

from PIL import Image

DEFAULT_MAX_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_PIXELS = 64_000_000
DEFAULT_MAX_DECODED_BYTES = 256 * 1024 * 1024

#: Bytes a pixel the clean-copy step holds at minimum: the RGBA working copy it converts to.
_WORKING_BYTES_PER_PIXEL = 4
#: Pillow modes deeper than the working copy (none today); everything else is measured at the working copy.
_MODE_BYTES = {"I;16": 2, "I;16L": 2, "I;16B": 2, "I": 4, "F": 4, "RGB": 3, "RGBA": 4, "CMYK": 4}


class ImageLimitError(ValueError):
    """An original refused by a limit. ``code`` is stable: too_large_bytes, too_many_pixels, decoded_too_large,
    gzip_too_large."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code

    def __reduce__(self) -> tuple[type[ImageLimitError], tuple[str, str]]:
        # its code and message across processes (MASK5: a CPU worker raises it)
        return type(self), (self.code, str(self))


@dataclass(frozen=True)
class Limits:
    max_bytes: int = DEFAULT_MAX_BYTES
    max_pixels: int = DEFAULT_MAX_PIXELS
    max_decoded_bytes: int = DEFAULT_MAX_DECODED_BYTES


def _positive(raw: str | None, default: int) -> int:
    if raw is None:
        return default
    try:
        n = int(raw.strip())
    except ValueError:
        return default
    return n if n > 0 else default


def current_limits(env: Mapping[str, str] | None = None) -> Limits:
    source = os.environ if env is None else env
    return Limits(
        max_bytes=_positive(source.get("MM_MAX_IMAGE_BYTES"), DEFAULT_MAX_BYTES),
        max_pixels=_positive(source.get("MM_MAX_IMAGE_PIXELS"), DEFAULT_MAX_PIXELS),
        max_decoded_bytes=_positive(source.get("MM_MAX_DECODED_BYTES"), DEFAULT_MAX_DECODED_BYTES),
    )


# Pillow's own guard is the second line of defence everywhere an image is opened in this process; state its bound
# rather than rely on the library default.
Image.MAX_IMAGE_PIXELS = current_limits().max_pixels


def check_image(original: bytes, *, limits: Limits | None = None) -> tuple[int, int] | None:
    """Refuse ``original`` if it breaks a limit; else its ``(width, height)``, or None when Pillow cannot read a
    header from it (the caller's decode then fails exactly as it always did). Decodes no pixel."""
    lim = limits or current_limits()
    if len(original) > lim.max_bytes:
        raise ImageLimitError(
            "too_large_bytes", f"{len(original)} bytes exceeds the {lim.max_bytes}-byte cap"
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(original)) as probe:
                width, height = probe.size
                mode = probe.mode
    except Image.DecompressionBombError as exc:
        raise ImageLimitError("too_many_pixels", f"Pillow refused the header: {exc}") from exc
    except Exception:  # not an image Pillow can read: not this module's business
        return None
    if width * height > lim.max_pixels:
        raise ImageLimitError(
            "too_many_pixels", f"{width}x{height} exceeds the {lim.max_pixels}-pixel cap"
        )
    decoded = (
        width
        * height
        * max(_WORKING_BYTES_PER_PIXEL, _MODE_BYTES.get(mode, _WORKING_BYTES_PER_PIXEL))
    )
    if decoded > lim.max_decoded_bytes:
        raise ImageLimitError(
            "decoded_too_large",
            f"{width}x{height} {mode} needs {decoded} bytes, cap {lim.max_decoded_bytes}",
        )
    return width, height


def gunzip_capped(data: bytes, max_bytes: int) -> bytes:
    """``gzip.decompress`` with an output ceiling: stops at ``max_bytes`` instead of expanding a bomb (a ~200 KB
    gzip can claim hundreds of MB). Concatenated members are read as one stream, like ``gzip.decompress``; a
    non-gzip input raises ``gzip.BadGzipFile`` and a truncated one ``EOFError``, as it does."""
    out = bytearray()
    rest = data
    while True:
        if not rest:
            raise EOFError("Compressed file ended before the end-of-stream marker was reached")
        stream = zlib.decompressobj(wbits=zlib.MAX_WBITS | 16)
        try:
            chunk = stream.decompress(rest, max_bytes - len(out) + 1)
        except zlib.error as exc:
            raise gzip.BadGzipFile(f"Not a gzipped file ({exc})") from exc
        out += chunk
        if len(out) > max_bytes:
            raise ImageLimitError("gzip_too_large", f"gzip output exceeds the {max_bytes}-byte cap")
        if not stream.eof:
            raise EOFError("Compressed file ended before the end-of-stream marker was reached")
        rest = stream.unused_data
        if not rest:
            return bytes(out)
