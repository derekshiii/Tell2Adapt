"""NPZ helpers compatible with numpy 1.x and numpy 2.x pickles.

The existing Tell2Adapt datasets were written with different numpy versions.
Older numpy cannot import ``numpy._core`` when unpickling a ``text_prompts``
object array produced by numpy 2.x. These helpers load numeric arrays without
pickle and only unpickle ``text_prompts``, remapping ``numpy._core`` to
``numpy.core`` when necessary.

Only use these helpers on trusted local experiment artifacts: unpickling can
execute arbitrary constructors.
"""

from __future__ import annotations

import io
import json
import pickle
import zipfile
from pathlib import Path
from typing import Any, BinaryIO, Dict, Union

import numpy as np

try:  # numpy >= 2
    import numpy._core  # noqa: F401

    _HAS_NUMPY_CORE = True
except Exception:  # numpy 1.x
    _HAS_NUMPY_CORE = False


class NumpyCompatUnpickler(pickle.Unpickler):
    """Unpickle numpy 2.x object arrays under numpy 1.x."""

    def find_class(self, module: str, name: str) -> Any:
        if not _HAS_NUMPY_CORE and module.startswith("numpy._core"):
            module = "numpy.core" + module[len("numpy._core") :]
        return super().find_class(module, name)


def decode_text_prompts(raw: Any) -> Dict[str, Any]:
    """Decode a pickled dict, a 0-d array, or a JSON string."""

    try:
        value = raw.item()
    except Exception:
        value = raw
    if isinstance(value, dict):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        decoded = json.loads(value)
        if not isinstance(decoded, dict):
            raise ValueError("text_prompts JSON must be an object")
        return decoded
    raise ValueError(f"unsupported text_prompts type: {type(value).__name__}")


def _read_header(fp: BinaryIO, version: tuple):
    reader = getattr(np.lib.format, "_read_array_header", None)
    if reader is not None:
        return reader(fp, version)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(fp)
    if version == (2, 0):
        return np.lib.format.read_array_header_2_0(fp)
    raise ValueError(f"unsupported .npy version: {version}")


def _load_npy_bytes(raw: bytes):
    stream = io.BytesIO(raw)
    version = np.lib.format.read_magic(stream)
    shape, fortran_order, dtype = _read_header(stream, version)
    if getattr(dtype, "hasobject", False):
        stream.seek(0)
        np.lib.format.read_magic(stream)
        _read_header(stream, version)
        return NumpyCompatUnpickler(stream).load()
    return np.load(io.BytesIO(raw), allow_pickle=False)


def load_text_prompts(path: Union[str, Path]) -> Dict[str, Any]:
    """Load ``text_prompts`` from an NPZ without unpickling other arrays."""

    with zipfile.ZipFile(path) as archive:
        if "text_prompts.npy" not in archive.namelist():
            raise KeyError(f"{path}: no text_prompts array")
        raw = archive.read("text_prompts.npy")
    return decode_text_prompts(_load_npy_bytes(raw))


def load_npz(
    path: Union[str, Path], *, include_prompts: bool = True
) -> Dict[str, Any]:
    """Load all numeric arrays, plus optionally ``text_prompts``."""

    data: Dict[str, Any] = {}
    with np.load(path, allow_pickle=False) as npz:
        for key in npz.files:
            if key == "text_prompts":
                continue
            data[key] = npz[key]
    if include_prompts:
        data["text_prompts"] = load_text_prompts(path)
    return data
