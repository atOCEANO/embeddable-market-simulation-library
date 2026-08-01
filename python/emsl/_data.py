"""Input helpers: turn a numpy array, a pandas DataFrame, or a parquet path into
the numeric arrays the engine takes. pandas and pyarrow are optional; they are
imported only when a DataFrame or a parquet path is passed.
"""

from __future__ import annotations

import numpy as np

_OHLCV = ["open", "high", "low", "close", "volume"]


def to_ohlcv(data):
    """Return a `(T, 5)` float64 OHLCV array from a numpy array, a pandas DataFrame
    with the columns open/high/low/close/volume, or a path to a parquet file.
    """
    if isinstance(data, np.ndarray):
        arr = np.ascontiguousarray(data, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 5:
            raise ValueError("array data must be (T, 5) OHLCV")
        return arr
    frame = _as_frame(data)
    missing = [c for c in _OHLCV if c not in frame.columns]
    if missing:
        raise ValueError(f"data is missing columns {missing}; need {_OHLCV}")
    _validate_index(frame)
    return np.ascontiguousarray(frame[_OHLCV].to_numpy(dtype=np.float64))


def _validate_index(frame):
    # the spec's input contract: a real (datetime) index must be sorted ascending
    # and unique. A default RangeIndex carries no timestamps, so it is exempt.
    import pandas as pd

    idx = frame.index
    if isinstance(idx, pd.RangeIndex):
        return
    if not idx.is_monotonic_increasing:
        raise ValueError("data index must be sorted ascending")
    if not idx.is_unique:
        raise ValueError("data index must be unique")


def to_float2d(data):
    """Return a `(T, F)` float64 array from a numpy array or a pandas DataFrame."""
    if isinstance(data, np.ndarray):
        return np.ascontiguousarray(data, dtype=np.float64)
    frame = _as_frame(data)
    return np.ascontiguousarray(frame.to_numpy(dtype=np.float64))


def index_of(data):
    """Return the row index of a DataFrame or parquet input as a numpy array, for
    stamping trades with times. Returns None for a numpy array, or when the frame
    carries only a default integer range index (no real timestamps to attach).
    """
    if isinstance(data, np.ndarray):
        return None
    import pandas as pd

    frame = _as_frame(data)
    if isinstance(frame.index, pd.RangeIndex):
        return None
    return frame.index.to_numpy()


def prepare(data):
    """Return `(ohlcv_array, index_or_None)` from one read of `data`.

    `to_ohlcv` and `index_of` each route a parquet path through `_as_frame`, so
    calling both decoded the file twice, doubling the load time and peak memory of
    the most convenient entry point and leaving room for the two reads to disagree
    if the file changed between them.
    """
    if isinstance(data, np.ndarray):
        return to_ohlcv(data), None
    frame = _as_frame(data)
    return to_ohlcv(frame), index_of(frame)


def _as_frame(data):
    # the type check comes before the import: on a pandas-free install an
    # unsupported argument used to report "pandas is required", sending the caller
    # after a dependency that would not have helped
    if isinstance(data, str) or hasattr(data, "__fspath__"):
        return _pandas().read_parquet(data)
    if type(data).__module__.split(".")[0] == "pandas":
        frame = _pandas().DataFrame
        if isinstance(data, frame):
            return data
    raise TypeError(
        f"unsupported data type {type(data).__name__}; "
        "pass a numpy array, a pandas DataFrame, or a parquet path"
    )


def _pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for DataFrame or parquet input; pip install pandas pyarrow"
        ) from exc
    return pd
