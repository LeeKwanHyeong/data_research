"""
file_io_util.py (refactored)

Cross‑platform, production‑ready file I/O utilities with:
- Safe atomic writes (temp + replace)
- Optional lock files for multi‑process safety
- JSON / YAML / NDJSON / TXT readers & writers
- Polars CSV/Parquet readers & writers tuned for large datasets
- Pickle/Joblib helpers
- Robust encoding fallback for text reads
- Path utilities (mkdirs, backups, hashing)

Python: 3.12+
Optional deps: PyYAML, joblib, polars

Design notes
------------
- All public APIs accept str | os.PathLike and return pathlib.Path or typed objects.
- No hard dependency on polars/PyYAML/joblib: functions fail with clear ImportError messages.
- Atomic write: write to a temp file in the same directory, then os.replace() (atomic on POSIX & NT ≥ Vista).
- Locking: simple lockfile implementation using create-exclusive + timeout; resilient on Windows.
"""
from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import os
import pickle
import shutil
import sys
import tempfile
import time
from dataclasses import is_dataclass, asdict
from hashlib import md5, sha256
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

__all__ = [
    # Core path utils
    "to_path",
    "ensure_dir",
    "timestamp_backup",
    "file_hash",
    # Locks & atomic
    "FileLock",
    "atomic_write",
    # Text & structured formats
    "read_text",
    "write_text",
    "read_json",
    "write_json",
    "read_ndjson",
    "write_ndjson",
    "read_yaml",
    "write_yaml",
    # Binary formats
    "read_pickle",
    "write_pickle",
    "read_joblib",
    "write_joblib",
    # Polars
    "pl_read_csv",
    "pl_write_csv",
    "pl_read_parquet",
    "pl_write_parquet",
]

# ------------------------------
# Path helpers
# ------------------------------

def to_path(path: os.PathLike[str] | str) -> Path:
    """Normalize to Path, expanding ~ and env vars."""
    if isinstance(path, Path):
        p = path
    else:
        p = Path(os.path.expandvars(os.path.expanduser(str(path))))
    return p


def ensure_dir(path: os.PathLike[str] | str) -> Path:
    """Ensure directory exists; return Path."""
    p = to_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------
# Backup & hashing
# ------------------------------

def timestamp_backup(src: os.PathLike[str] | str, *, suffix_fmt: str = "%Y%m%d-%H%M%S") -> Path:
    """Create a timestamped backup alongside `src`.

    Returns the backup path. If source does not exist, returns target path without copying.
    """
    src_p = to_path(src)
    ts = time.strftime(suffix_fmt, time.localtime())
    dst = src_p.with_name(f"{src_p.name}.{ts}.bak")
    if src_p.exists():
        shutil.copy2(src_p, dst)
    return dst


def file_hash(path: os.PathLike[str] | str, *, algo: str = "sha256", chunk_size: int = 4 * 1024 * 1024) -> str:
    """Compute file hash (sha256|md5) streaming in chunks."""
    p = to_path(path)
    h = sha256() if algo.lower() == "sha256" else md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------
# Simple cross‑platform lockfile
# ------------------------------
class FileLock:
    """Minimal advisory lock using an exclusive lockfile.

    Usage:
        with FileLock(path).acquire(timeout=10):
            ...
    """

    def __init__(self, target: os.PathLike[str] | str, *, lock_suffix: str = ".lock") -> None:
        self.target = to_path(target)
        self.lock_path = self.target.with_suffix(self.target.suffix + lock_suffix)
        self._fd: int | None = None

    @contextlib.contextmanager
    def acquire(self, *, timeout: float | None = None, poll_interval: float = 0.1) -> Iterator[None]:
        start = time.monotonic()
        while True:
            try:
                # O_CREAT|O_EXCL fails if exists; 0o644 perms
                self._fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o644)
                break
            except FileExistsError:
                if timeout is not None and (time.monotonic() - start) >= timeout:
                    raise TimeoutError(f"Lock timeout: {self.lock_path}")
                time.sleep(poll_interval)
        try:
            yield
        finally:
            try:
                if self._fd is not None:
                    os.close(self._fd)
                    self._fd = None
                if self.lock_path.exists():
                    self.lock_path.unlink(missing_ok=True)
            except Exception:
                # Never raise during cleanup
                pass


# ------------------------------
# Atomic write helper
# ------------------------------
@contextlib.contextmanager
def atomic_write(dest: os.PathLike[str] | str, *, mode: str = "wb", newline: str | None = None) -> Iterator[io.BufferedWriter | io.TextIOWrapper]:
    """Write to a temp file in the same dir, then atomically replace dest.

    - `mode` must be a write mode (e.g., "wb" or "w").
    - For text mode, `newline` is forwarded to open().
    """
    dst = to_path(dest)
    ensure_dir(dst.parent)
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=f".{dst.name}.", dir=str(dst.parent))
    os.close(tmp_fd)  # will reopen with correct mode
    tmp_path = Path(tmp_name)
    f: io.IOBase | None = None
    try:
        if "b" in mode:
            f = tmp_path.open(mode)
        else:
            f = tmp_path.open(mode, encoding="utf-8", newline=newline)
        yield f  # type: ignore[misc]
        f.flush()
        os.fsync(f.fileno())
        f.close()
        os.replace(tmp_path, dst)
    except Exception:
        with contextlib.suppress(Exception):
            if f and not f.closed:
                f.close()
            tmp_path.unlink(missing_ok=True)
        raise


# ------------------------------
# Text & structured formats
# ------------------------------

def _fallback_read_text(p: Path, *, encodings: Sequence[str] = ("utf-8", "utf-8-sig", "cp949", "euc-kr")) -> str:
    last_exc: Exception | None = None
    for enc in encodings:
        try:
            return p.read_text(encoding=enc)
        except Exception as e:  # noqa: PERF203 (small list of encodings)
            last_exc = e
            continue
    assert last_exc is not None
    raise last_exc


def read_text(path: os.PathLike[str] | str) -> str:
    """Read text with robust encoding fallback."""
    p = to_path(path)
    return _fallback_read_text(p)


def write_text(path: os.PathLike[str] | str, text: str, *, newline: str | None = "\n", lock: bool = False) -> Path:
    """Write text atomically (UTF‑8)."""
    p = to_path(path)
    lock_ctx = FileLock(p).acquire(timeout=10) if lock else contextlib.nullcontext()
    with lock_ctx:
        with atomic_write(p, mode="w", newline=newline) as f:
            assert isinstance(f, io.TextIOBase)
            f.write(text)
    return p


# JSON

def _json_default(o: Any) -> Any:
    if is_dataclass(o):
        return asdict(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def read_json(path: os.PathLike[str] | str) -> Any:
    p = to_path(path)
    txt = _fallback_read_text(p)
    return json.loads(txt)


def write_json(
    path: os.PathLike[str] | str,
    obj: Any,
    *,
    indent: int | None = 2,
    sort_keys: bool = False,
    ensure_ascii: bool = False,
    lock: bool = False,
) -> Path:
    p = to_path(path)
    data = json.dumps(obj, default=_json_default, indent=indent, sort_keys=sort_keys, ensure_ascii=ensure_ascii)
    return write_text(p, data, lock=lock)


# NDJSON

def read_ndjson(path: os.PathLike[str] | str) -> list[dict[str, Any]]:
    p = to_path(path)
    lines = _fallback_read_text(p).splitlines()
    out: list[dict[str, Any]] = []
    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON on line {i}: {e}") from e
    return out


def write_ndjson(path: os.PathLike[str] | str, records: Iterable[Mapping[str, Any]], *, lock: bool = False) -> Path:
    p = to_path(path)
    lock_ctx = FileLock(p).acquire(timeout=10) if lock else contextlib.nullcontext()
    with lock_ctx:
        with atomic_write(p, mode="w", newline="\n") as f:
            assert isinstance(f, io.TextIOBase)
            for rec in records:
                f.write(json.dumps(rec, default=_json_default, ensure_ascii=False))
                f.write("\n")
    return p


# YAML (optional)

def _require_yaml():
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("PyYAML is required for YAML operations: pip install pyyaml") from e
    return yaml


def read_yaml(path: os.PathLike[str] | str) -> Any:
    yaml = _require_yaml()
    p = to_path(path)
    return yaml.safe_load(_fallback_read_text(p))


def write_yaml(path: os.PathLike[str] | str, obj: Any, *, lock: bool = False) -> Path:
    yaml = _require_yaml()
    p = to_path(path)
    data = yaml.safe_dump(obj, sort_keys=False, allow_unicode=True)
    return write_text(p, data, lock=lock)


# ------------------------------
# Binary formats: pickle / joblib
# ------------------------------

def read_pickle(path: os.PathLike[str] | str) -> Any:
    p = to_path(path)
    with p.open("rb") as f:
        return pickle.load(f)


def write_pickle(path: os.PathLike[str] | str, obj: Any, *, protocol: int = pickle.HIGHEST_PROTOCOL, lock: bool = False) -> Path:
    p = to_path(path)
    lock_ctx = FileLock(p).acquire(timeout=10) if lock else contextlib.nullcontext()
    with lock_ctx:
        with atomic_write(p, mode="wb") as f:
            assert isinstance(f, io.BufferedIOBase)
            pickle.dump(obj, f, protocol=protocol)
    return p


def _require_joblib():
    try:
        import joblib  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("joblib is required for joblib operations: pip install joblib") from e
    return joblib


def read_joblib(path: os.PathLike[str] | str) -> Any:
    joblib = _require_joblib()
    return joblib.load(to_path(path))


def write_joblib(path: os.PathLike[str] | str, obj: Any, *, compress: int | tuple[int, str] | None = 3, lock: bool = False) -> Path:
    joblib = _require_joblib()
    p = to_path(path)
    tmp_dir = to_path(tempfile.mkdtemp(prefix="joblib_tmp_"))
    try:
        # joblib.dump is atomic-ish when dumping to a temp folder first
        tmp_path = tmp_dir / "obj.joblib"
        joblib.dump(obj, tmp_path, compress=compress)
        ensure_dir(p.parent)
        shutil.move(str(tmp_path), str(p))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return p


# ------------------------------
# Polars helpers (optional import)
# ------------------------------

def _require_polars():
    try:
        import polars as pl  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("polars is required for this operation: pip install polars") from e
    return pl


# CSV

def pl_read_csv(
    path: os.PathLike[str] | str,
    *,
    has_header: bool = True,
    separator: str = ",",
    infer_schema_length: int | None = 50_000,
    try_utf_fallbacks: bool = True,
    rechunk: bool = True,
    low_memory: bool = True,
    storage_options: dict[str, Any] | None = None,
    dtypes: Mapping[str, Any] | None = None,
) -> "pl.DataFrame":
    """Read CSV using Polars with sane defaults for large files.

    If `try_utf_fallbacks` is True, retries with cp949/euc-kr encodings if UTF‑8 fails (Windows/KR friendly).
    """
    pl = _require_polars()
    p = to_path(path)

    encodings: list[str] = ["utf8"]
    if try_utf_fallbacks:
        encodings += ["utf8-lossy", "utf-8", "utf-8-sig", "cp949", "euc-kr"]

    last_exc: Exception | None = None
    for enc in encodings:
        try:
            return pl.read_csv(
                source=str(p),
                has_header=has_header,
                separator=separator,
                infer_schema_length=infer_schema_length,
                try_parse_dates=True,
                encoding=enc,
                rechunk=rechunk,
                low_memory=low_memory,
                storage_options=storage_options,
                dtypes=dtypes,
            )
        except Exception as e:
            last_exc = e
            continue
    assert last_exc is not None
    raise last_exc


def pl_write_csv(
    df: "pl.DataFrame",
    path: os.PathLike[str] | str,
    *,
    include_header: bool = True,
    separator: str = ",",
    quote: str = '"',
    datetime_format: str | None = None,
    float_precision: int | None = None,
    lock: bool = False,
) -> Path:
    pl = _require_polars()
    p = to_path(path)
    lock_ctx = FileLock(p).acquire(timeout=30) if lock else contextlib.nullcontext()
    with lock_ctx:
        # Use atomic write to avoid partial files
        with atomic_write(p, mode="wb") as f:
            df.write_csv(
                file=f,
                include_header=include_header,
                separator=separator,
                quote=quote,
                datetime_format=datetime_format,
                float_precision=float_precision,
            )
    return p


# Parquet

def pl_read_parquet(
    path: os.PathLike[str] | str,
    *,
    use_statistics: bool = True,
    low_memory: bool = True,
    row_count_name: str | None = None,
    storage_options: dict[str, Any] | None = None,
    columns: Sequence[str] | None = None,
) -> "pl.DataFrame":
    pl = _require_polars()
    p = to_path(path)
    return pl.read_parquet(
        source=str(p),
        use_statistics=use_statistics,
        low_memory=low_memory,
        row_count_name=row_count_name,
        storage_options=storage_options,
        columns=columns,
    )


def pl_write_parquet(
    df: "pl.DataFrame",
    path: os.PathLike[str] | str,
    *,
    compression: str = "zstd",
    compression_level: int | None = None,
    statistics: bool = True,
    row_group_size: int | None = 512 * 1024,  # ~512k rows/group by default
    use_pyarrow: bool | None = None,
    lock: bool = False,
) -> Path:
    pl = _require_polars()
    p = to_path(path)
    lock_ctx = FileLock(p).acquire(timeout=60) if lock else contextlib.nullcontext()
    with lock_ctx:
        with atomic_write(p, mode="wb") as f:
            df.write_parquet(
                file=f,
                compression=compression,
                compression_level=compression_level,
                statistics=statistics,
                row_group_size=row_group_size,
                use_pyarrow=use_pyarrow,
            )
    return p


# ------------------------------
# Small convenience: dataclass JSON
# ------------------------------

def dataclass_to_json_file(path: os.PathLike[str] | str, dc_obj: Any, **kwargs: Any) -> Path:
    if not is_dataclass(dc_obj):
        raise TypeError("dc_obj must be a dataclass instance")
    return write_json(path, asdict(dc_obj), **kwargs)


def json_file_to_dataclass(path: os.PathLike[str] | str, dc_type: type, **kwargs: Any) -> Any:
    data = read_json(path)
    if not isinstance(data, Mapping):
        raise TypeError("JSON root must be an object for dataclass reconstruction")
    return dc_type(**data)  # type: ignore[misc]


# ------------------------------
# CLI (optional quick smoke test)
# ------------------------------
if __name__ == "__main__":
    p = Path("./_demo.txt")
    write_text(p, "hello world", lock=True)
    print("read:", read_text(p))
    print("hash:", file_hash(p))
    with FileLock(p).acquire(timeout=1):
        print("locked ok")
    timestamp_backup(p)
    print("backup created.")
