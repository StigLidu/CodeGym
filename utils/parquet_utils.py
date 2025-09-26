# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

#!/usr/bin/env python3
from __future__ import annotations
"""
Usage:
    python jsonl_to_parquet.py path/to/input.jsonl
"""

import sys
import pyarrow as pa
import pyarrow.json as paj
import pyarrow.parquet as pq
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Any, Optional
from typing import Any, Mapping, Sequence, Optional, List, Dict
import numpy as np

def jsonl_to_parquet(jsonl_path: str, block_size: int = 64 << 20):  # 64 MB
    jsonl_path = Path(jsonl_path)
    parquet_path = jsonl_path.with_suffix(".parquet")

    # 1) Use one block of data to infer schema (only one I/O)
    schema = paj.read_json(
        jsonl_path,
        read_options=paj.ReadOptions(block_size=block_size)
    ).schema

    # 2) Construct a streaming JSON reader
    reader = paj.open_json(
        jsonl_path,
        read_options=paj.ReadOptions(block_size=block_size),
        parse_options=paj.ParseOptions(explicit_schema=schema)  # Fixed schema, avoid type drift
    )

    # 3) Stream write a single Parquet file
    with pq.ParquetWriter(
            parquet_path,
            schema=schema,
            compression="zstd",
            version="2.6"
    ) as writer:
        for batch in reader:             # Here batch is RecordBatch
            writer.write_batch(batch)    # use write_batch instead of write_table

    print(f"✅ Done: {parquet_path.resolve()}")

def json_str_list_to_parquet(json_str_list, parquet_path: str, block_size: int = 64 << 20):
    # Join list[str] into a memory buffer in JSONL format
    buf = ("\n".join(json_str_list) + "\n").encode("utf-8")

    # 1) Use one block of data to infer schema (only one I/O)
    schema = paj.read_json(
        pa.BufferReader(buf),
        read_options=paj.ReadOptions(block_size=block_size)
    ).schema

    # 2) Construct a streaming JSON reader
    reader = paj.open_json(
        pa.BufferReader(buf),
        read_options=paj.ReadOptions(block_size=block_size),
        parse_options=paj.ParseOptions(explicit_schema=schema)
    )

    # 3) Stream write a single Parquet file
    parquet_path = Path(parquet_path)
    with pq.ParquetWriter(
            parquet_path,
            schema=schema,
            compression="zstd",
            version="2.6"
    ) as writer:
        for batch in reader:  # batch is RecordBatch
            writer.write_batch(batch)

    print(f"✅ Done: {parquet_path.resolve()}")


# ---------- Recursive type inference ----------
def _as_list_like(v):
    # Treat np.ndarray as list, avoid being treated as "scalar object" later
    if np is not None and isinstance(v, np.ndarray):
        return v.tolist()
    return v

def _infer_type(v: Any) -> pa.DataType:
    v = _as_list_like(v)
    if v is None:
        return pa.null()
    if isinstance(v, bool):
        return pa.bool_()
    if isinstance(v, int):
        return pa.int64()
    if isinstance(v, float):
        return pa.float64()
    if isinstance(v, str):
        return pa.string()
    if isinstance(v, dict):
        fields = [pa.field(k, _infer_type(vk)) for k, vk in v.items()]
        return pa.struct(fields)
    if isinstance(v, (list, tuple)):
        elems = [_as_list_like(e) for e in v if e is not None]
        elem_type = pa.string() if not elems else _infer_type(elems[0])
        for e in elems[1:]:
            elem_type = _merge_types(elem_type, _infer_type(e))
        return pa.list_(elem_type)
    # Other rare types, downgrade to string
    return pa.string()

# ---------- Merge two types (for merging multiple samples) ----------
def _merge_types(a: pa.DataType, b: pa.DataType) -> pa.DataType:
    if a.equals(b) or pa.types.is_null(b):
        return a
    if pa.types.is_null(a):
        return b

    # Scalar numbers: int + float -> float
    if pa.types.is_integer(a) and pa.types.is_floating(b):
        return pa.float64()
    if pa.types.is_floating(a) and pa.types.is_integer(b):
        return pa.float64()

    # String can handle most incompatible cases
    if pa.types.is_string(a) or pa.types.is_string(b):
        return pa.string()

    # struct: merge by field name
    if pa.types.is_struct(a) and pa.types.is_struct(b):
        a_fields = {f.name: f.type for f in a}
        b_fields = {f.name: f.type for f in b}
        all_keys = set(a_fields) | set(b_fields)
        merged_fields = []
        for k in all_keys:
            ta = a_fields.get(k, pa.null())
            tb = b_fields.get(k, pa.null())
            merged_fields.append(pa.field(k, _merge_types(ta, tb)))
        return pa.struct(merged_fields)

    # list: merge element types
    if pa.types.is_list(a) and pa.types.is_list(b):
        return pa.list_(_merge_types(a.value_type, b.value_type))

    # Other incompatible: downgrade to string
    return pa.string()

# ---------- Infer schema from multiple rows ----------
def _infer_schema_from_rows(rows: Sequence[Mapping[str, Any]]) -> pa.Schema:
    keys = set()
    for r in rows:
        keys.update(r.keys())
    fields: List[pa.Field] = []
    for k in sorted(keys):
        t: pa.DataType = pa.null()
        for r in rows:
            if k in r:
                t = _merge_types(t, _infer_type(r[k]))
        fields.append(pa.field(k, t))
    return pa.schema(fields)

# ---------- Recursively normalize a Python value to match the given Arrow type ----------
def _normalize_to_type(v: Any, t: pa.DataType) -> Any:
    v = _as_list_like(v)
    if v is None or pa.types.is_null(t):
        return None

    if pa.types.is_struct(t):
        if not isinstance(v, dict):
            return None
        out: Dict[str, Any] = {}
        for f in t:  # type: ignore
            out[f.name] = _normalize_to_type(v.get(f.name), f.type)
        return out

    if pa.types.is_list(t):
        if not isinstance(v, (list, tuple)):
            return None
        elem_t = t.value_type  # type: ignore
        return [_normalize_to_type(_as_list_like(e), elem_t) for e in v]

    if pa.types.is_boolean(t):
        return bool(v) if isinstance(v, (bool, int)) else None
    if pa.types.is_integer(t):
        try:
            return int(v)
        except Exception:
            return None
    if pa.types.is_floating(t):
        try:
            return float(v)
        except Exception:
            return None
    if pa.types.is_string(t):
        return str(v)

    return v

def _to_table(pylist, schema=None) -> pa.Table:
    if not pylist:
        return pa.table({})

    if schema is None:
        schema = _infer_schema_from_rows(pylist)

    cols = {}
    for field in schema:
        col_vals = [_normalize_to_type(row.get(field.name), field.type) for row in pylist]
        arr = pa.array(col_vals, type=field.type, from_pandas=False)
        if not arr.type.equals(field.type):
            arr = pa.compute.cast(arr, target_type=field.type, safe=False)
        assert arr.type.equals(field.type), f"{field.name}: {arr.type} != {field.type}"
        cols[field.name] = arr

    # Use schema to constrain once (optional)
    tbl = pa.table(cols, schema=schema)
    return tbl

# ---------- Stream write ----------
def list_of_dicts_to_parquet(
    data: Iterable[Mapping[str, Any]],
    parquet_path: str | Path,
    *,
    chunk_size: int = 100_000,
    schema: Optional[pa.Schema] = None,
    compression: str = "zstd",
    version: str = "2.6",
) -> Path:
    #TODO: have bugs when dicts contain lists
    parquet_path = Path(parquet_path)

    # Pre-read first block (only once)
    buffer: List[Mapping[str, Any]] = []
    it = iter(data)
    try:
        for _ in range(chunk_size):
            buffer.append(next(it))
    except StopIteration:
        pass

    if not buffer:
        empty = pa.table({}) if schema is None else pa.Table.from_arrays(
            [pa.array([], type=f.type) for f in schema],
            names=[f.name for f in schema]
        )
        pq.write_table(empty, parquet_path, compression=compression, version=version)
        return parquet_path.resolve()

    # ☆☆☆ Use our own schema to infer (or use the user-provided schema)
    if schema is None:
        schema = _infer_schema_from_rows(buffer)
        print(schema)

    with pq.ParquetWriter(parquet_path, schema=schema, compression=compression, version=version) as writer:
        # Write first block
        first_table = _to_table(buffer, schema=schema)
        for batch in first_table.to_batches():
            writer.write_batch(batch)

        # Stream subsequent blocks
        buffer.clear()
        for row in it:
            buffer.append(row)
            if len(buffer) >= chunk_size:
                tbl = _to_table(buffer, schema=schema)
                for batch in tbl.to_batches():
                    writer.write_batch(batch)
                buffer.clear()

        if buffer:
            tbl = _to_table(buffer, schema=schema)
            for batch in tbl.to_batches():
                writer.write_batch(batch)

    print(f"✅ Done: {parquet_path.resolve()}")
    return parquet_path.resolve()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: jsonl_to_parquet.py <input.jsonl>")
    jsonl_to_parquet(sys.argv[1])