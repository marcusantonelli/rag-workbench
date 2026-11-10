"""The C interface and the C# wrapper return exactly what the Python module returns."""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

shammy = pytest.importorskip("shammy")

ROOT = Path(__file__).resolve().parents[1] / "shammy"
LIB = ROOT / "target" / "release" / {"win32": "shammy.dll", "darwin": "libshammy.dylib"}.get(sys.platform, "libshammy.so")
pytestmark = pytest.mark.skipif(not LIB.exists(), reason="build the C library: cargo build --release -p shammy-ffi")

DIMS, ROWS = 256, 6000


@pytest.fixture(scope="module")
def fixture(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("native")
    rng = np.random.default_rng(5)
    centers = rng.standard_normal((40, DIMS)).astype(np.float32)
    data = centers[rng.integers(0, 40, ROWS)] + 0.6 * rng.standard_normal((ROWS, DIMS)).astype(np.float32)
    types = [("pdf", "md", "docx")[i % 3] for i in range(ROWS)]
    with shammy.Index.create(d / "kb.shammy", DIMS, model="modèle-test ✓") as ix:
        ix.add(range(1000, 1000 + ROWS), data, doc_ids=[i // 20 for i in range(ROWS)], file_types=types)
        ix.commit()
        ix.delete_docs([3, 4])
    queries = (data[rng.integers(0, ROWS, 8)] + 0.3 * rng.standard_normal((8, DIMS))).astype("<f4")
    queries.tofile(d / "queries.f32")
    ix = shammy.Index.open(d / "kb.shammy")
    cases = []
    for qi in range(8):
        for name, kw in (("plain", {}), ("pdf", {"file_types": ["pdf"]}), ("docs", {"doc_ids": [5, 6, 7, 250]}),
                         ("both", {"file_types": ["md", "docx"], "doc_ids": list(range(100, 140))}),
                         ("exact", {"exact": True}), ("threads", {"threads": 3}), ("short", {"query_len": 128})):
            kw = dict(kw)
            n = kw.pop("query_len", DIMS)
            hits = ix.search(queries[qi][:n], k=10, **kw)
            cases.append({"name": f"{name}-{qi}", "query": qi, "query_len": n, "k": 10, "exact": kw.get("exact", False),
                          "threads": kw.get("threads", 0), **{k: v for k, v in kw.items() if k in ("file_types", "doc_ids")},
                          "hits": [[h.id, h.doc_id, h.score] for h in hits]})
    info = ix.info()
    (d / "expected.json").write_text(json.dumps({
        "dims": DIMS, "live_rows": info["live_rows"], "model": info["model"], "file_types": info["file_types"],
        "some_id": 1000 + 777, "cases": cases}))
    ix.close()
    return d


class Hit(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint64), ("doc_id", ctypes.c_uint32), ("score", ctypes.c_float)]


class Params(ctypes.Structure):
    _fields_ = [("threads", ctypes.c_uint32), ("exact", ctypes.c_uint32), ("stage1", ctypes.c_uint64),
                ("stage2", ctypes.c_uint64), ("finish", ctypes.c_uint64)]


class Info(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint32) for n in ("dims", "sketch_dims", "lossless", "segments", "file_type_count", "reserved")] + \
               [(n, ctypes.c_uint64) for n in ("generation", "rows", "live_rows", "file_bytes", "instruction_hash")] + \
               [("model", ctypes.c_char * 128)]


def test_c_interface_matches_python(fixture):
    lib = ctypes.CDLL(str(LIB))
    lib.shammy_last_error.restype = ctypes.c_char_p
    lib.shammy_version.restype = ctypes.c_char_p
    lib.shammy_search.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t,
                                  ctypes.POINTER(ctypes.c_char_p), ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint32),
                                  ctypes.c_size_t, ctypes.POINTER(Params), ctypes.POINTER(Hit), ctypes.POINTER(ctypes.c_size_t)]
    lib.shammy_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(Info)]
    lib.shammy_close.argtypes = [ctypes.c_void_p]
    assert lib.shammy_version()

    expected = json.loads((fixture / "expected.json").read_text())
    queries = np.fromfile(fixture / "queries.f32", dtype="<f4").reshape(-1, DIMS)
    reader = ctypes.c_void_p()
    assert lib.shammy_open(str(fixture / "kb.shammy").encode(), ctypes.byref(reader)) == 0
    info = Info()
    assert lib.shammy_info(reader, ctypes.byref(info)) == 0
    assert (info.dims, info.live_rows, info.model.decode()) == (DIMS, expected["live_rows"], expected["model"])

    for case in expected["cases"]:
        q = np.ascontiguousarray(queries[case["query"]][:case["query_len"]])
        types = [t.encode() for t in case.get("file_types", [])]
        docs = case.get("doc_ids", [])
        hits = (Hit * case["k"])()
        count = ctypes.c_size_t()
        params = Params(threads=case["threads"], exact=int(case["exact"]))
        code = lib.shammy_search(reader, q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(q), case["k"],
                                 (ctypes.c_char_p * len(types))(*types) if types else None, len(types),
                                 (ctypes.c_uint32 * len(docs))(*docs) if docs else None, len(docs),
                                 ctypes.byref(params), hits, ctypes.byref(count))
        assert code == 0, lib.shammy_last_error()
        got = [[h.id, h.doc_id, h.score] for h in hits[:count.value]]
        assert got == [[i, d, np.float32(s)] for i, d, s in case["hits"]], case["name"]

    # errors come back as codes with a message, never a crash
    assert lib.shammy_search(reader, None, DIMS, 5, None, 0, None, 0, None, (Hit * 5)(), ctypes.byref(ctypes.c_size_t())) == -1
    assert b"null" in lib.shammy_last_error()
    lib.shammy_close(reader)
    assert lib.shammy_open(str(fixture / "nope.shammy").encode(), ctypes.byref(reader)) == -2 and not reader.value
    assert lib.shammy_last_error()


@pytest.mark.skipif(shutil.which("dotnet") is None, reason="needs the .NET SDK")
def test_csharp_wrapper_matches_python(fixture):
    env = {**os.environ, "DOTNET_CLI_TELEMETRY_OPTOUT": "1", "DOTNET_NOLOGO": "1",
           "LD_LIBRARY_PATH": f"{LIB.parent}{os.pathsep}{os.environ.get('LD_LIBRARY_PATH', '')}",
           "DYLD_LIBRARY_PATH": str(LIB.parent), "PATH": f"{LIB.parent}{os.pathsep}{os.environ['PATH']}"}
    project = ROOT / "dotnet" / "Shammy.Tests"
    empty = fixture / "no-packages"   # the projects have no package dependencies, so no feed is needed
    empty.mkdir(exist_ok=True)
    build = subprocess.run(["dotnet", "build", str(project), "-c", "Release", f"-p:RestoreSources={empty}"],
                           capture_output=True, text=True, env=env, timeout=600)
    assert build.returncode == 0, build.stdout + build.stderr
    dll = project / "bin" / "Release" / "net8.0" / "Shammy.Tests.dll"
    run = subprocess.run(["dotnet", str(dll), str(fixture)], capture_output=True, text=True, env=env, timeout=600)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "0 failed" in run.stdout
