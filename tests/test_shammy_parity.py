"""The Rust engine against the NumPy reference (shammy/python/shammy_ref.py).

Needs the built tool: (cd shammy && cargo build --release). Skipped when it isn't there.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / "shammy"
BIN = ROOT / "target" / "release" / ("shammy.exe" if sys.platform == "win32" else "shammy")
sys.path.insert(0, str(ROOT / "python"))

pytestmark = pytest.mark.skipif(not BIN.exists(), reason="shammy is not built (cd shammy && cargo build --release)")


def run(*args: object) -> str:
    r = subprocess.run([str(BIN), *map(str, args)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def make(n: int, dims: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    decay = 1.0 / np.sqrt(1 + np.arange(dims) / (dims / 8))
    groups = rng.standard_normal((n // 8 + 1, dims)) * decay
    x = groups[np.arange(n) // 8] + 0.3 * rng.standard_normal((n, dims)) * decay
    return (x * rng.uniform(0.5, 3.0, size=(n, 1))).astype(np.float32)      # not unit length on purpose


@pytest.mark.parametrize("dims,sketch", [(64, "quarter"), (256, "half"), (1024, "quarter")])
def test_engine_writes_exactly_the_reference_codes(tmp_path, dims, sketch):
    import shammy_ref as ref

    x = make(300, dims, dims)
    x[7] = 0                                                # an all-zero vector must not break anything
    (tmp_path / "v.f32").write_bytes(x.tobytes())
    ids = (np.arange(300, dtype="<u8") * 3 + 11)
    (tmp_path / "ids.u64").write_bytes(ids.tobytes())
    out = tmp_path / "t.shammy"
    run("build", out, "--dims", dims, "--vectors", tmp_path / "v.f32", "--ids", tmp_path / "ids.u64",
        "--sketch", sketch, "--model", "ref-test")

    f = ref.read(out)
    sd = dims // 4 if sketch == "quarter" else dims // 2
    assert (f["dims"], f["sketch_dims"], f["rows"], f["generation"], f["model"]) == (dims, sd, 300, 1, "ref-test")
    seg = f["segments"][0]
    want = ref.encode(x, sd)
    assert np.array_equal(seg["ids"], ids)
    assert np.array_equal(seg["sketch"], want["sketch"]), "sketch bits differ"
    assert np.array_equal(seg["bits"], want["bits"]), "sign bits differ"
    assert np.array_equal(seg["q8"], want["q8"]), "8-bit codes differ"
    assert np.array_equal(seg["f16"], want["f16"]), "float16 vectors differ"
    np.testing.assert_allclose(seg["corr"], want["corr"], rtol=1e-6)
    np.testing.assert_allclose(seg["q8corr"], want["q8corr"], rtol=1e-6)


def test_engine_search_matches_the_reference_search(tmp_path):
    import shammy_ref as ref

    dims = 256
    x = make(800, dims, 5)
    (tmp_path / "v.f32").write_bytes(x.tobytes())
    docs = (np.arange(800, dtype="<u4") // 20)
    (tmp_path / "docs.u32").write_bytes(docs.tobytes())
    out = tmp_path / "t.shammy"
    run("build", out, "--dims", dims, "--vectors", tmp_path / "v.f32", "--docs", tmp_path / "docs.u32")
    run("delete", out, 3, 17)
    q = make(5, dims, 6) * 0.2 + x[[10, 200, 399, 640, 777]]
    (tmp_path / "q.f32").write_bytes(q.astype(np.float32).tobytes())

    f = ref.read(out)
    assert f["live_rows"] == 760 and f["segments"][0]["tomb"].sum() == 40
    for row in range(5):
        got = [(int(a), float(c)) for a, _, c in (line.split("\t") for line in
               run("search", out, "--query", tmp_path / "q.f32", "--row", row, "-k", 10,
                   "--stage1", 120, "--stage2", 60, "--finish", 30).splitlines())]
        want = ref.search(f, q[row], k=10, stage1=120, stage2=60, finish=30)
        assert [g[0] for g in got] == [w[0] for w in want], f"query {row}"
        np.testing.assert_allclose([g[1] for g in got], [w[1] for w in want], atol=2e-6)
        assert not {i for i, _ in got} & set(range(60, 80)) and not {i for i, _ in got} & set(range(340, 360))

    assert "OK" in run("verify", out)
    assert "760 live" in run("info", out)
