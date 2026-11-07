from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))

from make_fixtures import make_all  # noqa: E402

TEST_EMBEDDING = {"embedding": {"backend": "hash-test", "model": "hash-test", "dims": 256, "native_dims": 256}}


@pytest.fixture(scope="session")
def corpus(tmp_path_factory) -> Path:
    return make_all(tmp_path_factory.mktemp("corpus"))


@pytest.fixture(scope="session")
def built_ws(tmp_path_factory, corpus):
    from rag.builder.engine import Builder
    from rag.core.workspace import Workspace

    ws = Workspace.init(tmp_path_factory.mktemp("ws") / "ws", TEST_EMBEDDING)
    stats = Builder(ws).build([corpus])
    assert stats.failed == 0, stats.errors
    return ws


@pytest.fixture()
def cfg():
    from rag.config import Config

    return Config().with_overrides(TEST_EMBEDDING)


@pytest.fixture()
def ctx(cfg, tmp_path):
    from rag.core.ingest.context import ExtractContext

    return ExtractContext(config=cfg, sha256="0" * 64, display_path="x", assets_root=tmp_path / "assets")


@pytest.fixture()
def searcher(built_ws):
    from rag.core.db.store import RagStore
    from rag.core.models.embed import make_embedder
    from rag.core.retrieve.search import Searcher

    store = RagStore(built_ws.rag_db, readonly=True)
    yield Searcher(store, make_embedder(built_ws.config["embedding"]), built_ws.config["search"])
    store.close()
