from __future__ import annotations

from rag.core.chunk.chunker import ChunkSettings, Chunker
from rag.core.chunk.identifiers import identifier_terms, query_identifiers, split_identifier
from rag.core.chunk.tokens import TokenCounter
from rag.core.models.embed import HashEmbedder
from rag.core.types import Block, ExtractedDoc

count = TokenCounter()


def _chunker(embedder=None, **kw):
    return Chunker(ChunkSettings(**kw), count, embedder)


def _sentence(i: int, topic: str) -> str:
    return f"Sentence {i} explains the {topic} procedure in detail for the operators on shift."


def test_long_section_packs_under_limit_with_overlap():
    paras = [" ".join(_sentence(i * 5 + j, "bonding") for j in range(5)) for i in range(12)]
    doc = ExtractedDoc(file_type="txt", title=None,
                       blocks=[Block(kind="heading", text="Bonding", level=1)] +
                              [Block(kind="paragraph", text=p, page=1) for p in paras])
    chunks = _chunker().chunk(doc, "a.txt")
    assert len(chunks) > 2
    assert all(c.n_tokens <= 500 for c in chunks)
    assert all(c.heading_path == "Bonding" for c in chunks)
    # consecutive chunks share overlapping text
    assert any(c1.text.split(". ")[-1][:30] in c2.text for c1, c2 in zip(chunks, chunks[1:]))
    assert chunks[0].embed_text.startswith("a.txt › Bonding\n")


def test_topic_shift_split_uses_embedder():
    sents = [_sentence(i, "bonding wire capillary") for i in range(25)] + \
            [f"Lid seal furnace zone {i} needs nitrogen purge and helium leak testing." for i in range(25)]
    doc = ExtractedDoc(file_type="txt", title=None, blocks=[Block(kind="paragraph", text=" ".join(sents))])
    emb = HashEmbedder({"model": "h", "dims": 256})
    chunks = _chunker(emb).chunk(doc, "a.txt")
    seal_chunks = [c for c in chunks if "furnace" in c.text]
    assert seal_chunks and all("capillary" not in c.text for c in seal_chunks[1:])


def test_heading_trail_and_levels():
    doc = ExtractedDoc(file_type="md", title=None, blocks=[
        Block(kind="heading", text="A", level=1), Block(kind="heading", text="B", level=2),
        Block(kind="paragraph", text="under b"), Block(kind="heading", text="C", level=2),
        Block(kind="paragraph", text="under c"), Block(kind="heading", text="D", level=1),
        Block(kind="paragraph", text="under d"),
    ])
    paths = [c.heading_path for c in _chunker().chunk(doc, "x.md")]
    assert paths == ["A › B", "A › C", "D"]


def test_big_table_becomes_pointer_chunk_when_named():
    rows = [[str(i), f"part {i}"] for i in range(40)]
    block = Block(kind="table", text="(table)", meta={"headers": ["n", "part"], "rows": rows,
                                                       "caption": "Parts", "table_name": "t_parts"})
    chunks = _chunker().chunk(ExtractedDoc(file_type="md", title=None, blocks=[block]), "x.md")
    assert len(chunks) == 1 and chunks[0].table_name == "t_parts" and "40 rows" in chunks[0].text


def test_oversized_inline_table_splits_by_rows_with_header():
    rows = [[f"value {i} " * 6, f"description {i} " * 6] for i in range(25)]
    from rag.core.util import markdown_table
    block = Block(kind="table", text=markdown_table(["a", "b"], rows), meta={"headers": ["a", "b"], "rows": rows})
    chunks = _chunker().chunk(ExtractedDoc(file_type="md", title=None, blocks=[block]), "x.md")
    assert len(chunks) > 1 and all(c.text.startswith("| a | b |") for c in chunks)
    assert all(c.n_tokens <= 500 for c in chunks)


def test_long_code_splits_with_signature_continuation():
    body = "\n".join([f"    x{i} = compute_value({i})  # step {i}" + ("\n" if i % 7 == 0 else "") for i in range(220)])
    text = f"```python\ndef big():\n{body}\n```"
    block = Block(kind="code", text=text, lang="python", symbol="m.big", line_start=10, line_end=240,
                  meta={"signature": "def big():"})
    chunks = _chunker().chunk(ExtractedDoc(file_type="py", title=None, blocks=[block]), "m.py")
    assert len(chunks) > 1
    assert all(c.n_tokens <= 820 for c in chunks)
    assert all("# continued: def big():" in c.text for c in chunks[1:])
    n_lines = len(text.split("\n")) - 2  # minus the fence lines
    assert chunks[0].line_start == 10 and chunks[-1].line_end == 10 + n_lines - 1


def test_identifier_helpers():
    assert split_identifier("GetLotYield") == ["get", "lot", "yield"]
    assert split_identifier("HTTPServer2_ok") == ["http", "server", "2", "ok"]
    assert "get lot yield" in identifier_terms("var y = repo.GetLotYield(lotId);")
    assert set(query_identifiers("Why does GetLotYield fail for lot 24A-0193?")) == {"GetLotYield", "24A-0193"}


def test_code_chunks_carry_identifier_forms():
    block = Block(kind="code", text="```csharp\npublic bool CanRelease(string lotId) {}\n```", lang="csharp",
                  symbol="A.B.CanRelease")
    ch = _chunker().chunk(ExtractedDoc(file_type="cs", title=None, blocks=[block]), "B.cs")[0]
    assert "identifiers: can release; lot id" in ch.embed_text
    assert ch.embed_text.startswith("B.cs › A.B.CanRelease\n")
