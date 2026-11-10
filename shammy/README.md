# Shammy

The purpose-built vector store for this pipeline (plan: "Shammy: low-level plan for a
purpose-built vector store"). Nothing here is used by `rag` yet.

## Milestone 0: prove the funnel

`prototype/m0_funnel.py` reads the vectors of an existing workspace or pack (read-only, no
re-embedding), encodes them as the plan describes and reports whether the three-stage search
keeps recall@20 at 0.99 or better against exact search.

```bash
pip install numpy sqlite-vec          # already there if rag-builder is installed
python shammy/prototype/m0_funnel.py /path/to/siemens-kb.ragpack
```

Optional, closer to real use (the embedding model must be running, e.g. in RAG Workbench):

```bash
python shammy/prototype/m0_funnel.py siemens-kb.ragpack --golden golden.jsonl --embed-url http://127.0.0.1:8081
```

It prints three tables and a verdict, and writes `m0_results.json`. Runtime: about a minute for
30,000 chunks; RAM is roughly 6.4 KB x chunks x queries / 1000 (lower `--max-rows` or `--queries`
if needed). The pack must hold real embeddings: a `hash-test` workspace proves nothing.
