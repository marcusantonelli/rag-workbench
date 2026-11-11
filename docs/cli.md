# The `rag` command

`rag` does everything the desktop app does, from a terminal or a script: build knowledge
bases, export and unpack packs, search, query tables and run evaluations. It reads and
writes the same files as the app, so you can build in one and inspect in the other.

## Getting it

| Where | How |
|---|---|
| Windows (installed app) | Turn on **Add the rag command to PATH** on this page, then open a new terminal. |
| Linux (.deb) | Already installed as `/usr/bin/rag`. |
| Linux (AppImage) | Not available; install the .deb, or use pip. |
| Any machine with Python 3.11+ | From the source (the git bundle): `pip install .` in the rag-builder folder. |

You can also skip the setup: the app's **Command line** page has a terminal that runs `rag`
commands (see "In the app" below).

Check it works:

```
rag --version
```

## Before you start: the embedding model

Building and searching turn text into vectors, so `rag` needs the embedding model the
knowledge base uses. It looks for llama-server at `http://127.0.0.1:8081` (set in each
workspace's `config.toml`).

- **Easiest:** keep RAG Workbench open with its embedding model running. The app serves it
  on port 8081 whenever that port is free, so `rag` uses it automatically.
- **Without the app:** start llama-server yourself:

  ```
  llama-server -m Qwen3-Embedding-8B-Q4_K_M.gguf --embedding --pooling last -ub 8192 -c 8192 --port 8081 -ngl 99
  ```

  Use `-ngl 0` for CPU only.
- **Just trying things out:** `rag init try-ws --embedder hash-test` makes a workspace that
  needs no model. Its search quality is only good enough for testing; never ship it.

Test the connection with `rag check-embedder WORKSPACE`.

Browsing commands (`rag tables`, `rag describe`, `rag sql`, `rag status`) never need the model.

## In the app

Command line, then Terminal, runs `rag` without any setup:

- Type what comes after `rag` (the prompt already says `rag`), e.g.
  `search my-kb.ragpack "leak test limits"`, and press Enter.
- Only `rag` commands run there: no other programs, pipes (`|`), `&&` or redirects (`>`).
  Anything else is refused before it runs.
- Commands run in the folder shown under **Runs in** (your home folder at first), so relative
  paths start there. **Change…** picks another folder.
- **File…**, **Folder…** and **Open knowledge base…** insert a path, quoted, where the cursor is.
- It always uses the app's embedding model when it's running, whatever its port.
- Up and down recall earlier commands, Tab completes a command name, Ctrl+C (or **Stop**)
  stops the running command, and Ctrl+L clears the screen.

## Workspaces and packs

A **workspace** is a folder you build into (`config.toml`, `rag.db`, `tables.db` and a small
hidden `.ragworkspace` marker that names it). A **pack**
is a `.ragpack` file: a checked, read-only snapshot of a workspace to copy to the retrieval
server. Commands that read (search, tables, sql, eval) accept either. Commands that write
(build, remove, export) need a workspace.

Workspaces you create in the app live in its data folder:

- Windows: `%LOCALAPPDATA%\com.ragworkbench.desktop\workspaces\`
- Linux: `~/.local/share/rag-workbench/workspaces/`

`rag list` shows what's there (name, documents, chunks, tables, model, last update, size);
`rag list D:\KB\packs` looks in another folder, including any `.ragpack` files in it.
`rag verify WS` (or `PACK`) checks one is whole.

## Common tasks

Build a knowledge base from folders, then export it:

```
rag init docs-kb
rag build docs-kb "D:\KB\Procedures" "D:\KB\Specs"
rag status docs-kb
rag export docs-kb docs-kb.ragpack
```

Run `rag build` again after documents change: only new and changed files are processed.
Add `--prune` to drop documents whose files were deleted.

Search, the way the agent will:

```
rag search docs-kb.ragpack "minimum clamp pull force for a 10 mm hose"
rag search docs-kb.ragpack "pressure test limits" --type pdf -k 10
rag search docs-kb.ragpack "pressure test limits" --json
```

Search one document, by its number or part of its file name:

```
rag search docs-kb "filter won't seat" --doc DOC-12
rag search docs-kb "torque for the cover screws" --doc WI-0300 --doc FX-100
```

### Procedures and the spots file

Every build also writes `structure.spots` next to `rag.db`: which chunks make up each procedure
and its steps (and the sub-steps of long steps), the warnings in each step, references ("see
Section 8.4", "see FX-100"), and each document's number and revision. With it, a search that hits
one step of a procedure returns more of the procedure, as much as fits in `[search]
procedure_tokens` (1500 by default):

- a short procedure comes back whole;
- in a longer one, the step that matched and the steps around it (`procedure_window`);
- in a step too long for that, the sub-steps around the match, or the chunks around it.

The source line names the document and what was included ("steps 7.32-7.34 of 40", "step 7.20 of
40, sub-steps 7.20.6-7.20.8"). The first result from a long procedure also lists the steps around
it, with the included ones in bold, and references are listed under the passage. At most
`procedure_hits` (3) results come from one procedure while other documents still have results to
give; hits in the step just next to one already returned widen that result instead.

A knowledge base built before 0.4.1 has no spots file. Make one from its stored chunks; nothing is
read or embedded again, and it takes seconds:

```
rag spots docs-kb
rag spots docs-kb --check
```

Packs exported from 0.4.1 carry the spots file; the retrieval server uses it when it's there and
searches exactly as before when it isn't. It counts as a procedure only when the numbered steps,
numbered list or flowchart read as instructions, so manuals and web pages stay plain sections.

Query tables from spreadsheets and documents:

```
rag tables docs-kb.ragpack
rag describe docs-kb.ragpack t_runin_results_2025q3
rag sql docs-kb.ragpack "SELECT part_number, SUM(qty_fail) FROM t_runin_results_2025q3 GROUP BY part_number"
```

Combine knowledge bases built with the same model, without embedding anything again:

```
rag merge all-docs line-a.ragpack line-b.ragpack --dry-run
rag merge all-docs line-a.ragpack line-b.ragpack
rag export all-docs all-docs.ragpack
```

`all-docs` is created if it doesn't exist. Files already there are skipped; when the same
file differs, the newer build wins (`--on-conflict keep` or `replace` to choose); a file
whose content is already there under another path is skipped (`--keep-copies` adds it).
Packs or workspaces embedded with a different model or vector size are refused.

Measure retrieval with known answers:

```
rag eval docs-kb.ragpack golden.jsonl --report eval.md
```

Install a pack on the retrieval server (checks every file's checksum):

```
rag unpack docs-kb.ragpack D:\RAG\docs-kb
```

## Command reference

| Command | What it does |
|---|---|
| `rag init WS [--name NAME] [--dims N] [--embedder NAME] [--vectors shammy]` | Create a workspace. `--dims 2048` or `1024` stores smaller vectors; it can't be changed later without a full rebuild. `--vectors shammy` keeps the vectors in `vectors.shammy` instead of inside `rag.db`. |
| `rag list [FOLDER] [--json]` | Workspaces and packs in a folder; the app's workspaces folder by default. |
| `rag verify WS\|PACK [--json]` | Check it's whole: marker, settings, database integrity, counts that must agree, tables, figures; packs also every checksum. Exits 1 if something is wrong. |
| `rag build WS PATH... [--prune] [--force] [--json]` | Read files and folders into the workspace. `--force` rebuilds everything; `--json` prints progress as JSON lines. |
| `rag remove WS PATTERN...` | Remove documents by path or glob, e.g. `"*/Archive/*"`. |
| `rag status WS` | Counts by type, failed files and embedding settings. |
| `rag export WS OUT.ragpack [--name NAME] [--include-originals] [--slim]` | Write a pack. Originals let the agent open source files but make the pack larger. `--slim` (Shammy workspaces) leaves out the float16 vector copies: about a third of the vector size, but results aren't rescored exactly and the pack can't be merged. |
| `rag unpack PACK DEST` | Extract a pack and verify it. |
| `rag merge DEST SRC... [--on-conflict newer\|keep\|replace] [--keep-copies] [--dry-run]` | Copy compatible packs or workspaces into DEST (created if missing). No re-embedding. |
| `rag convert WS [--to shammy\|sqlite-vec]` | Move a workspace's vectors to the other store. Nothing is embedded again. Packs exported from a Shammy workspace are format `ragpack/2`, which the retrieval server can't read yet. |
| `rag compact WS` | Rewrite `vectors.shammy` without deleted rows. Builds do this on their own when a quarter of the rows are deleted. |
| `rag spots WS [--check] [--json]` | Build the spots file (procedures, steps, references) from the stored chunks; `--check` only says whether it's there and current. Builds, merges and exports do this on their own. |
| `rag search WS\|PACK "question" [-k N] [--type pdf] [--doc DOC-12] [--json] [--rerank] [--rerank-url URL]` | The passages the agent would get, with sources. `--doc` (repeatable) keeps only those documents, by number or part of the path. `--rerank` reorders the best 20 candidates with a reranker model at `--rerank-url` (default `http://127.0.0.1:8082`, or `RAG_RERANK_URL`); `--no-rerank` overrides a workspace that has it on. |
| `rag tables WS\|PACK` | List tables. |
| `rag describe WS\|PACK TABLE` | A table's columns, value ranges and sample rows. |
| `rag sql WS\|PACK "SELECT ..." [--json]` | Read-only SQL over the tables, as the agent runs it. |
| `rag eval WS\|PACK golden.jsonl [--report R.md] [--out R.json]` | Recall@5, recall@20 and MRR by question type, with the retrieved chunks in the report. |
| `rag check-embedder WS` | Embed a test sentence; shows the model, dimensions and speed. |

`rag COMMAND --help` lists every option.

## Golden question files

One JSON object per line. `expect` says where the answer is; any match counts.

```
{"id": "q1", "type": "spec", "question": "What is the pressure test limit for the H-39 housing?", "expect": [{"file": "*SOP-0412*", "page": 3, "contains": "0.5 bar"}]}
{"id": "q2", "question": "What does a replacement pump cost?", "expect": []}
```

`file` is a glob on the document's path. Optional keys: `page`, `sheet`, `line`, `kind`
(`prose`, `table`, `figure`, `code`, `table_card`) and `contains` (text the chunk must
include). Questions with `"expect": []` aren't scored; ask them through the agent to check
it says it can't find the answer.

## In scripts

`rag` exits with 0 on success, 1 when `rag build` finished but some files failed (add
`--allow-failures` to accept that), and 2 for errors such as an unreachable model, a damaged
pack or a rejected SQL query. Errors go to stderr as one line starting with `error:`.

`--json` on `build`, `search` and `sql` gives machine-readable output.

Set `RAG_EMBEDDING_URL` (for example `http://gpu-box:8081`) to point every workspace at one
llama-server without editing their `config.toml` files.

## Troubleshooting

- **`error: cannot reach llama-server at http://127.0.0.1:8081`**: open the app and start
  the embedding model, or start llama-server yourself (see above). If another program
  holds port 8081, the app uses a different port; set `RAG_EMBEDDING_URL` to the address
  shown on the Command line page, or use the app's terminal, which finds it automatically.
- **`embedded with different settings`**: the workspace was built with another model or
  vector size. Use the same model, or rebuild with `rag build WS PATH --force`.
- **`is not a workspace`**: the folder has no `.ragworkspace` marker and isn't an older
  workspace either. Check the path with `rag list` on its parent folder.
- **`can't merge: embedded differently`**: the two were built with different models or
  vector sizes; their vectors can't be compared. Rebuild one from its files instead.
- **`'rag' is not recognized`** (Windows): open a new terminal after turning the command on;
  terminals opened before don't see the change.

## Picture descriptions

Builds can ask a vision model to describe each photo and screenshot, so pictures are found by
what they show. In the desktop app, turn it on under **Settings → Picture descriptions** (it
downloads Qwen3-VL-4B, about 3 GB, and runs it beside the embedding model; builds in the app and
its terminal use it automatically). From the command line, run a vision model yourself and point
builds at it:

```bash
llama-server -m Qwen3VL-4B-Instruct-Q4_K_M.gguf --mmproj mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf \
  -c 8192 -np 1 --port 8083 -ngl 99
export RAG_DESCRIBE_URL=http://127.0.0.1:8083     # or [figures] describe_url in config.toml
rag build my-workspace docs/
```

Any OpenAI-compatible chat endpoint that accepts images works. Documents with pictures that were
built without descriptions are read again by the next build once descriptions are on; others are
left alone. If the model stops answering mid-build, the build finishes, the document is flagged,
and the next build tries again.
