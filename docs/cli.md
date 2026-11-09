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

A **workspace** is a folder you build into (`config.toml`, `rag.db`, `tables.db`). A **pack**
is a `.ragpack` file: a checked, read-only snapshot of a workspace to copy to the retrieval
server. Commands that read (search, tables, sql, eval) accept either. Commands that write
(build, remove, export) need a workspace.

Workspaces you create in the app live in its data folder:

- Windows: `%LOCALAPPDATA%\com.ragworkbench.desktop\workspaces\`
- Linux: `~/.local/share/rag-workbench/workspaces/`

## Common tasks

Build a knowledge base from folders, then export it:

```
rag init plant-kb
rag build plant-kb "D:\KB\Procedures" "D:\KB\Specs"
rag status plant-kb
rag export plant-kb plant-kb.ragpack
```

Run `rag build` again after documents change: only new and changed files are processed.
Add `--prune` to drop documents whose files were deleted.

Search, the way the agent will:

```
rag search plant-kb.ragpack "minimum pull strength for 5 mil aluminum wire"
rag search plant-kb.ragpack "seal leak limits" --type pdf -k 10
rag search plant-kb.ragpack "seal leak limits" --json
```

Query tables from spreadsheets and documents:

```
rag tables plant-kb.ragpack
rag describe plant-kb.ragpack t_burnin_results_2025q3
rag sql plant-kb.ragpack "SELECT part_number, SUM(qty_fail) FROM t_burnin_results_2025q3 GROUP BY part_number"
```

Measure retrieval with known answers:

```
rag eval plant-kb.ragpack golden.jsonl --report eval.md
```

Install a pack on the retrieval server (checks every file's checksum):

```
rag unpack plant-kb.ragpack D:\RAG\plant-kb
```

## Command reference

| Command | What it does |
|---|---|
| `rag init WS [--dims N] [--embedder NAME]` | Create a workspace. `--dims 2048` or `1024` stores smaller vectors; it can't be changed later without a full rebuild. |
| `rag build WS PATH... [--prune] [--force] [--json]` | Read files and folders into the workspace. `--force` rebuilds everything; `--json` prints progress as JSON lines. |
| `rag remove WS PATTERN...` | Remove documents by path or glob, e.g. `"*/Archive/*"`. |
| `rag status WS` | Counts by type, failed files and embedding settings. |
| `rag export WS OUT.ragpack [--name NAME] [--include-originals]` | Write a pack. Originals let the agent open source files but make the pack larger. |
| `rag unpack PACK DEST` | Extract a pack and verify it. |
| `rag search WS\|PACK "question" [-k N] [--type pdf] [--json]` | The passages the agent would get, with sources. |
| `rag tables WS\|PACK` | List tables. |
| `rag describe WS\|PACK TABLE` | A table's columns, value ranges and sample rows. |
| `rag sql WS\|PACK "SELECT ..." [--json]` | Read-only SQL over the tables, as the agent runs it. |
| `rag eval WS\|PACK golden.jsonl [--report R.md] [--out R.json]` | Recall@5, recall@20 and MRR by question type, with the retrieved chunks in the report. |
| `rag check-embedder WS` | Embed a test sentence; shows the model, dimensions and speed. |

`rag COMMAND --help` lists every option.

## Golden question files

One JSON object per line. `expect` says where the answer is; any match counts.

```
{"id": "q1", "type": "spec", "question": "What is the helium leak limit for TO-39?", "expect": [{"file": "*SOP-0412*", "page": 3, "contains": "5e-8"}]}
{"id": "q2", "question": "What does a wire bonder cost?", "expect": []}
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
- **`'rag' is not recognized`** (Windows): open a new terminal after turning the command on;
  terminals opened before don't see the change.
