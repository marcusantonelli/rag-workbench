# RAG Workbench (desktop app)

A native app for building, inspecting and testing knowledge bases:

- **Open `.ragpack` files** (double-click, drag onto the window, or Open pack) and search
  them the way the agent will: every passage shows its file, page and section, how it ranked
  in the vector and keyword searches, and the exact text the MCP tool hands the agent.
- **Browse** each document's chunks in order, look at extracted tables and run the same
  read-only SQL the agent can run.
- **Evaluate** retrieval with a golden question file: recall@5, recall@20 and MRR, by question
  type, and the retrieved chunks for every question with the correct ones marked.
- **Build** knowledge bases from files and folders (incremental, cancellable), **merge** in
  other packs or workspaces built with the same model (no re-embedding), and **export**
  them as `.ragpack` files for the retrieval server. **Check workspace** (About tab) runs
  the same checks as `rag verify`.
- **Manage the embedding model**: the first run downloads llama.cpp and
  Qwen3-Embedding-8B (Q4_K_M, 4.7 GB), or uses a `.gguf` you already have, and runs it
  on the best hardware it finds.

- **Use `rag` in a terminal**: the installers include the command-line tool. The app's
  Command line page has a built-in terminal that runs `rag` commands (and nothing else),
  the guide ([docs/cli.md](../docs/cli.md)), and on Windows a switch that adds `rag` to
  your PATH.

The crawler stays a separate tool (`rag-crawl`).

## Install

Download the installer from the GitHub Actions run or release (see "Building the installers"):

- **Windows**: `RAG Workbench_0.4.1_x64-setup.exe` installs for the current user without admin
  rights. The `.msi` is for machine-wide or managed installs. The installers are not code
  signed yet, so SmartScreen will warn; choose More info, then Run anyway.
- **Linux**: `.deb` for Debian/Ubuntu (`sudo apt install ./RAG.Workbench_0.4.1_amd64.deb`), or the
  `.AppImage` for anything else (`chmod +x` and run it).

From 0.4.1 the app updates itself: a few seconds after it starts it checks the project's latest
GitHub release, and when a newer one exists a bar at the top shows the version and what's new.
**Update and restart** downloads it, checks its signature, stops the engine and the model
servers, installs and restarts; it waits while a build, merge or export runs. Settings → Updates
turns the check off or runs it by hand. 0.4.1 itself has to be installed by hand once. The
`.AppImage` and the Windows installers update this way; a `.deb` install is updated with a new
`.deb`.

On first start, Settings shows the hardware it found. **Download and start** fetches llama.cpp
for this machine and the model (resumable, checksums verified), then starts it. Offline
machines: copy `Qwen3-Embedding-8B-Q4_K_M.gguf` over and choose **Use a .gguf file I already
have**.

## Hardware selection

| Machine | llama.cpp build | Runs on |
|---|---|---|
| Windows + NVIDIA | CUDA 13.5 (driver 580+) or CUDA 12.4 | the NVIDIA GPU |
| Windows or Linux + AMD, Intel or other GPU (integrated too) | Vulkan | that GPU |
| Linux + NVIDIA | Vulkan (llama.cpp has no Linux CUDA binaries) | the NVIDIA GPU |
| No GPU | CPU | the CPU |

After installing, the app asks llama.cpp which devices it can actually use
(`llama-server --list-devices`). If none is usable, or the GPU reports less free memory
than the model needs (model size + 1.5 GB), the model runs on the CPU, and Settings says why.
**Force CPU** runs on the CPU regardless (leaving the GPU free). **Use the GPU even if it
reports too little free memory** overrides the size check. **Use my own llama-server** points
the app at a server you run elsewhere.

The llama.cpp build is pinned (`b12207`) so every install behaves the same.

## Where things are kept

| | Windows | Linux |
|---|---|---|
| Settings, models, llama.cpp, opened packs, new workspaces | `%LOCALAPPDATA%\com.ragworkbench.desktop` | `~/.local/share/rag-workbench` |
| The program (per-user install) | `%LOCALAPPDATA%\RAG Workbench` | `/usr/lib/RAG Workbench` |

Data is kept apart from the program so uninstalling never deletes models or workspaces;
the Windows uninstaller asks whether to delete the data folder too.

Opened packs are unpacked (and checksum-verified) once into `packs/` and reused; the 12
most recent are kept. Set `RAG_WORKBENCH_HOME` to use another folder (portable installs).

## The `rag` command

The installers ship `rag` next to the engine (same files, no Python needed):

- **Linux .deb**: installed as `/usr/bin/rag`.
- **Windows**: Command line page, **Add the rag command to PATH** (edits your user PATH; no
  admin rights). Open a new terminal afterwards. Turning it off removes the entry; uninstalling
  the app leaves a harmless PATH entry behind, so turn it off first if you care.
- **AppImage**: can't add commands to the PATH; use the .deb or pip.

While the app's embedding model is running it listens on `127.0.0.1:8081` (when that port is
free), the address workspaces use by default, so `rag build` and `rag search` use it with no
setup. See [docs/cli.md](../docs/cli.md), which is also the app's Command line page.

## How it fits together

```text
┌──────────── RAG Workbench (Tauri) ─────────────┐
│ React UI  ── invoke("engine_call") ──┐         │
│           ◄─ "engine-event" ─────────┤ Rust    │
└──────────────────────────────────────┼─────────┘
                       JSON lines over stdio
                                       ▼
                   rag-engine (the Python Builder, frozen with PyInstaller)
                   knowledge bases, search, SQL, eval, builds, exports, jobs
                                       │ HTTP 127.0.0.1
                                       ▼
                   llama-server + Qwen3-Embedding-8B (started and watched by the engine)
```

- `src/`: the React UI. `api.ts` talks to the engine over Tauri IPC in the app, or over HTTP
  in a browser during development.
- `src-tauri/`: the native shell (`src/lib.rs` starts the engine, relays calls and events,
  and stops the engine on exit).
- `../src/rag/app/`: the engine. `service.py` (every method the UI can call),
  `runtime.py` (hardware, downloads, llama-server), `transport.py` (stdio and HTTP).
- `engine/rag-engine.spec` and `scripts/build_engine.py`: freeze the engine and the `rag`
  command (two programs sharing one folder) and smoke-test both.
- `src-tauri/linux/rag`: the `/usr/bin/rag` launcher in the .deb.
- `../docs/cli.md`: the command-line guide, bundled into the app's Command line page.
- `../src/rag/app/terminal.py`: the in-app terminal. It splits the command line itself (no
  shell), accepts only rag's subcommands, and always runs the bundled `rag`.

## Developing

Prerequisites: Python 3.11+ with rag-builder installed (`pip install -e ..`), Node 20+, and
Rust (for the native shell). On Linux also:
`sudo apt install libwebkit2gtk-4.1-dev libayatana-appindicator3-dev librsvg2-dev libxdo-dev`.

**In a browser** (fastest; no Rust needed):

```bash
python -m rag.app --http 8765     # terminal 1: the engine
npm install && npm run dev        # terminal 2: open http://localhost:5173
```

File dialogs become text prompts (type a path) and drag-and-drop is off; everything else is
the same.

**As the native app**, with the engine running from your Python:

```bash
RAG_ENGINE_PYTHON=$(which python) npm run tauri:dev
```

(Windows PowerShell: `$env:RAG_ENGINE_PYTHON = (Get-Command python).Source; npm run tauri:dev`.)

## Building the installers

The GitHub Actions workflow `.github/workflows/desktop.yml` builds the Windows `.exe`/`.msi`
and Linux `.deb`/`.AppImage` on every `v*` tag (attached to a draft release) and from the
Actions tab (Run workflow). Push the repository to GitHub to use it.

To build locally on the target OS:

```bash
python desktop/scripts/build_engine.py     # freezes the engine and smoke-tests it
cd desktop && npm ci
npx tauri build                            # Windows: nsis + msi; Linux: deb + appimage
```

Installers land in `desktop/src-tauri/target/release/bundle/`. Windows installers have to be
built on Windows.

### Releasing with updates

The installed apps check `releases/latest/download/latest.json` on GitHub (and, with pre-releases
on, the `update-feed` release first), so a release is offered as an update when:

1. The repository has the secrets `TAURI_SIGNING_PRIVATE_KEY` (the contents of the private key
   made with `npx tauri signer generate`) and `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`. The build
   then signs the installers, and the `release` job writes `latest.json` from the signatures and
   `docs/release-notes/<version>.md` and attaches it to the draft release. Without the secrets
   the installers are built as before and no update is published.
2. The draft is published. A normal release is offered to everyone. A pre-release is offered only
   to apps with **Settings → Updates → Also offer pre-releases** on (see below).

When a release is published, the `Update feed` workflow (`.github/workflows/update-feed.yml`)
copies its `latest.json` to `prerelease.json` on a fixed release tagged `update-feed`, unless that
already offers a newer version. Apps with pre-releases on read that file first (it carries normal
releases too) and fall back to the normal feed; every other app reads GitHub's "latest" release,
which never points at a pre-release or at `update-feed`. The workflow creates `update-feed` (as a
pre-release, so it never becomes "latest") the first time. Leave it in place.

**Pre-release versions.** Tag them with a numeric suffix: `v0.4.3-1`, `v0.4.3-2`, and so on. The
Windows `.msi` builder only accepts numbers after the hyphen, so `v0.4.3-beta.1` fails to build.
The release workflow marks any tag with a hyphen as a pre-release. Put the same version
(`0.4.3-1`) in `pyproject.toml`, `src/rag/__init__.py`, `package.json`, `tauri.conf.json` and
`Cargo.toml`, and write `docs/release-notes/0.4.3-1.md`. The app shows a small **Pre-release**
mark beside its name while a pre-release is installed. `0.4.3-1` counts as older than `0.4.3`,
so the normal release replaces it.

The public half of the key is in `src-tauri/tauri.conf.json` (`plugins.updater.pubkey`); an
update signed with any other key is refused. Keep a backup of the private key: without it,
installed apps can't be updated and have to be reinstalled by hand.

## Troubleshooting

- **"llama-server exited"**: Settings → Show log. On Linux, the Vulkan build needs the Vulkan
  loader (`sudo apt install libvulkan1 mesa-vulkan-drivers`); turn on Force CPU to work
  without it.
- **Downloads fail on a company network**: set `HTTPS_PROXY` before starting the app, or
  download the `.gguf` elsewhere and use the file option. llama.cpp builds can be copied
  into the data folder's `runtime/` too.
- **"Different embedding model"**: the pack was built with a different model name than the
  one loaded. Load the same model; searching with another model returns unrelated passages.
- **The engine stopped**: the app restarts it on the next action; its last output is in the
  error message.
