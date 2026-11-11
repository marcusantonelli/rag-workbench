"""Everything the desktop UI can ask the engine to do.

Methods are plain functions taking JSON params and returning JSON results; transport.py
carries them over stdio (inside the app) or HTTP (browser dev mode). Slow work (builds,
exports, evaluations, downloads) runs as a job: the call returns a job id at once and the
job reports progress through "job" events.
"""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from rag import __version__
from rag.app.paths import Settings
from rag.app.runtime import Cancelled, Runtime, RuntimeErrorWithHint
from rag.core import spots
from rag.core.util import slugify

EmitFn = Callable[[str, dict[str, Any]], None]


class ServiceError(Exception):
    """An error the UI shows as is: a code to branch on, a message and an optional hint."""

    def __init__(self, code: str, message: str, hint: str = "", data: dict[str, Any] | None = None):
        super().__init__(message)
        self.code, self.message, self.hint, self.data = code, message, hint, data or {}

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "hint": self.hint, "data": self.data}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------------------------------------
# Jobs
# ------------------------------------------------------------------------------------------------
@dataclass
class Job:
    id: str
    kind: str
    title: str
    kb: str | None = None
    state: str = "running"                 # running | done | failed | cancelled
    started_at: str = field(default_factory=_now)
    finished_at: str | None = None
    progress: dict[str, Any] = field(default_factory=dict)
    log: list[dict[str, Any]] = field(default_factory=list)
    result: Any = None
    error: dict[str, Any] | None = None
    cancel: threading.Event = field(default_factory=threading.Event)

    def snapshot(self, with_result: bool = False) -> dict[str, Any]:
        out = {"id": self.id, "kind": self.kind, "title": self.title, "kb": self.kb, "state": self.state,
               "started_at": self.started_at, "finished_at": self.finished_at, "progress": self.progress,
               "error": self.error, "log_size": len(self.log)}
        if with_result:
            out["result"] = self.result
        return out


# ------------------------------------------------------------------------------------------------
# Knowledge bases
# ------------------------------------------------------------------------------------------------
@dataclass
class KnowledgeBase:
    id: str
    kind: str                  # pack | workspace
    path: Path                 # the .ragpack file or the workspace folder
    root: Path                 # folder holding rag.db (the unpacked copy for packs)
    name: str
    manifest: dict[str, Any] | None = None
    building: bool = False

    @property
    def readonly(self) -> bool:
        return self.kind == "pack"

    def workspace(self):
        from rag.core.workspace import Workspace

        return Workspace.open(self.root)

    @contextmanager
    def store(self):
        from rag.core.db.store import RagStore

        s = RagStore(self.root / "rag.db", readonly=True)
        try:
            yield s
        finally:
            s.close()


def _kb_id(path: Path) -> str:
    return "kb_" + hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:10]


class Service:
    def __init__(self, settings: Settings | None = None, emit: EmitFn | None = None):
        self.settings = settings or Settings()
        self._emit = emit or (lambda name, data: None)
        self.runtime = Runtime(self.settings, emit=self._emit)
        from rag.app.runtime import HELPERS, HelperModel

        self.helpers = {kind: HelperModel(kind, self.runtime, emit=self._helper_changed) for kind in HELPERS}
        self.vision, self.reranker = self.helpers["vision"], self.helpers["rerank"]
        from rag.app.terminal import Terminal

        self.terminal = Terminal(self.emit, self.runtime.url)
        self.kbs: dict[str, KnowledgeBase] = {}
        self.jobs: dict[str, Job] = {}
        self._lock = threading.RLock()
        self.methods: dict[str, Callable[..., Any]] = {
            "app.info": self.app_info,
            "app.open_path": self.open_path,
            "app.write_text": self.write_text,
            "runtime.status": self.runtime_status,
            "runtime.detect": self.runtime_detect,
            "runtime.setup": self.runtime_setup,
            "runtime.start": self.runtime_start,
            "runtime.stop": self.runtime_stop,
            "runtime.configure": self.runtime_configure,
            **{f"{kind}.{verb}": (lambda kind, verb: lambda **p: self._helper_call(kind, verb, **p))(kind, verb)
               for kind in ("vision", "rerank") for verb in ("status", "setup", "start", "stop", "configure")},
            "kb.recent": self.kb_recent,
            "kb.forget": self.kb_forget,
            "kb.open": self.kb_open,
            "kb.create": self.kb_create,
            "kb.close": self.kb_close,
            "kb.list": self.kb_list,
            "kb.summary": self.kb_summary,
            "kb.spots": self.kb_spots,
            "kb.documents": self.kb_documents,
            "kb.document": self.kb_document,
            "kb.chunk": self.kb_chunk,
            "kb.search": self.kb_search,
            "kb.procedure": self.kb_procedure,
            "kb.tables": self.kb_tables,
            "kb.table": self.kb_table,
            "kb.sql": self.kb_sql,
            "kb.asset": self.kb_asset,
            "kb.open_source": self.kb_open_source,
            "kb.build": self.kb_build,
            "kb.remove_documents": self.kb_remove_documents,
            "kb.export": self.kb_export,
            "kb.verify": self.kb_verify,
            "kb.merge": self.kb_merge,
            "kb.convert": self.kb_convert,
            "kb.discover": self.kb_discover,
            "eval.run": self.eval_run,
            "eval.markdown": self.eval_markdown,
            "cli.status": self.cli_status,
            "cli.set_enabled": self.cli_set_enabled,
            "terminal.run": self.terminal_run,
            "terminal.stop": self.terminal_stop,
            "jobs.list": self.jobs_list,
            "jobs.get": self.jobs_get,
            "jobs.cancel": self.jobs_cancel,
        }

    # ---------------------------------------------------------------------------- plumbing
    def emit(self, name: str, data: dict[str, Any]) -> None:
        try:
            self._emit(name, data)
        except Exception:  # noqa: BLE001 - a broken listener must not break the engine
            pass

    def call(self, method: str, params: dict[str, Any] | None) -> Any:
        fn = self.methods.get(method)
        if fn is None:
            raise ServiceError("unknown_method", f"Unknown method {method!r}.")
        try:
            return fn(**(params or {}))
        except ServiceError:
            raise
        except TypeError as exc:
            if "argument" in str(exc):
                raise ServiceError("bad_params", f"{method}: {exc}") from exc
            raise
        except RuntimeErrorWithHint as exc:
            raise ServiceError("runtime", str(exc), exc.hint) from exc
        except FileNotFoundError as exc:
            raise ServiceError("not_found", str(exc)) from exc

    def startup(self) -> None:
        rt = self.settings.runtime
        if rt.get("mode") == "managed" and rt.get("autostart", True):
            threading.Thread(target=self._autostart, daemon=True).start()
        for helper in self.helpers.values():
            if helper.cfg.get("enabled") and not helper.status()["needs_setup"]:
                threading.Thread(target=self._helper_autostart, args=(helper,), daemon=True).start()

    def _helper_autostart(self, helper) -> None:
        try:
            helper.start()
        except Exception as exc:  # noqa: BLE001
            self.emit("log", {"level": "warning", "message": f"{helper.spec['label']} autostart failed: {exc}"})

    def _helper_changed(self, name: str, data: dict[str, Any]) -> None:
        """Builds describe pictures, and searches rerank, while the helper model is ready (here and
        in the app's terminal): its address goes in RAG_DESCRIBE_URL / RAG_RERANK_URL."""
        from rag.app.runtime import HELPERS

        var = HELPERS[name]["env"]
        url = data.get("url") if data.get("ready") else None
        if url:
            os.environ[var] = url
        else:
            os.environ.pop(var, None)
        self._emit(name, data)

    def _autostart(self) -> None:
        try:
            self.runtime.start()
        except Exception as exc:  # noqa: BLE001
            self.emit("log", {"level": "warning", "message": f"llama-server autostart failed: {exc}"})

    def shutdown(self) -> None:
        self.terminal.shutdown()
        for job in list(self.jobs.values()):
            job.cancel.set()
        for helper in self.helpers.values():
            helper.stop()
        self.runtime.stop()

    def _kb(self, kb: str) -> KnowledgeBase:
        found = self.kbs.get(kb)
        if found is None:
            raise ServiceError("kb_not_open", "That knowledge base is not open.")
        return found

    def _job(self, kind: str, title: str, work: Callable[[Job], Any], kb: str | None = None) -> dict[str, Any]:
        job = Job(id="job_" + uuid.uuid4().hex[:10], kind=kind, title=title, kb=kb)
        self.jobs[job.id] = job
        last_emit = [0.0]

        def report(update: dict[str, Any] | None = None, log: dict[str, Any] | None = None, force: bool = False):
            if update:
                job.progress.update(update)
            if log:
                job.log.append({"t": _now(), **log})
                if len(job.log) > 2000:
                    del job.log[:500]
            now = time.monotonic()
            if force or now - last_emit[0] > 0.2:
                last_emit[0] = now
                self.emit("job", job.snapshot())

        job.report = report  # type: ignore[attr-defined]

        def run() -> None:
            try:
                job.result = work(job)
                job.state = "cancelled" if job.cancel.is_set() else "done"
            except (Cancelled,) as exc:
                job.state, job.error = "cancelled", {"code": "cancelled", "message": str(exc), "hint": ""}
            except ServiceError as exc:
                job.state, job.error = "failed", exc.to_dict()
            except RuntimeErrorWithHint as exc:
                job.state, job.error = "failed", {"code": "runtime", "message": str(exc), "hint": exc.hint}
            except Exception as exc:  # noqa: BLE001
                job.state = "failed"
                job.error = {"code": "error", "message": f"{type(exc).__name__}: {exc}", "hint": "",
                             "trace": traceback.format_exc()[-4000:]}
            finally:
                job.finished_at = _now()
                self.emit("job", job.snapshot(with_result=True))

        threading.Thread(target=run, name=job.id, daemon=True).start()
        return job.snapshot()

    # ---------------------------------------------------------------------------- app
    def app_info(self) -> dict[str, Any]:
        return {"version": __version__, "data_dir": str(self.settings.root),
                "workspaces_dir": str(self.settings.workspaces_dir), "platform": sys.platform,
                "python": sys.version.split()[0], "frozen": bool(getattr(sys, "frozen", False)),
                "shammy": self._shammy_version()}

    @staticmethod
    def _shammy_version() -> str | None:
        """Version of the Shammy vector engine, or None when this build doesn't include it."""
        try:
            import shammy
        except ImportError:
            return None
        return shammy.__version__

    def open_path(self, path: str, reveal: bool = False) -> dict[str, Any]:
        p = Path(path)
        if not p.exists():
            raise ServiceError("not_found", f"{path} does not exist.")
        target = p.parent if reveal and p.is_file() else p
        if sys.platform == "win32":
            if reveal and p.is_file():
                subprocess.Popen(["explorer", "/select,", str(p)])
            else:
                os.startfile(str(target))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(p)] if reveal else ["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {"opened": str(target)}

    def write_text(self, path: str, text: str) -> dict[str, Any]:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return {"path": str(p), "bytes": p.stat().st_size}

    # ---------------------------------------------------------------------------- runtime
    def runtime_status(self) -> dict[str, Any]:
        return self.runtime.status()

    def runtime_detect(self) -> dict[str, Any]:
        self.runtime.hardware(refresh=True)
        return self.runtime.status()

    def runtime_setup(self, download_model: bool = True, model_path: str | None = None,
                      model_id: str | None = None) -> dict[str, Any]:
        def work(job: Job) -> Any:
            def progress(p: dict[str, Any]) -> None:
                msg = p.get("message") or " ".join(str(x) for x in (p.get("stage"), p.get("label")) if x)
                job.report(p, log={"message": msg} if p.get("stage") != "download" else None)  # type: ignore[attr-defined]

            return self.runtime.setup(download_model=download_model, model_path=model_path, model_id=model_id,
                                      progress=progress, should_stop=job.cancel.is_set)

        return self._job("runtime_setup", "Set up the embedding model", work)

    def runtime_start(self) -> dict[str, Any]:
        return self.runtime.start()

    def runtime_stop(self) -> dict[str, Any]:
        return self.runtime.stop()

    def runtime_configure(self, **changes: Any) -> dict[str, Any]:
        try:
            return self.runtime.configure(**changes)
        except ValueError as exc:
            raise ServiceError("bad_params", str(exc)) from exc

    # ---------------------------------------------------------------------------- helper models
    def _helper_call(self, kind: str, verb: str, **params: Any) -> dict[str, Any]:
        """vision.* and rerank.*: status, setup (a job: download a listed model or use your own
        file, then start it), start, stop, configure (enabled, choice of a downloaded model)."""
        helper = self.helpers[kind]
        if verb == "status":
            return helper.status()
        if verb == "start":
            return helper.start()
        if verb == "stop":
            return helper.stop()
        if verb == "configure":
            try:
                return helper.configure(**params)
            except ValueError as exc:
                raise ServiceError("bad_params", str(exc)) from exc
        choice, model_path = params.get("choice"), params.get("model_path")
        unknown = set(params) - {"choice", "model_path"}
        if unknown:
            raise ServiceError("bad_params", f"{kind}.setup: unknown {', '.join(sorted(unknown))}")
        if model_path and not helper.spec["own_file"]:
            raise ServiceError("bad_params", "Choose one of the listed picture models.",
                               "A picture model needs its matching image encoder, so only listed models are offered.")
        if model_path and not (Path(model_path).is_file() and model_path.lower().endswith(".gguf")):
            raise ServiceError("bad_params", f"{model_path} is not a .gguf file.")

        def work(job: Job) -> Any:
            def progress(p: dict[str, Any]) -> None:
                msg = p.get("message") or " ".join(str(x) for x in (p.get("stage"), p.get("label")) if x)
                job.report(p, log={"message": msg} if p.get("stage") != "download" else None)  # type: ignore[attr-defined]

            return helper.setup(choice=choice, model_path=model_path, progress=progress, should_stop=job.cancel.is_set)

        return self._job(f"{kind}_setup", f"Set up the {helper.spec['label']}", work)

    # ---------------------------------------------------------------------------- knowledge bases
    def kb_recent(self) -> list[dict[str, Any]]:
        out = []
        for r in self.settings.data.get("recent", []):
            out.append({**r, "exists": Path(r["path"]).exists()})
        return out

    def kb_forget(self, path: str) -> list[dict[str, Any]]:
        self.settings.forget(path)
        return self.kb_recent()

    def kb_list(self) -> list[dict[str, Any]]:
        return [self.kb_summary(k) for k in list(self.kbs)]

    def kb_open(self, path: str) -> dict[str, Any]:
        p = Path(path).expanduser()
        if not p.exists():
            raise ServiceError("not_found", f"{p} does not exist.")
        kid = _kb_id(p)
        with self._lock:
            if kid in self.kbs:
                return self.kb_summary(kid)
        if p.is_file():
            kb = self._open_pack(p, kid)
        else:
            from rag.core.workspace import NotAWorkspace, Workspace

            try:
                ws = Workspace.open(p)
            except NotAWorkspace as exc:
                raise ServiceError("not_workspace", str(exc),
                                   "Open a .ragpack file, or a workspace folder made by RAG Workbench or rag init.") from exc
            kb = KnowledgeBase(id=kid, kind="workspace", path=p, root=p, name=ws.name)
        with self._lock:
            self.kbs[kid] = kb
        self.settings.remember(str(p), kb.kind, kb.name, _now())
        return self.kb_summary(kid)

    def _open_pack(self, p: Path, kid: str) -> KnowledgeBase:
        from rag.core.db.pack import PackError, read_manifest, unpack_pack

        if not zipfile.is_zipfile(p):
            raise ServiceError("not_pack", f"{p.name} is not a .ragpack file.")
        try:
            manifest = read_manifest(p)
        except PackError as exc:
            raise ServiceError("not_pack", str(exc)) from exc
        st = p.stat()
        key = hashlib.sha1(f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}|{manifest.get('pack_id')}".encode()).hexdigest()[:16]
        dest = self.settings.packs_dir / key
        marker = dest / ".verified"
        if not marker.exists():
            shutil.rmtree(dest, ignore_errors=True)
            try:
                unpack_pack(p, dest)
            except PackError as exc:
                raise ServiceError("pack_corrupt", str(exc),
                                   "The file may be damaged or incomplete; copy it again.") from exc
            marker.write_text(_now(), encoding="utf-8")
            self._trim_pack_cache(keep=dest)
        return KnowledgeBase(id=kid, kind="pack", path=p, root=dest, name=manifest.get("name") or p.stem,
                             manifest=manifest)

    def _trim_pack_cache(self, keep: Path, limit: int = 12) -> None:
        folders = sorted((d for d in self.settings.packs_dir.iterdir() if d.is_dir()),
                         key=lambda d: d.stat().st_mtime, reverse=True)
        in_use = {kb.root.resolve() for kb in self.kbs.values()} | {keep.resolve()}
        for d in folders[limit:]:
            if d.resolve() not in in_use:
                shutil.rmtree(d, ignore_errors=True)

    def kb_create(self, name: str, folder: str | None = None, dims: int = 4096,
                  backend: str = "llama-server", vectors: str = "sqlite-vec") -> dict[str, Any]:
        from rag.core.db.vectors import BACKENDS, VectorStoreError, check_dims
        from rag.core.workspace import Workspace

        if vectors not in BACKENDS:
            raise ServiceError("bad_params", f"Unknown vector store {vectors!r}.")
        if vectors == "shammy":
            if self._shammy_version() is None:
                raise ServiceError("unavailable", "This build doesn't include the Shammy engine.",
                                   "Choose sqlite-vec; the knowledge base can be moved to Shammy later.")
            try:
                check_dims(int(dims))
            except VectorStoreError as exc:
                raise ServiceError("bad_params", str(exc)) from exc

        if not name.strip():
            raise ServiceError("bad_params", "Give the knowledge base a name.")
        root = Path(folder) if folder else self.settings.workspaces_dir / slugify(name, 60)
        if (root / "rag.db").exists():
            raise ServiceError("exists", f"{root} already holds a knowledge base.", "Open it instead.")
        if root.exists() and any(root.iterdir()):
            raise ServiceError("exists", f"{root} is not empty.", "Choose an empty or new folder.")
        rt = self.settings.runtime
        overrides: dict[str, Any] = {"embedding": {"backend": backend, "dims": int(dims)},
                                     "vectors": {"backend": vectors}}
        if backend == "llama-server":
            overrides["embedding"]["model"] = rt.get("model_id")
        elif backend == "hash-test":
            overrides["embedding"].update({"model": "hash-test", "native_dims": int(dims)})
        Workspace.init(root, overrides, name=name.strip())
        summary = self.kb_open(str(root))
        return self.kb_summary(summary["id"])

    def kb_close(self, kb: str) -> dict[str, Any]:
        k = self._kb(kb)
        if k.building:
            raise ServiceError("busy", "A build is running in this knowledge base.", "Cancel it first.")
        self.kbs.pop(kb, None)
        return {"closed": kb}

    def _embedding_cfg(self, k: KnowledgeBase) -> dict[str, Any]:
        cfg = dict(k.workspace().config["embedding"])
        with k.store() as s:
            meta = s.meta()
        # A built knowledge base is defined by what it was embedded with.
        if meta.get("embed_model"):
            cfg["model"] = meta["embed_model"]
            cfg["backend"] = meta.get("embed_backend") or cfg.get("backend")
            cfg["dims"] = int(meta.get("embed_dims") or cfg["dims"])
        return cfg

    def _runtime_check(self, cfg: dict[str, Any]) -> dict[str, Any]:
        if cfg.get("backend") != "llama-server":
            return {"needs_runtime": False, "ready": True, "match": True}
        rt = self.runtime.status()
        loaded = self.settings.runtime.get("model_id")
        external = self.settings.runtime.get("mode") == "external"
        match = external or loaded == cfg.get("model")
        return {"needs_runtime": True, "ready": rt["ready"], "match": match, "kb_model": cfg.get("model"),
                "runtime_model": None if external else loaded}

    def _embedder(self, k: KnowledgeBase, allow_mismatch: bool = False):
        from rag.core.models.embed import make_embedder

        cfg = self._embedding_cfg(k)
        check = self._runtime_check(cfg)
        if check["needs_runtime"]:
            if not check["ready"]:
                raise ServiceError("runtime_not_ready", "The embedding model isn't running.",
                                   "Start it from the status bar, or finish setup in Settings.")
            if not check["match"] and not allow_mismatch:
                raise ServiceError(
                    "model_mismatch",
                    f"This knowledge base was embedded with {cfg.get('model')}, but the running model is "
                    f"{check['runtime_model']}.",
                    "Load the same model in Settings. Searching with a different model gives meaningless results.",
                    data=check)
            cfg["url"] = self.runtime.url()
        return make_embedder(cfg)

    def kb_summary(self, kb: str) -> dict[str, Any]:
        from rag.core.db.tables import TablesStore

        k = self._kb(kb)
        with k.store() as s:
            meta = s.meta()
            counts = s.counts()
            library = s.conn.execute("SELECT COUNT(*) FROM documents WHERE library_path IS NOT NULL").fetchone()[0]
        ts = TablesStore(k.root / "tables.db", readonly=True)
        try:
            n_tables = len(ts.catalog())
        finally:
            ts.close()
        cfg = self._embedding_cfg(k)
        size = (k.path.stat().st_size if k.kind == "pack" else
                sum(f.stat().st_size for f in (k.root / "rag.db", k.root / "tables.db", k.root / "vectors.shammy")
                    if f.exists()))
        return {
            "id": k.id, "kind": k.kind, "name": k.name, "path": str(k.path), "root": str(k.root),
            "readonly": k.readonly, "building": k.building, "size_bytes": size,
            "vector_store": counts.get("vector_store", "sqlite-vec"),
            "manifest": {key: v for key, v in (k.manifest or {}).items() if key != "files"} or None,
            "counts": {**counts, "tables": n_tables, "with_originals": library},
            "embedding": {"backend": cfg.get("backend"), "model": cfg.get("model"), "dims": cfg.get("dims"),
                          "query_instruction": meta.get("query_instruction")},
            "runtime": self._runtime_check(cfg),
            "last_build_at": meta.get("last_build_at"), "built_with": meta.get("app_version"),
            "config_fingerprint": meta.get("config_fingerprint"),
            # 1: file › headings in each embedded chunk; 2: also the document's title, number, revision
            "embed_context": int(meta.get("embed_context") or (1 if counts.get("chunks") else 0)) or None,
            "structure": spots.status(k.root),
        }

    def kb_spots(self, kb: str) -> dict[str, Any]:
        """(Re)build a knowledge base's spots file from its stored chunks. Nothing is read or embedded."""
        k = self._kb(kb)
        if k.building:
            raise ServiceError("busy", "This knowledge base is being built; its structure is rebuilt when the build ends.")
        try:
            spots.build(k.root / "rag.db")
        except Exception as exc:  # noqa: BLE001
            raise ServiceError("spot_failed", f"The structure couldn't be built: {exc}") from exc
        return self.kb_summary(kb)

    # ---------------------------------------------------------------------------- browsing
    def kb_documents(self, kb: str, query: str = "", file_type: str | None = None,
                     status: str | None = None) -> list[dict[str, Any]]:
        k = self._kb(kb)
        sql = ("SELECT doc_id, display_path, file_type, title, n_pages, size_bytes, status, warnings, n_chunks, "
               "extractor, ingested_at, library_path IS NOT NULL AS has_original FROM documents WHERE 1=1")
        args: list[Any] = []
        if query:
            sql += " AND (display_path LIKE ? OR title LIKE ?)"
            args += [f"%{query}%", f"%{query}%"]
        if file_type:
            sql += " AND file_type = ?"
            args.append(file_type)
        if status:
            sql += " AND status = ?"
            args.append(status)
        sql += " ORDER BY display_path"
        with k.store() as s:
            rows = [dict(r) for r in s.conn.execute(sql, args)]
            errors: dict[str, str] = {}
            if self._has_table(s, "ingest_errors"):
                for r in s.conn.execute("SELECT path, message FROM ingest_errors ORDER BY rowid"):
                    errors[r["path"]] = r["message"]   # latest message per file wins
        for r in rows:
            r["warnings"] = json.loads(r["warnings"]) if r.get("warnings") else []
            r["has_original"] = bool(r["has_original"])
            r["error"] = errors.get(r["display_path"]) if r["status"] == "failed" else None
        return rows

    @staticmethod
    def _has_table(store, name: str) -> bool:
        return store.conn.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (name,)).fetchone() is not None

    @staticmethod
    def _chunk_out(r: dict[str, Any]) -> dict[str, Any]:
        keep = ("chunk_id", "doc_id", "ordinal", "kind", "text", "heading_path", "symbol", "page_start", "page_end",
                "line_start", "line_end", "sheet", "cell_range", "table_name", "ocr_conf", "asset_path", "n_tokens")
        out = {key: r.get(key) for key in keep}
        if "display_path" in r:
            out["file"] = r["display_path"]
            out["file_type"] = r.get("file_type")
        return out

    def kb_document(self, kb: str, doc_id: int, offset: int = 0, limit: int = 500) -> dict[str, Any]:
        k = self._kb(kb)
        with k.store() as s:
            doc = s.conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
            if doc is None:
                raise ServiceError("not_found", "No such document.")
            chunks = [self._chunk_out(dict(r)) for r in s.conn.execute(
                "SELECT * FROM chunks WHERE doc_id = ? ORDER BY ordinal LIMIT ? OFFSET ?", (doc_id, limit, offset))]
        d = dict(doc)
        d["warnings"] = json.loads(d["warnings"]) if d.get("warnings") else []
        return {"document": d, "chunks": chunks}

    def kb_chunk(self, kb: str, chunk_id: int, context: int = 1) -> dict[str, Any]:
        k = self._kb(kb)
        with k.store() as s:
            row = s.conn.execute(
                "SELECT c.*, d.display_path, d.file_type, d.title FROM chunks c JOIN documents d USING(doc_id) "
                "WHERE c.chunk_id = ?", (chunk_id,)).fetchone()
            if row is None:
                raise ServiceError("not_found", "No such chunk.")
            r = dict(row)
            around = [self._chunk_out(dict(x)) for x in s.conn.execute(
                "SELECT * FROM chunks WHERE doc_id = ? AND ordinal BETWEEN ? AND ? ORDER BY ordinal",
                (r["doc_id"], r["ordinal"] - context, r["ordinal"] + context))]
        out = self._chunk_out(r)
        out["embed_text"] = r["embed_text"]
        out["title"] = r.get("title")
        return {"chunk": out, "context": around}

    def kb_search(self, kb: str, query: str, k: int | None = None, file_type: str | None = None,
                  expand: bool = True, allow_mismatch: bool = False, candidates: int = 30,
                  rerank: bool | None = None, rerank_url: str | None = None,
                  documents: list[str] | None = None) -> dict[str, Any]:
        from rag.core.retrieve.search import Searcher, rerank_settings

        if not query.strip():
            raise ServiceError("bad_params", "Type a question to search for.")
        kbo = self._kb(kb)
        embedder = self._embedder(kbo, allow_mismatch=allow_mismatch)
        config = kbo.workspace().config
        cfg = config["search"]
        if rerank is None and self.reranker.cfg.get("enabled") and self.reranker.url():
            rerank = True             # the app's reranker is on: every search uses it unless asked not to
        with kbo.store() as s:
            searcher = Searcher(s, embedder, cfg, reranker=rerank_settings(config.data, rerank, rerank_url or None))
            try:
                try:
                    res = searcher.search(query, k=k, file_type=file_type or None, expand=expand,
                                          documents=[d for d in (documents or []) if d and d.strip()] or None)
                except Exception as exc:
                    from rag.core.models.embed import EmbedderError

                    if isinstance(exc, EmbedderError):
                        raise ServiceError("embed_failed", str(exc), "Check the model status in Settings.") from exc
                    raise
                out = res.to_dict()
                for r in out["results"]:
                    r["ranks"] = res.ranks.get(r["chunk_id"])
                for t in out["tables"]:
                    t["ranks"] = res.ranks.get(t["chunk_id"])
                top = res.candidates[:candidates]
                rows = searcher._load(top)
                out["candidates"] = [{**self._chunk_out(rows[c]), "ranks": res.ranks.get(c)} for c in top if c in rows]
                out["markdown"] = res.to_markdown()
                out["engine"] = res.engine
            finally:
                searcher.close()      # Windows: an open reader keeps vectors.shammy from being replaced
        return out

    def kb_procedure(self, kb: str, procedure: int, from_step: str | None = None, after_chunk: int | None = None,
                     max_tokens: int = 4000) -> dict[str, Any]:
        """A page of a procedure (what `get_procedure_steps` gives an agent)."""
        from rag.core.retrieve.search import Searcher

        kbo = self._kb(kb)
        with kbo.store() as s:
            searcher = Searcher(s, None, kbo.workspace().config["search"])
            try:
                return searcher.procedure(int(procedure), from_=from_step or None, after_chunk=after_chunk,
                                          max_tokens=int(max_tokens))
            except ValueError as exc:
                raise ServiceError("bad_params", str(exc)) from exc
            finally:
                searcher.close()

    def kb_tables(self, kb: str) -> list[dict[str, Any]]:
        from rag.core.db.tables import TablesStore

        k = self._kb(kb)
        ts = TablesStore(k.root / "tables.db", readonly=True)
        try:
            return ts.catalog()
        finally:
            ts.close()

    def kb_table(self, kb: str, name: str, samples: int = 25) -> dict[str, Any]:
        from rag.core.db.tables import TablesStore

        k = self._kb(kb)
        ts = TablesStore(k.root / "tables.db", readonly=True)
        try:
            return ts.describe(name, samples=samples)
        except KeyError as exc:
            raise ServiceError("not_found", str(exc)) from exc
        finally:
            ts.close()

    def kb_sql(self, kb: str, sql: str, limit: int = 500) -> dict[str, Any]:
        from rag.core.db.sqlguard import QueryRejected, run_query

        k = self._kb(kb)
        try:
            return run_query(k.root / "tables.db", sql, row_limit=limit, time_limit_ms=5000)
        except QueryRejected as exc:
            raise ServiceError("sql_rejected", str(exc)) from exc

    def kb_asset(self, kb: str, path: str) -> dict[str, Any]:
        k = self._kb(kb)
        target = (k.root / path).resolve()
        if k.root.resolve() not in target.parents or not target.is_file():
            raise ServiceError("not_found", "No such asset.")
        if target.stat().st_size > 20 * 1024 * 1024:
            raise ServiceError("too_large", "Asset is too large to preview.")
        mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        return {"data_url": f"data:{mime};base64," + base64.b64encode(target.read_bytes()).decode()}

    def kb_open_source(self, kb: str, doc_id: int, reveal: bool = False) -> dict[str, Any]:
        k = self._kb(kb)
        with k.store() as s:
            row = s.conn.execute("SELECT source_key, library_path, display_path FROM documents WHERE doc_id = ?",
                                 (doc_id,)).fetchone()
        if row is None:
            raise ServiceError("not_found", "No such document.")
        candidates = []
        if row["library_path"]:
            candidates.append(k.root / row["library_path"])
        if k.kind == "workspace" and row["source_key"]:
            candidates.append(Path(row["source_key"]))
        for c in candidates:
            if c.exists():
                return self.open_path(str(c), reveal=reveal)
        raise ServiceError("no_original", f"The original of {row['display_path']} isn't available here.",
                           "Packs only carry originals when exported with 'Include original files'.")

    # ---------------------------------------------------------------------------- building
    def kb_build(self, kb: str, inputs: list[str], force: bool = False, prune: bool = False,
                 allow_mismatch: bool = False) -> dict[str, Any]:
        from rag.builder.engine import Builder
        from rag.core.db.store import EmbeddingMismatch

        k = self._kb(kb)
        if k.readonly:
            raise ServiceError("readonly", "Packs are read-only snapshots.",
                               "Build into a workspace, then export a new pack.")
        if k.building:
            raise ServiceError("busy", "A build is already running here.")
        paths = [Path(p) for p in inputs]
        missing = [str(p) for p in paths if not p.exists()]
        if not paths:
            raise ServiceError("bad_params", "Add at least one file or folder.")
        if missing:
            raise ServiceError("not_found", "Not found: " + ", ".join(missing[:5]))
        embedder = self._embedder(k, allow_mismatch=allow_mismatch)

        def work(job: Job) -> Any:
            k.building = True
            self.emit("kb", {"id": k.id, "building": True})
            # what the Build and Documents views show while a build runs
            counters: dict[str, Any] = {
                "files_total": 0, "files_done": 0, "chunks": 0, "failed": 0, "added": 0, "unchanged": 0,
                "bytes_total": 0, "bytes_done": 0, "bytes_skipped": 0, "started": time.time(), "work_started": None,
            }
            current: dict[str, Any] = {}
            if self.settings.vision.get("enabled") and self.vision.server.state == "starting":
                # pictures read before the model is up would be read again by the next build
                job.report({"current": None}, log={"message": "Waiting for the picture model to load"}, force=True)
                try:
                    self.vision.server.wait(job.cancel.is_set, timeout=600)
                except RuntimeErrorWithHint:
                    job.report(log={"message": "The picture model didn't start; pictures won't be described this time"})

            def file_state() -> dict[str, Any]:
                return {"current": current.get("path"), "stage": current.get("stage"),
                        "stage_done": current.get("stage_done"), "stage_total": current.get("stage_total"),
                        "page": current.get("page"), "pages": current.get("pages"), "pictures": current.get("pictures"),
                        "file_chunks": current.get("chunks"), "file_embedded": current.get("embedded"),
                        "file_bytes": current.get("bytes")}

            def progress(e: dict[str, Any]) -> None:
                ev = e.get("event")
                if ev == "start":
                    counters["files_total"] = e.get("files", 0)
                    counters["bytes_total"] = int(e.get("bytes") or 0)
                    job.report({**counters, "current": None}, log={"message": f"{e.get('files', 0)} files to check"},
                               force=True)
                elif ev in ("file_done", "file_unchanged", "file_failed", "file_removed"):
                    if ev != "file_removed":
                        counters["files_done"] = e.get("index", counters["files_done"])
                        size = int(e.get("bytes") or 0)
                        counters["bytes_done"] += size
                        if ev == "file_unchanged":
                            counters["bytes_skipped"] += size     # took no time: left out of the estimate
                            counters["unchanged"] += 1
                    if ev == "file_done":
                        counters["chunks"] += int(e.get("chunks", 0))
                        counters["added"] += 1
                    if ev == "file_failed":
                        counters["failed"] += 1
                    current.clear()
                    job.report({**counters, **file_state()},
                               log={"event": ev, "path": e.get("path"), "chunks": e.get("chunks"), "error": e.get("error"),
                                    "seconds": e.get("seconds")})
                elif ev == "file_start":
                    if counters["work_started"] is None:
                        counters["work_started"] = time.time()
                    current.clear()
                    current.update(path=e.get("path"), stage="extract", bytes=e.get("bytes"))
                    job.report({**counters, **file_state()})
                elif ev == "stage":
                    st = e.get("stage")
                    current["path"] = e.get("path")
                    if st == "page":
                        current.update(stage="page", page=e.get("page"), pages=e.get("pages"))
                    elif st == "ocr":
                        current.update(stage="ocr", page=e.get("page"))
                    elif st == "describe":
                        current.update(stage="describe", pictures=e.get("done"))
                    elif st == "chunked":
                        current.update(stage="chunk", chunks=e.get("chunks"))
                    elif st == "embed":
                        current.update(stage="embed", embedded=e.get("done"), chunks=e.get("total"),
                                       stage_done=e.get("done"), stage_total=e.get("total"))
                    else:
                        current.update(stage=st)
                    job.report(file_state())
                elif ev == "skipped":
                    job.report(log={"event": "skipped", "message": e.get("reason")})
                elif ev == "cancelled":
                    job.report(log={"event": "cancelled", "message": "Stopped after the current file."}, force=True)

            try:
                builder = Builder(k.workspace(), embedder=embedder, progress=progress, should_stop=job.cancel.is_set)
                try:
                    stats = builder.build(paths, force=force, prune=prune)
                except EmbeddingMismatch as exc:
                    raise ServiceError("model_mismatch", str(exc)) from exc
                out = {key: v for key, v in stats.__dict__.items()}
                out["cancelled"] = builder.cancelled
                return out
            finally:
                k.building = False
                self.emit("kb", {"id": k.id, "building": False})

        return self._job("build", f"Build {k.name}", work, kb=k.id)

    def kb_remove_documents(self, kb: str, paths: list[str] | None = None,
                            doc_ids: list[int] | None = None) -> dict[str, Any]:
        import glob

        from rag.builder.engine import Builder

        k = self._kb(kb)
        if k.readonly:
            raise ServiceError("readonly", "Packs are read-only snapshots.")
        if k.building:
            raise ServiceError("busy", "A build is running here.")
        patterns = list(paths or [])
        if doc_ids:
            with k.store() as s:
                marks = ",".join("?" * len(doc_ids))
                patterns += [glob.escape(r["display_path"]) for r in s.conn.execute(
                    f"SELECT display_path FROM documents WHERE doc_id IN ({marks})", list(doc_ids))]
        if not patterns:
            return {"removed": 0}
        removed = Builder(k.workspace()).remove(patterns)
        return {"removed": removed}

    def kb_export(self, kb: str, out: str, include_originals: bool = False, name: str | None = None) -> dict[str, Any]:
        from rag.core.db.pack import export_pack

        k = self._kb(kb)
        if k.kind != "workspace":
            raise ServiceError("readonly", "This is already a pack.")
        if k.building:
            raise ServiceError("busy", "Wait for the build to finish before exporting.")
        target = Path(out)
        if target.suffix.lower() != ".ragpack":
            target = target.with_suffix(".ragpack")

        def work(job: Job) -> Any:
            job.report({"stage": "export", "path": str(target)}, log={"message": f"Writing {target.name}"}, force=True)
            manifest = export_pack(k.workspace(), target, include_originals=include_originals, name=name or k.name)
            manifest.pop("files", None)
            return {"path": str(target), "size_bytes": target.stat().st_size, "manifest": manifest}

        return self._job("export", f"Export {target.name}", work, kb=k.id)

    def kb_merge(self, kb: str, sources: list[str], on_conflict: str = "newer", keep_copies: bool = False,
                 dry_run: bool = False) -> dict[str, Any]:
        """Copy compatible packs or workspaces into this workspace. A dry run answers at once; a merge is a job."""
        from rag.core.db.merge import MergeError, merge

        k = self._kb(kb)
        if k.readonly:
            raise ServiceError("readonly", "Packs are read-only snapshots.",
                               "Merge into a workspace (New knowledge base), then export a new pack.")
        if k.building:
            raise ServiceError("busy", "Wait for the build to finish before merging.")
        paths = [Path(p) for p in sources]
        if not paths:
            raise ServiceError("bad_params", "Choose at least one knowledge base to merge in.")
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise ServiceError("not_found", "Not found: " + ", ".join(missing[:5]))
        if dry_run:
            try:
                return merge(k.path, paths, on_conflict=on_conflict, keep_copies=keep_copies, dry_run=True)
            except MergeError as exc:
                raise ServiceError("bad_params", str(exc)) from exc

        def work(job: Job) -> Any:
            k.building = True
            self.emit("kb", {"id": k.id, "building": True})
            counters = {"files_total": 0, "files_done": 0}

            def progress(e: dict[str, Any]) -> None:
                counters["files_total"], counters["files_done"] = e["total"], e["index"]
                job.report({**counters, "current": e["path"], "stage": e["source"]},
                           log={"event": e["decision"], "path": e["path"]} if e["decision"] in ("add", "replace")
                           else None)

            try:
                res = merge(k.path, paths, on_conflict=on_conflict, keep_copies=keep_copies, progress=progress,
                            should_stop=job.cancel.is_set)
            except MergeError as exc:
                raise ServiceError("bad_params", str(exc)) from exc
            finally:
                k.building = False
                self.emit("kb", {"id": k.id, "building": False})
            return res

        return self._job("merge", f"Merge into {k.name}", work, kb=k.id)

    def kb_convert(self, kb: str, to: str) -> dict[str, Any]:
        """Move a workspace's vectors to the other store (sqlite-vec or shammy), as a job. Nothing is embedded again."""
        from rag.core.db.convert import convert_workspace
        from rag.core.db.vectors import VectorStoreError

        k = self._kb(kb)
        if k.readonly:
            raise ServiceError("readonly", "Packs are read-only snapshots.", "Convert the workspace, then export a new pack.")
        if k.building:
            raise ServiceError("busy", "Wait for the build to finish before moving the vectors.")
        if to == "shammy" and self._shammy_version() is None:
            raise ServiceError("unavailable", "This build doesn't include the Shammy engine.")

        def work(job: Job) -> Any:
            k.building = True
            self.emit("kb", {"id": k.id, "building": True})
            try:
                return convert_workspace(k.workspace(), to,
                                         progress=lambda done, total: job.report({"files_done": done, "files_total": total,
                                                                                  "stage": "convert"}))
            except VectorStoreError as exc:
                raise ServiceError("bad_params", str(exc)) from exc
            finally:
                k.building = False
                self.emit("kb", {"id": k.id, "building": False})

        return self._job("convert", f"Move vectors of {k.name} to {'Shammy' if to == 'shammy' else 'sqlite-vec'}", work, kb=k.id)

    def kb_verify(self, kb: str) -> dict[str, Any]:
        """Check a pack's checksums and contents, or a workspace's marker, databases and links."""
        from rag.core.check import check_path

        k = self._kb(kb)
        if k.building:
            raise ServiceError("busy", "Wait for the build to finish before checking.")

        def work(job: Job) -> Any:
            job.report({"stage": "verify"}, force=True)
            res = check_path(k.path)
            errors = [p["message"] for p in res["problems"] if p["level"] == "error"]
            warnings = [p["message"] for p in res["problems"] if p["level"] == "warning"]
            if errors:
                res["message"] = "; ".join(errors[:3]).capitalize() + "."
            elif k.kind == "pack":
                res["message"] = f"All files match their checksums ({res.get('checksums', '')}) and the databases are whole."
            else:
                res["message"] = "The marker, settings and databases are whole and consistent."
            if warnings:
                res["message"] += " Also: " + "; ".join(warnings[:3]) + "."
            return res

        return self._job("verify", f"Check {k.name}", work, kb=k.id)

    def kb_discover(self, folder: str | None = None) -> dict[str, Any]:
        """Workspaces and packs in a folder (default: the workspaces folder), whether opened or not."""
        from rag.core.listing import list_knowledge_bases

        f = Path(folder) if folder else self.settings.workspaces_dir
        rows = list_knowledge_bases(f, sizes=False)
        open_paths = {str(k.path) for k in self.kbs.values()}
        for r in rows:
            r["open"] = r["path"] in open_paths
        return {"folder": str(f), "items": rows}

    # ---------------------------------------------------------------------------- evaluation
    def eval_run(self, kb: str, golden: str, show: int = 6, allow_mismatch: bool = False) -> dict[str, Any]:
        from rag.core.retrieve.search import Searcher
        from rag.eval import EvalCancelled, run_eval

        k = self._kb(kb)
        golden_path = Path(golden)
        if not golden_path.is_file():
            raise ServiceError("not_found", f"{golden} not found.")
        embedder = self._embedder(k, allow_mismatch=allow_mismatch)
        cfg = k.workspace().config["search"]

        def work(job: Job) -> Any:
            def progress(e: dict[str, Any]) -> None:
                job.report({"done": e["index"] - 1, "total": e["total"], "current": e.get("id")})

            with k.store() as s:
                try:
                    report = run_eval(Searcher(s, embedder, cfg), golden_path, show=show, progress=progress,
                                      should_stop=job.cancel.is_set)
                except EvalCancelled as exc:
                    raise Cancelled("evaluation cancelled") from exc
                except (ValueError, KeyError) as exc:
                    raise ServiceError("bad_golden", f"Couldn't read {golden_path.name}: {exc}",
                                       "Each line must be JSON with question and expect.") from exc
            report["golden"] = str(golden_path)
            report["kb"] = {"name": k.name, "path": str(k.path)}
            job.report({"done": report["overall"]["n"], "total": report["overall"]["n"]}, force=True)
            return report

        return self._job("eval", f"Evaluate {golden_path.name}", work, kb=k.id)

    def eval_markdown(self, report: dict[str, Any]) -> str:
        from rag.eval import report_markdown

        return report_markdown(report)

    # ---------------------------------------------------------------------------- command line
    def cli_status(self) -> dict[str, Any]:
        from rag.app import cli_install

        out = cli_install.status()
        out["embedding_url"] = self.runtime.url()
        return out

    def cli_set_enabled(self, enabled: bool) -> dict[str, Any]:
        from rag.app import cli_install

        try:
            return cli_install.set_enabled(bool(enabled))
        except (RuntimeError, OSError) as exc:
            raise ServiceError("cli", str(exc)) from exc

    def terminal_run(self, command: str, cwd: str | None = None) -> dict[str, Any]:
        from rag.app.terminal import CommandRejected

        try:
            return self.terminal.run(command, cwd)
        except CommandRejected as exc:
            raise ServiceError("rejected", str(exc)) from exc

    def terminal_stop(self, run_id: str) -> dict[str, Any]:
        return self.terminal.stop(run_id)

    # ---------------------------------------------------------------------------- jobs
    def jobs_list(self) -> list[dict[str, Any]]:
        return [j.snapshot() for j in sorted(self.jobs.values(), key=lambda j: j.started_at, reverse=True)]

    def jobs_get(self, job: str, log_from: int = 0) -> dict[str, Any]:
        j = self.jobs.get(job)
        if j is None:
            raise ServiceError("not_found", "No such job.")
        out = j.snapshot(with_result=True)
        out["log"] = j.log[log_from:]
        return out

    def jobs_cancel(self, job: str) -> dict[str, Any]:
        j = self.jobs.get(job)
        if j is None:
            raise ServiceError("not_found", "No such job.")
        j.cancel.set()
        return j.snapshot()
