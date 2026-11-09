"""The embedding runtime the desktop app manages: llama.cpp's llama-server plus the GGUF model.

Hardware selection ("auto"):
  * Windows + NVIDIA GPU  -> the CUDA build (13.5 with driver >= 580, else 12.4)
  * any other GPU         -> the Vulkan build (AMD, Intel, NVIDIA on Linux; integrated GPUs too)
  * no GPU                -> the CPU build
After download, ``llama-server --list-devices`` reports what the build can really use. If no
GPU shows up, or the GPU doesn't have room for the model, the server runs on the CPU and the
app says why. The "Force CPU" switch (accel = "cpu") runs the same build with no offload.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from collections import deque
from pathlib import Path
from typing import Any, Callable

from rag.app.paths import Settings

PINNED_LLAMA_BUILD = "b12207"
RELEASE_DOWNLOAD = "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{asset}"
RELEASE_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{tag}"

DEFAULT_MODEL = {
    "id": "Qwen/Qwen3-Embedding-8B-GGUF:Q4_K_M",
    "file": "Qwen3-Embedding-8B-Q4_K_M.gguf",
    "url": "https://huggingface.co/Qwen/Qwen3-Embedding-8B-GGUF/resolve/main/Qwen3-Embedding-8B-Q4_K_M.gguf",
    "approx_bytes": 4_680_000_000,
}

USER_AGENT = "rag-workbench (llama.cpp and model downloader)"
_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
_DEVICE_RE = re.compile(r"^\s*([A-Za-z]+\d*):\s*(.+?)\s*\((\d+)\s*MiB,\s*(\d+)\s*MiB free\)")

ProgressFn = Callable[[dict[str, Any]], None]


class Cancelled(RuntimeError):
    pass


class RuntimeErrorWithHint(RuntimeError):
    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.hint = hint


# ----------------------------------------------------------------------------------------------
# Hardware
# ----------------------------------------------------------------------------------------------
def _run(cmd: list[str], timeout: float = 15) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, creationflags=_NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _cpu_name() -> str:
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if sys.platform == "win32":
        out = _run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"])
        if out and out.strip():
            return out.strip().splitlines()[0]
    return platform.processor() or platform.machine()


def _ram_gb() -> float | None:
    try:
        if sys.platform.startswith("linux"):
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1024 / 1024, 1)
        if sys.platform == "win32":
            import ctypes

            class MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = MemStatus()
            st.dwLength = ctypes.sizeof(MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))  # type: ignore[attr-defined]
            return round(st.ullTotalPhys / 1024 ** 3, 1)
    except (OSError, ValueError, AttributeError):
        pass
    return None


def _nvidia() -> list[dict[str, Any]]:
    exe = shutil.which("nvidia-smi")
    if exe is None and sys.platform == "win32":
        cand = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        exe = str(cand) if cand.exists() else None
    if exe is None:
        return []
    out = _run([exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"])
    gpus = []
    for line in (out or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                gpus.append({"name": parts[0], "memory_mb": int(float(parts[1])), "driver": parts[2]})
            except ValueError:
                continue
    return gpus


_PCI_VENDORS = {"0x10de": "NVIDIA", "0x1002": "AMD", "0x1022": "AMD", "0x8086": "Intel", "0x5143": "Qualcomm"}


def _other_gpus() -> list[str]:
    names: list[str] = []
    if sys.platform == "win32":
        out = _run(["powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }"])
        names = [ln.strip() for ln in (out or "").splitlines() if ln.strip()]
        # Remote desktop / basic display adapters can't run compute.
        names = [n for n in names if not re.search(r"basic display|remote display|hyper-v|virtual", n, re.I)]
    elif sys.platform.startswith("linux"):
        out = _run(["lspci", "-mm"])
        if out:
            for line in out.splitlines():
                if re.search(r'"(VGA compatible controller|3D controller|Display controller)"', line):
                    fields = re.findall(r'"([^"]*)"', line)
                    if len(fields) >= 3:
                        names.append(f"{fields[1]} {fields[2]}".strip())
        else:
            for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
                vendor = card / "device" / "vendor"
                if vendor.exists() and "-" not in card.name:
                    v = vendor.read_text().strip()
                    names.append(f"{_PCI_VENDORS.get(v, v)} GPU ({card.name})")
        names = [n for n in names if not re.search(r"virtio|vmware|cirrus|bochs|qxl|matrox|aspeed", n, re.I)]
    return names


def detect_hardware() -> dict[str, Any]:
    arch = platform.machine().lower()
    arch = {"amd64": "x64", "x86_64": "x64", "aarch64": "arm64"}.get(arch, arch)
    os_name = "windows" if sys.platform == "win32" else "linux" if sys.platform.startswith("linux") else sys.platform
    nvidia = _nvidia()
    gpus = _other_gpus()
    for g in nvidia:
        if not any(g["name"].lower() in n.lower() or n.lower() in g["name"].lower() for n in gpus):
            gpus.append(g["name"])
    return {"os": os_name, "arch": arch, "cpu": _cpu_name(), "threads": os.cpu_count(), "ram_gb": _ram_gb(),
            "nvidia": nvidia, "gpus": gpus}


def choose_variant(hw: dict[str, Any], accel: str) -> tuple[str, str]:
    """Which llama.cpp build to install, and why."""
    if accel == "cpu" and not hw["gpus"]:
        return "cpu", "Force CPU is on and no GPU was found."
    if hw["os"] == "windows" and hw["nvidia"]:
        driver = hw["nvidia"][0].get("driver", "0")
        try:
            major = int(driver.split(".")[0])
        except ValueError:
            major = 0
        if hw["arch"] == "x64":
            cuda = "cuda-13.5" if major >= 580 else "cuda-12.4"
            return cuda, f"NVIDIA {hw['nvidia'][0]['name']} (driver {driver}): CUDA build."
    if hw["gpus"]:
        return "vulkan", f"{hw['gpus'][0]}: Vulkan build (falls back to CPU if the GPU can't be used)."
    if accel == "gpu":
        return "vulkan", "GPU requested but none detected; trying the Vulkan build."
    return "cpu", "No GPU detected: CPU build."


def release_assets(tag: str, variant: str, os_name: str, arch: str) -> list[str]:
    """Archive names for one llama.cpp build (the CUDA runtime DLLs come separately)."""
    if os_name == "windows":
        if variant.startswith("cuda-"):
            ver = variant.split("-", 1)[1]
            return [f"llama-{tag}-bin-win-cuda-{ver}-{arch}.zip", f"cudart-llama-bin-win-cuda-{ver}-{arch}.zip"]
        return [f"llama-{tag}-bin-win-{variant}-{arch}.zip"]
    if os_name == "linux":
        if variant == "vulkan":
            return [f"llama-{tag}-bin-ubuntu-vulkan-{arch}.tar.gz"]
        return [f"llama-{tag}-bin-ubuntu-{arch}.tar.gz"]
    raise RuntimeErrorWithHint(f"No prebuilt llama.cpp for {os_name}",
                               "Install llama-server yourself and choose 'Use my own llama-server' in Settings.")


# ----------------------------------------------------------------------------------------------
# Downloads
# ----------------------------------------------------------------------------------------------
def _open(url: str, *, headers: dict[str, str] | None = None, method: str = "GET", timeout: float = 60):
    req = urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT, **(headers or {})})
    return urllib.request.urlopen(req, timeout=timeout)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # noqa: D401
        return None


def expected_sha256(url: str) -> str | None:
    """The published SHA-256 of a download, when the host states one."""
    try:
        if "huggingface.co" in url:
            opener = urllib.request.build_opener(_NoRedirect)
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
            try:
                resp = opener.open(req, timeout=30)
                headers = resp.headers
            except urllib.error.HTTPError as exc:   # the 302 to the CDN carries the headers
                headers = exc.headers
            etag = (headers.get("X-Linked-ETag") or headers.get("ETag") or "").strip('"')
            return etag.lower() if re.fullmatch(r"[0-9a-fA-F]{64}", etag) else None
        m = re.search(r"/releases/download/([^/]+)/([^/]+)$", url)
        if m and "github.com" in url:
            with _open(RELEASE_API.format(tag=m.group(1)), headers={"Accept": "application/vnd.github+json"},
                       timeout=30) as resp:
                data = json.load(resp)
            for asset in data.get("assets", []):
                if asset.get("name") == m.group(2) and str(asset.get("digest", "")).startswith("sha256:"):
                    return asset["digest"].split(":", 1)[1].lower()
    except (OSError, ValueError, urllib.error.URLError):
        return None
    return None


def download(url: str, dest: Path, *, progress: ProgressFn | None = None, should_stop: Callable[[], bool] | None = None,
             label: str = "", expected_bytes: int | None = None, verify: bool = True) -> Path:
    """Download with resume (a .part file survives an interrupted run) and SHA-256 check."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    try:
        resp = _open(url, headers=headers, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and have:          # already complete
            resp = None
        else:
            raise RuntimeErrorWithHint(f"Download failed ({exc.code}) for {url}",
                                       "Check the internet connection or proxy (HTTPS_PROXY), then retry.") from exc
    except urllib.error.URLError as exc:
        raise RuntimeErrorWithHint(f"Can't reach {url.split('/')[2]}: {exc.reason}",
                                   "Check the internet connection or proxy (HTTPS_PROXY), then retry.") from exc
    if resp is not None:
        with resp:
            if have and resp.status != 206:     # server ignored the range: start over
                have = 0
            length = resp.headers.get("Content-Length")
            total = (have + int(length)) if length else expected_bytes
            mode = "ab" if have else "wb"
            done = have
            last = 0.0
            with open(part, mode) as fh:
                while True:
                    if should_stop and should_stop():
                        raise Cancelled("download cancelled")
                    block = resp.read(1 << 20)
                    if not block:
                        break
                    fh.write(block)
                    done += len(block)
                    now = time.monotonic()
                    if progress and now - last > 0.25:
                        last = now
                        progress({"stage": "download", "label": label, "done": done, "total": total})
            if progress:
                progress({"stage": "download", "label": label, "done": done, "total": total or done})
    if verify:
        want = expected_sha256(url)
        if want:
            if progress:
                progress({"stage": "verify", "label": label})
            h = hashlib.sha256()
            with open(part, "rb") as fh:
                for block in iter(lambda: fh.read(1 << 22), b""):
                    if should_stop and should_stop():
                        raise Cancelled("cancelled")
                    h.update(block)
            if h.hexdigest() != want:
                part.unlink(missing_ok=True)
                raise RuntimeErrorWithHint(f"Checksum mismatch for {dest.name}; the download was discarded.",
                                           "Retry the download.")
    part.replace(dest)
    return dest


def _extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                target = (dest / member).resolve()
                if root not in target.parents and target != root:
                    raise RuntimeErrorWithHint(f"unsafe path in {archive.name}: {member}")
            zf.extractall(dest)
    else:
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                target = (dest / member.name).resolve()
                if root not in target.parents and target != root:
                    raise RuntimeErrorWithHint(f"unsafe path in {archive.name}: {member.name}")
            tf.extractall(dest, filter="data") if hasattr(tarfile, "data_filter") else tf.extractall(dest)


def find_server(folder: Path) -> Path | None:
    name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    hits = sorted(folder.rglob(name), key=lambda p: len(p.parts))
    return hits[0] if hits else None


def _server_env(exe: Path) -> dict[str, str]:
    env = dict(os.environ)
    if sys.platform.startswith("linux"):
        env["LD_LIBRARY_PATH"] = str(exe.parent) + (os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    elif sys.platform == "win32":
        env["PATH"] = str(exe.parent) + os.pathsep + env.get("PATH", "")
    return env


def list_devices(exe: Path) -> tuple[list[dict[str, Any]], str]:
    """GPUs llama.cpp can use with this build: [{id, name, total_mb, free_mb}], plus raw output."""
    try:
        out = subprocess.run([str(exe), "--list-devices"], capture_output=True, text=True, timeout=60,
                             env=_server_env(exe), creationflags=_NO_WINDOW)
    except (OSError, subprocess.SubprocessError) as exc:
        return [], f"could not run llama-server: {exc}"
    text = (out.stdout or "") + (out.stderr or "")
    devices = []
    for line in text.splitlines():
        m = _DEVICE_RE.match(line)
        if m:
            devices.append({"id": m.group(1), "name": m.group(2), "total_mb": int(m.group(3)),
                            "free_mb": int(m.group(4))})
    return devices, text.strip()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ----------------------------------------------------------------------------------------------
# The managed server
# ----------------------------------------------------------------------------------------------
class Runtime:
    def __init__(self, settings: Settings, emit: Callable[[str, dict[str, Any]], None] | None = None):
        self.settings = settings
        self.emit = emit or (lambda name, data: None)
        self.root = settings.root / "runtime"
        self.models_dir = settings.root / "models"
        self.root.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._hw: dict[str, Any] | None = None
        self._devices: list[dict[str, Any]] | None = None
        self._proc: subprocess.Popen | None = None
        self._log: deque[str] = deque(maxlen=400)
        self._state = "stopped"          # stopped | starting | ready | failed
        self._error: str | None = None
        self._device_note = ""
        self._url: str | None = None
        self._lock = threading.RLock()
        self._kill_stale()

    # -- facts ------------------------------------------------------------------------------
    @property
    def cfg(self) -> dict[str, Any]:
        return self.settings.runtime

    def hardware(self, refresh: bool = False) -> dict[str, Any]:
        if self._hw is None or refresh:
            self._hw = detect_hardware()
        return self._hw

    @property
    def build_tag(self) -> str:
        return self.cfg.get("llama_build") or PINNED_LLAMA_BUILD

    def build_dir(self, variant: str) -> Path:
        return self.root / f"llama-{self.build_tag}-{variant}"

    def installed_variant(self) -> str | None:
        v = self.cfg.get("variant")
        return v if v and find_server(self.build_dir(v)) else None

    def server_exe(self) -> Path | None:
        v = self.installed_variant()
        return find_server(self.build_dir(v)) if v else None

    def model_path(self) -> Path | None:
        p = self.cfg.get("model_path")
        return Path(p) if p and Path(p).is_file() else None

    def url(self) -> str | None:
        if self.cfg.get("mode") == "external":
            return self.cfg.get("url")
        return self._url if self._state == "ready" else None

    def status(self) -> dict[str, Any]:
        hw = self.hardware()
        recommended, why = choose_variant(hw, self.cfg.get("accel", "auto"))
        model = self.model_path()
        managed = self.cfg.get("mode") != "external"
        return {
            "mode": self.cfg.get("mode"),
            "state": self._state if managed else "external",
            "ready": self.url() is not None,
            "url": self.url(),
            "error": self._error,
            "accel": self.cfg.get("accel", "auto"),
            "variant": self.installed_variant(),
            "recommended_variant": recommended,
            "recommended_reason": why,
            "llama_build": self.build_tag,
            "devices": self._devices,
            "device_note": self._device_note,
            "model_path": str(model) if model else None,
            "model_id": self.cfg.get("model_id"),
            "model_size": model.stat().st_size if model else None,
            "default_model": DEFAULT_MODEL,
            "needs_setup": managed and (self.installed_variant() is None or model is None),
            "hardware": hw,
            "external_url": self.cfg.get("url"),
            "log_tail": list(self._log)[-60:],
        }

    # -- setup ------------------------------------------------------------------------------
    def setup(self, *, download_model: bool, model_path: str | None, model_id: str | None,
              progress: ProgressFn, should_stop: Callable[[], bool]) -> dict[str, Any]:
        hw = self.hardware(refresh=True)
        variant, why = choose_variant(hw, self.cfg.get("accel", "auto"))
        progress({"stage": "plan", "message": why})
        if find_server(self.build_dir(variant)) is None:
            self._install_build(variant, hw, progress, should_stop)
        self.settings.update_runtime(variant=variant)
        exe = find_server(self.build_dir(variant))
        if exe is None:
            raise RuntimeErrorWithHint("llama-server was not found in the downloaded build.")
        progress({"stage": "devices", "message": "Checking which devices llama.cpp can use"})
        self._devices, _ = list_devices(exe)

        if model_path:
            p = Path(model_path)
            if not p.is_file() or p.suffix.lower() != ".gguf":
                raise RuntimeErrorWithHint(f"{model_path} is not a .gguf file.")
            self.settings.update_runtime(model_path=str(p), model_id=model_id or self._guess_model_id(p))
        elif download_model:
            dest = self.models_dir / DEFAULT_MODEL["file"]
            if not dest.exists():
                download(DEFAULT_MODEL["url"], dest, progress=progress, should_stop=should_stop,
                         label=DEFAULT_MODEL["file"], expected_bytes=DEFAULT_MODEL["approx_bytes"])
            self.settings.update_runtime(model_path=str(dest), model_id=DEFAULT_MODEL["id"])
        if self.model_path() is None:
            raise RuntimeErrorWithHint("No embedding model yet.", "Download the default model or choose a .gguf file.")
        progress({"stage": "start", "message": "Starting llama-server"})
        self.start(wait=True, should_stop=should_stop)
        return self.status()

    @staticmethod
    def _guess_model_id(path: Path) -> str:
        """Name recorded in packs built with this file; matches the CLI's default for the same model."""
        m = re.match(r"(?i)qwen3-embedding-(0\.6|4|8)b-(.+)\.gguf$", path.name)
        if m:
            return f"Qwen/Qwen3-Embedding-{m.group(1)}B-GGUF:{m.group(2).upper() if m.group(2).lower() != 'f16' else 'f16'}"
        return path.stem

    def _install_build(self, variant: str, hw: dict[str, Any], progress: ProgressFn,
                       should_stop: Callable[[], bool]) -> None:
        tag = self.build_tag
        target = self.build_dir(variant)
        staging = self.root / f".staging-{tag}-{variant}"
        shutil.rmtree(staging, ignore_errors=True)
        downloads = self.root / "downloads"
        for asset in release_assets(tag, variant, hw["os"], hw["arch"]):
            archive = downloads / asset
            if not archive.exists():
                download(RELEASE_DOWNLOAD.format(tag=tag, asset=asset), archive, progress=progress,
                         should_stop=should_stop, label=asset)
            progress({"stage": "extract", "label": asset})
            _extract(archive, staging)
        exe = find_server(staging)
        if exe is None:
            raise RuntimeErrorWithHint(f"llama-server is missing from the {variant} build.")
        if sys.platform != "win32":
            for f in exe.parent.iterdir():
                if f.is_file() and (f.name.startswith("llama-") or f.suffix == ""):
                    f.chmod(f.stat().st_mode | 0o111)
        shutil.rmtree(target, ignore_errors=True)
        staging.replace(target)

    # -- process ----------------------------------------------------------------------------
    def _pidfile(self) -> Path:
        return self.root / "llama-server.pid"

    def _kill_stale(self) -> None:
        """Stop a llama-server left behind by an app that crashed."""
        pf = self._pidfile()
        if not pf.exists():
            return
        try:
            pid = int(pf.read_text().strip())
        except ValueError:
            pf.unlink(missing_ok=True)
            return
        try:
            if sys.platform.startswith("linux"):
                cmd = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="ignore")
                if "llama-server" in cmd:
                    os.kill(pid, 15)
            elif sys.platform == "win32":
                out = _run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"]) or ""
                if "llama-server" in out.lower():
                    _run(["taskkill", "/PID", str(pid), "/T", "/F"])
        except (OSError, ProcessLookupError, PermissionError):
            pass
        pf.unlink(missing_ok=True)

    def _gpu_plan(self, model: Path) -> tuple[bool, str]:
        if self.cfg.get("accel") == "cpu":
            return False, "Force CPU is on."
        devices = self._devices
        if devices is None:
            exe = self.server_exe()
            devices = list_devices(exe)[0] if exe else []
            self._devices = devices
        if not devices:
            return False, "llama.cpp found no usable GPU with this build: running on the CPU."
        need_mb = model.stat().st_size // (1024 * 1024) + 1536
        best = max(devices, key=lambda d: d["free_mb"])
        if best["free_mb"] < need_mb and self.cfg.get("accel") != "gpu":
            return False, (f"{best['name']} has {best['free_mb']} MiB free; the model needs about {need_mb} MiB: "
                           "running on the CPU. Turn Force GPU on in Settings to try anyway.")
        return True, f"Running on {best['name']} ({best['id']})."

    def start(self, *, wait: bool = False, should_stop: Callable[[], bool] | None = None) -> dict[str, Any]:
        with self._lock:
            if self.cfg.get("mode") == "external":
                return self.status()
            if self._proc is not None and self._proc.poll() is None:
                return self.status()
            exe, model = self.server_exe(), self.model_path()
            if exe is None or model is None:
                self._state, self._error = "stopped", None
                return self.status()
            use_gpu, note = self._gpu_plan(model)
            self._device_note = note
            port = int(self.cfg.get("port") or 0) or _free_port()
            ctx = str(int(self.cfg.get("ctx") or 8192))
            cmd = [str(exe), "-m", str(model), "--embedding", "--pooling", "last", "-c", ctx, "-b", ctx, "-ub", ctx,
                   "--host", "127.0.0.1", "--port", str(port)]
            cmd += ["-ngl", "99"] if use_gpu else ["-ngl", "0", "--device", "none"]
            self._log.clear()
            self._log.append("$ " + " ".join(cmd))
            self._state, self._error, self._url = "starting", None, None
            self.emit("runtime", self.status())
            try:
                self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                              encoding="utf-8", errors="replace", env=_server_env(exe),
                                              creationflags=_NO_WINDOW, bufsize=1)
            except OSError as exc:
                self._state, self._error = "failed", f"Could not start llama-server: {exc}"
                self.emit("runtime", self.status())
                return self.status()
            self._pidfile().write_text(str(self._proc.pid))
            threading.Thread(target=self._pump, args=(self._proc,), daemon=True).start()
            threading.Thread(target=self._watch, args=(self._proc, f"http://127.0.0.1:{port}"), daemon=True).start()
        if wait:
            deadline = time.monotonic() + 600
            while self._state == "starting" and time.monotonic() < deadline:
                if should_stop and should_stop():
                    self.stop()
                    raise Cancelled("cancelled")
                time.sleep(0.25)
            if self._state != "ready":
                raise RuntimeErrorWithHint(self._error or "llama-server did not become ready.",
                                           "See the server log in Settings.")
        return self.status()

    def _pump(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            self._log.append(line.rstrip())

    def _watch(self, proc: subprocess.Popen, url: str) -> None:
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            try:
                with _open(url + "/health", timeout=5) as resp:
                    if resp.status == 200:
                        with self._lock:
                            if self._proc is proc:
                                self._state, self._url = "ready", url
                        self.emit("runtime", self.status())
                        break
            except (urllib.error.URLError, OSError, ValueError):
                pass
            time.sleep(0.5)
        else:
            with self._lock:
                if self._proc is proc:
                    self._state, self._error = "failed", "llama-server did not become ready in 10 minutes."
            self.emit("runtime", self.status())
            return
        if proc.poll() is not None or self._state != "ready":
            # Exited before (or instead of) becoming ready.
            if self._proc is proc and self._state != "stopped":
                tail = " | ".join(list(self._log)[-6:])
                self._state = "failed"
                self._error = f"llama-server exited (code {proc.returncode}). {tail}"[:800]
                self.emit("runtime", self.status())
            return
        proc.wait()
        with self._lock:
            if self._proc is proc and self._state == "ready":
                self._state, self._url = "failed", None
                self._error = f"llama-server stopped unexpectedly (code {proc.returncode})."
                self.emit("runtime", self.status())

    def stop(self) -> dict[str, Any]:
        with self._lock:
            proc, self._proc = self._proc, None
            self._state, self._url, self._error = "stopped", None, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._pidfile().unlink(missing_ok=True)
        self.emit("runtime", self.status())
        return self.status()

    def restart(self) -> dict[str, Any]:
        self.stop()
        return self.start()

    def configure(self, **changes: Any) -> dict[str, Any]:
        allowed = {"accel", "mode", "url", "ctx", "port", "autostart", "llama_build", "model_id"}
        bad = set(changes) - allowed
        if bad:
            raise ValueError(f"unknown runtime settings: {', '.join(sorted(bad))}")
        if "accel" in changes and changes["accel"] not in ("auto", "gpu", "cpu"):
            raise ValueError("accel must be auto, gpu or cpu")
        if "mode" in changes and changes["mode"] not in ("managed", "external"):
            raise ValueError("mode must be managed or external")
        running = self._proc is not None and self._proc.poll() is None
        self.settings.update_runtime(**changes)
        if "llama_build" in changes:
            self._devices = None
        if running and set(changes) & {"accel", "ctx", "port", "mode", "llama_build"}:
            self.stop()
            if self.cfg.get("mode") == "managed":
                self.start()
        return self.status()
