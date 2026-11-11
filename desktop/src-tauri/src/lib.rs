//! RAG Workbench shell: a window around the web UI, plus the engine process.
//!
//! The engine (Python, bundled with PyInstaller) speaks JSON lines over stdio:
//! requests `{"id", "method", "params"}`, responses `{"id", "result" | "error"}`, and
//! unsolicited `{"event", "data"}` messages, which are forwarded to the UI as
//! "engine-event". The UI calls the engine through the `engine_call` command.

use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{channel, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tauri::{AppHandle, Emitter, Manager, RunEvent, State};

#[cfg(windows)]
use std::os::windows::process::CommandExt;

const CALL_TIMEOUT: Duration = Duration::from_secs(15 * 60);

struct Engine {
    app: AppHandle,
    stdin: Mutex<Option<ChildStdin>>,
    child: Mutex<Option<Child>>,
    pending: Mutex<HashMap<u64, Sender<Value>>>,
    next_id: AtomicU64,
    alive: AtomicBool,
    stderr_tail: Mutex<Vec<String>>,
}

fn engine_command(app: &AppHandle) -> Result<Command, String> {
    // Development: RAG_ENGINE_PYTHON=/path/to/python runs `python -m rag.app --stdio`.
    if let Ok(python) = std::env::var("RAG_ENGINE_PYTHON") {
        let mut cmd = Command::new(python);
        cmd.args(["-m", "rag.app", "--stdio"]);
        return Ok(cmd);
    }
    let exe_name = if cfg!(windows) { "rag-engine.exe" } else { "rag-engine" };
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(dir) = app.path().resource_dir() {
        candidates.push(dir.join("engine").join(exe_name));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join("engine").join(exe_name));
        }
    }
    for c in &candidates {
        if c.exists() {
            let mut cmd = Command::new(c);
            cmd.arg("--stdio");
            return Ok(cmd);
        }
    }
    if cfg!(debug_assertions) {
        // `cargo tauri dev` without RAG_ENGINE_PYTHON: try the Python on PATH.
        let mut cmd = Command::new(if cfg!(windows) { "python" } else { "python3" });
        cmd.args(["-m", "rag.app", "--stdio"]);
        return Ok(cmd);
    }
    Err(format!(
        "The engine is missing (looked for {}). Reinstall RAG Workbench.",
        candidates.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")
    ))
}

impl Engine {
    fn new(app: AppHandle) -> Arc<Self> {
        Arc::new(Engine {
            app,
            stdin: Mutex::new(None),
            child: Mutex::new(None),
            pending: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
            alive: AtomicBool::new(false),
            stderr_tail: Mutex::new(Vec::new()),
        })
    }

    fn start(self: &Arc<Self>) -> Result<(), String> {
        let mut cmd = engine_command(&self.app)?;
        cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
        cmd.env("PYTHONUNBUFFERED", "1").env("PYTHONIOENCODING", "utf-8");
        #[cfg(windows)]
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
        let mut child = cmd.spawn().map_err(|e| format!("Couldn't start the engine: {e}"))?;
        let stdout = child.stdout.take().ok_or("engine stdout unavailable")?;
        let stderr = child.stderr.take().ok_or("engine stderr unavailable")?;
        *self.stdin.lock().unwrap() = child.stdin.take();
        *self.child.lock().unwrap() = Some(child);
        self.alive.store(true, Ordering::SeqCst);

        let me = Arc::clone(self);
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                let Ok(msg) = serde_json::from_str::<Value>(&line) else { continue };
                if let Some(id) = msg.get("id").and_then(Value::as_u64) {
                    if let Some(tx) = me.pending.lock().unwrap().remove(&id) {
                        let _ = tx.send(msg);
                    }
                } else if msg.get("event").is_some() {
                    let _ = me.app.emit("engine-event", msg);
                }
            }
            me.alive.store(false, Ordering::SeqCst);
            let tail = me.stderr_tail.lock().unwrap().join("\n");
            for (_, tx) in me.pending.lock().unwrap().drain() {
                let _ = tx.send(json!({"error": {"code": "engine_exit", "message": "The engine stopped.", "hint": tail}}));
            }
            let _ = me.app.emit("engine-event", json!({"event": "engine_exit", "data": {"stderr": tail}}));
        });

        let me = Arc::clone(self);
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines() {
                let Ok(line) = line else { break };
                eprintln!("[engine] {line}");
                let mut tail = me.stderr_tail.lock().unwrap();
                tail.push(line);
                if tail.len() > 60 {
                    let drop = tail.len() - 60;
                    tail.drain(..drop);
                }
            }
        });
        Ok(())
    }

    fn call(self: &Arc<Self>, method: &str, params: Value) -> Result<Value, Value> {
        if !self.alive.load(Ordering::SeqCst) {
            // First call, or the engine died: (re)start it.
            self.start().map_err(|m| json!({"code": "engine_start", "message": m}))?;
        }
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = channel();
        self.pending.lock().unwrap().insert(id, tx);
        let line = json!({"id": id, "method": method, "params": params}).to_string() + "\n";
        {
            let mut guard = self.stdin.lock().unwrap();
            let Some(stdin) = guard.as_mut() else {
                self.pending.lock().unwrap().remove(&id);
                return Err(json!({"code": "engine_exit", "message": "The engine isn't running."}));
            };
            if stdin.write_all(line.as_bytes()).and_then(|_| stdin.flush()).is_err() {
                self.pending.lock().unwrap().remove(&id);
                self.alive.store(false, Ordering::SeqCst);
                return Err(json!({"code": "engine_exit", "message": "The engine stopped; try again."}));
            }
        }
        match rx.recv_timeout(CALL_TIMEOUT) {
            Ok(msg) => {
                if let Some(err) = msg.get("error") {
                    Err(err.clone())
                } else {
                    Ok(msg.get("result").cloned().unwrap_or(Value::Null))
                }
            }
            Err(RecvTimeoutError::Timeout) => {
                self.pending.lock().unwrap().remove(&id);
                Err(json!({"code": "timeout", "message": format!("{method} took too long.")}))
            }
            Err(RecvTimeoutError::Disconnected) => Err(json!({"code": "engine_exit", "message": "The engine stopped."})),
        }
    }

    fn shutdown(&self) {
        if let Some(stdin) = self.stdin.lock().unwrap().as_mut() {
            let _ = stdin.write_all(b"{\"id\": 0, \"method\": \"app.shutdown\"}\n");
            let _ = stdin.flush();
        }
        *self.stdin.lock().unwrap() = None; // EOF also ends the engine
        if let Some(mut child) = self.child.lock().unwrap().take() {
            for _ in 0..50 {
                if let Ok(Some(_)) = child.try_wait() {
                    return;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

#[tauri::command]
async fn engine_call(engine: State<'_, Arc<Engine>>, method: String, params: Option<Value>) -> Result<Value, Value> {
    let engine = Arc::clone(engine.inner());
    tauri::async_runtime::spawn_blocking(move || engine.call(&method, params.unwrap_or_else(|| json!({}))))
        .await
        .map_err(|e| json!({"code": "error", "message": e.to_string()}))?
}

/// .ragpack files the app was launched with (double-click / "Open with").
/// Stop the engine and the model servers it runs, before an update replaces their files. The next
/// engine call starts it again (if the update is cancelled or fails).
#[tauri::command]
async fn engine_stop(engine: State<'_, Arc<Engine>>) -> Result<(), String> {
    let engine = Arc::clone(engine.inner());
    tauri::async_runtime::spawn_blocking(move || engine.shutdown()).await.map_err(|e| e.to_string())
}

#[tauri::command]
fn startup_paths() -> Vec<String> {
    std::env::args()
        .skip(1)
        .filter(|a| a.to_lowercase().ends_with(".ragpack") && std::path::Path::new(a).is_file())
        .collect()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_process::init())
        .setup(|app| {
            let engine = Engine::new(app.handle().clone());
            if let Err(e) = engine.start() {
                eprintln!("{e}");
            }
            app.manage(engine);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![engine_call, engine_stop, startup_paths])
        .build(tauri::generate_context!())
        .expect("error while building RAG Workbench");

    app.run(|handle, event| {
        if let RunEvent::Exit = event {
            if let Some(engine) = handle.try_state::<Arc<Engine>>() {
                engine.shutdown();
            }
        }
    });
}
