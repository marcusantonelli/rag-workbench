import { Cpu, Download, FolderOpen, HardDrive, Play, RotateCw, Square } from "lucide-react";
import { useEffect, useState } from "react";
import { call } from "../api";
import { useApp, useJob } from "../state";
import { Button, Notice, Progress, Switch } from "../components/ui";
import { basename, bytes } from "../format";
import { GGUF_FILTER, pickPaths } from "../platform";
import { PicturesPanel } from "./PicturesPanel";

const VARIANT_LABEL: Record<string, string> = {
  "cuda-12.4": "CUDA 12.4 (NVIDIA)",
  "cuda-13.5": "CUDA 13.5 (NVIDIA)",
  vulkan: "Vulkan (AMD, Intel, NVIDIA)",
  cpu: "CPU only",
};

let lastSetup: string | null = null;

export function SettingsView() {
  const { runtime, info, fail, refreshRuntime, startJob } = useApp();
  const [setupId, setSetupId] = useState<string | null>(lastSetup);
  const setup = useJob(setupId);
  const [source, setSource] = useState<"download" | "file">("download");
  const [ggufPath, setGgufPath] = useState("");
  const [modelId, setModelId] = useState("");
  const [extUrl, setExtUrl] = useState(runtime?.external_url ?? "http://127.0.0.1:8081");
  const [showLog, setShowLog] = useState(false);

  useEffect(() => {
    if (setup && setup.state !== "running") refreshRuntime();
  }, [setup?.state]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!runtime) return null;

  const configure = async (changes: Record<string, unknown>) => {
    try {
      await call("runtime.configure", changes);
      refreshRuntime();
    } catch (e) {
      fail(e);
    }
  };
  const act = async (method: string) => {
    try {
      await call(method);
      refreshRuntime();
    } catch (e) {
      fail(e);
    }
  };
  const runSetup = async (opts: Record<string, unknown>) => {
    const j = await startJob("runtime.setup", opts);
    if (j) {
      setSetupId(j.id);
      lastSetup = j.id;
    }
  };

  const managed = runtime.mode === "managed";
  const sp = setup?.progress ?? {};
  const gpuNames = runtime.hardware.gpus;
  const wantsOtherBuild = managed && runtime.variant && runtime.variant !== runtime.recommended_variant && runtime.accel !== "cpu";

  return (
    <div className="settings">
      <h1>Settings</h1>

      <section className="panel">
        <h2>Embedding model</h2>
        <p className="muted">
          Searching and building turn text into vectors with this model. It must be the model a knowledge base was built
          with; packs record which one.
        </p>

        <div className="status-grid">
          <div>
            <span className="field-label">State</span>
            <span className={`state state-${runtime.state}`}>
              {runtime.mode === "external" ? "Using your llama-server" :
                runtime.needs_setup ? "Not set up" :
                ({ stopped: "Stopped", starting: "Loading…", ready: "Ready", failed: "Failed" } as Record<string, string>)[runtime.state] ?? runtime.state}
            </span>
          </div>
          <div>
            <span className="field-label">Model</span>
            <span>{runtime.model_path ? `${basename(runtime.model_path)} (${bytes(runtime.model_size)})` : "None yet"}</span>
          </div>
          <div>
            <span className="field-label">Recorded as</span>
            <code>{runtime.model_id}</code>
          </div>
          <div>
            <span className="field-label">llama.cpp</span>
            <span>{runtime.variant ? `${runtime.llama_build}, ${VARIANT_LABEL[runtime.variant] ?? runtime.variant}` : "Not installed"}</span>
          </div>
          <div className="span2">
            <span className="field-label">Running on</span>
            <span>{runtime.device_note || "Decided when the model starts"}</span>
          </div>
        </div>
        {runtime.error && <Notice tone="error" title="The model stopped">{runtime.error}</Notice>}

        {managed && !runtime.needs_setup && (
          <div className="row">
            {runtime.state !== "ready" && runtime.state !== "starting" && (
              <Button kind="primary" icon={<Play size={16} />} onClick={() => act("runtime.start")}>Start</Button>
            )}
            {(runtime.state === "ready" || runtime.state === "starting") && (
              <Button icon={<Square size={16} />} onClick={() => act("runtime.stop")}>Stop</Button>
            )}
            {runtime.state === "ready" && (
              <Button kind="ghost" icon={<RotateCw size={16} />} onClick={async () => { await act("runtime.stop"); await act("runtime.start"); }}>Restart</Button>
            )}
            <Button kind="ghost" onClick={() => setShowLog((v) => !v)}>{showLog ? "Hide log" : "Show log"}</Button>
          </div>
        )}
        {showLog && <pre className="log">{runtime.log_tail.join("\n") || "(nothing yet)"}</pre>}
      </section>

      <PicturesPanel />

      <section className="panel">
        <h2>Hardware</h2>
        <div className="hw">
          <div className="hw-item">
            <Cpu size={18} />
            <span>
              <strong>{runtime.hardware.cpu}</strong>
              <span className="muted">
                {runtime.hardware.threads} threads{runtime.hardware.ram_gb ? `, ${runtime.hardware.ram_gb} GB memory` : ""}
              </span>
            </span>
          </div>
          {gpuNames.length === 0 && (
            <div className="hw-item"><HardDrive size={18} /><span className="muted">No GPU found</span></div>
          )}
          {gpuNames.map((g) => {
            const dev = runtime.devices?.find((d) => d.name.toLowerCase().includes(g.toLowerCase().split(" ").slice(-2).join(" ")) || g.toLowerCase().includes(d.name.toLowerCase()));
            return (
              <div className="hw-item" key={g}>
                <HardDrive size={18} />
                <span>
                  <strong>{g}</strong>
                  <span className="muted">{dev ? `${bytes(dev.free_mb * 1048576)} free of ${bytes(dev.total_mb * 1048576)} for llama.cpp (${dev.id})` : ""}</span>
                </span>
              </div>
            );
          })}
        </div>
        <p className="muted">
          Recommended build: {VARIANT_LABEL[runtime.recommended_variant]}. {runtime.recommended_reason}
        </p>
        <div className="row options">
          <Switch
            checked={runtime.accel === "cpu"}
            onChange={(v) => configure({ accel: v ? "cpu" : "auto" })}
            label="Force CPU (slower; leaves the GPU free)"
          />
          <Switch
            checked={runtime.accel === "gpu"}
            disabled={runtime.accel === "cpu"}
            onChange={(v) => configure({ accel: v ? "gpu" : "auto" })}
            label="Use the GPU even if it reports too little free memory"
          />
        </div>
        {wantsOtherBuild && (
          <Notice
            tone="info"
            title={`A ${VARIANT_LABEL[runtime.recommended_variant]} build fits this machine better`}
            action={<Button onClick={() => runSetup({ download_model: false })}>Install it</Button>}
          >
            You're using {VARIANT_LABEL[runtime.variant!]}.
          </Notice>
        )}
      </section>

      {managed && (
        <section className="panel">
          <h2>{runtime.needs_setup ? "Set up the model" : "Change the model"}</h2>
          <div className="choice-list">
            <label className={`choice ${source === "download" ? "chosen" : ""}`}>
              <input type="radio" name="src" checked={source === "download"} onChange={() => setSource("download")} />
              <span>
                <span className="choice-title">Download {runtime.default_model.id.split("/")[1].replace("-GGUF:", " ")}</span>
                <span className="choice-note">
                  {(runtime.default_model.approx_bytes / 1e9).toFixed(1)} GB from Hugging Face, plus llama.cpp for this machine (about 30 to
                  600 MB). Interrupted downloads resume.
                </span>
              </span>
            </label>
            <label className={`choice ${source === "file" ? "chosen" : ""}`}>
              <input type="radio" name="src" checked={source === "file"} onChange={() => setSource("file")} />
              <span>
                <span className="choice-title">Use a .gguf file I already have</span>
                <span className="choice-note">For offline machines. It must be the same model your packs were built with.</span>
              </span>
            </label>
          </div>
          {source === "file" && (
            <div className="form">
              <div className="path-row">
                <code className="path">{ggufPath || "No file chosen"}</code>
                <Button onClick={async () => {
                  const p = await pickPaths({ title: "Choose a GGUF model", filters: GGUF_FILTER });
                  if (p?.[0]) {
                    setGgufPath(p[0]);
                    const m = basename(p[0]).match(/^qwen3-embedding-(0\.6|4|8)b-(.+)\.gguf$/i);
                    setModelId(m ? `Qwen/Qwen3-Embedding-${m[1]}B-GGUF:${m[2].toLowerCase() === "f16" ? "f16" : m[2].toUpperCase()}` : basename(p[0]).replace(/\.gguf$/i, ""));
                  }
                }}>Choose…</Button>
              </div>
              <label className="field">
                <span className="field-label">Recorded as</span>
                <input value={modelId} onChange={(e) => setModelId(e.target.value)} />
                <span className="field-help">Packs store this name. Keep it identical to the name used when they were built.</span>
              </label>
            </div>
          )}
          <div className="row end">
            <Button
              kind="primary"
              icon={<Download size={16} />}
              disabled={setup?.state === "running" || (source === "file" && !ggufPath)}
              onClick={() => runSetup(source === "download" ? { download_model: true } : { download_model: false, model_path: ggufPath, model_id: modelId })}
            >
              {setup?.state === "running" ? "Setting up…" : source === "download" ? "Download and start" : "Use this file"}
            </Button>
          </div>
          {setup?.state === "running" && (
            <div className="setup-progress">
              <Progress value={sp.stage === "download" ? sp.done : undefined} total={sp.stage === "download" ? sp.total : undefined} label="Setup progress" />
              <span className="muted">
                {sp.stage === "download"
                  ? `Downloading ${sp.label}: ${bytes(sp.done)}${sp.total ? ` of ${bytes(sp.total)}` : ""}`
                  : sp.stage === "verify" ? `Checking ${sp.label}` : sp.message ?? sp.stage ?? "Starting"}
              </span>
              <Button kind="ghost" onClick={() => call("jobs.cancel", { job: setup.id })}>Cancel</Button>
            </div>
          )}
          {setup?.state === "failed" && <Notice tone="error" title={setup.error?.message ?? "Setup failed"}>{setup.error?.hint}</Notice>}
          {setup?.state === "done" && <Notice tone="success" title="The embedding model is ready" />}
        </section>
      )}

      <section className="panel">
        <h2>Your own llama-server</h2>
        <p className="muted">Point the app at a llama-server you run yourself, for example on a GPU machine on the network.</p>
        <div className="row options">
          <Switch checked={!managed} onChange={(v) => configure({ mode: v ? "external" : "managed", url: extUrl })} label="Use my own llama-server" />
        </div>
        {!managed && (
          <div className="path-row">
            <input value={extUrl} onChange={(e) => setExtUrl(e.target.value)} aria-label="llama-server URL" />
            <Button onClick={() => configure({ url: extUrl })}>Save</Button>
          </div>
        )}
      </section>

      {info && (
        <section className="panel">
          <h2>About</h2>
          <p>RAG Workbench {info.version}</p>
          <div className="path-row">
            <code className="path">{info.data_dir}</code>
            <Button icon={<FolderOpen size={16} />} onClick={() => call("app.open_path", { path: info.data_dir }).catch(fail)}>Open data folder</Button>
          </div>
        </section>
      )}
    </div>
  );
}
