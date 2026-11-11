import { Download, FileUp, Play, Square } from "lucide-react";
import { useEffect, useState } from "react";
import { call, onEvent } from "../api";
import { useApp, useJob } from "../state";
import { Button, Notice, Progress, Switch } from "../components/ui";
import { bytes } from "../format";
import { GGUF_FILTER, pickPaths } from "../platform";

export interface HelperStatus {
  kind: "vision" | "rerank";
  enabled: boolean;
  state: "stopped" | "starting" | "ready" | "failed";
  ready: boolean;
  url: string | null;
  error: string | null;
  model_id: string | null;
  size_bytes: number | null;
  needs_setup: boolean;
  choices: { id: string; label: string; note: string; bytes: number; downloaded: boolean }[];
  own_file: boolean;
  device_note: string;
  log_tail: string[];
}

/** Status of a helper model (vision | rerank), kept current from engine events. */
export function useHelper(kind: "vision" | "rerank") {
  const [st, setSt] = useState<HelperStatus | null>(null);
  const refresh = () => call<HelperStatus>(`${kind}.status`).then(setSt).catch(() => {});
  useEffect(() => {
    refresh();
    return onEvent((name, data) => {
      if (name === kind) setSt(data as HelperStatus);
    });
  }, [kind]); // eslint-disable-line react-hooks/exhaustive-deps
  return [st, setSt, refresh] as const;
}

const STATE: Record<string, string> = { stopped: "Stopped", starting: "Loading…", ready: "Ready", failed: "Failed" };
const lastSetup: Record<string, string | null> = {};

/** Settings: an optional model that runs next to the embedding model (picture descriptions, reranking). */
export function HelperPanel({ kind, title, children, toggle }: {
  kind: "vision" | "rerank";
  title: string;
  children: React.ReactNode;
  toggle: string;
}) {
  const { fail, startJob } = useApp();
  const [st, setSt, refresh] = useHelper(kind);
  const [setupId, setSetupId] = useState<string | null>(lastSetup[kind] ?? null);
  const setup = useJob(setupId);
  const [pick, setPick] = useState<string>("");
  const [showLog, setShowLog] = useState(false);

  useEffect(() => {
    if (setup && setup.state !== "running") refresh();
  }, [setup?.state]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!st) return null;

  const current = st.choices.find((c) => c.id === st.model_id);
  const selected = pick || st.model_id || st.choices[0]?.id || "";
  const choice = st.choices.find((c) => c.id === selected);
  const running = setup?.state === "running";

  const act = async (method: string, params: Record<string, unknown> = {}) => {
    try {
      setSt(await call<HelperStatus>(method, params));
    } catch (e) {
      fail(e);
    }
  };
  const runSetup = async (params: Record<string, unknown>) => {
    const j = await startJob(`${kind}.setup`, params);
    if (j) {
      setSetupId(j.id);
      lastSetup[kind] = j.id;
      setPick("");
    }
  };
  const useOwn = async () => {
    const ps = await pickPaths({ title: "Choose a reranker model (.gguf)", filters: GGUF_FILTER });
    if (ps?.[0]) runSetup({ model_path: ps[0] });
  };
  const sp = setup?.progress ?? {};

  return (
    <section className="panel">
      <h2>{title}</h2>
      <p className="muted">{children}</p>

      {!st.needs_setup && (
        <div className="status-grid">
          <div>
            <span className="field-label">State</span>
            <span className={`state state-${st.enabled ? st.state : "stopped"}`}>{st.enabled ? STATE[st.state] ?? st.state : "Off"}</span>
          </div>
          <div>
            <span className="field-label">Model</span>
            <span>{current?.label ?? st.model_id} ({bytes(st.size_bytes)})</span>
          </div>
          <div className="span2">
            <span className="field-label">Running on</span>
            <span>{st.device_note || "Decided when the model starts"}</span>
          </div>
        </div>
      )}
      {st.error && st.enabled && <Notice tone="error" title={`The ${kind === "vision" ? "picture model" : "reranker"} stopped`}>{st.error}</Notice>}
      {setup?.state === "failed" && <Notice tone="error" title={setup.error?.message ?? "Setup failed"}>{setup.error?.hint}</Notice>}

      {running ? (
        <>
          <Progress value={sp.done} total={sp.total} label={sp.label ?? sp.message ?? "Setting up"} />
          <Button kind="ghost" onClick={() => call("jobs.cancel", { job: setup!.id })}>Cancel</Button>
        </>
      ) : (
        <>
          <div className="row">
            <label className="field">
              <span className="field-label">Model</span>
              <select value={selected} onChange={(e) => setPick(e.target.value)}>
                {st.choices.map((c) => (
                  <option key={c.id} value={c.id}>{c.label} · {bytes(c.bytes)}{c.downloaded ? " · downloaded" : ""}</option>
                ))}
                {st.model_id && !current && <option value={st.model_id}>{st.model_id} (your file)</option>}
              </select>
            </label>
            {choice && !choice.downloaded && (
              <Button kind={st.needs_setup ? "primary" : undefined} icon={<Download size={16} />} onClick={() => runSetup({ choice: choice.id })}>
                Download {bytes(choice.bytes)} and use it
              </Button>
            )}
            {choice && choice.downloaded && choice.id !== st.model_id && (
              <Button onClick={() => (st.needs_setup ? runSetup({ choice: choice.id }) : act(`${kind}.configure`, { choice: choice.id }))}>
                Switch to this model
              </Button>
            )}
            {st.own_file && <Button kind="ghost" icon={<FileUp size={16} />} onClick={useOwn}>Use my own .gguf</Button>}
          </div>
          {choice?.note && <p className="muted small">{choice.note}</p>}
          {!st.needs_setup && (
            <div className="row">
              <Switch checked={st.enabled} onChange={(on) => act(`${kind}.configure`, { enabled: on })} label={toggle} />
              {st.enabled && st.state !== "ready" && st.state !== "starting" && (
                <Button icon={<Play size={16} />} onClick={() => act(`${kind}.start`)}>Start</Button>
              )}
              {st.enabled && (st.state === "ready" || st.state === "starting") && (
                <Button kind="ghost" icon={<Square size={16} />} onClick={() => act(`${kind}.stop`)}>Stop for now</Button>
              )}
              <Button kind="ghost" onClick={() => setShowLog((v) => !v)}>{showLog ? "Hide log" : "Show log"}</Button>
            </div>
          )}
          {showLog && <pre className="log">{st.log_tail.join("\n") || "(nothing yet)"}</pre>}
        </>
      )}
    </section>
  );
}
