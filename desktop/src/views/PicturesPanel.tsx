import { Download, Image, Play, Square } from "lucide-react";
import { useEffect, useState } from "react";
import { call, onEvent } from "../api";
import { useApp, useJob } from "../state";
import { Button, Notice, Progress, Switch } from "../components/ui";
import { bytes } from "../format";

interface VisionStatus {
  enabled: boolean;
  state: "stopped" | "starting" | "ready" | "failed";
  ready: boolean;
  url: string | null;
  error: string | null;
  model_id: string | null;
  size_bytes: number | null;
  needs_setup: boolean;
  default_model: { id: string; file: string; approx_bytes: number; mmproj_bytes: number };
  device_note: string;
  log_tail: string[];
}

const STATE: Record<string, string> = { stopped: "Stopped", starting: "Loading…", ready: "Ready", failed: "Failed" };
let lastSetup: string | null = null;

/** Settings: the optional picture model that describes photos and screenshots during builds. */
export function PicturesPanel() {
  const { fail, startJob } = useApp();
  const [st, setSt] = useState<VisionStatus | null>(null);
  const [setupId, setSetupId] = useState<string | null>(lastSetup);
  const setup = useJob(setupId);
  const [showLog, setShowLog] = useState(false);

  const refresh = () => call<VisionStatus>("vision.status").then(setSt).catch(() => {});
  useEffect(() => {
    refresh();
    return onEvent((name, data) => {
      if (name === "vision") setSt(data as VisionStatus);
    });
  }, []);
  useEffect(() => {
    if (setup && setup.state !== "running") refresh();
  }, [setup?.state]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!st) return null;

  const act = async (method: string, params: Record<string, unknown> = {}) => {
    try {
      setSt(await call<VisionStatus>(method, params));
    } catch (e) {
      fail(e);
    }
  };
  const runSetup = async () => {
    const j = await startJob("vision.setup", {});
    if (j) {
      setSetupId(j.id);
      lastSetup = j.id;
    }
  };
  const total = st.default_model.approx_bytes + st.default_model.mmproj_bytes;
  const sp = setup?.progress ?? {};

  return (
    <section className="panel">
      <h2><Image size={18} /> Picture descriptions</h2>
      <p className="muted">
        A small vision model looks at each photo, screenshot and diagram picture while a knowledge base is built and
        writes a sentence or two about what it shows, so pictures can be found by what's in them. It runs beside the
        embedding model and only during builds. Documents with pictures are read again the next time you build.
      </p>

      {st.needs_setup ? (
        setup?.state === "running" ? (
          <>
            <Progress value={sp.done} total={sp.total} label={sp.label ?? "Setting up"} />
            <Button kind="ghost" onClick={() => call("jobs.cancel", { job: setup.id })}>Cancel</Button>
          </>
        ) : (
          <>
            {setup?.state === "failed" && <Notice tone="error" title={setup.error?.message ?? "Setup failed"}>{setup.error?.hint}</Notice>}
            <div className="row">
              <Button kind="primary" icon={<Download size={16} />} onClick={runSetup}>
                Download and turn on ({bytes(total)})
              </Button>
              <span className="muted small">{st.default_model.id}</span>
            </div>
          </>
        )
      ) : (
        <>
          <div className="status-grid">
            <div>
              <span className="field-label">State</span>
              <span className={`state state-${st.enabled ? st.state : "stopped"}`}>
                {st.enabled ? STATE[st.state] ?? st.state : "Off"}
              </span>
            </div>
            <div>
              <span className="field-label">Model</span>
              <span>{st.model_id} ({bytes(st.size_bytes)})</span>
            </div>
            <div className="span2">
              <span className="field-label">Running on</span>
              <span>{st.device_note || "Decided when the model starts"}</span>
            </div>
          </div>
          {st.error && <Notice tone="error" title="The picture model stopped">{st.error}</Notice>}
          <div className="row">
            <Switch checked={st.enabled} onChange={(on) => act("vision.configure", { enabled: on })}
                    label="Describe pictures when building" />
            {st.enabled && st.state !== "ready" && st.state !== "starting" && (
              <Button icon={<Play size={16} />} onClick={() => act("vision.start")}>Start</Button>
            )}
            {st.enabled && (st.state === "ready" || st.state === "starting") && (
              <Button kind="ghost" icon={<Square size={16} />} onClick={() => act("vision.stop")}>Stop for now</Button>
            )}
            <Button kind="ghost" onClick={() => setShowLog((v) => !v)}>{showLog ? "Hide log" : "Show log"}</Button>
          </div>
          {showLog && <pre className="log">{st.log_tail.join("\n") || "(nothing yet)"}</pre>}
        </>
      )}
    </section>
  );
}
