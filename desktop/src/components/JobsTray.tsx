import { useApp } from "../state";
import { Progress } from "./ui";
import { call } from "../api";
import { bytes } from "../format";

/** Running jobs (builds, exports, evaluations, downloads) with progress and a stop button. */
export function JobsTray() {
  const { jobs, setScreen, setTab } = useApp();
  const running = Object.values(jobs).filter((j) => j.state === "running");
  if (running.length === 0) return null;
  return (
    <div className="jobs-tray">
      {running.map((j) => {
        const p = j.progress ?? {};
        let value: number | undefined;
        let total: number | undefined;
        let detail = p.current ? String(p.current).split(/[\\/]/).pop() : p.message ?? "";
        if (j.kind === "build") {
          value = p.files_done;
          total = p.files_total;
          detail = total ? `${value ?? 0} of ${total} files` : "Scanning files";
        } else if (j.kind === "merge") {
          value = p.files_done;
          total = p.files_total;
          detail = total ? `${value ?? 0} of ${total} documents` : "Opening";
        } else if (j.kind === "eval") {
          value = p.done;
          total = p.total;
          detail = total ? `${value ?? 0} of ${total} questions` : "Starting";
        } else if (j.kind === "runtime_setup" && p.stage === "download") {
          value = p.done;
          total = p.total;
          detail = `${p.label ?? "Downloading"}: ${bytes(p.done)}${p.total ? ` of ${bytes(p.total)}` : ""}`;
        }
        const go = () => {
          if (j.kind === "runtime_setup") setScreen({ kind: "settings" });
          else if (j.kb) {
            setScreen({ kind: "kb", id: j.kb });
            setTab(j.kb, j.kind === "eval" ? "evaluate" : j.kind === "verify" ? "about" : "build");
          }
        };
        return (
          <div key={j.id} className="job">
            <button className="job-main" onClick={go}>
              <span className="job-title">{j.title}</span>
              <span className="job-detail">{detail}</span>
            </button>
            <Progress value={value} total={total} label={j.title} />
            <button className="job-stop" onClick={() => call("jobs.cancel", { job: j.id })}>
              Stop
            </button>
          </div>
        );
      })}
    </div>
  );
}
