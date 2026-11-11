import { FilePlus, FolderPlus, Package, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { call } from "../api";
import { useApp, useJob } from "../state";
import { duration, fileFraction, secondsLeft, stageText } from "../buildProgress";
import type { KbSummary } from "../types";
import { Button, Notice, Progress, Switch } from "../components/ui";
import { RuntimeGate } from "./KbView";
import { MergePanel } from "./MergePanel";
import { basename, bytes, count } from "../format";
import { onFileDrop, pickPaths, pickSavePath } from "../platform";

const lastBuild: Record<string, string> = {};
const lastExport: Record<string, string> = {};

function loadInputs(path: string): string[] {
  try {
    return JSON.parse(localStorage.getItem(`inputs:${path}`) ?? "[]");
  } catch {
    return [];
  }
}
function saveInputs(path: string, inputs: string[]) {
  try {
    localStorage.setItem(`inputs:${path}`, JSON.stringify(inputs));
  } catch {
    /* storage unavailable: the list just isn't remembered */
  }
}

export function BuildView({ kb }: { kb: KbSummary }) {
  const { startJob, notify, fail, openKb } = useApp();
  const [inputs, setInputs] = useState<string[]>(() => loadInputs(kb.path));
  const [prune, setPrune] = useState(false);
  const [force, setForce] = useState(false);
  const [dropping, setDropping] = useState(false);
  const [buildId, setBuildId] = useState<string | null>(lastBuild[kb.id] ?? null);
  const [exportId, setExportId] = useState<string | null>(lastExport[kb.id] ?? null);
  const [includeOriginals, setIncludeOriginals] = useState(false);
  const { jobs } = useApp();
  // follow a build of this knowledge base started anywhere (another tab, before navigating away)
  const runningId = Object.values(jobs).find((j) => j.kb === kb.id && j.kind === "build" && j.state === "running")?.id;
  const build = useJob(runningId ?? buildId);
  const exp = useJob(exportId);

  const add = (paths: string[]) =>
    setInputs((cur) => {
      const next = [...cur, ...paths.filter((p) => !cur.includes(p) && !p.toLowerCase().endsWith(".ragpack"))];
      saveInputs(kb.path, next);
      return next;
    });
  const removeInput = (p: string) =>
    setInputs((cur) => {
      const next = cur.filter((x) => x !== p);
      saveInputs(kb.path, next);
      return next;
    });

  useEffect(() => onFileDrop((paths) => add(paths), setDropping), [kb.path]); // eslint-disable-line react-hooks/exhaustive-deps

  const start = async () => {
    if (force && !window.confirm("Rebuild everything? Every file is read and embedded again.")) return;
    const j = await startJob("kb.build", { kb: kb.id, inputs, prune, force });
    if (j) {
      setBuildId(j.id);
      lastBuild[kb.id] = j.id;
    }
  };

  const doExport = async () => {
    const path = await pickSavePath({
      title: "Export a pack",
      defaultPath: `${kb.name.replace(/[^\w.-]+/g, "-").toLowerCase()}.ragpack`,
      filters: [{ name: "RAG pack", extensions: ["ragpack"] }],
    });
    if (!path) return;
    const j = await startJob("kb.export", { kb: kb.id, out: path, include_originals: includeOriginals, name: kb.name });
    if (j) {
      setExportId(j.id);
      lastExport[kb.id] = j.id;
    }
  };

  useEffect(() => {
    if (exp?.state === "done") notify({ tone: "success", title: `Exported ${basename(exp.result.path)}` });
  }, [exp?.state]); // eslint-disable-line react-hooks/exhaustive-deps

  const p = build?.progress ?? {};
  const [now, setNow] = useState(() => Date.now() / 1000);
  useEffect(() => {
    if (build?.state !== "running") return;
    const t = window.setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => window.clearInterval(t);
  }, [build?.state]);
  const left = secondsLeft(p, now);
  const fileFrac = fileFraction(p);
  const log = useMemo(() => (build?.log ?? []).filter((e) => e.event !== "file_unchanged").slice(-200).reverse(), [build?.log]);
  const running = build?.state === "running";
  const r = build?.state !== "running" ? build?.result : null;

  return (
    <div className={`build-view ${dropping ? "dropping" : ""}`}>
      <RuntimeGate kb={kb} />
      <section className="panel">
        <h2>Sources</h2>
        <p className="muted">
          Files and folders to read. Folders are searched recursively for PDF, Word, Excel, CSV, text, Markdown, HTML,
          SQL, C# and Python files. Rebuilding only processes new or changed files.
        </p>
        {inputs.length === 0 ? (
          <div className="dropzone">Drop files or folders here, or add them below.</div>
        ) : (
          <ul className="inputs">
            {inputs.map((i) => (
              <li key={i}>
                <code>{i}</code>
                <button className="icon-btn" onClick={() => removeInput(i)} aria-label={`Remove ${i}`} disabled={running}>
                  <X size={14} />
                </button>
              </li>
            ))}
          </ul>
        )}
        <div className="row">
          <Button icon={<FolderPlus size={16} />} disabled={running} onClick={async () => {
            const ps = await pickPaths({ title: "Add folders", directory: true, multiple: true });
            if (ps) add(ps);
          }}>Add folder</Button>
          <Button icon={<FilePlus size={16} />} disabled={running} onClick={async () => {
            const ps = await pickPaths({ title: "Add files", multiple: true });
            if (ps) add(ps);
          }}>Add files</Button>
        </div>
        <div className="row options">
          <Switch checked={prune} onChange={setPrune} label="Remove documents whose files are gone" disabled={running} />
          <Switch checked={force} onChange={setForce} label="Rebuild everything from scratch" disabled={running} />
        </div>
        <div className="row end">
          <Button kind="primary" onClick={start} disabled={running || inputs.length === 0 || !kb.runtime.ready && kb.runtime.needs_runtime}>
            {running ? "Building…" : kb.counts.documents ? "Update knowledge base" : "Build knowledge base"}
          </Button>
        </div>
      </section>

      {build && (
        <section className="panel">
          <h2>{running ? "Building" : build.state === "done" ? (r?.cancelled ? "Build stopped" : "Build finished") : build.state === "cancelled" ? "Build stopped" : "Build failed"}</h2>
          {running && (
            <>
              <Progress value={p.files_done} total={p.files_total} label="Files processed" />
              <p className="build-now">
                <span>{p.files_total ? `${p.files_done ?? 0} of ${count(p.files_total, "file")}` : "Looking for files"}</span>
                <span className="muted">
                  {p.started ? `${duration(now - p.started)} so far` : ""}
                  {left != null ? `, about ${duration(left)} left` : ""}
                  {p.unchanged ? ` · ${p.unchanged} unchanged` : ""}
                  {p.chunks ? ` · ${count(p.chunks, "chunk")} added` : ""}
                </span>
              </p>
              {p.current && (
                <div className="build-file">
                  <p>
                    <strong>{basename(p.current)}</strong>
                    <span className="muted"> {stageText(p)}</span>
                    {p.file_chunks != null && p.stage !== "embed" && p.stage !== "chunk" ? <span className="muted"> · {count(p.file_chunks, "chunk")}</span> : null}
                  </p>
                  {fileFrac != null && <div className="file-bar" role="progressbar" aria-valuemin={0} aria-valuemax={100}
                    aria-valuenow={Math.round(fileFrac * 100)} aria-label={`Progress on ${basename(p.current)}`}>
                    <span style={{ width: `${Math.round(fileFrac * 100)}%` }} />
                  </div>}
                </div>
              )}
              <Button kind="ghost" onClick={() => call("jobs.cancel", { job: build.id })}>Stop after this file</Button>
            </>
          )}
          {build.state === "failed" && <Notice tone="error" title={build.error?.message ?? "Build failed"}>{build.error?.hint}</Notice>}
          {r && (
            <p className="build-result">
              {[
                r.added && `${r.added} added`, r.updated && `${r.updated} updated`, r.unchanged && `${r.unchanged} unchanged`,
                r.removed && `${r.removed} removed`, r.failed && `${r.failed} failed`, `${count(r.chunks, "new chunk")}`,
                r.reused_vectors && `${r.reused_vectors} vectors reused`, `${Math.round(r.seconds)} s`,
              ].filter(Boolean).join(", ")}
            </p>
          )}
          {log.length > 0 && (
            <ol className="build-log">
              {log.map((e, i) => (
                <li key={i} className={e.event === "file_failed" ? "bad" : ""}>
                  <span className="log-what">
                    {e.event === "file_done" ? "Added" : e.event === "file_failed" ? "Failed" : e.event === "file_removed" ? "Removed" : e.event === "skipped" ? "Skipped" : ""}
                  </span>
                  <span className="log-path">{e.path ?? e.message}</span>
                  <span className="log-extra">{e.event === "file_done" ? count(e.chunks, "chunk") + (e.seconds != null ? ` in ${duration(e.seconds)}` : "") : e.error ?? ""}</span>
                </li>
              ))}
            </ol>
          )}
        </section>
      )}

      <MergePanel kb={kb} busy={running || kb.building} />

      <section className="panel">
        <h2>Export a pack</h2>
        <p className="muted">
          A .ragpack is one file holding the databases, settings and a checksum for every file: the thing to copy to the
          retrieval server.
        </p>
        <div className="row">
          <Switch checked={includeOriginals} onChange={setIncludeOriginals} label="Include original files (lets the agent open sources; makes the pack larger)" />
        </div>
        <div className="row end">
          <Button kind="primary" icon={<Package size={16} />} onClick={doExport} disabled={running || kb.counts.documents === 0 || exp?.state === "running"}>
            {exp?.state === "running" ? "Exporting…" : "Export pack…"}
          </Button>
        </div>
        {exp?.state === "running" && <Progress label="Exporting" />}
        {exp?.state === "failed" && <Notice tone="error" title={exp.error?.message ?? "Export failed"}>{exp.error?.hint}</Notice>}
        {exp?.state === "done" && (
          <Notice
            tone="success"
            title={`Exported ${basename(exp.result.path)}`}
            action={
              <>
                <Button onClick={() => call("app.open_path", { path: exp.result.path, reveal: true }).catch(fail)}>Show in folder</Button>
                <Button onClick={() => openKb(exp.result.path)}>Open pack</Button>
              </>
            }
          >
            {bytes(exp.result.size_bytes)}, {count(exp.result.manifest.counts.documents, "document")}, {count(exp.result.manifest.counts.chunks, "chunk")}.
          </Notice>
        )}
      </section>
    </div>
  );
}
