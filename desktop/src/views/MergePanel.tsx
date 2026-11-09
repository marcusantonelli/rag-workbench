import { Check, FilePlus, FolderPlus, GitMerge, X } from "lucide-react";
import { useEffect, useState } from "react";
import { call, EngineCallError } from "../api";
import { useApp, useJob } from "../state";
import type { KbSummary } from "../types";
import { Button, Notice, Progress, Switch } from "../components/ui";
import { basename, count } from "../format";
import { pickPaths } from "../platform";

type Policy = "newer" | "keep" | "replace";

interface SourcePlan {
  path: string;
  name: string;
  compatible: boolean;
  reason: string;
  documents: number;
  add: number;
  replace: number;
  identical: number;
  copies: number;
  kept: number;
  failed_skipped: number;
  chunks: number;
  tables: number;
  renamed_tables: [string, string][];
  conflicts: { path: string; decision: "replace" | "kept" }[];
}

interface MergeResult {
  dry_run: boolean;
  cancelled: boolean;
  sources: SourcePlan[];
}

const lastMerge: Record<string, string> = {};

const POLICIES: { value: Policy; label: string }[] = [
  { value: "newer", label: "Keep whichever was built from the newer file" },
  { value: "keep", label: "Keep the version already here" },
  { value: "replace", label: "Take the incoming version" },
];

function skipped(s: SourcePlan): string {
  return [
    s.identical && `${s.identical} already here`,
    s.copies && `${s.copies} same content under another path`,
    s.kept && `${s.kept} kept as they are here`,
    s.failed_skipped && `${s.failed_skipped} that failed to build`,
  ].filter(Boolean).join(", ");
}

function PlanTable({ plan, done }: { plan: MergeResult; done: boolean }) {
  return (
    <table className="grid compact merge-plan">
      <thead>
        <tr>
          <th>Knowledge base</th>
          <th className="num">{done ? "Added" : "New"}</th>
          <th className="num">{done ? "Replaced" : "Replaces"}</th>
          <th className="num">Chunks</th>
          <th className="num">Tables</th>
          <th>Skipped</th>
        </tr>
      </thead>
      <tbody>
        {plan.sources.map((s) =>
          s.compatible ? (
            <tr key={s.path}>
              <td><b>{s.name}</b><div className="muted small">{count(s.documents, "document")}</div></td>
              <td className="num">{s.add}</td>
              <td className="num">{s.replace}</td>
              <td className="num">{s.chunks}</td>
              <td className="num">{s.tables}</td>
              <td className="cell-text">
                {skipped(s) || "nothing"}
                {s.renamed_tables.map(([a, b]) => <div key={a}>Table <code>{a}</code> renamed to <code>{b}</code></div>)}
              </td>
            </tr>
          ) : (
            <tr key={s.path} className="bad">
              <td><b>{s.name}</b></td>
              <td colSpan={5}>Can't merge: {s.reason}</td>
            </tr>
          ),
        )}
      </tbody>
    </table>
  );
}

/** Copy compatible packs or workspaces into this workspace, without embedding anything again. */
export function MergePanel({ kb, busy }: { kb: KbSummary; busy: boolean }) {
  const { kbs, startJob, notify } = useApp();
  const [sources, setSources] = useState<string[]>([]);
  const [policy, setPolicy] = useState<Policy>("newer");
  const [keepCopies, setKeepCopies] = useState(false);
  const [plan, setPlan] = useState<MergeResult | null>(null);
  const [checking, setChecking] = useState(false);
  const [checkError, setCheckError] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(lastMerge[kb.id] ?? null);
  const job = useJob(jobId);
  const running = job?.state === "running";

  // Check what would happen whenever the choices change (nothing is written).
  useEffect(() => {
    setPlan(null);
    setCheckError(null);
    if (!sources.length) return;
    let alive = true;
    setChecking(true);
    call<MergeResult>("kb.merge", { kb: kb.id, sources, on_conflict: policy, keep_copies: keepCopies, dry_run: true })
      .then((r) => alive && setPlan(r))
      .catch((e) => alive && setCheckError(e instanceof EngineCallError ? e.message : String(e)))
      .finally(() => alive && setChecking(false));
    return () => {
      alive = false;
    };
  }, [kb.id, sources, policy, keepCopies, kb.counts.documents]);

  useEffect(() => {
    if (job?.state !== "done") return;
    const r = job.result as MergeResult;
    const added = r.sources.reduce((n, s) => n + s.add + s.replace, 0);
    notify({ tone: "success", title: `Merged ${count(added, "document")} into ${kb.name}` });
    setSources([]);
  }, [job?.state]); // eslint-disable-line react-hooks/exhaustive-deps

  const addSources = (paths: string[]) =>
    setSources((cur) => [...cur, ...paths.filter((p) => !cur.includes(p) && p !== kb.path)]);

  const others = kbs.filter((k) => k.id !== kb.id && !sources.includes(k.path));
  const toCopy = plan?.sources.reduce((n, s) => n + (s.compatible ? s.add + s.replace : 0), 0) ?? 0;

  const merge = async () => {
    const j = await startJob("kb.merge", { kb: kb.id, sources, on_conflict: policy, keep_copies: keepCopies });
    if (j) {
      setJobId(j.id);
      lastMerge[kb.id] = j.id;
    }
  };

  const p = job?.progress ?? {};
  const result = job?.state === "done" ? (job.result as MergeResult) : null;

  return (
    <section className="panel">
      <h2>Merge other knowledge bases</h2>
      <p className="muted">
        Copies the documents, vectors, tables and figures of packs or workspaces built with the same embedding model.
        Nothing is read or embedded again, so it takes seconds and doesn't need the model running.
      </p>
      {sources.length > 0 && (
        <ul className="inputs">
          {sources.map((s) => (
            <li key={s}>
              <code>{s}</code>
              <button className="icon-btn" onClick={() => setSources((c) => c.filter((x) => x !== s))} aria-label={`Remove ${s}`} disabled={running}>
                <X size={14} />
              </button>
            </li>
          ))}
        </ul>
      )}
      <div className="row">
        <Button icon={<FilePlus size={16} />} disabled={running || busy} onClick={async () => {
          const ps = await pickPaths({ title: "Packs to merge in", multiple: true, filters: [{ name: "RAG pack", extensions: ["ragpack"] }] });
          if (ps) addSources(ps);
        }}>Add pack</Button>
        <Button icon={<FolderPlus size={16} />} disabled={running || busy} onClick={async () => {
          const ps = await pickPaths({ title: "Workspace folders to merge in", directory: true, multiple: true });
          if (ps) addSources(ps);
        }}>Add workspace</Button>
        {others.length > 0 && (
          <select aria-label="Add an open knowledge base" value="" disabled={running || busy}
                  onChange={(e) => e.target.value && addSources([e.target.value])}>
            <option value="">Add an open knowledge base…</option>
            {others.map((k) => <option key={k.id} value={k.path}>{k.name} ({k.kind})</option>)}
          </select>
        )}
      </div>

      {sources.length > 0 && (
        <>
          <div className="row options merge-options">
            <label className="field">
              <span className="field-label">When the same file differs</span>
              <select value={policy} onChange={(e) => setPolicy(e.target.value as Policy)} disabled={running}>
                {POLICIES.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
              </select>
            </label>
            <Switch checked={keepCopies} onChange={setKeepCopies} disabled={running}
                    label="Also add files whose content is already here under another path" />
          </div>
          {checking && <p className="muted">Checking…</p>}
          {checkError && <Notice tone="error" title="Can't merge these">{checkError}</Notice>}
          {plan && !running && !result && <PlanTable plan={plan} done={false} />}
          {plan && plan.sources.some((s) => s.conflicts.length) && !running && !result && (
            <details className="merge-conflicts">
              <summary>{count(plan.sources.reduce((n, s) => n + s.conflicts.length, 0), "file")} in both with different content</summary>
              <ul>
                {plan.sources.flatMap((s) => s.conflicts).map((c) => (
                  <li key={c.path}><code>{c.path}</code>: {c.decision === "replace" ? "incoming version" : "this version kept"}</li>
                ))}
              </ul>
            </details>
          )}
          <div className="row end">
            <Button kind="primary" icon={<GitMerge size={16} />} onClick={merge}
                    disabled={running || busy || checking || !plan || toCopy === 0}>
              {running ? "Merging…" : toCopy ? `Merge ${count(toCopy, "document")}` : "Nothing to merge"}
            </Button>
          </div>
        </>
      )}

      {running && (
        <>
          <Progress value={p.files_done} total={p.files_total} label="Documents checked" />
          <p className="build-now">
            <span>{p.files_total ? `${p.files_done ?? 0} of ${count(p.files_total, "document")} from ${p.stage}` : "Opening"}</span>
            {p.current && <span className="muted">{basename(p.current)}</span>}
          </p>
          <Button kind="ghost" onClick={() => call("jobs.cancel", { job: job!.id })}>Stop</Button>
        </>
      )}
      {job?.state === "failed" && <Notice tone="error" title={job.error?.message ?? "Merge failed"}>{job.error?.hint}</Notice>}
      {result && (
        <>
          <p className="state state-ready"><Check size={16} /> {result.cancelled ? "Merge stopped; what was copied stays." : "Merge finished."}</p>
          <PlanTable plan={result} done />
        </>
      )}
    </section>
  );
}
