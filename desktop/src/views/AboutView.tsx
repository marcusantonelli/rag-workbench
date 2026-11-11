import { ArrowLeftRight, FolderOpen, ShieldCheck } from "lucide-react";
import { useEffect, useState } from "react";
import { call } from "../api";
import { useApp, useJob } from "../state";
import type { KbSummary } from "../types";
import { Button, Notice, Progress } from "../components/ui";
import { bytes, kindLabel, when } from "../format";

/** How questions are embedded for this knowledge base, editable in a workspace. Only questions
 * change (nothing is embedded again), so trying a different instruction is cheap: edit, then
 * compare with an evaluation. */
function QueryTemplate({ kb }: { kb: KbSummary }) {
  const { fail, notify, refreshKb } = useApp();
  const stored = kb.embedding.query_instruction ?? "";
  const editable = stored.replaceAll("{query}", "${query}");
  const [editing, setEditing] = useState(false);
  const [text, setText] = useState(editable);
  const [saving, setSaving] = useState(false);
  useEffect(() => setText(editable), [editable]);
  const count = text.split("${query}").length - 1;
  const problem = count === 0 ? "Put ${query} where the question goes." : count > 1 ? "Use ${query} only once." : null;
  const save = async () => {
    setSaving(true);
    try {
      await call("kb.query_template", { kb: kb.id, template: text });
      await refreshKb(kb.id);
      notify({ tone: "success", title: "Query template saved: searches use it from now on" });
      setEditing(false);
    } catch (e) {
      fail(e);
    } finally {
      setSaving(false);
    }
  };
  if (!editing) {
    return (
      <span className="query-template">
        {stored ? <code>{editable}</code> : "None (questions are embedded as typed)"}
        {kb.kind === "workspace" && !kb.building && (
          <Button kind="ghost" onClick={() => setEditing(true)}>Edit</Button>
        )}
      </span>
    );
  }
  return (
    <div className="query-template editing">
      <textarea value={text} onChange={(e) => setText(e.target.value)} rows={3} spellCheck={false}
        aria-label="Query template" aria-invalid={!!problem} />
      <p className={problem ? "bad small" : "muted small"}>
        {problem ?? <>A question is embedded as: <code>{text.replace("${query}", "how do I replace the seal?")}</code></>}
      </p>
      <p className="muted small">
        For experienced users. Only questions are embedded this way, so nothing is embedded again, but results change:
        compare with an evaluation before and after. Exported packs carry it to the retrieval server.
      </p>
      <div className="row">
        <Button kind="primary" disabled={!!problem || saving || text === editable} onClick={save}>{saving ? "Saving…" : "Save"}</Button>
        <Button kind="ghost" onClick={() => { setText(editable); setEditing(false); }}>Cancel</Button>
      </div>
    </div>
  );
}

export function AboutView({ kb }: { kb: KbSummary }) {
  const { startJob, fail, info, refreshKb, notify } = useApp();
  const [verifyId, setVerifyId] = useState<string | null>(null);
  const verify = useJob(verifyId);
  const [convertId, setConvertId] = useState<string | null>(null);
  const convert = useJob(convertId);
  const onShammy = kb.vector_store === "shammy";
  const powerOfTwo = (kb.embedding.dims & (kb.embedding.dims - 1)) === 0 && kb.embedding.dims >= 64;
  const canConvert = kb.kind === "workspace" && !!info?.shammy && (onShammy || powerOfTwo);

  useEffect(() => {
    if (convert?.state !== "done") return;
    const r = convert.result as { to: string; vectors: number };
    notify({ tone: "success", title: `${r.vectors.toLocaleString()} vectors moved to ${r.to === "shammy" ? "Shammy" : "sqlite-vec"}` });
    refreshKb(kb.id);
  }, [convert?.state]); // eslint-disable-line react-hooks/exhaustive-deps

  const m = kb.manifest;
  const [spotBusy, setSpotBusy] = useState(false);
  const st = kb.structure;
  const rebuildSpot = async () => {
    setSpotBusy(true);
    try {
      const r = await call<KbSummary>("kb.spots", { kb: kb.id });
      const n = r.structure?.procedures ?? 0;
      const sec = r.structure?.sections ?? 0;
      notify({ tone: "success", title: `Structure built: ${n} procedure${n === 1 ? "" : "s"}, ${sec} section${sec === 1 ? "" : "s"}` });
      refreshKb(kb.id);
    } catch (e) {
      fail(e);
    } finally {
      setSpotBusy(false);
    }
  };
  const structure = (
    <span key="s" className="fact-with-action">
      {!st?.present ? "Not built yet" : !st.current ? "Out of date (the chunks changed since it was built)" :
        `${st.procedures ?? 0} procedures, ${st.sections ?? 0} sections, ${st.steps ?? 0} steps, ${st.links ?? 0} references` +
        (st.outdated ? " (made by an older version, without sections: rebuild to add them)" : "")}
      {!kb.building && (
        <Button kind="ghost" onClick={rebuildSpot} disabled={spotBusy}>
          {spotBusy ? "Building…" : st?.present && st.current && !st.outdated ? "Rebuild" : "Build structure"}
        </Button>
      )}
    </span>
  );

  const rows: [string, React.ReactNode][] = [
    ["Location", <code key="p">{kb.path}</code>],
    ["Kind", kb.kind === "pack" ? "Pack (read-only snapshot)" : "Workspace (can be rebuilt and exported)"],
    ["Size", bytes(kb.size_bytes) + (kb.building ? " (building…)" : "")],
    ["Embedding model", <code key="m">{kb.embedding.model}</code>],
    ["Vector size", `${kb.embedding.dims} dimensions`],
    ["Vector store", onShammy ? "Shammy (vectors.shammy)" : "sqlite-vec (inside rag.db)"],
    ["Embedding backend", kb.embedding.backend],
    ["Query template", <QueryTemplate key="q" kb={kb} />],
    ["Last built", kb.last_build_at ? `${when(kb.last_build_at)} (${new Date(kb.last_build_at).toLocaleString()})` : "Not built yet"],
    ["Built with", kb.built_with ? `rag-builder ${kb.built_with}` : ""],
    ["Embedded text", kb.embed_context === 2 ? "Document name, number and revision, file and headings" :
      kb.embed_context === 1 ? "File and headings (made before 0.4.1; a new knowledge base also names the document)" : ""],
    ["Structure", structure],
  ];
  if (m) {
    rows.push(["Pack exported", m.built_at ? new Date(m.built_at).toLocaleString() : ""]);
    rows.push(["Original files", m.includes_originals ? "Included" : "Not included"]);
    rows.push(["Pack id", <code key="id">{m.pack_id}</code>]);
  }

  return (
    <div className="about-view">
      <dl className="facts">
        {rows.filter(([, v]) => v !== "" && v != null).map(([k, v]) => (
          <div key={k} className="fact">
            <dt>{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>

      <section className="panel">
        <h2>Contents</h2>
        {kb.building && <p className="muted small">A build is running: these numbers and the size update as files are added.</p>}
        <div className="breakdown">
          <div>
            <h3>Documents by type</h3>
            <table className="grid compact">
              <tbody>
                {Object.entries(kb.counts.documents_by_type).sort((a, b) => b[1] - a[1]).map(([t, n]) => (
                  <tr key={t}><td>{t.toUpperCase()}</td><td className="num">{n.toLocaleString()}</td></tr>
                ))}
                {kb.counts.failed > 0 && <tr className="bad"><td>Failed to read</td><td className="num">{kb.counts.failed}</td></tr>}
              </tbody>
            </table>
          </div>
          <div>
            <h3>Chunks by kind</h3>
            <table className="grid compact">
              <tbody>
                {Object.entries(kb.counts.chunks_by_kind).sort((a, b) => b[1] - a[1]).map(([k, n]) => (
                  <tr key={k}><td>{kindLabel(k)}</td><td className="num">{n.toLocaleString()}</td></tr>
                ))}
                <tr><td>Vectors</td><td className="num">{kb.counts.vectors.toLocaleString()}</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <div className="row">
        <Button icon={<FolderOpen size={16} />} onClick={() => call("app.open_path", { path: kb.path, reveal: kb.kind === "pack" }).catch(fail)}>
          Show in folder
        </Button>
        <Button
          icon={<ShieldCheck size={16} />}
          disabled={verify?.state === "running" || kb.building}
          onClick={async () => {
            const j = await startJob("kb.verify", { kb: kb.id });
            if (j) setVerifyId(j.id);
          }}
        >
          {verify?.state === "running" ? "Checking…" : kb.kind === "pack" ? "Check file integrity" : "Check workspace"}
        </Button>
        {canConvert && (
          <Button
            icon={<ArrowLeftRight size={16} />}
            disabled={convert?.state === "running" || kb.building}
            onClick={async () => {
              const j = await startJob("kb.convert", { kb: kb.id, to: onShammy ? "sqlite-vec" : "shammy" });
              if (j) setConvertId(j.id);
            }}
          >
            {convert?.state === "running" ? "Moving vectors…" : onShammy ? "Move vectors back to sqlite-vec" : "Move vectors to Shammy"}
          </Button>
        )}
      </div>
      {canConvert && convert?.state !== "running" && (
        <p className="muted small">
          {onShammy
            ? "Packs exported from this workspace need a retrieval server with Shammy support. Moving back embeds nothing again."
            : "Shammy stores the vectors about a fifth smaller and searches large collections much faster. Nothing is embedded again, and packs exported afterwards need a retrieval server with Shammy support."}
        </p>
      )}
      {convert?.state === "running" && (
        <Progress value={convert.progress?.files_done} total={convert.progress?.files_total} label="Vectors moved" />
      )}
      {convert?.state === "failed" && <Notice tone="error" title={convert.error?.message ?? "Couldn't move the vectors"}>{convert.error?.hint}</Notice>}
      {verify?.state === "done" && (
        <Notice
          tone={!verify.result.ok ? "error" : verify.result.problems?.length ? "warn" : "success"}
          title={verify.result.ok ? (kb.kind === "pack" ? "Pack is intact" : "Workspace is intact")
            : kb.kind === "pack" ? "Pack is damaged" : "Workspace has problems"}
        >
          {verify.result.message}
        </Notice>
      )}
    </div>
  );
}
