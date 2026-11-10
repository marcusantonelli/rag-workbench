import { ArrowLeftRight, FolderOpen, ShieldCheck } from "lucide-react";
import { useEffect, useState } from "react";
import { call } from "../api";
import { useApp, useJob } from "../state";
import type { KbSummary } from "../types";
import { Button, Notice, Progress } from "../components/ui";
import { bytes, kindLabel, when } from "../format";

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

  const rows: [string, React.ReactNode][] = [
    ["Location", <code key="p">{kb.path}</code>],
    ["Kind", kb.kind === "pack" ? "Pack (read-only snapshot)" : "Workspace (can be rebuilt and exported)"],
    ["Size", bytes(kb.size_bytes)],
    ["Embedding model", <code key="m">{kb.embedding.model}</code>],
    ["Vector size", `${kb.embedding.dims} dimensions`],
    ["Vector store", onShammy ? "Shammy (vectors.shammy)" : "sqlite-vec (inside rag.db)"],
    ["Embedding backend", kb.embedding.backend],
    ["Query instruction", kb.embedding.query_instruction ? <code key="q">{kb.embedding.query_instruction}</code> : "None"],
    ["Last built", kb.last_build_at ? `${when(kb.last_build_at)} (${new Date(kb.last_build_at).toLocaleString()})` : "Not built yet"],
    ["Built with", kb.built_with ? `rag-builder ${kb.built_with}` : ""],
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
