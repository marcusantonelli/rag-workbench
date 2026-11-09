import { Check, Download, FileText } from "lucide-react";
import { Fragment, useEffect, useMemo, useState } from "react";
import { call } from "../api";
import { useApp, useJob } from "../state";
import type { EvalChunk, EvalReport, KbSummary } from "../types";
import { Button, Empty, Notice, Progress, Stat } from "../components/ui";
import { Passage } from "../components/Passage";
import { RuntimeGate } from "./KbView";
import { basename, pct, queryTerms } from "../format";
import { GOLDEN_FILTER, pickPaths, pickSavePath } from "../platform";
import { focusChunk } from "../focus";

const lastJob: Record<string, string> = {};
const lastGolden: Record<string, string> = {};

function rankTone(r: number | null): string {
  if (r == null) return "miss";
  if (r === 1) return "first";
  if (r <= 5) return "top5";
  if (r <= 20) return "top20";
  return "miss";
}

export function EvalView({ kb }: { kb: KbSummary }) {
  const { startJob, fail, notify, setTab } = useApp();
  const [golden, setGolden] = useState(lastGolden[kb.id] ?? "");
  const [jobId, setJobId] = useState<string | null>(lastJob[kb.id] ?? null);
  const job = useJob(jobId);
  const report: EvalReport | null = job?.state === "done" ? (job.result as EvalReport) : null;
  const [selected, setSelected] = useState<string | null>(null);
  const [onlyProblems, setOnlyProblems] = useState(false);

  useEffect(() => setSelected(null), [jobId]);

  const choose = async () => {
    const p = await pickPaths({ title: "Choose a golden question file", filters: GOLDEN_FILTER });
    if (p?.[0]) {
      setGolden(p[0]);
      lastGolden[kb.id] = p[0];
    }
  };
  const run = async () => {
    const j = await startJob("eval.run", { kb: kb.id, golden, show: 6 });
    if (j) {
      setJobId(j.id);
      lastJob[kb.id] = j.id;
    }
  };
  const save = async (format: "md" | "json") => {
    if (!report) return;
    const path = await pickSavePath({
      title: "Save the report",
      defaultPath: `${basename(golden).replace(/\.jsonl?$/, "")}-report.${format}`,
      filters: [{ name: format === "md" ? "Markdown" : "JSON", extensions: [format] }],
    });
    if (!path) return;
    try {
      const text = format === "md" ? await call<string>("eval.markdown", { report }) : JSON.stringify(report, null, 2);
      await call("app.write_text", { path, text });
      notify({ tone: "success", title: `Saved ${basename(path)}` });
    } catch (e) {
      fail(e);
    }
  };

  const questions = useMemo(() => {
    const qs = report?.questions ?? [];
    return onlyProblems ? qs.filter((q) => q.first_hit_rank == null || q.first_hit_rank > 1) : qs;
  }, [report, onlyProblems]);
  const current = report?.questions.find((q) => q.id === selected) ?? questions[0];
  const unans = report?.unanswerable.find((q) => q.id === selected);

  return (
    <div className="eval-view">
      <RuntimeGate kb={kb} />
      <div className="eval-bar">
        <div className="path-row grow">
          <span className="field-label">Questions</span>
          <code className="path">{golden || "No file chosen"}</code>
          <Button onClick={choose}>Choose file…</Button>
        </div>
        <Button kind="primary" onClick={run} disabled={!golden || job?.state === "running"}>
          {job?.state === "running" ? "Evaluating…" : report ? "Run again" : "Run evaluation"}
        </Button>
      </div>
      <p className="muted small">
        A .jsonl file, one question per line with where its answer is:{" "}
        <code>{`{"id": "q1", "question": "...", "expect": [{"file": "*FactSheet*", "page": 2, "contains": "SECS/GEM"}]}`}</code>.
        Questions with <code>"expect": []</code> are listed as not answerable.
      </p>

      {job?.state === "running" && (
        <div className="eval-running">
          <Progress value={job.progress.done} total={job.progress.total} label="Evaluation progress" />
          <span className="muted">{job.progress.total ? `${job.progress.done ?? 0} of ${job.progress.total} questions` : "Starting"}</span>
        </div>
      )}
      {job?.state === "failed" && <Notice tone="error" title={job.error?.message ?? "Evaluation failed"}>{job.error?.hint}</Notice>}

      {!job && <Empty title="Measure retrieval with known answers">Recall@5 is the share of questions whose answer is in the first five results, about what the agent receives.</Empty>}

      {report && (
        <>
          <section className="eval-summary">
            <Stat value={pct(report.overall["recall@5"])} label="answer in the top 5" />
            <Stat value={pct(report.overall["recall@20"])} label="answer in the top 20" />
            <Stat value={report.overall["mrr@10"].toFixed(2)} label="mean reciprocal rank" />
            <Stat value={`${Math.round(report.latency_ms.embed_p50 + report.latency_ms.search_p50)} ms`} label="typical query time" />
            <div className="eval-save">
              <Button kind="ghost" icon={<FileText size={16} />} onClick={() => save("md")}>Save report</Button>
              <Button kind="ghost" icon={<Download size={16} />} onClick={() => save("json")}>Save JSON</Button>
            </div>
          </section>
          {Object.keys(report.by_type).length > 1 && (
            <table className="grid by-type">
              <thead><tr><th>Question type</th><th>Questions</th><th>Top 5</th><th>Top 20</th><th>MRR</th></tr></thead>
              <tbody>
                {Object.entries(report.by_type).map(([t, s]) => (
                  <tr key={t}><td>{t}</td><td>{s.n}</td><td>{pct(s["recall@5"])}</td><td>{pct(s["recall@20"])}</td><td>{s["mrr@10"].toFixed(2)}</td></tr>
                ))}
              </tbody>
            </table>
          )}

          <div className="split eval-split">
            <section className="q-pane">
              <label className="check">
                <input type="checkbox" checked={onlyProblems} onChange={(e) => setOnlyProblems(e.target.checked)} />
                Only questions not answered first
              </label>
              <ol className="q-list">
                {questions.map((q) => (
                  <li key={q.id}>
                    <button className={`q-row ${current?.id === q.id && !unans ? "selected" : ""}`} onClick={() => setSelected(q.id)}>
                      <span className={`rank-badge rank-${rankTone(q.first_hit_rank)}`}>
                        {q.first_hit_rank == null ? "miss" : `#${q.first_hit_rank}`}
                      </span>
                      <span className="q-text">{q.question}</span>
                    </button>
                  </li>
                ))}
              </ol>
              {report.unanswerable.length > 0 && (
                <>
                  <h3 className="q-group">Not answerable from these documents</h3>
                  <ol className="q-list">
                    {report.unanswerable.map((q) => (
                      <li key={q.id}>
                        <button className={`q-row ${unans?.id === q.id ? "selected" : ""}`} onClick={() => setSelected(q.id)}>
                          <span className="rank-badge rank-none">none</span>
                          <span className="q-text">{q.question}</span>
                        </button>
                      </li>
                    ))}
                  </ol>
                </>
              )}
            </section>

            <section className="q-detail">
              {unans ? (
                <>
                  <h2>{unans.question}</h2>
                  <p className="muted">
                    Nothing in the knowledge base answers this. Below is what the agent would be handed; it should say
                    it can't find the answer rather than use these.
                  </p>
                  <RetrievedList kbId={kb.id} chunks={unans.retrieved ?? []} terms={queryTerms(unans.question)}
                    onShow={(id) => { focusChunk({ kb: kb.id, chunkId: id }); setTab(kb.id, "documents"); }} />
                </>
              ) : current ? (
                <>
                  <h2>{current.question}</h2>
                  <p className="muted">
                    {current.first_hit_rank == null
                      ? "No correct passage in the top 20."
                      : `First correct passage at rank ${current.first_hit_rank}.`}{" "}
                    Expected: {(current.expect ?? []).map((e) => Object.entries(e).map(([k, v]) => `${k} ${v}`).join(", ")).join("; or ")}
                  </p>
                  <RetrievedList kbId={kb.id} chunks={current.retrieved ?? []} terms={queryTerms(current.question)}
                    onShow={(id) => { focusChunk({ kb: kb.id, chunkId: id }); setTab(kb.id, "documents"); }} />
                </>
              ) : null}
            </section>
          </div>
        </>
      )}
    </div>
  );
}

function RetrievedList({ chunks, terms, onShow }: { kbId: string; chunks: EvalChunk[]; terms: string[]; onShow: (id: number) => void }) {
  let prev = 0;
  return (
    <ol className="retrieved">
      {chunks.map((c) => {
        const gap = c.rank > prev + 1 ? <li className="gap muted">Ranks {prev + 1} to {c.rank - 1} not shown</li> : null;
        prev = c.rank;
        return (
          <Fragment key={c.chunk_id}>
            {gap}
            <li className={`retrieved-item ${c.match ? "match" : ""}`}>
              <div className="chunk-head">
                <span className="chunk-no">{c.rank}</span>
                {c.match && <span className="match-flag"><Check size={14} /> Correct</span>}
                <span className="chunk-where">{[c.file.split(/[\\/]/).pop(), c.where, c.heading].filter(Boolean).join(", ")}</span>
                <button className="link-btn" onClick={() => onShow(c.chunk_id)}>Show in document</button>
              </div>
              <Passage text={c.text.length > 2500 ? c.text.slice(0, 2500) + " …" : c.text} terms={terms} kind={c.kind} />
            </li>
          </Fragment>
        );
      })}
    </ol>
  );
}
