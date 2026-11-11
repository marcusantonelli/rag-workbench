import { Search as SearchIcon, FileText, ExternalLink } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { call, EngineCallError } from "../api";
import { useHelper } from "./HelperPanel";
import { useApp } from "../state";
import type { KbSummary, ProcedurePage, SearchHit, SearchResponse } from "../types";
import { Button, Empty, Notice, Switch } from "../components/ui";
import { Passage } from "../components/Passage";
import { HitFigure } from "../components/FigureImage";
import { RankStrip } from "../components/RankStrip";
import { RuntimeGate } from "./KbView";
import { kindLabel, location, queryTerms } from "../format";
import { focusChunk } from "../focus";

const lastQuery: Record<string, string> = {};
const lastResult: Record<string, SearchResponse | null> = {};

function hitLocation(h: SearchHit): string {
  return location({
    page_start: h.source.page,
    page_end: h.source.page_end,
    line_start: h.source.lines ? Number(h.source.lines.split("-")[0]) : null,
    line_end: h.source.lines ? Number(h.source.lines.split("-")[1] ?? h.source.lines) : null,
    sheet: h.source.sheet,
    cell_range: h.source.range,
  });
}

function stored(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function remember(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* a convenience only */
  }
}

function procedureNote(p: NonNullable<SearchHit["procedure"]>): string {
  const name = p.title || p.label || (p.kind === "flowchart" ? "the flowchart" : "the procedure");
  const what = p.kind === "flowchart" ? "flowchart" : "procedure";
  const of = `of the ${what} ${name} (${p.steps} ${p.kind === "flowchart" ? "page" : "step"}${p.steps === 1 ? "" : "s"})`;
  if (p.whole) return `The whole ${what} ${name} is included: ${p.steps} ${p.kind === "flowchart" ? "page" : "step"}${p.steps === 1 ? "" : "s"}`;
  if (p.level === "substeps") return `Step ${p.step} is long, so ${p.from === p.to ? `its sub-step ${p.from}` : `its sub-steps ${p.from} to ${p.to}`} ${of}`;
  if (p.level === "chunks") return `Part of step ${p.step}${p.from && p.from !== p.step ? ` (sub-step ${p.from})` : ""}, which is too long to include whole, ${of}`;
  if (p.from === p.to) return `Step ${p.from} ${of}`;
  return `Steps ${p.from} to ${p.to} ${of}`;
}

export function SearchView({ kb }: { kb: KbSummary }) {
  const { fail, setTab } = useApp();
  const [query, setQuery] = useState(lastQuery[kb.id] ?? "");
  const [res, setRes] = useState<SearchResponse | null>(lastResult[kb.id] ?? null);
  const [selected, setSelected] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<EngineCallError | null>(null);
  const [fileType, setFileType] = useState("");
  const [onlyDocs, setOnlyDocs] = useState("");
  const [whole, setWhole] = useState<{ chunkId: number; page: ProcedurePage } | null>(null);
  const [wholeBusy, setWholeBusy] = useState(false);
  const [expand, setExpand] = useState(true);
  const [agentView, setAgentView] = useState(false);
  const [showCandidates, setShowCandidates] = useState(false);
  const [reranker] = useHelper("rerank");
  const managed = !!reranker && !reranker.needs_setup && reranker.enabled;     // the app runs one
  const [rerankChoice, setRerank] = useState<boolean | null>(null);           // null = follow Settings
  const rerank = rerankChoice ?? (managed || stored("rag.rerank") === "1");
  const [rerankUrl, setRerankUrl] = useState(() => stored("rag.rerankUrl") || "http://127.0.0.1:8082");
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => input.current?.focus(), [kb.id]);
  const terms = useMemo(() => queryTerms(res?.query ?? ""), [res?.query]);
  const types = Object.keys(kb.counts.documents_by_type ?? {}).sort();

  const run = async (allowMismatch = false) => {
    if (!query.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const r = await call<SearchResponse>("kb.search", {
        kb: kb.id, query, file_type: fileType || null, expand, allow_mismatch: allowMismatch,
        documents: onlyDocs.split(",").map((d) => d.trim()).filter(Boolean),
        rerank, rerank_url: rerank && !managed ? rerankUrl.trim() : null,
      });
      setRes(r);
      setSelected(0);
      lastQuery[kb.id] = query;
      lastResult[kb.id] = r;
    } catch (e) {
      if (e instanceof EngineCallError) setError(e);
      else fail(e);
    } finally {
      setBusy(false);
    }
  };

  const hit = res?.results[selected];
  const shownWhole = whole && hit && whole.chunkId === hit.chunk_id ? whole.page : null;
  const showWhole = async (h: SearchHit) => {
    if (!h.procedure) return;
    setWholeBusy(true);
    try {
      const page = await call<ProcedurePage>("kb.procedure", { kb: kb.id, procedure: h.procedure.id, max_tokens: 1000000 });
      setWhole({ chunkId: h.chunk_id, page });
    } catch (e) {
      fail(e);
    } finally {
      setWholeBusy(false);
    }
  };

  return (
    <div className="search-view">
      <RuntimeGate kb={kb} />
      <form
        className="search-bar"
        onSubmit={(e) => {
          e.preventDefault();
          run();
        }}
      >
        <SearchIcon size={18} className="search-icon" aria-hidden="true" />
        <input
          ref={input}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask what your agent would ask, e.g. how do I replace the filter?"
          aria-label="Search question"
        />
        <select value={fileType} onChange={(e) => setFileType(e.target.value)} aria-label="File type">
          <option value="">All file types</option>
          {types.map((t) => (
            <option key={t} value={t}>
              {t.toUpperCase()} only
            </option>
          ))}
        </select>
        <input
          className="doc-filter"
          value={onlyDocs}
          onChange={(e) => setOnlyDocs(e.target.value)}
          placeholder="Only documents (e.g. DOC-12, WI-1234)"
          aria-label="Only these documents: document numbers or parts of file names, separated by commas"
          title="Document numbers (from the structure) or parts of file names, separated by commas"
          spellCheck={false}
        />
        <Button kind="primary" type="submit" disabled={busy || !query.trim()}>
          {busy ? "Searching…" : "Search"}
        </Button>
      </form>
      <div className="search-options">
        <Switch checked={expand} onChange={setExpand} label="Add the surrounding steps or chunks (as the agent gets them)" />
        <Switch checked={agentView} onChange={setAgentView} label="As the agent sees it" />
        <Switch checked={showCandidates} onChange={setShowCandidates} label="Show the 30 candidates before the cut" />
        <Switch
          checked={rerank}
          onChange={(on) => {
            setRerank(on);
            if (!managed) remember("rag.rerank", on ? "1" : "0");
          }}
          label={managed ? "Rerank the best 20" : "Rerank the best 20 with my own reranker server"}
        />
        {rerank && !managed && (
          <label className="rerank-url">
            <span className="muted small">Reranker address</span>
            <input
              value={rerankUrl}
              onChange={(e) => {
                setRerankUrl(e.target.value);
                remember("rag.rerankUrl", e.target.value);
              }}
              placeholder="http://127.0.0.1:8082"
              aria-label="Reranker address"
              spellCheck={false}
            />
          </label>
        )}
        {res && (
          <span className="timings">
            Embedding {res.timings_ms.embed} ms, search {res.timings_ms.search} ms
            {res.timings_ms.rerank != null && `, rerank ${res.timings_ms.rerank} ms`}
          </span>
        )}
      </div>
      {res?.rerank && !res.rerank.applied && (
        <Notice tone="warn" title="Not reranked: these results are in the usual order">
          {res.rerank.error}{" "}
          {managed ? "The reranker may still be loading; see Settings → Reranker." :
            "Download one in Settings → Reranker, or turn reranking off."}
        </Notice>
      )}

      {error && (
        <Notice
          tone={error.code === "model_mismatch" ? "error" : "warn"}
          title={error.message}
          action={
            error.code === "model_mismatch" ? (
              <Button onClick={() => run(true)}>Search anyway</Button>
            ) : undefined
          }
        >
          {error.hint}
        </Notice>
      )}

      {!res && !error && (
        <Empty title="Search this knowledge base">
          Results show the passages retrieval returns, where each one came from, and how it ranked in the meaning
          (vector) and word (keyword) searches.
        </Empty>
      )}

      {res && agentView && (
        <section className="agent-view">
          <p className="muted">
            The text the MCP tool hands the agent for this question: {res.results.length} passages
            {res.tables.length ? ` and ${res.tables.length} table suggestions` : ""}.
          </p>
          <pre>{res.markdown || "(no results)"}</pre>
        </section>
      )}

      {res && !agentView && (
        <div className="split">
          <ol className="hit-list" aria-label="Results">
            {res.results.length === 0 && <li className="muted">No passages matched.</li>}
            {res.results.map((h, i) => (
              <li key={h.chunk_id}>
                <button className={`hit ${i === selected ? "selected" : ""}`} onClick={() => setSelected(i)}>
                  <span className="hit-tag">{h.tag}</span>
                  <span className="hit-body">
                    <span className="hit-file">{h.source.file.split(/[\\/]/).pop()}</span>
                    <span className="hit-where">
                      {[h.source.document, hitLocation(h), h.source.heading_path, kindLabel(h.kind)].filter(Boolean).join(", ")}
                    </span>
                    <span className="hit-snippet">{h.text.replace(/[#*|>`]/g, " ").slice(0, 160)}</span>
                  </span>
                  <RankStrip ranks={h.ranks} />
                </button>
              </li>
            ))}
            {res.tables.length > 0 && (
              <li className="table-hints">
                <h3>Tables that may answer this</h3>
                {res.tables.map((t) => (
                  <button key={t.table_name} className="table-hint" onClick={() => setTab(kb.id, "tables")}>
                    <code>{t.table_name}</code>
                    <span>{t.summary}</span>
                  </button>
                ))}
              </li>
            )}
          </ol>

          <article className="evidence" aria-label="Selected passage">
            {hit ? (
              <>
                <header className="evidence-head">
                  <div>
                    <h2>{hit.source.file.split(/[\\/]/).pop()}</h2>
                    <p className="evidence-where">
                      {[hit.source.document && `${hit.source.document}${hit.source.revision ? ` rev ${hit.source.revision}` : ""}`,
                        hitLocation(hit), hit.source.heading_path, hit.source.symbol, kindLabel(hit.kind)]
                        .filter(Boolean)
                        .join(", ")}
                    </p>
                    <p className="evidence-path">{hit.source.file}</p>
                  </div>
                  <div className="evidence-actions">
                    <Button
                      kind="ghost"
                      icon={<FileText size={16} />}
                      onClick={() => {
                        focusChunk({ kb: kb.id, chunkId: hit.chunk_id });
                        setTab(kb.id, "documents");
                      }}
                    >
                      Show in document
                    </Button>
                    <OpenSourceButton kb={kb} chunkId={hit.chunk_id} />
                  </div>
                </header>
                {hit.procedure && (
                  <p className="evidence-note">
                    {shownWhole
                      ? `The whole procedure ${hit.procedure.title || hit.procedure.label || ""}: ${shownWhole.chunk_ids.length} chunks. `
                      : `${procedureNote(hit.procedure)} (${hit.chunk_ids?.length ?? 1} chunk${(hit.chunk_ids?.length ?? 1) === 1 ? "" : "s"}, matched in chunk ${hit.chunk_id}). `}
                    {!hit.procedure.whole && (
                      shownWhole
                        ? <button className="link-button" onClick={() => setWhole(null)}>Show only what the agent got</button>
                        : <button className="link-button" onClick={() => showWhole(hit)} disabled={wholeBusy}>
                            {wholeBusy ? "Loading…" : "Show the whole procedure"}
                          </button>
                    )}
                  </p>
                )}
                {hit.outline && (
                  <details className="evidence-outline">
                    <summary>Where this sits in the procedure</summary>
                    <ol>
                      {hit.outline.before && <li className="muted">…</li>}
                      {hit.outline.steps.map((x, i) => (
                        <li key={i} className={x.shown ? "shown" : undefined}>{x.title}</li>
                      ))}
                      {hit.outline.after && <li className="muted">…</li>}
                    </ol>
                  </details>
                )}
                {hit.related && hit.related.length > 0 && (
                  <p className="evidence-note">
                    Refers to:{" "}
                    {hit.related.map((r) => r.kind === "section" ? `section ${r.title || r.ref}` :
                      `${r.ref}${r.file ? ` (${r.file})` : " (not in this knowledge base)"}`).join("; ")}
                  </p>
                )}
                {!hit.procedure && hit.chunk_ids && hit.chunk_ids.length > 1 && (
                  <p className="evidence-note">
                    Chunk {hit.chunk_id} matched; its neighbours from the same section are included (
                    {hit.chunk_ids.length} chunks).
                  </p>
                )}
                {hit.kind === "figure" && <HitFigure kb={kb.id} chunkId={hit.chunk_id} />}
                <Passage text={shownWhole ? shownWhole.text : hit.text} terms={terms} kind={hit.kind} />
              </>
            ) : (
              <Empty title="No passage selected" />
            )}
          </article>
        </div>
      )}

      {res && showCandidates && (
        <section className="candidates">
          <h3>Candidates after fusing both searches</h3>
          <p className="muted">
            The first {Math.min(res.candidates.length, 30)} in fused order, duplicates removed. The agent receives the
            top {res.results.length}.
          </p>
          <table className="grid">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Meaning</th>
                <th>Words</th>
                <th>File</th>
                <th>Where</th>
                <th>Text</th>
              </tr>
            </thead>
            <tbody>
              {res.candidates.map((c) => (
                <tr key={c.chunk_id} className={res.results.some((r) => r.chunk_id === c.chunk_id) ? "kept" : ""}>
                  <td>{c.ranks?.rank}</td>
                  <td>{c.ranks?.vector ?? ""}</td>
                  <td>{c.ranks?.keyword ?? ""}</td>
                  <td title={c.file}>{c.file?.split(/[\\/]/).pop()}</td>
                  <td>{location(c)}</td>
                  <td className="cell-text">{c.text.slice(0, 140)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </div>
  );
}

function OpenSourceButton({ kb, chunkId }: { kb: KbSummary; chunkId: number }) {
  const { fail } = useApp();
  return (
    <Button
      kind="ghost"
      icon={<ExternalLink size={16} />}
      onClick={async () => {
        try {
          const c = await call<{ chunk: { doc_id: number } }>("kb.chunk", { kb: kb.id, chunk_id: chunkId, context: 0 });
          await call("kb.open_source", { kb: kb.id, doc_id: c.chunk.doc_id });
        } catch (e) {
          fail(e);
        }
      }}
    >
      Open original
    </Button>
  );
}
