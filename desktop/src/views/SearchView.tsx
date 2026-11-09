import { Search as SearchIcon, FileText, ExternalLink } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { call, EngineCallError } from "../api";
import { useApp } from "../state";
import type { KbSummary, SearchHit, SearchResponse } from "../types";
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

export function SearchView({ kb }: { kb: KbSummary }) {
  const { fail, setTab } = useApp();
  const [query, setQuery] = useState(lastQuery[kb.id] ?? "");
  const [res, setRes] = useState<SearchResponse | null>(lastResult[kb.id] ?? null);
  const [selected, setSelected] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<EngineCallError | null>(null);
  const [fileType, setFileType] = useState("");
  const [expand, setExpand] = useState(true);
  const [agentView, setAgentView] = useState(false);
  const [showCandidates, setShowCandidates] = useState(false);
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
          placeholder="Ask what your agent would ask, e.g. what's new in Opcenter Execution Semiconductor 8.6"
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
        <Button kind="primary" type="submit" disabled={busy || !query.trim()}>
          {busy ? "Searching…" : "Search"}
        </Button>
      </form>
      <div className="search-options">
        <Switch checked={expand} onChange={setExpand} label="Add neighbouring chunks (as the agent gets them)" />
        <Switch checked={agentView} onChange={setAgentView} label="As the agent sees it" />
        <Switch checked={showCandidates} onChange={setShowCandidates} label="Show the 30 candidates before the cut" />
        {res && (
          <span className="timings">
            Embedding {res.timings_ms.embed} ms, search {res.timings_ms.search} ms
          </span>
        )}
      </div>

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
                      {[hitLocation(h), h.source.heading_path, kindLabel(h.kind)].filter(Boolean).join(", ")}
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
                      {[hitLocation(hit), hit.source.heading_path, hit.source.symbol, kindLabel(hit.kind)]
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
                {hit.chunk_ids && hit.chunk_ids.length > 1 && (
                  <p className="evidence-note">
                    Chunk {hit.chunk_id} matched; its neighbours from the same section are included (
                    {hit.chunk_ids.length} chunks).
                  </p>
                )}
                {hit.kind === "figure" && <HitFigure kb={kb.id} chunkId={hit.chunk_id} />}
                <Passage text={hit.text} terms={terms} kind={hit.kind} />
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
