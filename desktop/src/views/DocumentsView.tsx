import { ExternalLink, RefreshCw, Trash2 } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { call } from "../api";
import { useApp } from "../state";
import type { Chunk, DocumentRow, KbSummary } from "../types";
import { Button, Empty, Notice } from "../components/ui";
import { Passage } from "../components/Passage";
import { FigureImage } from "../components/FigureImage";
import { bytes, count, kindLabel, location } from "../format";
import { takeFocus, type ChunkFocus } from "../focus";

export function DocumentsView({ kb }: { kb: KbSummary }) {
  const { fail, notify, refreshKb } = useApp();
  const [docs, setDocs] = useState<DocumentRow[]>([]);
  const [filter, setFilter] = useState("");
  const [type, setType] = useState("");
  const [failedOnly, setFailedOnly] = useState(false);
  const [docId, setDocId] = useState<number | null>(null);
  const [detail, setDetail] = useState<{ document: any; chunks: Chunk[] } | null>(null);
  const [focusId, setFocusId] = useState<number | null>(null);
  const chunkRefs = useRef<Record<number, HTMLElement | null>>({});

  const load = useCallback(async () => {
    try {
      setDocs(await call<DocumentRow[]>("kb.documents", {
        kb: kb.id, query: filter, file_type: type || null, status: failedOnly ? "failed" : null,
      }));
    } catch (e) {
      fail(e);
    }
  }, [kb.id, filter, type, failedOnly, fail]);

  useEffect(() => {
    const t = setTimeout(load, 150);
    return () => clearTimeout(t);
  }, [load, kb.counts.documents]);

  const applyFocus = useCallback(async (f: ChunkFocus) => {
    try {
      const c = await call<{ chunk: Chunk }>("kb.chunk", { kb: kb.id, chunk_id: f.chunkId, context: 0 });
      setDocId(c.chunk.doc_id);
      setFocusId(f.chunkId);
    } catch (e) {
      fail(e);
    }
  }, [kb.id, fail]);

  useEffect(() => {
    const f = takeFocus(kb.id);
    if (f) applyFocus(f);
    const onFocus = (e: Event) => {
      const f2 = (e as CustomEvent<ChunkFocus>).detail;
      if (f2.kb === kb.id) {
        takeFocus(kb.id);
        applyFocus(f2);
      }
    };
    window.addEventListener("kb-focus-chunk", onFocus);
    return () => window.removeEventListener("kb-focus-chunk", onFocus);
  }, [kb.id, applyFocus]);

  useEffect(() => {
    if (docId == null) {
      setDetail(null);
      return;
    }
    call<{ document: any; chunks: Chunk[] }>("kb.document", { kb: kb.id, doc_id: docId, limit: 2000 })
      .then(setDetail)
      .catch(fail);
  }, [kb.id, docId, fail]);

  useEffect(() => {
    if (focusId != null && detail) {
      chunkRefs.current[focusId]?.scrollIntoView({ block: "center" });
    }
  }, [focusId, detail]);

  const types = Object.keys(kb.counts.documents_by_type ?? {}).sort();
  const doc = detail?.document;

  const remove = async () => {
    if (!doc || !window.confirm(`Remove ${doc.display_path} from ${kb.name}? The file itself isn't touched.`)) return;
    try {
      const r = await call<{ removed: number }>("kb.remove_documents", { kb: kb.id, doc_ids: [doc.doc_id] });
      notify({ tone: "success", title: `Removed ${r.removed} document` });
      setDocId(null);
      load();
      refreshKb(kb.id);
    } catch (e) {
      fail(e);
    }
  };

  return (
    <div className="split docs-view">
      <section className="doc-list-pane">
        <div className="filters">
          <input value={filter} onChange={(e) => setFilter(e.target.value)} placeholder="Filter by name" aria-label="Filter documents" />
          <button className="icon-btn" title="Refresh the list" aria-label="Refresh the list" onClick={() => { refreshKb(kb.id); load(); }}>
            <RefreshCw size={15} />
          </button>
          <select value={type} onChange={(e) => setType(e.target.value)} aria-label="File type">
            <option value="">All types</option>
            {types.map((t) => (
              <option key={t} value={t}>{t.toUpperCase()} ({kb.counts.documents_by_type[t]})</option>
            ))}
          </select>
          {kb.counts.failed > 0 && (
            <label className="check">
              <input type="checkbox" checked={failedOnly} onChange={(e) => setFailedOnly(e.target.checked)} />
              Failed only ({kb.counts.failed})
            </label>
          )}
        </div>
        <ul className="doc-list">
          {docs.length === 0 && <li className="muted pad">No documents{filter || type ? " match" : " yet"}.</li>}
          {docs.map((d) => {
            const parts = d.display_path.split(/[\\/]/);
            return (
              <li key={d.doc_id}>
                <button className={`doc-row ${d.doc_id === docId ? "selected" : ""}`} onClick={() => { setDocId(d.doc_id); setFocusId(null); }}>
                  <span className={`type-chip type-${d.file_type}`}>{d.file_type}</span>
                  <span className="doc-text">
                    <span className="doc-name">{d.title || parts[parts.length - 1]}</span>
                    <span className="doc-dir">{d.title ? parts[parts.length - 1] : parts.slice(0, -1).join("/")}</span>
                  </span>
                  <span className="doc-meta">
                    {d.status !== "ok" ? <span className={`status status-${d.status}`}>{d.status}</span> : count(d.n_chunks, "chunk")}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      </section>

      <section className="doc-detail">
        {!doc ? (
          <Empty title="Pick a document">See its chunks in order, exactly as they were stored and embedded.</Empty>
        ) : (
          <>
            <header className="evidence-head">
              <div>
                <h2>{doc.title || doc.display_path.split(/[\\/]/).pop()}</h2>
                <p className="evidence-where">
                  {[doc.display_path, doc.n_pages ? count(doc.n_pages, "page") : null, bytes(doc.size_bytes),
                    count(detail!.chunks.length, "chunk"), doc.extractor?.split("@")[0]].filter(Boolean).join(", ")}
                </p>
              </div>
              <div className="evidence-actions">
                <Button kind="ghost" icon={<ExternalLink size={16} />} onClick={() => call("kb.open_source", { kb: kb.id, doc_id: doc.doc_id }).catch(fail)}>
                  Open original
                </Button>
                {!kb.readonly && (
                  <Button kind="ghost" icon={<Trash2 size={16} />} onClick={remove}>Remove</Button>
                )}
              </div>
            </header>
            {doc.status === "failed" && (
              <Notice tone="error" title="This file couldn't be read">{docs.find((d) => d.doc_id === doc.doc_id)?.error}</Notice>
            )}
            {doc.warnings?.length > 0 && (
              <Notice tone="warn" title="Read with warnings">
                <ul className="plain">{doc.warnings.map((w: string, i: number) => <li key={i}>{w}</li>)}</ul>
              </Notice>
            )}
            <ol className="chunk-list">
              {detail!.chunks.map((c) => (
                <li
                  key={c.chunk_id}
                  ref={(el) => { chunkRefs.current[c.chunk_id] = el; }}
                  className={`chunk ${c.chunk_id === focusId ? "focused" : ""}`}
                >
                  <div className="chunk-head">
                    <span className="chunk-no">{c.ordinal + 1}</span>
                    <span className="chunk-kind">{kindLabel(c.kind)}</span>
                    <span className="chunk-where">{[location(c), c.heading_path, c.symbol].filter(Boolean).join(", ")}</span>
                    <span className="chunk-tokens" title={`Chunk ${c.chunk_id}`}>{c.n_tokens} tokens</span>
                  </div>
                  {c.asset_path && <FigureImage kb={kb.id} path={c.asset_path} />}
                  <Passage text={c.text} kind={c.kind} />
                  {c.ocr_conf != null && c.ocr_conf < 0.8 && (
                    <p className="chunk-note">Read by OCR with {Math.round(c.ocr_conf * 100)}% confidence.</p>
                  )}
                </li>
              ))}
            </ol>
          </>
        )}
      </section>
    </div>
  );
}
