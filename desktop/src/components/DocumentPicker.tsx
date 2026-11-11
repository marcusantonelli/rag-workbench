import { ChevronDown, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { call } from "../api";
import type { DocumentRow } from "../types";

/** Pick the documents a search is limited to: a list with a search field and checkboxes. Each
 * choice is a document's path; typed text can also be kept as a pattern ("WI-12" matches every
 * document whose number or name contains it), as the old text box did. */
export function DocumentPicker({ kb, value, onChange }: { kb: string; value: string[]; onChange: (v: string[]) => void }) {
  const [open, setOpen] = useState(false);
  const [docs, setDocs] = useState<DocumentRow[] | null>(null);
  const [filter, setFilter] = useState("");
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open || docs) return;
    call<DocumentRow[]>("kb.documents", { kb })
      .then((rows) => setDocs(rows.filter((d) => d.status !== "failed")))
      .catch(() => setDocs([]));
  }, [open, docs, kb]);
  useEffect(() => setDocs(null), [kb]);
  useEffect(() => {
    if (!open) return;
    const away = (e: MouseEvent) => ref.current && !ref.current.contains(e.target as Node) && setOpen(false);
    const key = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", away);
    document.addEventListener("keydown", key);
    return () => {
      document.removeEventListener("mousedown", away);
      document.removeEventListener("keydown", key);
    };
  }, [open]);

  const f = filter.trim().toLowerCase();
  const shown = useMemo(() => (docs ?? []).filter((d) => !f || d.display_path.toLowerCase().includes(f)
    || (d.title ?? "").toLowerCase().includes(f)).slice(0, 300), [docs, f]);
  const paths = new Set((docs ?? []).map((d) => d.display_path));
  const patterns = value.filter((v) => !paths.has(v));
  const toggle = (v: string) => onChange(value.includes(v) ? value.filter((x) => x !== v) : [...value, v]);
  const name = (p: string) => p.split(/[\\/]/).pop() ?? p;
  const label = value.length === 0 ? "All documents"
    : value.length === 1 ? (paths.has(value[0]) ? name(value[0]) : `Matching “${value[0]}”`)
    : `${value.length} documents`;

  return (
    <div className="doc-picker" ref={ref}>
      <button type="button" className={`doc-picker-button${value.length ? " active" : ""}`} aria-haspopup="listbox"
        aria-expanded={open} onClick={() => setOpen(!open)} title={value.join("\n") || "Search every document"}>
        <span>{label}</span>
        <ChevronDown size={14} aria-hidden="true" />
      </button>
      {value.length > 0 && (
        <button type="button" className="doc-picker-clear" aria-label="Search every document" onClick={() => onChange([])}>
          <X size={14} />
        </button>
      )}
      {open && (
        <div className="doc-picker-panel">
          <input autoFocus value={filter} onChange={(e) => setFilter(e.target.value)} spellCheck={false}
            placeholder="Find a document by name, title or number" aria-label="Find a document" />
          {f && !value.includes(filter.trim()) && (
            <button type="button" className="doc-picker-pattern" onClick={() => { toggle(filter.trim()); setFilter(""); }}>
              Any document whose number or name contains “{filter.trim()}”
            </button>
          )}
          <div className="doc-picker-list" role="listbox" aria-multiselectable="true">
            {patterns.map((v) => (
              <label key={`p:${v}`} className="doc-picker-row">
                <input type="checkbox" checked onChange={() => toggle(v)} />
                <span>Matching “{v}”</span>
              </label>
            ))}
            {docs == null && <p className="muted small">Loading…</p>}
            {docs != null && shown.length === 0 && <p className="muted small">No documents match.</p>}
            {shown.map((d) => (
              <label key={d.doc_id} className="doc-picker-row">
                <input type="checkbox" checked={value.includes(d.display_path)} onChange={() => toggle(d.display_path)} />
                <span>
                  <span className="doc-picker-name">{name(d.display_path)}</span>
                  {d.title && d.title !== name(d.display_path) && <span className="muted small"> {d.title}</span>}
                </span>
              </label>
            ))}
          </div>
          <div className="doc-picker-foot">
            <span className="muted small">{value.length ? `${value.length} chosen` : "Searching every document"}</span>
            {value.length > 0 && <button type="button" className="linklike" onClick={() => onChange([])}>Clear</button>}
          </div>
        </div>
      )}
    </div>
  );
}
