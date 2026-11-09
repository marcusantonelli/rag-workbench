import { Play } from "lucide-react";
import { useEffect, useState } from "react";
import { call, EngineCallError } from "../api";
import { useApp } from "../state";
import type { KbSummary, SqlResult, TableCatalogRow, TableDescription } from "../types";
import { Button, Empty, Notice } from "../components/ui";

function ResultGrid({ columns, rows }: { columns: string[]; rows: unknown[][] }) {
  return (
    <div className="grid-wrap">
      <table className="grid">
        <thead>
          <tr>{columns.map((c) => <th key={c}>{c}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>{r.map((v, j) => <td key={j} className={typeof v === "number" ? "num" : typeof v === "string" && v.length <= 24 ? "nowrap" : ""}>{v == null ? "" : String(v)}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function TablesView({ kb }: { kb: KbSummary }) {
  const { fail } = useApp();
  const [tables, setTables] = useState<TableCatalogRow[]>([]);
  const [name, setName] = useState<string | null>(null);
  const [desc, setDesc] = useState<TableDescription | null>(null);
  const [sql, setSql] = useState("");
  const [result, setResult] = useState<SqlResult | null>(null);
  const [sqlError, setSqlError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    call<TableCatalogRow[]>("kb.tables", { kb: kb.id }).then(setTables).catch(fail);
  }, [kb.id, kb.counts.tables, fail]);

  useEffect(() => {
    if (!name) return;
    call<TableDescription>("kb.table", { kb: kb.id, name }).then((d) => {
      setDesc(d);
      setSql(`SELECT *\nFROM "${d.table_name}"\nLIMIT 50`);
      setResult(null);
      setSqlError(null);
    }).catch(fail);
  }, [kb.id, name, fail]);

  const run = async () => {
    setRunning(true);
    setSqlError(null);
    try {
      setResult(await call<SqlResult>("kb.sql", { kb: kb.id, sql }));
    } catch (e) {
      setResult(null);
      setSqlError(e instanceof EngineCallError ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  };

  if (tables.length === 0) {
    return (
      <Empty title="No tables in this knowledge base">
        Spreadsheets, CSV files and large tables inside documents are stored here as SQL tables the agent can query.
      </Empty>
    );
  }

  return (
    <div className="split tables-view">
      <ul className="table-list" aria-label="Tables">
        {tables.map((t) => (
          <li key={t.table_name}>
            <button className={`doc-row ${t.table_name === name ? "selected" : ""}`} onClick={() => setName(t.table_name)}>
              <span className="doc-text">
                <code className="doc-name">{t.table_name}</code>
                <span className="doc-dir">
                  {[t.source_path.split(/[\\/]/).pop(), t.sheet && `sheet ${t.sheet}`, t.page && `p. ${t.page}`].filter(Boolean).join(", ")}
                </span>
              </span>
              <span className="doc-meta">{t.n_rows.toLocaleString()} rows</span>
            </button>
          </li>
        ))}
      </ul>
      <section className="table-detail">
        {!desc ? (
          <Empty title="Pick a table">See its columns and try the SQL your agent could run against it.</Empty>
        ) : (
          <>
            <header className="evidence-head">
              <div>
                <h2><code>{desc.table_name}</code></h2>
                <p className="evidence-where">
                  {[desc.source_path, desc.sheet && `sheet ${desc.sheet}`, desc.cell_range, desc.page && `p. ${desc.page}`,
                    `${desc.n_rows.toLocaleString()} rows`].filter(Boolean).join(", ")}
                </p>
                {desc.caption && <p>{desc.caption}</p>}
              </div>
            </header>
            <table className="grid columns">
              <thead><tr><th>Column</th><th>Type</th><th>Original header</th><th>Range or examples</th></tr></thead>
              <tbody>
                {desc.columns.map((c) => (
                  <tr key={c.name}>
                    <td><code>{c.name}</code></td>
                    <td>{c.type}</td>
                    <td>{c.header}</td>
                    <td className="cell-text">
                      {c.min != null && c.max != null && c.type !== "text" ? `${c.min} to ${c.max}` : (c.examples ?? []).slice(0, 3).join(", ")}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="sql-box">
              <label className="field-label" htmlFor="sql">Read-only SQL</label>
              <textarea
                id="sql"
                value={sql}
                spellCheck={false}
                onChange={(e) => setSql(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) run();
                }}
              />
              <div className="sql-actions">
                <span className="muted">Ctrl+Enter runs it. Only SELECT is allowed, as for the agent.</span>
                <Button kind="primary" icon={<Play size={16} />} onClick={run} disabled={running}>Run query</Button>
              </div>
            </div>
            {sqlError && <Notice tone="error" title="Query rejected">{sqlError}</Notice>}
            {result && (
              <>
                <p className="muted">
                  {result.row_count.toLocaleString()} rows in {result.elapsed_ms} ms{result.truncated ? " (first 500 shown)" : ""}
                </p>
                <ResultGrid columns={result.columns} rows={result.rows} />
              </>
            )}
          </>
        )}
      </section>
    </div>
  );
}
