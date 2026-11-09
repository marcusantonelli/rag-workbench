import { Database, Package } from "lucide-react";
import { useEffect, useState } from "react";
import { useApp } from "../state";
import { Button, Notice } from "../components/ui";
import { PACK_FILTER, pickPaths } from "../platform";
import { NewKbDialog } from "./NewKbDialog";
import { count, when } from "../format";
import { call } from "../api";

interface Found {
  path: string;
  kind: "pack" | "workspace";
  name: string;
  documents: number | null;
  chunks: number | null;
  tables: number | null;
  updated: string | null;
  note: string;
}

export function Home() {
  const { recent, openKb, runtime, setScreen, refreshRecent, info, kbs } = useApp();
  const [creating, setCreating] = useState(false);
  const [found, setFound] = useState<{ folder: string; items: Found[] } | null>(null);
  useEffect(() => {
    call<{ folder: string; items: Found[] }>("kb.discover").then(setFound).catch(() => setFound(null));
  }, [kbs.length, creating]);

  const recentPaths = new Set(recent.map((r) => r.path));
  const notRecent = (found?.items ?? []).filter((r) => !recentPaths.has(r.path));

  const openPack = async () => {
    const p = await pickPaths({ title: "Open a knowledge base", filters: PACK_FILTER });
    if (p?.[0]) openKb(p[0]);
  };

  return (
    <div className="home">
      <section className="home-hero">
        <h1>Build, inspect and test your knowledge bases.</h1>
        <p>
          Open a <code>.ragpack</code> to search it and see exactly which passages your agent will receive, or turn a
          folder of documents into a new pack.
        </p>
        <div className="home-actions">
          <Button kind="primary" icon={<Package size={16} />} onClick={openPack}>
            Open a pack
          </Button>
          <Button icon={<Database size={16} />} onClick={() => setCreating(true)}>
            New knowledge base
          </Button>
        </div>
        <p className="home-drop">You can also drop a .ragpack file anywhere in this window.</p>
      </section>

      {runtime?.needs_setup && (
        <Notice
          tone="warn"
          title="Set up the embedding model to search and build"
          action={<Button kind="primary" onClick={() => setScreen({ kind: "settings" })}>Set up</Button>}
        >
          Searching needs the same model your packs were built with ({runtime.default_model.id}). The app downloads
          llama.cpp and the {Math.round(runtime.default_model.approx_bytes / 1e9 * 10) / 10} GB model once, or uses a
          .gguf file you already have.
        </Notice>
      )}

      <section className="home-recent">
        <h2>Recent</h2>
        {recent.length === 0 ? (
          <p className="muted">Nothing opened yet.</p>
        ) : (
          <ul className="recent-list">
            {recent.map((r) => (
              <li key={r.path} className={r.exists ? "" : "missing"}>
                <button className="recent-main" onClick={() => openKb(r.path)} disabled={!r.exists} title={r.path}>
                  {r.kind === "pack" ? <Package size={16} /> : <Database size={16} />}
                  <span className="recent-text">
                    <span className="recent-name">{r.name}</span>
                    <span className="recent-path">{r.exists ? r.path : `Moved or deleted: ${r.path}`}</span>
                  </span>
                  <span className="recent-when">{when(r.opened_at)}</span>
                </button>
                <button
                  className="link-btn"
                  onClick={async () => {
                    await call("kb.forget", { path: r.path });
                    refreshRecent();
                  }}
                >
                  Remove
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>
      {notRecent.length > 0 && (
        <section className="home-recent">
          <h2>{recent.length ? "Also in your workspaces folder" : "In your workspaces folder"}</h2>
          <p className="muted small" title={found!.folder}>{found!.folder}</p>
          <ul className="recent-list">
            {notRecent.map((r) => (
              <li key={r.path}>
                <button className="recent-main" onClick={() => openKb(r.path)} title={r.path}>
                  {r.kind === "pack" ? <Package size={16} /> : <Database size={16} />}
                  <span className="recent-text">
                    <span className="recent-name">{r.name}</span>
                    <span className="recent-path">
                      {r.documents != null ? `${count(r.documents, "document")}, ${count(r.chunks ?? 0, "chunk")}, ${count(r.tables ?? 0, "table")}` : ""}
                      {r.note ? ` (${r.note})` : ""}
                    </span>
                  </span>
                  <span className="recent-when">{r.updated ? when(r.updated) : ""}</span>
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}
      {info && <p className="home-foot">Version {info.version}. Data folder: {info.data_dir}</p>}
      {creating && <NewKbDialog onClose={() => setCreating(false)} />}
    </div>
  );
}
