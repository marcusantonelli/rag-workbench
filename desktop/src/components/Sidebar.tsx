import { Database, FolderOpen, Package, Plus, Settings, SquareTerminal, X } from "lucide-react";
import { useState } from "react";
import { useApp } from "../state";
import { PACK_FILTER, pickPaths } from "../platform";
import { NewKbDialog } from "../views/NewKbDialog";
import { RuntimeBadge } from "./RuntimeBadge";
import { JobsTray } from "./JobsTray";
import { PrereleaseTag } from "../updates";

export function Sidebar() {
  const { kbs, screen, setScreen, openKb, closeKb, info } = useApp();
  const [creating, setCreating] = useState(false);

  const open = async () => {
    const paths = await pickPaths({ title: "Open a knowledge base", filters: PACK_FILTER });
    if (paths?.[0]) await openKb(paths[0]);
  };
  const openFolder = async () => {
    const paths = await pickPaths({ title: "Open a workspace folder", directory: true });
    if (paths?.[0]) await openKb(paths[0]);
  };

  return (
    <aside className="sidebar">
      <button className="brand" onClick={() => setScreen({ kind: "home" })}>
        <span className="brand-mark" aria-hidden="true" />
        <span>RAG Workbench</span>
        <PrereleaseTag version={info?.version} />
      </button>

      <div className="side-actions">
        <button className="side-action" onClick={open}>
          <Package size={16} />
          <span>Open pack</span>
        </button>
        <button className="side-action" onClick={() => setCreating(true)}>
          <Plus size={16} />
          <span>New knowledge base</span>
        </button>
      </div>

      <nav className="kb-list" aria-label="Open knowledge bases">
        {kbs.length === 0 && <p className="side-empty">Open a .ragpack file or create a knowledge base to start.</p>}
        {kbs.map((k) => {
          const active = screen.kind === "kb" && screen.id === k.id;
          return (
            <div key={k.id} className={`kb-item ${active ? "active" : ""}`}>
              <button className="kb-item-main" onClick={() => setScreen({ kind: "kb", id: k.id })} title={k.path}>
                {k.kind === "pack" ? <Package size={16} /> : <Database size={16} />}
                <span className="kb-item-text">
                  <span className="kb-item-name">{k.name}</span>
                  <span className="kb-item-sub">
                    {k.building ? "Building…" : `${k.counts.documents.toLocaleString()} documents`}
                  </span>
                </span>
              </button>
              <button className="icon-btn kb-close" onClick={() => closeKb(k.id)} aria-label={`Close ${k.name}`}>
                <X size={14} />
              </button>
            </div>
          );
        })}
        <button className="side-link" onClick={openFolder}>
          <FolderOpen size={14} />
          <span>Open a workspace folder</span>
        </button>
      </nav>

      <div className="side-foot">
        <JobsTray />
        <RuntimeBadge />
        <button
          className={`side-action ${screen.kind === "cli" ? "active" : ""}`}
          onClick={() => setScreen({ kind: "cli" })}
        >
          <SquareTerminal size={16} />
          <span>Command line</span>
        </button>
        <button
          className={`side-action ${screen.kind === "settings" ? "active" : ""}`}
          onClick={() => setScreen({ kind: "settings" })}
        >
          <Settings size={16} />
          <span>Settings</span>
        </button>
      </div>
      {creating && <NewKbDialog onClose={() => setCreating(false)} />}
    </aside>
  );
}
