import { Database, Package } from "lucide-react";
import { useApp, type Tab } from "../state";
import type { KbSummary } from "../types";
import { Button, Notice } from "../components/ui";
import { SearchView } from "./SearchView";
import { DocumentsView } from "./DocumentsView";
import { TablesView } from "./TablesView";
import { EvalView } from "./EvalView";
import { BuildView } from "./BuildView";
import { AboutView } from "./AboutView";
import { bytes, count, when } from "../format";

const TABS: { id: Tab; label: string; workspaceOnly?: boolean }[] = [
  { id: "search", label: "Search" },
  { id: "documents", label: "Documents" },
  { id: "tables", label: "Tables" },
  { id: "evaluate", label: "Evaluate" },
  { id: "build", label: "Build", workspaceOnly: true },
  { id: "about", label: "Details" },
];

export function RuntimeGate({ kb }: { kb: KbSummary }) {
  const { runtime, setScreen } = useApp();
  const rt = kb.runtime;
  if (!rt.needs_runtime || (rt.ready && rt.match)) return null;
  if (!rt.match) {
    return (
      <Notice
        tone="error"
        title="Different embedding model"
        action={<Button onClick={() => setScreen({ kind: "settings" })}>Open settings</Button>}
      >
        This knowledge base was embedded with <code>{rt.kb_model}</code>, but the app's model is{" "}
        <code>{rt.runtime_model}</code>. Searches with a different model return unrelated passages.
      </Notice>
    );
  }
  return (
    <Notice
      tone="warn"
      title={runtime?.needs_setup ? "The embedding model isn't set up yet" : "The embedding model isn't running"}
      action={
        <Button kind="primary" onClick={() => setScreen({ kind: "settings" })}>
          {runtime?.needs_setup ? "Set up" : "Open settings"}
        </Button>
      }
    >
      Searching, evaluating and building need {rt.kb_model}. Browsing documents and tables works without it.
    </Notice>
  );
}

export function KbView({ kb }: { kb: KbSummary }) {
  const { tabs, setTab } = useApp();
  const available = TABS.filter((t) => !t.workspaceOnly || kb.kind === "workspace");
  const tab = available.some((t) => t.id === tabs[kb.id]) ? tabs[kb.id] : kb.counts.documents === 0 && kb.kind === "workspace" ? "build" : "search";

  return (
    <div className="kb-view">
      <header className="kb-head">
        <div className="kb-title">
          {kb.kind === "pack" ? <Package size={22} /> : <Database size={22} />}
          <div>
            <h1>{kb.name}</h1>
            <p className="kb-facts">
              <span>{kb.kind === "pack" ? "Pack, read-only" : "Workspace"}</span>
              <span>{count(kb.counts.documents, "document")}</span>
              <span>{count(kb.counts.chunks, "chunk")}</span>
              {kb.counts.tables > 0 && <span>{count(kb.counts.tables, "table")}</span>}
              <span>{bytes(kb.size_bytes)}</span>
              {kb.last_build_at && <span>Built {when(kb.last_build_at)}</span>}
            </p>
          </div>
        </div>
      </header>
      <nav className="tabs" role="tablist" aria-label={`${kb.name} views`}>
        {available.map((t) => (
          <button
            key={t.id}
            role="tab"
            aria-selected={tab === t.id}
            className={`tab ${tab === t.id ? "active" : ""}`}
            onClick={() => setTab(kb.id, t.id)}
          >
            {t.label}
            {t.id === "build" && kb.building && <span className="tab-dot" aria-label="running" />}
          </button>
        ))}
      </nav>
      <div className="kb-body" role="tabpanel">
        {tab === "search" && <SearchView kb={kb} />}
        {tab === "documents" && <DocumentsView kb={kb} />}
        {tab === "tables" && <TablesView kb={kb} />}
        {tab === "evaluate" && <EvalView kb={kb} />}
        {tab === "build" && kb.kind === "workspace" && <BuildView kb={kb} />}
        {tab === "about" && <AboutView kb={kb} />}
      </div>
    </div>
  );
}
