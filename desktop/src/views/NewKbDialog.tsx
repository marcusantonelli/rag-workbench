import { useState } from "react";
import { useApp } from "../state";
import { Button, Dialog } from "../components/ui";
import { pickPaths } from "../platform";

const STORES = [
  { id: "sqlite-vec", label: "Standard (sqlite-vec)", note: "Works with every retrieval server. Exact search." },
  { id: "shammy", label: "Shammy", note: "About a fifth smaller and much faster on large collections. Packs need a retrieval server with Shammy support." },
];

const SIZES = [
  { dims: 4096, label: "Full detail (4096)", note: "Best retrieval. About 16 KB per chunk." },
  { dims: 2048, label: "Balanced (2048)", note: "Half the size; a small quality cost." },
  { dims: 1024, label: "Compact (1024)", note: "Quarter size, for very large collections." },
];

export function NewKbDialog({ onClose }: { onClose: () => void }) {
  const { createKb, info, runtime } = useApp();
  const [name, setName] = useState("");
  const [dims, setDims] = useState(4096);
  const [folder, setFolder] = useState("");
  const [vectors, setVectors] = useState("sqlite-vec");
  const [busy, setBusy] = useState(false);

  const submit = async () => {
    if (!name.trim()) return;
    setBusy(true);
    const kb = await createKb(name.trim(), dims, folder || undefined, vectors);
    setBusy(false);
    if (kb) onClose();
  };

  return (
    <Dialog
      title="New knowledge base"
      onClose={onClose}
      footer={
        <>
          <Button kind="ghost" onClick={onClose}>Cancel</Button>
          <Button kind="primary" onClick={submit} disabled={!name.trim() || busy}>
            {busy ? "Creating…" : "Create knowledge base"}
          </Button>
        </>
      }
    >
      <form
        className="form"
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
      >
        <label className="field">
          <span className="field-label">Name</span>
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Product manuals" />
        </label>

        <fieldset className="field">
          <legend className="field-label">Vector size</legend>
          <div className="choice-list">
            {SIZES.map((s) => (
              <label key={s.dims} className={`choice ${dims === s.dims ? "chosen" : ""}`}>
                <input type="radio" name="dims" checked={dims === s.dims} onChange={() => setDims(s.dims)} />
                <span>
                  <span className="choice-title">{s.label}</span>
                  <span className="choice-note">{s.note}</span>
                </span>
              </label>
            ))}
          </div>
          <span className="field-help">Fixed once the knowledge base is built; changing it means a full rebuild.</span>
        </fieldset>

        {info?.shammy && (
          <fieldset className="field">
            <legend className="field-label">Vector store</legend>
            <div className="choice-list">
              {STORES.map((s) => (
                <label key={s.id} className={`choice ${vectors === s.id ? "chosen" : ""}`}>
                  <input type="radio" name="vectors" checked={vectors === s.id} onChange={() => setVectors(s.id)} />
                  <span>
                    <span className="choice-title">{s.label}</span>
                    <span className="choice-note">{s.note}</span>
                  </span>
                </label>
              ))}
            </div>
            <span className="field-help">Can be changed later without embedding anything again (Details tab).</span>
          </fieldset>
        )}

        <div className="field">
          <span className="field-label">Folder</span>
          <div className="path-row">
            <code className="path">{folder || `${info?.workspaces_dir ?? "App data"}/${name ? name.toLowerCase().replace(/[^a-z0-9]+/g, "_") : "…"}`}</code>
            <Button
              onClick={async () => {
                const p = await pickPaths({ title: "Choose an empty folder", directory: true });
                if (p?.[0]) setFolder(p[0]);
              }}
            >
              Choose…
            </Button>
          </div>
        </div>
        <p className="field-help">
          Embedded with {runtime?.model_id ?? "the app's embedding model"}. Packs exported from it need the same model to
          search.
        </p>
      </form>
    </Dialog>
  );
}
