import { Check, Copy } from "lucide-react";
import { useCallback, useEffect, useState, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import guide from "../../../docs/cli.md?raw";
import { call } from "../api";
import { useApp } from "../state";
import { Notice, Switch } from "../components/ui";

interface CliStatus {
  mode: "windows" | "linux" | "appimage" | "development";
  available: boolean;
  on_path: boolean;
  can_toggle: boolean;
  path: string | null;
  note: string;
  embedding_url: string | null;
}

function textOf(node: ReactNode): string {
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(textOf).join("");
  if (node && typeof node === "object" && "props" in node) return textOf((node as any).props.children);
  return "";
}

/** A code block with a copy button (commands are meant to be pasted into a terminal). */
function CodeBlock({ children }: { children?: ReactNode }) {
  const [copied, setCopied] = useState(false);
  const text = textOf(children).replace(/\n$/, "");
  return (
    <div className="code-block">
      <pre>{children}</pre>
      <button
        className="copy-btn"
        aria-label="Copy"
        onClick={async () => {
          try {
            await navigator.clipboard.writeText(text);
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
          } catch {
            /* clipboard unavailable */
          }
        }}
      >
        {copied ? <Check size={14} /> : <Copy size={14} />}
        <span>{copied ? "Copied" : "Copy"}</span>
      </button>
    </div>
  );
}

export function CommandLineView() {
  const { fail, runtime } = useApp();
  const [status, setStatus] = useState<CliStatus | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(() => {
    call<CliStatus>("cli.status").then(setStatus).catch(fail);
  }, [fail]);
  useEffect(load, [load, runtime?.state]);

  const toggle = async (enabled: boolean) => {
    setBusy(true);
    try {
      setStatus(await call<CliStatus>("cli.set_enabled", { enabled }));
    } catch (e) {
      fail(e, "Couldn't change the PATH");
    } finally {
      setBusy(false);
    }
  };

  const port = status?.embedding_url ? new URL(status.embedding_url).port : null;

  return (
    <div className="cli-view">
      <h1>Command line</h1>
      <section className="panel">
        <h2>The rag command on this computer</h2>
        {!status ? (
          <p className="muted">Checking…</p>
        ) : (
          <>
            {status.can_toggle ? (
              <div className="row options">
                <Switch checked={status.on_path} disabled={busy} onChange={toggle} label="Add the rag command to PATH" />
              </div>
            ) : (
              <p className={status.on_path ? "state state-ready" : "muted"}>
                {status.on_path ? "Ready to use in a terminal." : "Not on the PATH."}
              </p>
            )}
            <p className="muted">{status.note}</p>
            {status.path && (
              <p className="small">
                Program: <code>{status.path}</code>
              </p>
            )}
          </>
        )}
        {status && (
          status.embedding_url ? (
            port === "8081" ? (
              <Notice tone="success" title="rag can use the app's embedding model">
                It's running at <code>{status.embedding_url}</code>, the address workspaces use by default, while this
                app is open.
              </Notice>
            ) : (
              <Notice tone="warn" title={`The embedding model is on port ${port}`}>
                Port 8081 was busy, so rag won't find it by default. Free the port and restart the model, or set
                <code> url = "{status.embedding_url}"</code> under <code>[embedding]</code> in the workspace's
                config.toml.
              </Notice>
            )
          ) : (
            <Notice tone="info" title="Start the embedding model to build and search from a terminal">
              Browsing commands (tables, describe, sql, status) work without it.
            </Notice>
          )
        )}
      </section>
      <article className="guide passage-sans">
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={{ pre: CodeBlock }}>
          {guide.replace(/^# .*\n/, "")}
        </ReactMarkdown>
      </article>
    </div>
  );
}
