import { FilePlus, FolderInput, FolderPlus, Square, Trash2 } from "lucide-react";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { call, EngineCallError } from "../api";
import { useApp } from "../state";
import { terminal, useTerminal } from "../terminalStore";
import { Button } from "./ui";
import { pickPaths } from "../platform";

const SUBCOMMANDS = ["build", "check-embedder", "describe", "eval", "export", "init", "remove", "search", "sql",
  "status", "tables", "unpack", "help", "--version"];

function quote(p: string): string {
  return /[\s"']/.test(p) ? `"${p.replace(/"/g, '\\"')}"` : p;
}

const EXAMPLES = (kb?: string) => [
  "help",
  kb ? `status ${quote(kb)}` : "status <workspace>",
  kb ? `search ${quote(kb)} "what's new in 8.6"` : 'search <pack> "question"',
  kb ? `tables ${quote(kb)}` : "tables <pack>",
];

/** A terminal that runs rag commands only: the prompt starts with a fixed "rag". */
export function Terminal() {
  const { kbs, refreshKb, info } = useApp();
  const { entries, cwd, history } = useTerminal();
  const [input, setInput] = useState("");
  const [histPos, setHistPos] = useState(-1);
  const [submitting, setSubmitting] = useState(false);
  const stick = useRef(true);   // follow new output unless the user scrolled up to read
  const inputRef = useRef<HTMLInputElement>(null);
  const outRef = useRef<HTMLDivElement>(null);
  const running = entries.find((e) => e.state === "running") ?? (submitting ? ({ runId: null } as const) : undefined);
  const shownCwd = cwd ?? "your home folder";

  useEffect(() => inputRef.current?.focus(), []);
  // Builds can change open knowledge bases: refresh them when a command finishes.
  useEffect(() => terminal.onExit(() => kbs.forEach((k) => refreshKb(k.id))), [kbs, refreshKb]);
  useLayoutEffect(() => {
    const el = outRef.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  });

  const run = async (raw: string) => {
    const command = raw.trim().replace(/^rag(\.exe)?\s+/i, "");
    if (!command || submitting) return;
    setInput("");
    setHistPos(-1);
    setSubmitting(true);
    stick.current = true;
    try {
      const r = await call<{ run_id: string; cwd: string; display: string }>("terminal.run", { command, cwd });
      terminal.start(command, r.cwd, r.run_id);
    } catch (e) {
      const msg = e instanceof EngineCallError ? e.message : String(e);
      terminal.start(command, cwd ?? "", null, msg);
    } finally {
      setSubmitting(false);
    }
  };

  const insert = (text: string) => {
    const el = inputRef.current;
    const at = el?.selectionStart ?? input.length;
    const before = input.slice(0, at);
    const pad = before && !before.endsWith(" ") ? " " : "";
    const next = before + pad + text + " " + input.slice(at).trimStart();
    setInput(next);
    requestAnimationFrame(() => {
      el?.focus();
      const pos = (before + pad + text + " ").length;
      el?.setSelectionRange(pos, pos);
    });
  };

  const onKey = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      if (!running) run(input);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      const pos = Math.min(histPos + 1, history.length - 1);
      if (pos >= 0) {
        setHistPos(pos);
        setInput(history[pos]);
      }
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      const pos = histPos - 1;
      setHistPos(Math.max(pos, -1));
      setInput(pos >= 0 ? history[pos] : "");
    } else if (e.key === "Tab") {
      const word = input.trimStart();
      if (!word.includes(" ")) {
        e.preventDefault();
        const hits = SUBCOMMANDS.filter((s) => s.startsWith(word));
        if (hits.length === 1) setInput(hits[0] + " ");
      }
    } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "c" && running && !window.getSelection()?.toString()) {
      e.preventDefault();
      if (running.runId) call("terminal.stop", { run_id: running.runId });
    } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "l") {
      e.preventDefault();
      terminal.clear();
    }
  };

  const firstKb = kbs[0]?.path;

  return (
    <div className="terminal-wrap">
      <div className="terminal-bar">
        <div className="path-row grow">
          <span className="field-label">Runs in</span>
          <code className="path" title={shownCwd}>{shownCwd}</code>
          <Button kind="ghost" icon={<FolderInput size={16} />} disabled={!!running} onClick={async () => {
            const p = await pickPaths({ title: "Working folder for rag commands", directory: true });
            if (p?.[0]) terminal.setCwd(p[0]);
          }}>Change…</Button>
        </div>
      </div>
      <div className="terminal-bar">
        <span className="field-label">Insert a path</span>
        <Button kind="ghost" icon={<FilePlus size={16} />} onClick={async () => {
          const p = await pickPaths({ title: "Insert a file path", multiple: true });
          if (p) insert(p.map(quote).join(" "));
        }}>File…</Button>
        <Button kind="ghost" icon={<FolderPlus size={16} />} onClick={async () => {
          const p = await pickPaths({ title: "Insert a folder path", directory: true, multiple: true });
          if (p) insert(p.map(quote).join(" "));
        }}>Folder…</Button>
        {kbs.length > 0 && (
          <select
            aria-label="Insert an open knowledge base"
            value=""
            onChange={(e) => {
              if (e.target.value) insert(quote(e.target.value));
            }}
          >
            <option value="">Open knowledge base…</option>
            {kbs.map((k) => <option key={k.id} value={k.path}>{k.name}</option>)}
          </select>
        )}
        <span className="grow" />
        <Button kind="ghost" icon={<Trash2 size={16} />} onClick={() => terminal.clear()} title="Ctrl+L">Clear</Button>
      </div>

      <div
        className="terminal"
        ref={outRef}
        role="log"
        aria-live="polite"
        onScroll={(e) => {
          const el = e.currentTarget;
          stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
        }}
        onClick={() => !window.getSelection()?.toString() && inputRef.current?.focus()}
      >
        {entries.length === 0 && (
          <div className="term-welcome">
            <p>Runs rag {info?.version ?? ""} commands, and only rag commands: type what comes after <b>rag</b>.</p>
            <p>Try:</p>
            <ul>
              {EXAMPLES(firstKb).map((ex) => (
                <li key={ex}><button className="term-example" onClick={() => setInput(ex)}>rag {ex}</button></li>
              ))}
            </ul>
            <p className="term-dim">Up and down recall earlier commands. Tab completes a command name. Ctrl+C stops a running command; Ctrl+L clears.</p>
          </div>
        )}
        {entries.map((e, i) => (
          <section key={e.runId ?? `r${i}`} className={`term-entry term-${e.state}`}>
            <div className="term-cmd"><span className="term-prompt">rag</span> {e.command}</div>
            {e.lines.map((ln, j) => (
              <div key={j} className={`term-line term-${ln.stream}`}>{ln.text || " "}</div>
            ))}
            {e.state !== "running" && e.state !== "rejected" && (
              <div className="term-status">
                {e.state === "done" && `Finished in ${e.seconds} s`}
                {e.state === "failed" && `Exited with code ${e.code} after ${e.seconds} s`}
                {e.state === "stopped" && `Stopped after ${e.seconds} s`}
              </div>
            )}
            {e.state === "running" && <div className="term-status term-busy">Running…</div>}
          </section>
        ))}
      </div>

      <form
        className={`term-input ${running ? "busy" : ""}`}
        onSubmit={(e) => {
          e.preventDefault();
          if (!running) run(input);
        }}
      >
        <span className="term-prompt" aria-hidden="true">rag</span>
        <input
          ref={inputRef}
          value={input}
          onChange={(e) => {
            setInput(e.target.value);
            setHistPos(-1);
          }}
          onKeyDown={onKey}
          spellCheck={false}
          autoCapitalize="off"
          autoComplete="off"
          aria-label="rag command"
          placeholder={running ? "Wait for the command to finish, or stop it" : "search my-kb.ragpack \"question\""}
        />
        {running ? (
          <Button kind="ghost" icon={<Square size={14} />} onClick={() => running.runId && call("terminal.stop", { run_id: running.runId })}>
            Stop
          </Button>
        ) : (
          <Button kind="primary" type="submit" disabled={!input.trim()}>Run</Button>
        )}
      </form>
    </div>
  );
}
