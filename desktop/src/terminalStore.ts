// Terminal state lives outside React so output keeps arriving (and stays) while you
// move around the app.

import { useSyncExternalStore } from "react";
import { onEvent } from "./api";

export interface TermLine {
  stream: "stdout" | "stderr" | "info";
  text: string;
}

export interface TermEntry {
  runId: string | null;       // null: rejected before it ran
  command: string;
  cwd: string;
  lines: TermLine[];
  state: "running" | "done" | "failed" | "stopped" | "rejected";
  code?: number;
  seconds?: number;
}

const MAX_LINES = 5000;

interface State {
  entries: TermEntry[];
  cwd: string | null;
  history: string[];
}

function load<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}
function save(key: string, value: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage unavailable: not remembered */
  }
}

let state: State = { entries: [], cwd: load<string | null>("terminal:cwd", null), history: load<string[]>("terminal:history", []) };
const listeners = new Set<() => void>();
const exitListeners = new Set<(e: TermEntry) => void>();

function set(next: Partial<State>) {
  state = { ...state, ...next };
  listeners.forEach((l) => l());
}

function updateRun(runId: string, fn: (e: TermEntry) => TermEntry) {
  set({ entries: state.entries.map((e) => (e.runId === runId ? fn(e) : e)) });
}

// Output can arrive before terminal.run's reply registers the run: hold it until then.
const early = new Map<string, any[]>();

function apply(data: any) {
  if (!state.entries.some((e) => e.runId === data.run_id)) {
    early.set(data.run_id, [...(early.get(data.run_id) ?? []), data]);
    return;
  }
  {
    if (data.type === "output") {
      updateRun(data.run_id, (e) => {
        const lines = e.lines.concat(data.lines as TermLine[]);
        return { ...e, lines: lines.length > MAX_LINES ? lines.slice(lines.length - MAX_LINES) : lines };
      });
    } else if (data.type === "exit") {
      updateRun(data.run_id, (e) => ({
        ...e,
        state: data.stopped ? "stopped" : data.code === 0 ? "done" : "failed",
        code: data.code,
        seconds: data.seconds,
      }));
      const entry = state.entries.find((e) => e.runId === data.run_id);
      if (entry) exitListeners.forEach((l) => l(entry));
    }
  }
}

let wired = false;
function wire() {
  if (wired) return;
  wired = true;
  onEvent((name, data) => {
    if (name === "terminal") apply(data);
  });
}

export const terminal = {
  get: () => state,
  subscribe(l: () => void): () => void {
    wire();
    listeners.add(l);
    return () => {
      listeners.delete(l);
    };
  },
  onExit(l: (e: TermEntry) => void): () => void {
    exitListeners.add(l);
    return () => {
      exitListeners.delete(l);
    };
  },
  start(command: string, cwd: string, runId: string | null, rejected?: string) {
    const entry: TermEntry = {
      runId, command, cwd, state: rejected ? "rejected" : "running",
      lines: rejected ? [{ stream: "stderr", text: rejected }] : [],
    };
    const history = [command, ...state.history.filter((h) => h !== command)].slice(0, 200);
    save("terminal:history", history);
    set({ entries: [...state.entries, entry].slice(-100), history });
    if (runId && early.has(runId)) {
      const held = early.get(runId)!;
      early.delete(runId);
      held.forEach(apply);
    }
  },
  setCwd(cwd: string | null) {
    save("terminal:cwd", cwd);
    set({ cwd });
  },
  clear() {
    set({ entries: state.entries.filter((e) => e.state === "running") });
  },
};

export function useTerminal(): State {
  return useSyncExternalStore(terminal.subscribe, terminal.get);
}
