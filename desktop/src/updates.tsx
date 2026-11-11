// Updates: the installed app checks GitHub releases for a newer signed version, shows what's new,
// and installs it only when asked. Never silent, never during a build.

import { useEffect, useState } from "react";
import { inTauri } from "./api";
import { useApp } from "./state";
import { Button } from "./components/ui";

export type UpdatePhase = "idle" | "checking" | "none" | "available" | "downloading" | "installing" | "error" | "unsupported";

export interface UpdateState {
  phase: UpdatePhase;
  version?: string;
  notes?: string;
  error?: string;
  got?: number;
  total?: number;
  checkedAt?: number;
  dismissed?: boolean;
}

type Pending = { version: string; body?: string; download: (cb: (e: any) => void) => Promise<void>; install: () => Promise<void> };

let state: UpdateState = { phase: inTauri ? "idle" : "unsupported" };
let pending: Pending | null = null;
const listeners = new Set<(s: UpdateState) => void>();

function set(next: Partial<UpdateState>) {
  state = { ...state, ...next };
  listeners.forEach((l) => l(state));
}

export function useUpdates(): UpdateState {
  const [s, setS] = useState(state);
  useEffect(() => {
    listeners.add(setS);
    return () => void listeners.delete(setS);
  }, []);
  return s;
}

const AUTO_KEY = "rag.updates.auto";

export function autoCheck(): boolean {
  try {
    return localStorage.getItem(AUTO_KEY) !== "0";
  } catch {
    return true;
  }
}

export function setAutoCheck(on: boolean) {
  try {
    localStorage.setItem(AUTO_KEY, on ? "1" : "0");
  } catch {
    /* the switch still works for this session */
  }
}

/** Ask GitHub for a newer version. A check made at start-up stays quiet when it fails. */
export async function checkForUpdate(manual: boolean): Promise<void> {
  if (!inTauri) {
    set({ phase: "unsupported" });
    return;
  }
  if (state.phase === "downloading" || state.phase === "installing") return;
  set({ phase: "checking", error: undefined });
  try {
    const { check } = await import("@tauri-apps/plugin-updater");
    const found = (await check()) as Pending | null;
    pending = found;
    if (found) set({ phase: "available", version: found.version, notes: found.body ?? "", checkedAt: Date.now(), dismissed: false });
    else set({ phase: "none", checkedAt: Date.now() });
  } catch (e) {
    set(manual ? { phase: "error", error: String(e) } : { phase: "idle" });
  }
}

/** Download, stop the engine and its model servers, install, restart. */
export async function installUpdate(): Promise<void> {
  if (!pending) return;
  let got = 0;
  set({ phase: "downloading", got: 0, total: undefined, error: undefined });
  try {
    await pending.download((ev: any) => {
      if (ev.event === "Started") set({ total: ev.data?.contentLength ?? undefined });
      if (ev.event === "Progress") {
        got += ev.data?.chunkLength ?? 0;
        set({ got });
      }
    });
    set({ phase: "installing" });
    const { invoke } = await import("@tauri-apps/api/core");
    await invoke("engine_stop");      // its files are about to be replaced; it starts again on demand
    await pending.install();
    const { relaunch } = await import("@tauri-apps/plugin-process");
    await relaunch();
  } catch (e) {
    set({ phase: "error", error: String(e) });
  }
}

/** Checks once, a few seconds after the app starts, unless switched off in Settings. */
export function useStartupCheck() {
  useEffect(() => {
    if (!inTauri || !autoCheck()) return;
    const t = window.setTimeout(() => void checkForUpdate(false), 5000);
    return () => window.clearTimeout(t);
  }, []);
}

/** The bar at the top of the window when a newer version is available. */
export function UpdateBanner() {
  const s = useUpdates();
  const { jobs } = useApp();
  const [open, setOpen] = useState(false);
  const busy = Object.values(jobs).some((j) => j.state === "running" && ["build", "merge", "export", "convert"].includes(j.kind));
  if (s.dismissed || !["available", "downloading", "installing"].includes(s.phase) && !(s.phase === "error" && s.version)) return null;
  return (
    <div className="update-banner" role="status">
      <div className="update-row">
        <span>
          <strong>RAG Workbench {s.version}</strong> is available.
          {s.phase === "downloading" && ` Downloading${s.total ? ` ${Math.round(((s.got ?? 0) / s.total) * 100)}%` : "…"}`}
          {s.phase === "installing" && " Installing; the app will restart."}
          {s.phase === "error" && ` The update didn't install: ${s.error}`}
        </span>
        <span className="update-actions">
          {s.notes ? <button className="link-button" onClick={() => setOpen(!open)}>{open ? "Hide what's new" : "What's new"}</button> : null}
          {(s.phase === "available" || s.phase === "error") && (
            <>
              <Button kind="primary" onClick={() => void installUpdate()} disabled={busy}
                title={busy ? "Waits until the running build, merge or export finishes" : undefined}>
                {busy ? "Update after the current job" : "Update and restart"}
              </Button>
              <Button kind="ghost" onClick={() => set({ dismissed: true })}>Later</Button>
            </>
          )}
        </span>
      </div>
      {open && s.notes && <pre className="update-notes">{s.notes}</pre>}
    </div>
  );
}
