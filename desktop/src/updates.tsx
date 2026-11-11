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

let state: UpdateState = { phase: inTauri ? "idle" : "unsupported" };
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

const PRE_KEY = "rag.updates.prerelease";
let preSession: boolean | null = null;

/** Also offer pre-releases (off unless chosen). */
export function wantPrereleases(): boolean {
  if (preSession != null) return preSession;
  try {
    return localStorage.getItem(PRE_KEY) === "1";
  } catch {
    return false;
  }
}

export function setWantPrereleases(on: boolean) {
  preSession = on;
  try {
    localStorage.setItem(PRE_KEY, on ? "1" : "0");
  } catch {
    /* kept for this session */
  }
}

/** "0.4.3-beta.1" is a pre-release; "0.4.3" isn't. */
export function isPrerelease(version?: string | null): boolean {
  return !!version && /^\d+\.\d+\.\d+-/.test(version.replace(/^v/, ""));
}

/** A small "Pre-release" mark beside a version. */
export function PrereleaseTag({ version }: { version?: string | null }) {
  return isPrerelease(version) ? <span className="prerelease-tag" title="A test version, published before the next normal release">Pre-release</span> : null;
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
    const { invoke } = await import("@tauri-apps/api/core");
    const found = await invoke<{ version: string; notes?: string | null } | null>("update_check", { prerelease: wantPrereleases() });
    if (found) set({ phase: "available", version: found.version, notes: found.notes ?? "", checkedAt: Date.now(), dismissed: false });
    else set({ phase: "none", checkedAt: Date.now() });
  } catch (e) {
    set(manual ? { phase: "error", error: String(e) } : { phase: "idle" });
  }
}

/** Download, stop the engine and its model servers, install, restart. */
export async function installUpdate(): Promise<void> {
  if (state.phase !== "available" && !(state.phase === "error" && state.version)) return;
  set({ phase: "downloading", got: 0, total: undefined, error: undefined });
  const { listen } = await import("@tauri-apps/api/event");
  const stop = await listen<{ got: number; total?: number | null }>("update-progress", (ev) => {
    set({ got: ev.payload.got, total: ev.payload.total ?? undefined });
    if (ev.payload.total && ev.payload.got >= ev.payload.total) set({ phase: "installing" });
  });
  try {
    const { invoke } = await import("@tauri-apps/api/core");
    // downloads, checks the signature, stops the engine and its model servers, installs
    await invoke("update_install");
    set({ phase: "installing" });
    const { relaunch } = await import("@tauri-apps/plugin-process");
    await relaunch();
  } catch (e) {
    set({ phase: "error", error: String(e) });
  } finally {
    stop();
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
          <strong>RAG Workbench {s.version}</strong> <PrereleaseTag version={s.version} /> is available.
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
