import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { call, EngineCallError, inTauri, onEvent } from "./api";
import type { AppInfo, Job, KbSummary, RecentKb, RuntimeStatus } from "./types";

export type Tab = "search" | "documents" | "tables" | "evaluate" | "build" | "about";
export type Screen = { kind: "home" } | { kind: "kb"; id: string } | { kind: "settings" } | { kind: "cli" };

export interface Toast {
  id: number;
  tone: "error" | "info" | "success";
  title: string;
  detail?: string;
}

interface AppState {
  info: AppInfo | null;
  engineError: string | null;
  runtime: RuntimeStatus | null;
  kbs: KbSummary[];
  recent: RecentKb[];
  jobs: Record<string, Job>;
  screen: Screen;
  tabs: Record<string, Tab>;
  toasts: Toast[];
  setScreen: (s: Screen) => void;
  setTab: (kb: string, t: Tab) => void;
  openKb: (path: string) => Promise<KbSummary | null>;
  createKb: (name: string, dims: number, folder?: string, vectors?: string) => Promise<KbSummary | null>;
  closeKb: (id: string) => Promise<void>;
  refreshKb: (id: string) => Promise<void>;
  refreshRuntime: () => Promise<void>;
  refreshRecent: () => Promise<void>;
  startJob: (method: string, params: Record<string, unknown>) => Promise<Job | null>;
  notify: (t: Omit<Toast, "id">) => void;
  fail: (e: unknown, title?: string) => void;
  dismiss: (id: number) => void;
}

const Ctx = createContext<AppState | null>(null);

export function useApp(): AppState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useApp outside AppProvider");
  return v;
}

export function AppProvider({ children }: { children: ReactNode }) {
  const [info, setInfo] = useState<AppInfo | null>(null);
  const [engineError, setEngineError] = useState<string | null>(null);
  const [runtime, setRuntime] = useState<RuntimeStatus | null>(null);
  const [kbs, setKbs] = useState<KbSummary[]>([]);
  const [recent, setRecent] = useState<RecentKb[]>([]);
  const [jobs, setJobs] = useState<Record<string, Job>>({});
  const [screen, setScreen] = useState<Screen>({ kind: "home" });
  const [tabs, setTabs] = useState<Record<string, Tab>>({});
  const [toasts, setToasts] = useState<Toast[]>([]);
  const toastId = useRef(0);
  const kbIds = useRef<string[]>([]);
  kbIds.current = kbs.map((k) => k.id);

  const dismiss = useCallback((id: number) => setToasts((t) => t.filter((x) => x.id !== id)), []);
  const notify = useCallback(
    (t: Omit<Toast, "id">) => {
      const id = ++toastId.current;
      setToasts((all) => [...all.slice(-3), { ...t, id }]);
      if (t.tone !== "error") setTimeout(() => dismiss(id), 5000);
    },
    [dismiss],
  );
  const fail = useCallback(
    (e: unknown, title?: string) => {
      const err = e instanceof EngineCallError ? e : null;
      notify({
        tone: "error",
        title: title ?? err?.message ?? String(e),
        detail: title ? [err?.message ?? String(e), err?.hint].filter(Boolean).join(" ") : err?.hint,
      });
    },
    [notify],
  );

  const refreshRuntime = useCallback(async () => {
    try {
      setRuntime(await call<RuntimeStatus>("runtime.status"));
    } catch (e) {
      /* shown by the engine banner */
    }
  }, []);
  const refreshRecent = useCallback(async () => {
    try {
      setRecent(await call<RecentKb[]>("kb.recent"));
    } catch {
      /* ignore */
    }
  }, []);

  const upsertKb = useCallback((s: KbSummary) => {
    setKbs((all) => (all.some((k) => k.id === s.id) ? all.map((k) => (k.id === s.id ? s : k)) : [...all, s]));
  }, []);

  const refreshKb = useCallback(
    async (id: string) => {
      try {
        upsertKb(await call<KbSummary>("kb.summary", { kb: id }));
      } catch {
        /* closed meanwhile */
      }
    },
    [upsertKb],
  );

  useEffect(() => {
    let alive = true;
    const boot = async () => {
      for (let attempt = 0; attempt < 60 && alive; attempt++) {
        try {
          const i = await call<AppInfo>("app.info");
          if (!alive) return;
          setInfo(i);
          setEngineError(null);
          const [rt, open, rec] = await Promise.all([
            call<RuntimeStatus>("runtime.status"),
            call<KbSummary[]>("kb.list"),
            call<RecentKb[]>("kb.recent"),
          ]);
          setRuntime(rt);
          setKbs(open);
          setRecent(rec);
          if (inTauri) {
            // Packs the app was opened with (double-click on a .ragpack).
            const { invoke } = await import("@tauri-apps/api/core");
            const paths = await invoke<string[]>("startup_paths").catch(() => []);
            for (const p of paths) {
              try {
                const s = await call<KbSummary>("kb.open", { path: p });
                setKbs((all) => (all.some((k) => k.id === s.id) ? all : [...all, s]));
                setScreen({ kind: "kb", id: s.id });
              } catch {
                /* shown when opened manually */
              }
            }
          }
          return;
        } catch (e) {
          setEngineError(e instanceof EngineCallError ? `${e.message} ${e.hint ?? ""}` : String(e));
          await new Promise((r) => setTimeout(r, 1000));
        }
      }
    };
    boot();
    const off = onEvent((name, data) => {
      if (name === "runtime") {
        setRuntime(data as RuntimeStatus);
        kbIds.current.forEach((id) => refreshKb(id)); // readiness shows on each knowledge base
      } else if (name === "job") {
        const j = data as Job;
        setJobs((all) => ({ ...all, [j.id]: { ...all[j.id], ...j } }));
        if (j.state !== "running" && j.kb) refreshKb(j.kb);
        if (j.kind === "runtime_setup" && j.state !== "running") refreshRuntime();
      } else if (name === "kb") {
        refreshKb((data as { id: string }).id);
      } else if (name === "engine_exit") {
        notify({
          tone: "error",
          title: "The engine stopped",
          detail: "It restarts on your next action. Open knowledge bases need reopening.",
        });
        setKbs([]);
        setScreen({ kind: "home" });
      }
    });
    return () => {
      alive = false;
      off();
    };
  }, [refreshKb, refreshRuntime, notify]);

  const openKb = useCallback(
    async (path: string) => {
      try {
        const s = await call<KbSummary>("kb.open", { path });
        upsertKb(s);
        setScreen({ kind: "kb", id: s.id });
        refreshRecent();
        return s;
      } catch (e) {
        fail(e, "Couldn't open that file");
        return null;
      }
    },
    [upsertKb, refreshRecent, fail],
  );

  const createKb = useCallback(
    async (name: string, dims: number, folder?: string, vectors: string = "sqlite-vec") => {
      try {
        const s = await call<KbSummary>("kb.create", { name, dims, folder: folder || null, vectors });
        upsertKb(s);
        setScreen({ kind: "kb", id: s.id });
        setTabs((t) => ({ ...t, [s.id]: "build" }));
        refreshRecent();
        return s;
      } catch (e) {
        fail(e, "Couldn't create the knowledge base");
        return null;
      }
    },
    [upsertKb, refreshRecent, fail],
  );

  const closeKb = useCallback(
    async (id: string) => {
      try {
        await call("kb.close", { kb: id });
        setKbs((all) => all.filter((k) => k.id !== id));
        setScreen((s) => (s.kind === "kb" && s.id === id ? { kind: "home" } : s));
      } catch (e) {
        fail(e);
      }
    },
    [fail],
  );

  const startJob = useCallback(
    async (method: string, params: Record<string, unknown>) => {
      try {
        const j = await call<Job>(method, params);
        // A quick job can finish (and its events arrive) before this reply does: keep the newer state.
        setJobs((all) => ({ ...all, [j.id]: { ...j, ...all[j.id] } }));
        return j;
      } catch (e) {
        fail(e);
        return null;
      }
    },
    [fail],
  );

  const setTab = useCallback((kb: string, t: Tab) => setTabs((all) => ({ ...all, [kb]: t })), []);

  const value = useMemo<AppState>(
    () => ({
      info, engineError, runtime, kbs, recent, jobs, screen, tabs, toasts,
      setScreen, setTab, openKb, createKb, closeKb, refreshKb, refreshRuntime, refreshRecent,
      startJob, notify, fail, dismiss,
    }),
    [info, engineError, runtime, kbs, recent, jobs, screen, tabs, toasts, setTab, openKb, createKb, closeKb,
      refreshKb, refreshRuntime, refreshRecent, startJob, notify, fail, dismiss],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

/** The latest state of one job (from events), with its log fetched when it finishes. */
export function useJob(id: string | null | undefined): Job | null {
  const { jobs } = useApp();
  const [full, setFull] = useState<Job | null>(null);
  const job = id ? jobs[id] ?? null : null;
  useEffect(() => {
    if (!id || !job) return;
    let alive = true;
    const t = setTimeout(async () => {
      try {
        const j = await call<Job>("jobs.get", { job: id });
        if (alive) setFull(j);
      } catch {
        /* ignore */
      }
    }, job.state === "running" ? 400 : 0);
    return () => {
      alive = false;
      clearTimeout(t);
    };
  }, [id, job?.state, job?.log_size]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!job) return null;
  return { ...full, ...job, result: job.result ?? full?.result, log: full?.log } as Job;
}
