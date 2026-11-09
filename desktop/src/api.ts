// One way to talk to the engine, two transports:
//  - inside the desktop app: Tauri commands relayed to the engine process over stdio
//  - in a browser (npm run dev + npm run engine:dev): HTTP + Server-Sent Events

import type { EngineError } from "./types";

type EventHandler = (name: string, data: any) => void;

export const inTauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
const ENGINE_URL = (import.meta.env.VITE_ENGINE_URL as string | undefined) ?? "http://127.0.0.1:8765";

export class EngineCallError extends Error implements EngineError {
  code: string;
  hint?: string;
  data?: Record<string, unknown>;
  constructor(err: EngineError) {
    super(err.message);
    this.code = err.code;
    this.hint = err.hint;
    this.data = err.data;
  }
}

function asError(raw: unknown): EngineCallError {
  if (raw && typeof raw === "object" && "message" in raw) return new EngineCallError(raw as EngineError);
  if (typeof raw === "string") {
    try {
      return asError(JSON.parse(raw));
    } catch {
      return new EngineCallError({ code: "error", message: raw });
    }
  }
  return new EngineCallError({ code: "error", message: String(raw) });
}

export async function call<T = any>(method: string, params: Record<string, unknown> = {}): Promise<T> {
  if (inTauri) {
    const { invoke } = await import("@tauri-apps/api/core");
    try {
      return await invoke<T>("engine_call", { method, params });
    } catch (e) {
      throw asError(e);
    }
  }
  let resp: Response;
  try {
    resp = await fetch(`${ENGINE_URL}/rpc`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ method, params }),
    });
  } catch {
    throw new EngineCallError({
      code: "engine_offline",
      message: "Can't reach the engine.",
      hint: "Start it with: python -m rag.app --http 8765",
    });
  }
  const body = await resp.json();
  if (body.error) throw new EngineCallError(body.error);
  return body.result as T;
}

const handlers = new Set<EventHandler>();
let wired = false;

function wire() {
  if (wired) return;
  wired = true;
  const dispatch = (name: string, data: any) => handlers.forEach((h) => h(name, data));
  if (inTauri) {
    import("@tauri-apps/api/event").then(({ listen }) => {
      listen<{ event: string; data: any }>("engine-event", (e) => dispatch(e.payload.event, e.payload.data));
    });
  } else {
    const connect = () => {
      const es = new EventSource(`${ENGINE_URL}/events`);
      es.onmessage = (m) => {
        try {
          const msg = JSON.parse(m.data);
          dispatch(msg.event, msg.data);
        } catch {
          /* ignore malformed */
        }
      };
      es.onerror = () => {
        es.close();
        setTimeout(connect, 2000);
      };
    };
    connect();
  }
}

export function onEvent(handler: EventHandler): () => void {
  wire();
  handlers.add(handler);
  return () => handlers.delete(handler);
}
