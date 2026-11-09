// File pickers and drag-and-drop. Native dialogs in the app; a path prompt in browser dev mode.

import { inTauri } from "./api";

export interface PickOptions {
  title?: string;
  directory?: boolean;
  multiple?: boolean;
  filters?: { name: string; extensions: string[] }[];
}

function promptPaths(title: string, multiple: boolean): string[] | null {
  const raw = window.prompt(`${title}\n(browser dev mode: type ${multiple ? "paths separated by ;" : "a path"})`);
  if (!raw) return null;
  return raw.split(";").map((s) => s.trim()).filter(Boolean);
}

export async function pickPaths(opts: PickOptions): Promise<string[] | null> {
  if (!inTauri) return promptPaths(opts.title ?? "Choose", !!opts.multiple);
  const { open } = await import("@tauri-apps/plugin-dialog");
  const res = await open({
    title: opts.title,
    directory: opts.directory,
    multiple: opts.multiple,
    filters: opts.filters,
  });
  if (res === null) return null;
  return Array.isArray(res) ? res : [res];
}

export async function pickSavePath(opts: {
  title?: string;
  defaultPath?: string;
  filters?: { name: string; extensions: string[] }[];
}): Promise<string | null> {
  if (!inTauri) {
    const raw = window.prompt(`${opts.title ?? "Save as"}\n(browser dev mode: type a path)`, opts.defaultPath ?? "");
    return raw?.trim() || null;
  }
  const { save } = await import("@tauri-apps/plugin-dialog");
  return (await save({ title: opts.title, defaultPath: opts.defaultPath, filters: opts.filters })) ?? null;
}

export type DropHandler = (paths: string[]) => void;

/** Native file drops (Tauri gives real paths; browsers don't, so dev mode has no drop). */
export function onFileDrop(handler: DropHandler, onHover?: (over: boolean) => void): () => void {
  if (!inTauri) return () => {};
  let unlisten: (() => void) | null = null;
  let cancelled = false;
  import("@tauri-apps/api/webview").then(({ getCurrentWebview }) =>
    getCurrentWebview()
      .onDragDropEvent((e) => {
        const p = e.payload as { type: string; paths?: string[] };
        if (p.type === "over" || p.type === "enter") onHover?.(true);
        else if (p.type === "leave") onHover?.(false);
        else if (p.type === "drop") {
          onHover?.(false);
          if (p.paths?.length) handler(p.paths);
        }
      })
      .then((u) => {
        if (cancelled) u();
        else unlisten = u;
      }),
  );
  return () => {
    cancelled = true;
    unlisten?.();
  };
}

export const PACK_FILTER = [{ name: "RAG pack", extensions: ["ragpack"] }];
export const GOLDEN_FILTER = [{ name: "Golden questions", extensions: ["jsonl", "json"] }];
export const GGUF_FILTER = [{ name: "GGUF model", extensions: ["gguf"] }];
