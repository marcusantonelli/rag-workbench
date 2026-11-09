// Cross-tab navigation: "show this chunk in its document" from search or evaluation.

export interface ChunkFocus {
  kb: string;
  docId?: number;
  chunkId: number;
}

let pending: ChunkFocus | null = null;

export function focusChunk(f: ChunkFocus) {
  pending = f;
  window.dispatchEvent(new CustomEvent("kb-focus-chunk", { detail: f }));
}

export function takeFocus(kb: string): ChunkFocus | null {
  if (pending && pending.kb === kb) {
    const f = pending;
    pending = null;
    return f;
  }
  return null;
}
