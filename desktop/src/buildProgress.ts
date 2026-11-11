/** Words for what a running build is doing to the current file, shared by the Build and Documents views. */

export type BuildProgress = Record<string, any>;

/** "Reading page 12 of 40", "Embedding 40 of 84 chunks", ... */
export function stageText(p: BuildProgress): string {
  switch (p.stage) {
    case "page":
      return p.pages ? `Reading page ${p.page} of ${p.pages}` : "Reading";
    case "ocr":
      return p.page ? `Reading text in the scan, page ${p.page}${p.pages ? ` of ${p.pages}` : ""}` : "Reading text in the scan";
    case "describe":
      return `Describing pictures (${p.pictures ?? 0} so far)`;
    case "tables":
      return "Storing tables";
    case "chunk":
      return p.file_chunks != null ? `Split into ${p.file_chunks} chunks` : "Splitting into chunks";
    case "embed":
      return p.stage_total ? `Embedding ${p.stage_done ?? 0} of ${p.stage_total} chunks` : "Embedding";
    case "write":
      return "Saving";
    default:
      return "Reading";
  }
}

/** 0..1 for the current file, when the stage has a measure. */
export function fileFraction(p: BuildProgress): number | null {
  if (p.stage === "page" && p.pages) return Math.min(1, (p.page ?? 0) / p.pages) * 0.6;
  if (p.stage === "ocr" && p.pages) return Math.min(1, (p.page ?? 0) / p.pages) * 0.6;
  if (p.stage === "chunk") return 0.62;
  if (p.stage === "embed" && p.stage_total) return 0.65 + 0.35 * Math.min(1, (p.stage_done ?? 0) / p.stage_total);
  return null;
}

/** Seconds left, from the bytes of files actually processed so far (unchanged files take no time). */
export function secondsLeft(p: BuildProgress, now: number): number | null {
  if (!p.work_started || !p.bytes_total) return null;
  const elapsed = now - p.work_started;
  // count how far the current file has got, or a large file being read makes the files finished
  // before it look like the whole rate
  const partial = (p.file_bytes ?? 0) * (fileFraction(p) ?? 0);
  const workBytes = (p.bytes_done ?? 0) - (p.bytes_skipped ?? 0) + partial;
  const remaining = p.bytes_total - (p.bytes_done ?? 0) - partial;
  if (elapsed < 10 || workBytes <= 0 || remaining <= 0) return null;
  return (elapsed / workBytes) * remaining;
}

export function duration(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s} s`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m} min`;
  const h = Math.floor(m / 60);
  return `${h} h ${m % 60} min`;
}
