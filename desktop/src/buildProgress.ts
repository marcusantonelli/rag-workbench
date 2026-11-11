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

/** 0..1 for the current file, from whatever it has measured so far: reading counts 60% (pages
 * read, also while pictures on them are described or the scan is read), splitting 2%, embedding
 * the rest. Null before anything is measured. */
export function fileFraction(p: BuildProgress): number | null {
  if (p.stage_total) return 0.65 + 0.35 * Math.min(1, (p.stage_done ?? 0) / p.stage_total);
  if (p.stage === "chunk" || p.file_chunks != null) return 0.62;
  if (p.pages) return Math.min(1, (p.page ?? 0) / p.pages) * 0.6;
  return null;
}

/** Seconds left on the file being read, by stage. While its pages are read: the pages left at the
 * pace of those read, plus embedding its likely chunks at the pace of the files before it. While
 * embedding: the chunks left at this file's pace (or the earlier files'). Null when there is
 * nothing to go on yet (a file without pages before it is split, the first seconds of a file). */
export function fileSecondsLeft(p: BuildProgress, now: number): number | null {
  if (!p.current || !p.file_started) return null;
  const spent = now - p.file_started;
  const perChunk = p.embed_chunks > 0 ? p.embed_seconds / p.embed_chunks : null;
  if (p.stage_total) {
    const todo = p.stage_total - (p.embed_reused ?? 0);
    const done = (p.stage_done ?? 0) - (p.embed_reused ?? 0);
    const pace = done >= 3 && p.embed_started ? (now - p.embed_started) / done : perChunk;
    return pace == null ? null : Math.max(0, todo - done) * pace;
  }
  if (p.file_chunks != null) {
    return perChunk == null ? null : p.file_chunks * perChunk;
  }
  if (p.pages && p.page) {
    if (spent < 10 || p.page < 2) return null;
    const read = (spent * (p.pages - p.page)) / p.page;
    const chunks = p.read_bytes > 0 && p.file_bytes ? (p.read_chunks / p.read_bytes) * p.file_bytes : null;
    return read + (perChunk != null && chunks != null ? perChunk * chunks : 0);
  }
  return null;
}

/** Seconds left for the whole build: what is left of the current file, plus the files still to
 * come at the rate of the files already read (by size; unchanged files take no time). Null until
 * there is enough to go on: 30 seconds of work and either three files read or a tenth of the work
 * measured (a single large file counts by how far it has got). */
export function secondsLeft(p: BuildProgress, now: number): number | null {
  if (!p.work_started || !p.bytes_total) return null;
  const elapsed = now - p.work_started;
  if (elapsed < 30) return null;
  const work = p.bytes_total - (p.bytes_skipped ?? 0);
  const finishedBytes = (p.bytes_done ?? 0) - (p.bytes_skipped ?? 0);
  const finishedFiles = (p.added ?? 0) + (p.failed ?? 0);
  const f = fileFraction(p) ?? 0;
  const curBytes = p.current ? (p.file_bytes ?? 0) : 0;
  const spentOnFile = p.current && p.file_started ? Math.max(0, now - p.file_started) : 0;
  const measured = finishedBytes + curBytes * f;
  if (work <= 0 || measured <= 0 || (finishedFiles < 3 && measured < 0.1 * work)) return null;
  // seconds per byte: from the finished files when there are any, else from the current file
  const rate = finishedBytes > 0 ? Math.max(1e-9, elapsed - spentOnFile) / finishedBytes
    : f > 0 && curBytes > 0 ? spentOnFile / (curBytes * f) : null;
  if (rate == null) return null;
  const rest = Math.max(0, p.bytes_total - (p.bytes_done ?? 0) - curBytes);
  const fileLeft = p.current ? fileSecondsLeft(p, now) ?? rate * curBytes * (1 - f) : 0;
  return fileLeft + rest * rate;
}

/** An estimate that moves only when the build does: it is worked out again each time the work done
 * changes, eased toward the new figure, and counts down in between. A long step with nothing to
 * measure (describing one large picture) no longer makes it climb every second; if the countdown
 * runs out before the build moves on, it says it is still estimating. */
export class Estimate {
  private shown: number | null = null;
  private at = 0;
  private mark = "";

  constructor(private readonly measure: (p: BuildProgress, now: number) => number | null) {}

  update(p: BuildProgress, now: number): number | null {
    const mark = [p.bytes_done, p.files_done, p.stage, p.page, p.stage_done, p.pictures, p.current].join("|");
    if (mark !== this.mark || this.shown == null) {
      const fileChanged = this.mark.split("|").pop() !== String(p.current ?? "");
      this.mark = mark;
      const fresh = this.measure(p, now);
      const current = this.shown == null ? null : this.shown - (now - this.at);
      // ease toward the new figure unless the two are far apart (then the new one is closer to the truth)
      this.shown = fresh == null ? (fileChanged ? null : current != null && current > 0 ? current : null)
        : current != null && current > 0 && fresh < current * 3 && fresh > current / 3 ? 0.6 * current + 0.4 * fresh
        : fresh;
      this.at = now;
    }
    if (this.shown == null) return null;
    const left = this.shown - (now - this.at);
    return left > 0 ? left : null;
  }

  reset(): void {
    this.shown = null;
    this.mark = "";
  }
}

export function duration(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s} s`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m} min`;
  const h = Math.floor(m / 60);
  return `${h} h ${m % 60} min`;
}
