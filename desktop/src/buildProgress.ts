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

/** Seconds left, from the bytes of files actually processed so far (unchanged files take no time). */
export function secondsLeft(p: BuildProgress, now: number): number | null {
  if (!p.work_started || !p.bytes_total) return null;
  const elapsed = now - p.work_started;
  // count how far the current file has got, or a large file being read makes the files finished
  // before it look like the whole rate
  const partial = (p.file_bytes ?? 0) * (fileFraction(p) ?? 0);
  const workBytes = (p.bytes_done ?? 0) - (p.bytes_skipped ?? 0) + partial;
  const remaining = p.bytes_total - (p.bytes_done ?? 0) - partial;
  // too little seen yet to say: under 20 seconds, or less than 1% of the work and no file finished
  const finished = (p.bytes_done ?? 0) - (p.bytes_skipped ?? 0) > 0;
  if (elapsed < 20 || workBytes <= 0 || remaining <= 0) return null;
  if (!finished && workBytes < 0.01 * (p.bytes_total - (p.bytes_skipped ?? 0))) return null;
  return (elapsed / workBytes) * remaining;
}

/** An estimate that moves only when the build does: it is worked out again each time the work done
 * changes, eased toward the new figure, and counts down in between. A long step with nothing to
 * measure (describing one large picture) no longer makes it climb every second; if the countdown
 * runs out before the build moves on, it says it is still estimating. */
export class Estimate {
  private shown: number | null = null;
  private at = 0;
  private mark = "";

  update(p: BuildProgress, now: number): number | null {
    const mark = [p.bytes_done, p.files_done, p.stage, p.page, p.stage_done, p.pictures, p.current].join("|");
    if (mark !== this.mark || this.shown == null) {
      this.mark = mark;
      const fresh = secondsLeft(p, now);
      const current = this.shown == null ? null : this.shown - (now - this.at);
      // ease toward the new figure unless the two are far apart (then the new one is closer to the truth)
      this.shown = fresh == null ? null
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
