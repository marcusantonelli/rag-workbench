import type { Ranks } from "../types";

const SCALE = 40; // vector and keyword searches each return their top 40

/**
 * Where a result came from: its position in the vector search and in the keyword
 * search, drawn on the same 1–40 track. A mark missing from a track means that
 * search didn't return the passage at all.
 */
export function RankStrip({ ranks }: { ranks: Ranks | null | undefined }) {
  if (!ranks) return null;
  const row = (label: string, r: number | null) => (
    <div className="rank-row">
      <span className="rank-label">{label}</span>
      <span className="rank-track" aria-hidden="true">
        {r != null && <span className="rank-mark" style={{ left: `${((Math.min(r, SCALE) - 1) / (SCALE - 1)) * 100}%` }} />}
      </span>
      <span className="rank-value">{r != null ? `#${r}` : "none"}</span>
    </div>
  );
  return (
    <div
      className="rank-strip"
      title={`Fused score ${ranks.fused}. Vector search rank ${ranks.vector ?? "none"}, keyword search rank ${ranks.keyword ?? "none"}.`}
    >
      {row("Meaning", ranks.vector)}
      {row("Words", ranks.keyword)}
    </div>
  );
}
