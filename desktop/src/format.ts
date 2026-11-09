export function bytes(n: number | null | undefined): string {
  if (n == null) return "";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = n;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v >= 100 || i === 0 ? v.toFixed(0) : v.toFixed(1)} ${units[i]}`;
}

export function count(n: number | null | undefined, one: string, many?: string): string {
  const v = n ?? 0;
  return `${v.toLocaleString()} ${v === 1 ? one : many ?? one + "s"}`;
}

export function when(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const diff = (Date.now() - d.getTime()) / 1000;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.round(diff / 60)} min ago`;
  if (diff < 86400) return `${Math.round(diff / 3600)} h ago`;
  return d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

export function pct(x: number | null | undefined): string {
  return x == null ? "" : `${Math.round(x * 100)}%`;
}

export function basename(p: string): string {
  return p.split(/[\\/]/).filter(Boolean).pop() ?? p;
}

/** "p. 3–4", "lines 10–42", "sheet Yields A1:F20" */
export function location(c: {
  page_start?: number | null;
  page_end?: number | null;
  line_start?: number | null;
  line_end?: number | null;
  sheet?: string | null;
  cell_range?: string | null;
}): string {
  if (c.page_start) return c.page_end && c.page_end !== c.page_start ? `p. ${c.page_start}–${c.page_end}` : `p. ${c.page_start}`;
  if (c.line_start) return c.line_end && c.line_end !== c.line_start ? `lines ${c.line_start}–${c.line_end}` : `line ${c.line_start}`;
  if (c.sheet) return [`sheet ${c.sheet}`, c.cell_range].filter(Boolean).join(" ");
  return "";
}

const KIND_LABEL: Record<string, string> = {
  prose: "Text",
  table: "Table",
  figure: "Figure",
  code: "Code",
  sql: "SQL",
  table_card: "Table summary",
  rows: "Table rows",
};

export function kindLabel(kind: string): string {
  return KIND_LABEL[kind] ?? kind;
}

const STOP = new Set(
  "the a an and or of to in on for with by from at as is are was were be been what which who whom how why when where does do did can could should would will shall may might this that these those it its into about than then there their they them our your you we i".split(
    " ",
  ),
);

export function queryTerms(q: string): string[] {
  const words = q.toLowerCase().match(/[\p{L}\p{N}][\p{L}\p{N}._-]*/gu) ?? [];
  return [...new Set(words.filter((w) => w.length > 2 && !STOP.has(w)))];
}
