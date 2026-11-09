import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type HastNode = { type: string; value?: string; tagName?: string; properties?: any; children?: HastNode[] };

/** rehype plugin: wrap query terms in <mark> inside text nodes (not inside code). */
function highlight(terms: string[]) {
  const escaped = terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).filter(Boolean);
  const re = escaped.length ? new RegExp(`(${escaped.join("|")})`, "gi") : null;
  return () => (tree: HastNode) => {
    if (!re) return;
    const walk = (node: HastNode) => {
      if (!node.children) return;
      if (node.tagName === "code" || node.tagName === "pre") return;
      const out: HastNode[] = [];
      for (const child of node.children) {
        if (child.type === "text" && child.value && re.test(child.value)) {
          re.lastIndex = 0;
          const parts = child.value.split(re);
          parts.forEach((part, i) => {
            if (!part) return;
            if (i % 2 === 1) out.push({ type: "element", tagName: "mark", properties: {}, children: [{ type: "text", value: part }] });
            else out.push({ type: "text", value: part });
          });
        } else {
          walk(child);
          out.push(child);
        }
      }
      node.children = out;
    };
    walk(tree);
  };
}

export function Passage({ text, terms = [], kind }: { text: string; terms?: string[]; kind?: string }) {
  if (kind === "code" || kind === "sql") {
    return <pre className="passage-code">{text}</pre>;
  }
  return (
    <div className="passage">
      <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[highlight(terms)]}>
        {text}
      </ReactMarkdown>
    </div>
  );
}
