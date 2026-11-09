import { useEffect, useState } from "react";
import { call } from "../api";

/** A figure cropped from a PDF or Word file, read from the knowledge base's assets. */
export function FigureImage({ kb, path }: { kb: string; path: string }) {
  const [src, setSrc] = useState<string | null>(null);
  useEffect(() => {
    setSrc(null);
    call<{ data_url: string }>("kb.asset", { kb, path }).then((r) => setSrc(r.data_url)).catch(() => setSrc(null));
  }, [kb, path]);
  return src ? <img className="figure" src={src} alt="Figure from the document" /> : null;
}

/** For a search hit: look up the chunk's image, if it is a figure. */
export function HitFigure({ kb, chunkId }: { kb: string; chunkId: number }) {
  const [path, setPath] = useState<string | null>(null);
  useEffect(() => {
    setPath(null);
    call<{ chunk: { asset_path: string | null } }>("kb.chunk", { kb, chunk_id: chunkId, context: 0 })
      .then((r) => setPath(r.chunk.asset_path))
      .catch(() => setPath(null));
  }, [kb, chunkId]);
  return path ? <FigureImage kb={kb} path={path} /> : null;
}
