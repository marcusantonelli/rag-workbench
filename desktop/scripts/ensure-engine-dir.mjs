// `tauri dev` and `tauri build` need the engine folder to exist (it's bundled as a resource).
// In development the app runs the engine from Python instead, so an empty folder is enough.
import { mkdirSync, existsSync, writeFileSync } from "node:fs";
const dir = new URL("../engine-dist/rag-engine/", import.meta.url);
if (!existsSync(dir)) {
  mkdirSync(dir, { recursive: true });
  writeFileSync(new URL("README.txt", dir), "Placeholder. Run python desktop/scripts/build_engine.py for a real build.\n");
}
