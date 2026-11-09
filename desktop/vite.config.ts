import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Tauri serves the built files; `npm run dev` + `npm run engine:dev` runs the same UI in a browser.
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  // The command-line guide lives in ../docs/cli.md and is bundled into the app.
  server: { port: 5173, strictPort: true, fs: { allow: [".."] } },
  build: { target: "es2021", outDir: "dist", sourcemap: false, chunkSizeWarningLimit: 1500 },
});
