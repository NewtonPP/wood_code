import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Production build outputs to ../web, which the FastAPI backend serves as static
// files. During development the dev server proxies API calls to uvicorn (:8000)
// so session cookies stay same-origin.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // ws: true so the /api/ingest/ws frame-stream WebSocket proxies to uvicorn.
      "/api": { target: "http://localhost:8000", changeOrigin: true, ws: true },
      "/ping": { target: "http://localhost:8000", changeOrigin: true },
    },
  },
  build: {
    outDir: "../web",
    emptyOutDir: true,
  },
});
