import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// Dev: vite serves the UI on :5173 and proxies API/WS to uvicorn on :8000,
// so the browser only ever talks to one origin.
export default defineConfig(({ mode }) => {
  const apiTarget = loadEnv(mode, ".", "").ANALYST_API_URL ?? "http://localhost:8000";
  return {
    plugins: [react()],
    server: {
      proxy: {
        "/runs": { target: apiTarget, ws: true },
        "/datasets": { target: apiTarget },
      },
    },
  };
});
