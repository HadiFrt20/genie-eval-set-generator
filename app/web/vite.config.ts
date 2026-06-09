import path from 'node:path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Vite config — dev proxies /api → backend (default :8000), build outputs to
// ../server/static. Override the port pair via env when 8000/5173 are taken:
//   VITE_PORT=9446 API_PORT=9445 npm run dev
const apiPort = process.env.API_PORT ?? '8000'
const vitePort = Number(process.env.VITE_PORT ?? 5173)

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
    },
  },
  build: {
    outDir: path.resolve(__dirname, '../server/static'),
    emptyOutDir: true,
  },
  server: {
    port: vitePort,
    strictPort: true,
    proxy: {
      '/api': {
        target: `http://localhost:${apiPort}`,
        changeOrigin: true,
      },
    },
  },
})
