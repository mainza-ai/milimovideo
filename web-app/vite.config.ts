import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    // The bundle budget script walks the manifest to measure the real initial
    // gzip cost of each route instead of guessing from filenames.
    manifest: true,
  },
})
