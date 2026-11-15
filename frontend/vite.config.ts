import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({command}) => ({
  base: command === 'build' ? '/app-assets/' : '/',
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:18080',
      '/demo': 'http://127.0.0.1:18080',
      '/demo.js': 'http://127.0.0.1:18080',
      '/demo.css': 'http://127.0.0.1:18080',
      '/workspace-legacy': 'http://127.0.0.1:18080',
      '/workspace.css': 'http://127.0.0.1:18080',
      '/workspace.js': 'http://127.0.0.1:18080',
    },
  },
}));
