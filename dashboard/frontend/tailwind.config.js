/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        'space-black': '#0a0e1a',
        'space-dark': '#0d1220',
        'panel-bg': '#111827',
        'panel-border': '#1e2d4a',
        'accent-cyan': '#00d4ff',
        'accent-gold': '#ffe66d',
        'accent-red': '#ff6b6b',
        'accent-green': '#4ecdc4',
        'text-primary': '#e2e8f0',
        'text-secondary': '#94a3b8',
        'text-muted': '#64748b',
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'Consolas', 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
