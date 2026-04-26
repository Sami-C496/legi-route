import type { Config } from 'tailwindcss'

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        paper: 'var(--paper)',
        vellum: 'var(--vellum)',
        rule: 'var(--rule)',
        ink: 'var(--ink)',
        slate: 'var(--slate)',
        marine: 'var(--marine)',
        signal: { red: '#C0392B', green: '#3F7D5C', off: '#3A3A3A' },
      },
      borderRadius: { sm: '4px', md: '6px', lg: '8px' },
      boxShadow: {
        soft: '0 1px 0 rgba(11, 18, 32, 0.04)',
      },
      fontFamily: {
        sans: ['"Inter Variable"', 'system-ui', 'sans-serif'],
        serif: ['"Fraunces"', 'Georgia', 'serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
      },
      maxWidth: {
        column: '920px',
      },
      keyframes: {
        pulse: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.35' },
        },
        'fade-in': {
          from: { opacity: '0', transform: 'translateY(2px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        blink: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0' },
        },
      },
      animation: {
        'signal-pulse': 'pulse 1.1s ease-in-out infinite',
        'fade-in': 'fade-in 0.2s ease-out',
        caret: 'blink 1s steps(1) infinite',
      },
    },
  },
  plugins: [],
} satisfies Config
