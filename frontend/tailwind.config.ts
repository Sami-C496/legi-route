import type { Config } from 'tailwindcss'

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        paper: '#F5F4EF',
        vellum: '#FFFFFF',
        rule: '#E5E2D8',
        ink: '#0B1220',
        slate: '#6B7280',
        marine: '#1A365D',
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
