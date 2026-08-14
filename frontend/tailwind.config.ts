/** @type {import('tailwindcss').Config} */
export default {
    darkMode: 'class',
    content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
    theme: {
      extend: {
        colors: {
          background: '#090d16',
          surface: '#111827',
          'surface-hover': '#1f293d',
          border: '#1f293d',
          accent: {
            DEFAULT: '#6366f1',
            hover: '#4f46e5',
            light: '#818cf8',
          },
        },
      },
    },
    plugins: [],
  };