/** @type {import('tailwindcss').Config} */
export default {
  prefix: "tw-",
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["DM Sans", "system-ui", "sans-serif"],
        display: ["Space Grotesk", "DM Sans", "system-ui", "sans-serif"],
      },
      boxShadow: {
        auth:
          "0 0 0 1px rgba(34, 211, 238, 0.08), 0 25px 80px -12px rgba(0, 0, 0, 0.65), 0 0 120px -30px rgba(34, 211, 238, 0.15)",
        "oauth-hover":
          "0 0 0 1px rgba(34, 211, 238, 0.25), 0 0 28px rgba(34, 211, 238, 0.12)",
      },
      animation: {
        "gradient-shift": "gradient-shift 14s ease infinite",
        "pulse-slow": "pulse-slow 10s ease-in-out infinite",
      },
      keyframes: {
        "gradient-shift": {
          "0%, 100%": { backgroundPosition: "0% 50%" },
          "50%": { backgroundPosition: "100% 50%" },
        },
        "pulse-slow": {
          "0%, 100%": { opacity: "0.35", transform: "scale(1)" },
          "50%": { opacity: "0.5", transform: "scale(1.06)" },
        },
      },
    },
  },
  plugins: [],
};
