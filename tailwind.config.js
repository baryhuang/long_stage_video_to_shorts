/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'Noto Sans TC', 'system-ui', 'sans-serif'],
      },
      colors: {
        // 按照 UX 需求文檔 v1.1 指定的顏色系統
        'primary-bg': '#FFFFFF',      // 背景色（Base）
        'border-gray': '#E5E7EB',     // 分割線 / 卡片邊框 (gray-200)
        'text-primary': '#111827',    // 字體主色（正文）(gray-900)
        'text-secondary': '#6B7280',  // 次文字色 (gray-500)
        'highlight-blue': '#3B82F6',  // 高亮藍（強調）(blue-500)
        'success-green': '#10B981',   // 成功綠（導出完成）(green-500)
        'card-bg': '#F9FAFB',         // 卡片背景 (gray-50)
      },
      maxWidth: {
        '3xl': '48rem', // 全局寬度限制
      },
      spacing: {
        '4': '16px',  // 卡片 padding
        '6': '24px',  // 區塊間距
      }
    },
  },
  plugins: [],
}