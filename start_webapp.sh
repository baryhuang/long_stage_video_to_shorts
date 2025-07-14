#!/bin/bash

# 教會事工 AI 高亮摘要工具 - 啟動腳本

echo "🚀 正在啟動教會事工 AI 高亮摘要工具..."
echo ""

# 檢查 Node.js
if ! command -v node &> /dev/null; then
    echo "❌ 請先安裝 Node.js: https://nodejs.org"
    exit 1
fi

# 檢查 uv
if ! command -v uv &> /dev/null; then
    echo "❌ 請先安裝 uv: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

echo "✅ 找到 Node.js 和 uv"

# 安裝前端依賴
echo "📦 安裝前端依賴..."
npm install

# 安裝後端依賴
echo "🐍 安裝後端依賴..."
uv pip install -r requirements-api.txt

echo ""
echo "🎬 正在啟動服務..."
echo "📡 後端 API: http://localhost:5000"
echo "🖥️  前端界面: http://localhost:3000"
echo ""
echo "⚡ 使用 Ctrl+C 停止所有服務"
echo ""

# 使用 trap 確保子進程在腳本退出時被殺死
trap 'kill $(jobs -p) 2>/dev/null' EXIT

# 並行啟動後端和前端
uv run python api.py &
npm run dev &

# 等待所有背景進程
wait