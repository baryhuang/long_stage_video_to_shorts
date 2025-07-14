#!/bin/bash

# 教會事工 AI 高亮摘要工具 - 簡化啟動腳本
# 項目路徑: /Users/buryhuang/git/long_stage_video_to_shorts

PROJECT_PATH="/Users/buryhuang/git/long_stage_video_to_shorts"
cd "$PROJECT_PATH"

echo "🚀 正在啟動教會事工 AI 高亮摘要工具..."
echo "📁 項目路徑: $PROJECT_PATH"
echo ""

# 檢查依賴
if ! command -v node &> /dev/null; then
    echo "❌ 請先安裝 Node.js: https://nodejs.org"
    exit 1
fi

if ! command -v uv &> /dev/null; then
    echo "❌ 請先安裝 uv: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

echo "✅ 檢查通過"

# 創建必要目錄
mkdir -p uploads

# 安裝依賴
echo "📦 安裝前端依賴..."
npm install

echo "🐍 安裝後端依賴..."
uv pip install -r requirements-api.txt

echo ""
echo "🎬 正在啟動服務..."
echo "📡 後端 API: http://localhost:5001"
echo "🖥️  前端界面: http://localhost:3000"
echo ""
echo "⚡ 使用 Ctrl+C 停止所有服務"
echo ""

# 清理函數
cleanup() {
    echo ""
    echo "🛑 正在停止服務..."
    kill $(jobs -p) 2>/dev/null
    exit 0
}

# 設置 trap
trap cleanup SIGINT SIGTERM

# 啟動後端 API
echo "🚀 啟動後端 API..."
uv run python api.py &
BACKEND_PID=$!

# 等待後端啟動
sleep 3

# 啟動前端
echo "🚀 啟動前端..."
npm run dev &
FRONTEND_PID=$!

# 等待所有背景進程
wait