# 教會事工 AI 高亮摘要工具 WebApp

基於您的 **UX 需求文檔 v1.1** 實現的現代化 WebApp，使用 **React + TypeScript + Tailwind CSS + Flask API**。

## 🎯 設計特色

### 嚴格按照 UX 需求文檔實現
- ✅ **精確的任務流**: 上傳視頻 → 上傳 JSON → 展示摘要 → 播放/導出
- ✅ **指定顏色系統**: 使用完整的 Tailwind 顏色規範 (#FFFFFF, #3B82F6, #10B981 等)
- ✅ **佈局結構**: 頂部橫條、上傳按鈕、視頻播放器、摘要列表、導出按鈕
- ✅ **交互點**: 視覺反馈、Toast 提示、片段跳轉播放、文件導出
- ✅ **字體間距**: Inter 字體、16px 正文、16px padding、24px 間距

### 現代前端技術棧
- **React 18** + **TypeScript** 
- **Vite** 構建工具 (快速開發)
- **Tailwind CSS** 樣式系統
- **純 Flask API** 後端 (輕量化)

## 🚀 快速開始

### 1. 設置 S3 Bucket (必須)
```bash
# 1. 創建 S3 bucket
aws s3 mb s3://church-highlights-videos --region us-east-1

# 2. 設置環境變量
cp env.example .env
# 編輯 .env 文件，填入您的 AWS 憑證

# 3. 上傳視頻到 S3
aws s3 cp "your-video.mp4" s3://church-highlights-videos/
```

詳細 S3 設置說明請參考 [S3_SETUP.md](S3_SETUP.md)

### 2. 啟動 WebApp

#### 一鍵啟動 (推薦)
```bash
./start_simple.sh
```

#### 手動啟動
```bash
# 安裝依賴
npm install
uv pip install -r requirements-api.txt

# 啟動後端 API (終端 1)
uv run python api.py

# 啟動前端 (終端 2) 
npm run dev
```

## 🌐 訪問地址

- **前端界面**: http://localhost:3000
- **後端 API**: http://localhost:5000
- **健康檢查**: http://localhost:5000/api/health

## 📁 項目結構

```
├── src/                    # React 前端源碼
│   ├── components/         # UI 組件
│   ├── hooks/             # React Hooks
│   ├── services/          # API 服務
│   ├── types/             # TypeScript 類型
│   └── utils/             # 工具函數
├── api.py                 # Flask API 後端
├── start_webapp.sh        # 一鍵啟動腳本
├── sample_highlights.json # 示例數據
└── uploads/              # 文件上傳目錄
```

## 🎨 設計系統實現

### 顏色系統 (完全符合 UX 需求)
| 用途 | HEX 值 | Tailwind 類名 |
|------|--------|---------------|
| 背景色 | #FFFFFF | `bg-primary-bg` |
| 邊框色 | #E5E7EB | `border-border-gray` |
| 主文字 | #111827 | `text-text-primary` |
| 次文字 | #6B7280 | `text-text-secondary` |
| 強調藍 | #3B82F6 | `bg-highlight-blue` |
| 成功綠 | #10B981 | `text-success-green` |
| 卡片背景 | #F9FAFB | `bg-card-bg` |

### 響應式設計
- **桌面**: 雙列上傳按鈕佈局
- **手機**: 單列堆疊佈局
- **全局寬度**: max-w-3xl 居中

## 🔧 功能特性

### 核心功能 (100% 按需求實現)
- [x] **S3 視頻選擇**: 從 S3 bucket 瀏覽和選擇視頻
- [x] **自動下載**: 選擇後自動下載到本地處理
- [x] **視頻轉錄**: 集成 AssemblyAI 自動轉錄功能
- [x] **JSON 解析**: 自動識別多種 highlights 格式
- [x] **片段播放**: 精確跳轉到指定時間段播放
- [x] **單一導出**: 每個片段導出為 .txt 文件
- [x] **批量導出**: 一鍵導出所有摘要
- [x] **錯誤處理**: Toast 提示和狀態管理
- [x] **空狀態**: 友好的引導界面

### 技術優勢
- **TypeScript**: 類型安全和更好的開發體驗
- **組件化**: 可維護的模塊化架構  
- **狀態管理**: React Hooks 管理應用狀態
- **API 分離**: 前後端解耦，易於擴展

## 📄 支援的 JSON 格式

WebApp 自動識別多種格式：

```json
// 格式 1: 標準 highlights
{
  "highlights": [
    {"start": 30.5, "end": 95.2, "title": "片段標題", "score": 92.5}
  ]
}

// 格式 2: segments 格式  
{
  "segments": [
    {"start_time": 30.5, "end_time": 95.2, "content": "片段內容"}
  ]
}

// 格式 3: 直接陣列
[
  {"start": 30.5, "end": 95.2, "title": "片段標題"}
]
```

## 🛠️ 開發命令

```bash
npm run dev      # 開發模式
npm run build    # 構建生產版本
npm run preview  # 預覽構建結果
```

## 🎯 與原需求的對比

| UX 需求 | 實現狀態 | 說明 |
|---------|----------|------|
| 極簡設計 | ✅ 完成 | 純白背景，清晰佈局 |
| 任務流 | ✅ 完成 | 精確按照流程圖實現 |
| 顏色系統 | ✅ 完成 | 使用指定的 HEX 值 |
| 字體間距 | ✅ 完成 | Inter 字體，16px 正文 |
| 交互點 | ✅ 完成 | 所有按鈕和反饋 |
| 響應式 | ✅ 完成 | 桌面/手機適配 |

現在您有了一個完全符合 UX 需求文檔 v1.1 的現代化 WebApp！🎉