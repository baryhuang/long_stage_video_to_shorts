# Google Gemini API 設置指南

本應用集成了 Google Gemini Pro 2.5 來分析轉錄內容並自動生成精華片段。

## 獲取 Google API Key

1. 前往 [Google AI Studio](https://makersuite.google.com/app/apikey)
2. 登入您的 Google 帳號
3. 點擊「Create API Key」
4. 選擇一個 Google Cloud 專案或創建新專案
5. 複製生成的 API Key

## 設置環境變量

在 `.env` 文件中設置您的 Google API Key：

```bash
# Google Gemini API
GOOGLE_API_KEY=your_google_api_key_here
```

將 `your_google_api_key_here` 替換為您從 Google AI Studio 獲取的實際 API Key。

## 使用說明

1. **選擇視頻**: 從 S3 bucket 選擇要分析的視頻文件
2. **開始轉錄**: 選擇語言（中文或英文）並啟動轉錄
3. **自動分析**: 轉錄完成後，Gemini 會自動分析內容並生成精華片段
4. **查看結果**: 在轉錄面板中查看 AI 生成的精華片段

## 功能特點

- **智能分析**: 使用 Gemini Pro 2.5 分析教會講道內容
- **精華提取**: 自動識別重要的神學概念、見證故事和實用建議
- **評分系統**: 為每個片段提供 0-100 的評分
- **詳細信息**: 包含標題、時間段、內容摘要和選擇理由

## 故障排除

如果 Gemini 分析失敗：

1. 確認 `GOOGLE_API_KEY` 已正確設置
2. 檢查 Google Cloud 專案是否已啟用 Generative AI API
3. 確認 API Key 有足夠的配額
4. 查看後端日誌以獲取詳細錯誤信息

## API 配額

Google Gemini API 有使用配額限制，請根據您的需求選擇適當的定價計劃。