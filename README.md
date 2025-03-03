# 長視頻精彩片段提取器 (Long Video to Shorts)

一個可以從長視頻中自動提取精彩片段的工具，並將其轉換為9:16豎屏格式，適合在社交媒體上分享。

## 功能特點

- 從長視頻中提取不超過3分鐘的精彩片段
- 使用AssemblyAI進行準確的中文語音識別與轉錄
- 使用Claude 3.7 AI識別視頻中最有價值的部分
- 生成9:16豎屏格式，適合社交媒體分享
- 自動跟蹤並放大說話者，使其佔據畫面2/3高度
- 在頂部空間添加標題和信息
- 在底部空間添加字幕
- 支持添加自定義logo

## 安裝

### 自動安裝（推薦）

使用提供的安裝腳本進行快速設置：

```bash
# 克隆存儲庫
git clone https://github.com/yourusername/long_stage_video_to_shorts.git
cd long_stage_video_to_shorts

# 運行安裝腳本
chmod +x setup.sh
./setup.sh
```

安裝腳本將創建虛擬環境、安裝所有依賴項，並設置環境文件。

### 手動安裝

1. 克隆此存儲庫：
   ```bash
   git clone https://github.com/yourusername/long_stage_video_to_shorts.git
   cd long_stage_video_to_shorts
   ```

2. 創建並激活虛擬環境（可選但推薦）：
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # 在Windows上使用: venv\Scripts\activate
   ```

3. 安裝依賴項：
   ```bash
   pip install -r requirements.txt
   ```

4. 設置API密鑰：
   ```bash
   cp .env.example .env
   ```
   然後編輯`.env`文件，添加您的AssemblyAI和Anthropic API密鑰。

### 驗證安裝

運行測試腳本以確認所有依賴項都已正確安裝：

```bash
python test_setup.py
```

## 使用方法

基本用法：

```bash
python highlight_generator.py input_video.mp4
```

這將從`input_video.mp4`中提取一個精彩片段，並將其保存為`input_video_highlight.mp4`。

高級選項：

```bash
python highlight_generator.py input_video.mp4 --output output.mp4 --max-duration 150 --logo logo.png
```

參數說明：
- `input_video.mp4`：輸入視頻文件
- `--output, -o`：輸出視頻文件路徑（默認為輸入文件名加_highlight）
- `--max-duration, -d`：最大片段長度，單位秒（默認為180）
- `--logo, -l`：要顯示在頂部區域的logo圖像路徑

## 工作原理

1. **語音識別與轉錄**：使用AssemblyAI API轉錄視頻中的語音為繁體中文文本，帶有時間戳信息。
   
2. **精彩片段識別**：使用Claude 3.7 AI分析轉錄文本，找出最有價值、最吸引人的部分。

3. **視頻處理**：
   - 從原視頻中提取所選時間段
   - 使用OpenCV進行人臉檢測和跟蹤
   - 放大顯示說話者，使其佔據畫面2/3高度
   - 將視頻轉換為9:16格式，中間為1:1比例的核心內容
   - 在頂部添加標題和logo
   - 在底部添加同步字幕

4. **輸出**：生成最終的9:16比例精彩片段視頻

## 獲取API密鑰

要使用此工具，您需要獲取以下API密鑰：

1. **AssemblyAI API密鑰**：
   - 訪問 [AssemblyAI](https://www.assemblyai.com/) 並創建一個帳戶
   - 在儀表板中獲取您的API密鑰

2. **Anthropic Claude API密鑰**：
   - 訪問 [Anthropic](https://www.anthropic.com/) 並申請API訪問權限
   - 獲取您的API密鑰

將這些密鑰添加到`.env`文件中。

## 故障排除

如果您遇到問題：

1. 確保您已安裝所有依賴項：`pip install -r requirements.txt`
2. 確保您的API密鑰已正確設置在`.env`文件中
3. 檢查您的視頻格式是否受支持（建議使用MP4格式）
4. 運行`python test_setup.py`檢查您的環境設置

## 依賴項

- Python 3.8+
- OpenCV
- MoviePy
- AssemblyAI API
- Anthropic Claude API
- 其他依賴項（見requirements.txt）

## 許可證

MIT 許可證 - 詳見LICENSE文件