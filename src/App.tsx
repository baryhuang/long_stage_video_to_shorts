import React, { useState, useRef } from 'react';
import { UploadState, Highlight } from './types';
import { useToast } from './hooks/useToast';
import { setVideoPath, setHighlightsPath, exportHighlight, exportAll } from './services/api';
import UploadButton from './components/UploadButton';
import S3VideoSelector from './components/S3VideoSelector';
import VideoPlayer, { VideoPlayerRef } from './components/VideoPlayer';
import HighlightsList from './components/HighlightsList';
import ToastContainer from './components/Toast';
import TranscriptionPanel from './components/TranscriptionPanel';
import { formatTime } from './utils/timeFormat';

const App: React.FC = () => {
  const [uploadState, setUploadState] = useState<UploadState>({
    video: { filename: null, path: null, uploaded: false, loading: false },
    highlights: { filename: null, path: null, uploaded: false, loading: false, data: [] },
    transcription: {
      taskId: null,
      status: 'idle',
      progress: 0,
      message: '',
      transcript: null,
      error: null
    }
  });

  const { toasts, addToast, removeToast } = useToast();
  const videoPlayerRef = useRef<VideoPlayerRef>(null);

  // 處理 S3 視頻選擇
  const handleS3VideoSelect = async (localPath: string, filename: string) => {
    setUploadState(prev => ({
      ...prev,
      video: { ...prev.video, loading: true }
    }));

    try {
      const result = await setVideoPath(localPath);
      setUploadState(prev => ({
        ...prev,
        video: {
          filename: result.filename,
          path: result.path,
          uploaded: true,
          loading: false
        }
      }));
      addToast(result.message);
    } catch (error) {
      setUploadState(prev => ({
        ...prev,
        video: { ...prev.video, loading: false }
      }));
      addToast(error instanceof Error ? error.message : '設置失敗', 'error');
    }
  };

  // 處理 JSON 高亮摘要文件選擇
  const handleHighlightsSelect = async (filePath: string) => {
    setUploadState(prev => ({
      ...prev,
      highlights: { ...prev.highlights, loading: true }
    }));

    try {
      const result = await setHighlightsPath(filePath);
      setUploadState(prev => ({
        ...prev,
        highlights: {
          filename: result.filename,
          path: result.path,
          uploaded: true,
          loading: false,
          data: result.highlights
        }
      }));
      addToast(result.message);
    } catch (error) {
      setUploadState(prev => ({
        ...prev,
        highlights: { ...prev.highlights, loading: false }
      }));
      addToast(error instanceof Error ? error.message : '設置失敗', 'error');
    }
  };

  // 播放片段
  const handlePlaySegment = (start: number, end: number) => {
    if (!uploadState.video.path) {
      addToast('請先選擇視頻文件', 'error');
      return;
    }

    // 使用 videoPlayerRef 調用播放方法
    if (videoPlayerRef.current) {
      videoPlayerRef.current.playSegment(start, end);
    }
  };

  // 導出單個高亮片段
  const handleExportHighlight = async (highlight: Highlight, index: number) => {
    try {
      const blob = await exportHighlight(highlight);
      
      // 創建下載
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      const startTime = formatTime(highlight.start).replace(':', '_');
      const endTime = formatTime(highlight.end).replace(':', '_');
      a.download = `highlight_${startTime}-${endTime}.txt`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
      addToast('摘要已導出');
    } catch (error) {
      addToast(error instanceof Error ? error.message : '導出失敗', 'error');
    }
  };

  // 導出全部摘要
  const handleExportAll = async () => {
    if (uploadState.highlights.data.length === 0) {
      addToast('沒有可導出的摘要', 'error');
      return;
    }

    try {
      const blob = await exportAll(uploadState.highlights.data);
      
      // 創建下載
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `all_highlights_${uploadState.highlights.data.length}_segments.txt`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
      addToast(`所有 ${uploadState.highlights.data.length} 個摘要已導出`);
    } catch (error) {
      addToast(error instanceof Error ? error.message : '導出失敗', 'error');
    }
  };

  // 處理轉錄狀態更新
  const handleTranscriptionUpdate = (update: any) => {
    setUploadState(prev => ({
      ...prev,
      transcription: {
        ...prev.transcription,
        ...update
      }
    }));
  };

  return (
    <div className="min-h-screen bg-primary-bg">
      <div className="container mx-auto max-w-3xl px-6 py-6">
        {/* 頂部橫條 - 按照 UX 需求文檔的佈局 */}
        <div className="text-center py-6 border-b border-border-gray mb-6">
          <h1 className="text-2xl font-semibold text-text-primary">
            ✝️ Church Highlights AI
          </h1>
        </div>

        {/* Toast 消息區域 */}
        <ToastContainer toasts={toasts} onRemoveToast={removeToast} />

        {/* 上傳按鈕區域 - 橫向居中佈局 */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6">
          <S3VideoSelector
            loading={uploadState.video.loading}
            uploaded={uploadState.video.uploaded}
            onVideoSelect={handleS3VideoSelect}
            onToast={addToast}
          />
          
          <UploadButton
            icon="📝"
            label="選擇 Highlights JSON"
            accept=".json"
            loading={uploadState.highlights.loading}
            uploaded={uploadState.highlights.uploaded}
            onFileSelect={handleHighlightsSelect}
          />
        </div>

        {/* 視頻播放器區域 */}
        <VideoPlayer 
          ref={videoPlayerRef}
          videoPath={uploadState.video.path}
        />

        {/* 轉錄面板 */}
        <TranscriptionPanel
          videoPath={uploadState.video.path}
          transcriptionState={uploadState.transcription}
          onTranscriptionUpdate={handleTranscriptionUpdate}
          onToast={addToast}
        />

        {/* 高亮摘要列表區域 */}
        <HighlightsList
          highlights={uploadState.highlights.data}
          onPlaySegment={handlePlaySegment}
          onExportHighlight={handleExportHighlight}
          onExportAll={handleExportAll}
        />

        {/* 空狀態提示 */}
        {!uploadState.video.uploaded && !uploadState.highlights.uploaded && (
          <div className="text-center py-12 text-text-secondary">
            <div className="text-4xl mb-4">📁</div>
            <p>請選擇視頻和高亮摘要文件</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;