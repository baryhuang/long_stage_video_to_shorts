import React, { useState, useEffect } from 'react';
import { startTranscription, getTranscriptionStatus, generateHighlights, exportAll } from '../services/api';

interface TranscriptionPanelProps {
  videoPath: string | null;
  transcriptionState: {
    taskId: string | null;
    status: 'idle' | 'running' | 'completed' | 'error';
    progress: number;
    message: string;
    transcript: string | null;
    error: string | null;
    highlights: any[] | null;
  };
  onTranscriptionUpdate: (update: any) => void;
  onToast: (message: string, type?: 'success' | 'error') => void;
  onPlaySegment?: (start: number, end: number) => void;
  onExportHighlight?: (highlight: any) => void;
}

const TranscriptionPanel: React.FC<TranscriptionPanelProps> = ({
  videoPath,
  transcriptionState,
  onTranscriptionUpdate,
  onToast,
  onPlaySegment,
  onExportHighlight
}) => {
  const [languageCode, setLanguageCode] = useState('en');
  const [generatingHighlights, setGeneratingHighlights] = useState(false);
  const [forceOverwrite, setForceOverwrite] = useState(false);
  
  // 輪詢轉錄狀態
  useEffect(() => {
    if (transcriptionState.taskId && transcriptionState.status === 'running') {
      const interval = setInterval(async () => {
        try {
          const status = await getTranscriptionStatus(transcriptionState.taskId!);
          onTranscriptionUpdate({
            status: status.status,
            progress: status.progress,
            message: status.message,
            transcript: status.transcript,
            error: status.error,
            highlights: status.highlights
          });
          
          if (status.status === 'completed') {
            onToast('轉錄完成！');
            clearInterval(interval);
          } else if (status.status === 'error') {
            onToast(`轉錄失敗: ${status.error}`, 'error');
            clearInterval(interval);
          }
        } catch (error) {
          console.error('獲取轉錄狀態失敗:', error);
          clearInterval(interval);
        }
      }, 2000);
      
      return () => clearInterval(interval);
    }
  }, [transcriptionState.taskId, transcriptionState.status, onTranscriptionUpdate, onToast]);

  const handleStartTranscription = async () => {
    if (!videoPath) {
      onToast('請先選擇視頻文件', 'error');
      return;
    }

    try {
      const result = await startTranscription(videoPath, languageCode, forceOverwrite);
      onTranscriptionUpdate({
        taskId: result.task_id,
        status: 'running',
        progress: 0,
        message: '正在啟動轉錄...',
        transcript: null,
        error: null,
        highlights: null
      });
      onToast(result.message);
    } catch (error) {
      onToast(error instanceof Error ? error.message : '啟動轉錄失敗', 'error');
    }
  };

  const handleGenerateHighlights = async () => {
    if (!videoPath) {
      onToast('請先選擇視頻文件', 'error');
      return;
    }

    if (transcriptionState.status !== 'completed' || !transcriptionState.transcript) {
      onToast('請先完成轉錄', 'error');
      return;
    }

    setGeneratingHighlights(true);
    try {
      const result = await generateHighlights(videoPath);
      onTranscriptionUpdate({
        ...transcriptionState,
        highlights: result.highlights
      });
      onToast(result.message);
    } catch (error) {
      onToast(error instanceof Error ? error.message : '生成精華片段失敗', 'error');
    } finally {
      setGeneratingHighlights(false);
    }
  };

  const handleExportAllHighlights = async () => {
    if (!transcriptionState.highlights || transcriptionState.highlights.length === 0) {
      onToast('沒有可導出的精華片段', 'error');
      return;
    }

    try {
      const blob = await exportAll(transcriptionState.highlights);
      
      // 創建下載
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `ai_highlights_${transcriptionState.highlights.length}_segments.txt`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
      onToast(`已導出 ${transcriptionState.highlights.length} 個 AI 精華片段`);
    } catch (error) {
      onToast(error instanceof Error ? error.message : '導出失敗', 'error');
    }
  };

  const getProgressBarColor = () => {
    switch (transcriptionState.status) {
      case 'running':
        return 'bg-highlight-blue';
      case 'completed':
        return 'bg-success-green';
      case 'error':
        return 'bg-red-500';
      default:
        return 'bg-gray-300';
    }
  };

  return (
    <div className="mb-6 p-4 bg-card-bg border border-border-gray rounded-lg">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold text-text-primary">
          🎤 視頻轉錄
        </h3>
        
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <label className="text-sm text-text-primary">語言:</label>
            <select
              value={languageCode}
              onChange={(e) => setLanguageCode(e.target.value)}
              className="px-3 py-1 border border-border-gray rounded text-sm bg-white"
              disabled={transcriptionState.status === 'running'}
            >
              <option value="en">English</option>
              <option value="zh">中文 (繁體/簡體)</option>
            </select>
          </div>
          
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="forceOverwrite"
              checked={forceOverwrite}
              onChange={(e) => setForceOverwrite(e.target.checked)}
              disabled={transcriptionState.status === 'running'}
              className="text-highlight-blue"
            />
            <label htmlFor="forceOverwrite" className="text-sm text-text-primary">
              強制覆蓋
            </label>
          </div>
          
          <button
            onClick={handleStartTranscription}
            disabled={!videoPath || transcriptionState.status === 'running'}
            className="
              px-4 py-2 bg-highlight-blue text-white
              rounded-md text-sm font-medium
              hover:bg-blue-600 transition-colors duration-200
              disabled:opacity-50 disabled:cursor-not-allowed
            "
          >
            {transcriptionState.status === 'running' ? '轉錄中...' : '開始轉錄'}
          </button>
        </div>
      </div>
      
      {/* 進度條 */}
      {transcriptionState.status !== 'idle' && (
        <div className="mb-4">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm text-text-secondary">
              {transcriptionState.message}
            </span>
            <span className="text-sm text-text-secondary">
              {transcriptionState.progress}%
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className={`h-2 rounded-full transition-all duration-300 ${getProgressBarColor()}`}
              style={{ width: `${transcriptionState.progress}%` }}
            />
          </div>
        </div>
      )}
      
      {/* 錯誤顯示 */}
      {transcriptionState.status === 'error' && transcriptionState.error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded text-red-800 text-sm">
          <strong>轉錄失敗:</strong> {transcriptionState.error}
        </div>
      )}
      
      {/* 轉錄結果 */}
      {transcriptionState.transcript && (
        <div className="mt-4">
          <div className="flex justify-between items-center mb-2">
            <h4 className="text-sm font-medium text-text-primary">轉錄結果:</h4>
            {transcriptionState.status === 'completed' && (
              <button
                onClick={handleGenerateHighlights}
                disabled={generatingHighlights}
                className="
                  px-3 py-1 bg-success-green text-white
                  rounded text-sm font-medium
                  hover:bg-green-600 transition-colors duration-200
                  disabled:opacity-50 disabled:cursor-not-allowed
                  flex items-center gap-2
                "
              >
                {generatingHighlights ? (
                  <>
                    <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-white"></div>
                    生成中...
                  </>
                ) : (
                  <>
                    🤖 生成精華片段
                  </>
                )}
              </button>
            )}
          </div>
          <div className="
            max-h-60 overflow-y-auto p-3 
            bg-white border border-border-gray rounded
            text-sm text-text-primary leading-relaxed
          ">
            <pre className="whitespace-pre-wrap font-sans">
              {transcriptionState.transcript}
            </pre>
          </div>
        </div>
      )}
      
      {/* Gemini 生成的精華片段 */}
      {transcriptionState.highlights && transcriptionState.highlights.length > 0 && (
        <div className="mt-4">
          <div className="flex justify-between items-center mb-2">
            <h4 className="text-sm font-medium text-text-primary">
              🤖 AI 精華片段分析 ({transcriptionState.highlights.length} 個片段):
            </h4>
            <button
              onClick={() => handleExportAllHighlights()}
              className="
                px-3 py-1 bg-gray-600 text-white
                rounded text-xs font-medium
                hover:bg-gray-700 transition-colors duration-200
                flex items-center gap-1
              "
            >
              📦 導出全部
            </button>
          </div>
          <div className="space-y-3">
            {transcriptionState.highlights.map((highlight: any, index: number) => (
              <div key={index} className="
                p-3 bg-white border border-border-gray rounded
                hover:shadow-sm transition-shadow
              ">
                <div className="flex justify-between items-start mb-2">
                  <h5 className="font-medium text-text-primary text-sm">
                    {highlight.title}
                  </h5>
                  <div className="flex items-center gap-2">
                    {highlight.score && (
                      <span className="text-xs px-2 py-1 bg-highlight-blue/10 text-highlight-blue rounded">
                        評分: {highlight.score}
                      </span>
                    )}
                    <span className="text-xs text-text-secondary">
                      {Math.floor(highlight.start / 60)}:{(highlight.start % 60).toString().padStart(2, '0')} - 
                      {Math.floor(highlight.end / 60)}:{(highlight.end % 60).toString().padStart(2, '0')}
                    </span>
                  </div>
                </div>
                {highlight.content && (
                  <p className="text-xs text-text-secondary mt-2">
                    {highlight.content}
                  </p>
                )}
                {highlight.reason && (
                  <p className="text-xs text-text-secondary mt-1 italic">
                    選擇理由: {highlight.reason}
                  </p>
                )}
                
                {/* 操作按鈕 */}
                <div className="flex justify-end gap-2 mt-3 pt-2 border-t border-border-gray">
                  {onPlaySegment && (
                    <button
                      onClick={() => onPlaySegment(highlight.start, highlight.end)}
                      className="
                        px-3 py-1 bg-highlight-blue text-white
                        rounded text-xs font-medium
                        hover:bg-blue-600 transition-colors duration-200
                        flex items-center gap-1
                      "
                    >
                      ▶️ 播放
                    </button>
                  )}
                  {onExportHighlight && (
                    <button
                      onClick={() => onExportHighlight(highlight)}
                      className="
                        px-3 py-1 bg-success-green text-white
                        rounded text-xs font-medium
                        hover:bg-green-600 transition-colors duration-200
                        flex items-center gap-1
                      "
                    >
                      📁 導出
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default TranscriptionPanel;