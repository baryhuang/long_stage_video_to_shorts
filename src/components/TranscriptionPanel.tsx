import React, { useState, useEffect } from 'react';
import { startTranscription, getTranscriptionStatus } from '../services/api';

interface TranscriptionPanelProps {
  videoPath: string | null;
  transcriptionState: {
    taskId: string | null;
    status: 'idle' | 'running' | 'completed' | 'error';
    progress: number;
    message: string;
    transcript: string | null;
    error: string | null;
  };
  onTranscriptionUpdate: (update: any) => void;
  onToast: (message: string, type?: 'success' | 'error') => void;
}

const TranscriptionPanel: React.FC<TranscriptionPanelProps> = ({
  videoPath,
  transcriptionState,
  onTranscriptionUpdate,
  onToast
}) => {
  const [languageCode, setLanguageCode] = useState('en');
  
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
            error: status.error
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
      const result = await startTranscription(videoPath, languageCode);
      onTranscriptionUpdate({
        taskId: result.task_id,
        status: 'running',
        progress: 0,
        message: '正在啟動轉錄...',
        transcript: null,
        error: null
      });
      onToast(result.message);
    } catch (error) {
      onToast(error instanceof Error ? error.message : '啟動轉錄失敗', 'error');
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
          <h4 className="text-sm font-medium text-text-primary mb-2">轉錄結果:</h4>
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
    </div>
  );
};

export default TranscriptionPanel;