import React, { useState, useEffect } from 'react';
import { listS3Videos, downloadS3Video } from '../services/api';

interface S3Video {
  key: string;
  name: string;
  size: number;
  last_modified: string;
  url: string;
}

interface S3VideoSelectorProps {
  loading: boolean;
  uploaded: boolean;
  onVideoSelect: (localPath: string, filename: string) => void;
  onToast: (message: string, type?: 'success' | 'error') => void;
}

const S3VideoSelector: React.FC<S3VideoSelectorProps> = ({
  loading,
  uploaded,
  onVideoSelect,
  onToast
}) => {
  const [videos, setVideos] = useState<S3Video[]>([]);
  const [showList, setShowList] = useState(false);
  const [loadingVideos, setLoadingVideos] = useState(false);
  const [downloadingKey, setDownloadingKey] = useState<string | null>(null);

  // 獲取 S3 視頻列表
  const fetchVideos = async () => {
    setLoadingVideos(true);
    try {
      const result = await listS3Videos();
      setVideos(result.videos);
      setShowList(true);
    } catch (error) {
      onToast(error instanceof Error ? error.message : '獲取視頻列表失敗', 'error');
    } finally {
      setLoadingVideos(false);
    }
  };

  // 下載並選擇視頻
  const handleVideoSelect = async (video: S3Video) => {
    setDownloadingKey(video.key);
    try {
      const result = await downloadS3Video(video.key);
      onVideoSelect(result.local_path, result.filename);
      onToast(result.message);
      setShowList(false);
    } catch (error) {
      onToast(error instanceof Error ? error.message : '下載視頻失敗', 'error');
    } finally {
      setDownloadingKey(null);
    }
  };

  // 格式化文件大小
  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // 格式化時間
  const formatDate = (isoString: string) => {
    const date = new Date(isoString);
    return date.toLocaleDateString('zh-TW') + ' ' + date.toLocaleTimeString('zh-TW');
  };

  const buttonClass = `
    flex flex-col items-center justify-center
    p-6 min-h-[120px]
    border-2 border-dashed border-border-gray
    rounded-lg bg-card-bg
    cursor-pointer transition-all duration-200
    hover:border-highlight-blue hover:bg-blue-50
    ${uploaded ? 'border-success-green bg-green-50' : ''}
    ${loading ? 'opacity-50 cursor-not-allowed' : ''}
  `;

  const displayLabel = loading ? '處理中...' : (uploaded ? '✓ 已選擇' : '選擇 S3 視頻');

  if (showList) {
    return (
      <div className="p-4 border-2 border-highlight-blue rounded-lg bg-blue-50">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-text-primary">
            🎬 S3 視頻列表
          </h3>
          <button
            onClick={() => setShowList(false)}
            className="text-sm text-text-secondary hover:text-text-primary"
          >
            ✕ 關閉
          </button>
        </div>
        
        {loadingVideos ? (
          <div className="text-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-highlight-blue mx-auto"></div>
            <p className="mt-2 text-text-secondary">載入視頻列表...</p>
          </div>
        ) : (
          <div className="max-h-80 overflow-y-auto space-y-2">
            {videos.length === 0 ? (
              <div className="text-center py-8 text-text-secondary">
                沒有找到視頻文件
              </div>
            ) : (
              videos.map((video) => (
                <div
                  key={video.key}
                  className="
                    flex items-center justify-between p-3 
                    bg-white border border-border-gray rounded
                    hover:bg-gray-50 transition-colors duration-200
                  "
                >
                  <div className="flex-1">
                    <div className="font-medium text-text-primary text-sm">
                      {video.name}
                    </div>
                    <div className="text-xs text-text-secondary">
                      {formatFileSize(video.size)} • {formatDate(video.last_modified)}
                    </div>
                  </div>
                  
                  <button
                    onClick={() => handleVideoSelect(video)}
                    disabled={downloadingKey === video.key}
                    className="
                      px-3 py-1 bg-highlight-blue text-white
                      rounded text-sm font-medium
                      hover:bg-blue-600 transition-colors duration-200
                      disabled:opacity-50 disabled:cursor-not-allowed
                    "
                  >
                    {downloadingKey === video.key ? '下載中...' : '選擇'}
                  </button>
                </div>
              ))
            )}
          </div>
        )}
      </div>
    );
  }

  return (
    <div 
      className={buttonClass}
      onClick={() => !loading && fetchVideos()}
    >
      <div className="text-2xl mb-2">🎬</div>
      <span className={`text-base ${uploaded ? 'text-success-green' : 'text-text-secondary'}`}>
        {displayLabel}
      </span>
    </div>
  );
};

export default S3VideoSelector;