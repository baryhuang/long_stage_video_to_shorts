import React, { useRef, useImperativeHandle, forwardRef, useMemo } from 'react';

interface VideoPlayerProps {
  videoPath: string | null;
  onTimeUpdate?: (currentTime: number) => void;
}

export interface VideoPlayerRef {
  playSegment: (start: number, end: number) => void;
}

const VideoPlayer = forwardRef<VideoPlayerRef, VideoPlayerProps>(({ videoPath, onTimeUpdate }, ref) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  
  // 檢測是否為音頻文件
  const isAudio = useMemo(() => {
    if (!videoPath) return false;
    const extension = videoPath.split('.').pop()?.toLowerCase();
    return ['mp3', 'wav', 'm4a', 'aac', 'ogg'].includes(extension || '');
  }, [videoPath]);

  const handleTimeUpdate = () => {
    const currentPlayer = isAudio ? audioRef.current : videoRef.current;
    if (currentPlayer && onTimeUpdate) {
      onTimeUpdate(currentPlayer.currentTime);
    }
  };

  // 播放指定片段的方法，供父組件調用
  const playSegment = (start: number, end: number) => {
    const currentPlayer = isAudio ? audioRef.current : videoRef.current;
    if (currentPlayer) {
      currentPlayer.currentTime = start;
      currentPlayer.play();

      // 設置結束時間監聽
      const handleEndTime = () => {
        if (currentPlayer && currentPlayer.currentTime >= end) {
          currentPlayer.pause();
          currentPlayer.removeEventListener('timeupdate', handleEndTime);
        }
      };

      currentPlayer.addEventListener('timeupdate', handleEndTime);
    }
  };

  // 暴露方法給父組件
  useImperativeHandle(ref, () => ({
    playSegment
  }));

  if (!videoPath) {
    return null;
  }

  return (
    <div className="mb-6 p-4 bg-card-bg border border-border-gray rounded-lg">
      {isAudio ? (
        <audio
          ref={audioRef}
          className="w-full rounded-lg"
          controls
          onTimeUpdate={handleTimeUpdate}
          src={videoPath.startsWith('http') ? videoPath : `file://${videoPath}`}
        >
          您的瀏覽器不支援音頻播放
        </audio>
      ) : (
        <video
          ref={videoRef}
          className="w-full max-h-[400px] rounded-lg"
          controls
          onTimeUpdate={handleTimeUpdate}
          src={videoPath.startsWith('http') ? videoPath : `file://${videoPath}`}
        >
          您的瀏覽器不支援影片播放
        </video>
      )}
    </div>
  );
});

VideoPlayer.displayName = 'VideoPlayer';

export default VideoPlayer;