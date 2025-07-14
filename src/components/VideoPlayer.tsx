import React, { useRef, useImperativeHandle, forwardRef } from 'react';

interface VideoPlayerProps {
  videoPath: string | null;
  onTimeUpdate?: (currentTime: number) => void;
}

export interface VideoPlayerRef {
  playSegment: (start: number, end: number) => void;
}

const VideoPlayer = forwardRef<VideoPlayerRef, VideoPlayerProps>(({ videoPath, onTimeUpdate }, ref) => {
  const videoRef = useRef<HTMLVideoElement>(null);

  const handleTimeUpdate = () => {
    if (videoRef.current && onTimeUpdate) {
      onTimeUpdate(videoRef.current.currentTime);
    }
  };

  // 播放指定片段的方法，供父組件調用
  const playSegment = (start: number, end: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = start;
      videoRef.current.play();

      // 設置結束時間監聽
      const handleEndTime = () => {
        if (videoRef.current && videoRef.current.currentTime >= end) {
          videoRef.current.pause();
          videoRef.current.removeEventListener('timeupdate', handleEndTime);
        }
      };

      videoRef.current.addEventListener('timeupdate', handleEndTime);
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
      <video
        ref={videoRef}
        className="w-full max-h-[400px] rounded-lg"
        controls
        onTimeUpdate={handleTimeUpdate}
        src={videoPath.startsWith('http') ? videoPath : `file://${videoPath}`}
      >
        您的瀏覽器不支援影片播放
      </video>
    </div>
  );
});

VideoPlayer.displayName = 'VideoPlayer';

export default VideoPlayer;