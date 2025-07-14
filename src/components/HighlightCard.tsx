import React from 'react';
import { Highlight } from '../types';
import { formatTime } from '../utils/timeFormat';

interface HighlightCardProps {
  highlight: Highlight;
  index: number;
  onPlay: (start: number, end: number) => void;
  onExport: (highlight: Highlight, index: number) => void;
}

const HighlightCard: React.FC<HighlightCardProps> = ({
  highlight,
  index,
  onPlay,
  onExport
}) => {
  const duration = Math.round(highlight.end - highlight.start);
  const startTime = formatTime(highlight.start);
  const endTime = formatTime(highlight.end);

  return (
    <div className="bg-card-bg border border-border-gray rounded-lg p-4 mb-4">
      <div className="text-sm text-text-secondary mb-2">
        {startTime} - {endTime} ({duration}秒)
      </div>
      
      <div className="text-base font-medium text-text-primary mb-3">
        ✨ {highlight.title}
      </div>
      
      <div className="flex gap-3">
        <button
          onClick={() => onPlay(highlight.start, highlight.end)}
          className="
            px-3 py-2 bg-highlight-blue text-white
            rounded-md text-sm font-medium
            hover:bg-blue-600 transition-colors duration-200
          "
        >
          ▶️ 播放片段
        </button>
        
        <button
          onClick={() => onExport(highlight, index)}
          className="
            px-3 py-2 bg-white text-text-secondary
            border border-border-gray rounded-md text-sm
            hover:bg-card-bg hover:text-text-primary
            transition-colors duration-200
          "
        >
          📄 導出文字
        </button>
      </div>
    </div>
  );
};

export default HighlightCard;