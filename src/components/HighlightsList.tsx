import React from 'react';
import { Highlight } from '../types';
import HighlightCard from './HighlightCard';

interface HighlightsListProps {
  highlights: Highlight[];
  onPlaySegment: (start: number, end: number) => void;
  onExportHighlight: (highlight: Highlight, index: number) => void;
  onExportAll: () => void;
}

const HighlightsList: React.FC<HighlightsListProps> = ({
  highlights,
  onPlaySegment,
  onExportHighlight,
  onExportAll
}) => {
  if (highlights.length === 0) {
    return null;
  }

  return (
    <div className="mt-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3 mb-4">
        <h2 className="text-lg font-semibold text-text-primary">
          🔦 高亮摘要片段
        </h2>
        <button
          onClick={onExportAll}
          className="
            px-4 py-2 bg-highlight-blue text-white
            rounded-md text-sm font-medium
            hover:bg-blue-600 transition-colors duration-200
            w-full sm:w-auto
          "
        >
          📤 導出全部摘要
        </button>
      </div>
      
      <div>
        {highlights.map((highlight, index) => (
          <HighlightCard
            key={index}
            highlight={highlight}
            index={index}
            onPlay={onPlaySegment}
            onExport={onExportHighlight}
          />
        ))}
      </div>
    </div>
  );
};

export default HighlightsList;