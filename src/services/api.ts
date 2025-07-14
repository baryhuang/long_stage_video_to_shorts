import { Highlight } from '../types';

const API_BASE = '/api';

export const setVideoPath = async (filePath: string): Promise<{ 
  filename: string; 
  path: string; 
  message: string 
}> => {
  const response = await fetch(`${API_BASE}/set-video-path`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ path: filePath }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || '設置失敗');
  }

  return response.json();
};

export const setHighlightsPath = async (filePath: string): Promise<{ 
  filename: string; 
  path: string;
  highlights: Highlight[]; 
  message: string 
}> => {
  const response = await fetch(`${API_BASE}/set-highlights-path`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ path: filePath }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || '設置失敗');
  }

  return response.json();
};

export const exportHighlight = async (highlight: Highlight): Promise<Blob> => {
  const data = {
    ...highlight,
    timestamp: new Date().toLocaleString('zh-TW')
  };

  const response = await fetch(`${API_BASE}/export-highlight`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    throw new Error('導出失敗');
  }

  return response.blob();
};

export const exportAll = async (highlights: Highlight[]): Promise<Blob> => {
  const data = {
    highlights,
    timestamp: new Date().toLocaleString('zh-TW')
  };

  const response = await fetch(`${API_BASE}/export-all`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    throw new Error('導出失敗');
  }

  return response.blob();
};

export const startTranscription = async (videoPath: string, languageCode: string = 'en', forceOverwrite: boolean = false): Promise<{
  task_id: string;
  message: string;
}> => {
  const response = await fetch(`${API_BASE}/start-transcription`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ 
      video_path: videoPath,
      language_code: languageCode,
      force_overwrite: forceOverwrite
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || '啟動轉錄失敗');
  }

  return response.json();
};

export const getTranscriptionStatus = async (taskId: string): Promise<{
  task_id: string;
  status: string;
  progress: number;
  message: string;
  transcript: string | null;
  error: string | null;
  highlights: any[] | null;
}> => {
  const response = await fetch(`${API_BASE}/transcription-status/${taskId}`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || '獲取轉錄狀態失敗');
  }

  return response.json();
};

export const listS3Videos = async (): Promise<{
  success: boolean;
  videos: Array<{
    key: string;
    name: string;
    size: number;
    last_modified: string;
    url: string;
  }>;
  count: number;
}> => {
  const response = await fetch(`${API_BASE}/list-s3-videos`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || '獲取 S3 視頻列表失敗');
  }

  return response.json();
};

export const downloadS3Video = async (key: string): Promise<{
  success: boolean;
  local_path: string;
  filename: string;
  message: string;
}> => {
  const response = await fetch(`${API_BASE}/download-s3-video`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ key }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || '下載 S3 視頻失敗');
  }

  return response.json();
};

export const generateHighlights = async (videoPath: string): Promise<{
  success: boolean;
  highlights: any[];
  message: string;
}> => {
  const response = await fetch(`${API_BASE}/generate-highlights`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ video_path: videoPath }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || '生成精華片段失敗');
  }

  return response.json();
};