// 按照 UX 需求文檔定義的數據結構

export interface Highlight {
  start: number;
  end: number;
  title: string;
  score: number;
}

export interface UploadState {
  video: {
    filename: string | null;
    path: string | null;
    uploaded: boolean;
    loading: boolean;
  };
  highlights: {
    filename: string | null;
    path: string | null;
    uploaded: boolean;
    loading: boolean;
    data: Highlight[];
  };
  transcription: {
    taskId: string | null;
    status: 'idle' | 'running' | 'completed' | 'error';
    progress: number;
    message: string;
    transcript: string | null;
    error: string | null;
  };
}

export interface ToastMessage {
  id: string;
  message: string;
  type: 'success' | 'error';
}