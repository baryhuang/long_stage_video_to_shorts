import React from 'react';

interface UploadButtonProps {
  icon: string;
  label: string;
  accept: string;
  loading: boolean;
  uploaded: boolean;
  onFileSelect: (filePath: string) => void;
}

const UploadButton: React.FC<UploadButtonProps> = ({
  icon,
  label,
  accept,
  loading,
  uploaded,
  onFileSelect
}) => {
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      // 在 Web 環境中，我們需要用戶手動輸入完整路徑
      // 或者使用 webkitRelativePath 等方式
      const fileName = file.name;
      const fullPath = prompt(`請輸入文件的完整路徑:\n文件名: ${fileName}`, `~/Documents/${fileName}`);
      
      if (fullPath) {
        // 展開 ~ 為用戶家目錄
        const expandedPath = fullPath.startsWith('~') 
          ? fullPath.replace('~', '/Users/buryhuang')
          : fullPath;
        onFileSelect(expandedPath);
      }
    }
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

  const displayLabel = loading ? '上傳中...' : (uploaded ? '✓ 已上傳' : label);

  return (
    <label className={buttonClass}>
      <div className="text-2xl mb-2">{icon}</div>
      <span className={`text-base ${uploaded ? 'text-success-green' : 'text-text-secondary'}`}>
        {displayLabel}
      </span>
      <input
        type="file"
        accept={accept}
        onChange={handleFileChange}
        disabled={loading}
        className="hidden"
      />
    </label>
  );
};

export default UploadButton;