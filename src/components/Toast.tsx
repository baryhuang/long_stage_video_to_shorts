import React, { useEffect } from 'react';
import { ToastMessage } from '../types';

interface ToastProps {
  toast: ToastMessage;
  onRemove: (id: string) => void;
}

const Toast: React.FC<ToastProps> = ({ toast, onRemove }) => {
  useEffect(() => {
    const timer = setTimeout(() => {
      onRemove(toast.id);
    }, 5000);

    return () => clearTimeout(timer);
  }, [toast.id, onRemove]);

  const bgClass = toast.type === 'success' 
    ? 'bg-green-50 text-green-800 border-green-200' 
    : 'bg-red-50 text-red-800 border-red-200';

  return (
    <div className={`
      p-3 border rounded-md text-sm mb-3
      ${bgClass}
      animate-in slide-in-from-top-2 duration-300
    `}>
      {toast.message}
    </div>
  );
};

interface ToastContainerProps {
  toasts: ToastMessage[];
  onRemoveToast: (id: string) => void;
}

const ToastContainer: React.FC<ToastContainerProps> = ({ toasts, onRemoveToast }) => {
  if (toasts.length === 0) return null;

  return (
    <div className="mb-6">
      {toasts.map(toast => (
        <Toast key={toast.id} toast={toast} onRemove={onRemoveToast} />
      ))}
    </div>
  );
};

export default ToastContainer;