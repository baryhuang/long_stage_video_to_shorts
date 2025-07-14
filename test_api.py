#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
測試後端 API 是否正常工作
"""

import requests
import json
import os

def test_health():
    """測試健康檢查"""
    try:
        response = requests.get('http://localhost:5000/api/health')
        print(f"健康檢查: {response.status_code}")
        print(f"響應: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"健康檢查失敗: {e}")
        return False

def test_upload_highlights():
    """測試上傳 highlights JSON"""
    try:
        # 使用示例文件
        with open('sample_highlights.json', 'rb') as f:
            files = {'highlights': f}
            response = requests.post('http://localhost:5000/api/upload-highlights', files=files)
        
        print(f"上傳 JSON: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"找到 {len(data['highlights'])} 個高亮片段")
            for i, h in enumerate(data['highlights']):
                print(f"  {i+1}. {h['title']} ({h['start']:.1f}s - {h['end']:.1f}s)")
        else:
            print(f"錯誤: {response.text}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"上傳測試失敗: {e}")
        return False

def main():
    print("🧪 測試後端 API...")
    print("=" * 40)
    
    if not test_health():
        print("❌ 後端服務未啟動，請先運行: uv run python api.py")
        return
    
    if not os.path.exists('sample_highlights.json'):
        print("❌ 找不到 sample_highlights.json 文件")
        return
    
    if test_upload_highlights():
        print("✅ 所有測試通過!")
    else:
        print("❌ 測試失敗")

if __name__ == '__main__':
    main()