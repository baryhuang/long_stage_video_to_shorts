#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用 uv 運行教會事工 AI 高亮摘要 WebApp
"""

import subprocess
import sys
import os
from pathlib import Path

def check_uv():
    """檢查 uv 是否已安裝"""
    try:
        subprocess.run(['uv', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_dependencies():
    """使用 uv 安裝依賴"""
    print("正在安裝 WebApp 依賴...")
    try:
        subprocess.run(['uv', 'pip', 'install', '-r', 'requirements-web.txt'], check=True)
        print("✓ 依賴安裝完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 依賴安裝失敗: {e}")
        return False

def run_webapp():
    """使用 uv 運行 Flask WebApp"""
    print("正在啟動教會事工 AI 高亮摘要工具...")
    print("WebApp 將在 http://localhost:5000 運行")
    print("按 Ctrl+C 停止服務")
    print("-" * 50)
    
    try:
        subprocess.run(['uv', 'run', 'python', 'web_app.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"✗ WebApp 啟動失敗: {e}")
        return False
    except KeyboardInterrupt:
        print("\n✓ WebApp 已停止")
        return True

def main():
    """主函數"""
    # 檢查 uv 是否已安裝
    if not check_uv():
        print("✗ 未找到 uv，請先安裝 uv:")
        print("  curl -LsSf https://astral.sh/uv/install.sh | sh")
        print("  或訪問: https://docs.astral.sh/uv/getting-started/installation/")
        sys.exit(1)
    
    print("✓ 找到 uv")
    
    # 檢查 requirements-web.txt 是否存在
    if not Path('requirements-web.txt').exists():
        print("✗ 未找到 requirements-web.txt 文件")
        sys.exit(1)
    
    # 安裝依賴
    if not install_dependencies():
        sys.exit(1)
    
    # 運行 WebApp
    run_webapp()

if __name__ == '__main__':
    main()