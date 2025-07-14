#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
教會事工 AI 高亮摘要工具 - 純 API 後端服務
簡化版本，專注於 API 端點和文件處理
"""

import os
import json
import traceback
import subprocess
import threading
import boto3
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tempfile
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# 載入環境變量
load_dotenv()

app = Flask(__name__)
CORS(app)  # 允許跨域請求

# 配置
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
ALLOWED_JSON_EXTENSIONS = {'json'}

# S3 配置
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'church-highlights-videos')
S3_REGION = os.getenv('AWS_REGION', 'us-east-1')

# 存儲轉錄任務狀態
transcription_tasks = {}

app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

# 初始化 S3 客戶端
# 優先使用環境變量，如果沒有則使用 AWS CLI 默認憑證
try:
    if os.getenv('AWS_ACCESS_KEY_ID') and os.getenv('AWS_SECRET_ACCESS_KEY'):
        s3_client = boto3.client(
            's3',
            region_name=S3_REGION,
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
    else:
        # 使用 AWS CLI 默認憑證或 IAM 角色
        s3_client = boto3.client('s3', region_name=S3_REGION)
except Exception as e:
    print(f"初始化 S3 客戶端失敗: {e}")
    s3_client = None

def allowed_file(filename, allowed_extensions):
    """檢查文件擴展名"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def parse_highlights_json(json_path):
    """解析 highlights JSON 文件"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        highlights = []
        
        # 支援多種 JSON 格式
        if 'highlights' in data:
            for h in data['highlights']:
                highlights.append({
                    'start': h['start'],
                    'end': h['end'],
                    'title': h['title'],
                    'score': h.get('score', 0)
                })
        elif 'segments' in data:
            for seg in data['segments']:
                highlights.append({
                    'start': seg['start_time'],
                    'end': seg['end_time'], 
                    'title': seg.get('content', '摘要片段'),
                    'score': 80.0
                })
        elif isinstance(data, list):
            for i, h in enumerate(data):
                if 'start' in h and 'end' in h:
                    highlights.append({
                        'start': h['start'],
                        'end': h['end'],
                        'title': h.get('title', h.get('content', f'片段 {i+1}')),
                        'score': h.get('score', 80.0)
                    })
        
        # 按時間排序
        highlights.sort(key=lambda x: x['start'])
        return highlights
        
    except Exception as e:
        raise ValueError(f"解析 JSON 文件失敗: {str(e)}")

@app.route('/api/set-video-path', methods=['POST'])
def set_video_path():
    """設置視頻文件路徑 API"""
    try:
        data = request.get_json()
        if not data or 'path' not in data:
            return jsonify({'error': '沒有提供文件路徑'}), 400
        
        video_path = data['path']
        
        # 檢查文件是否存在
        if not os.path.exists(video_path):
            return jsonify({'error': f'文件不存在: {video_path}'}), 400
        
        # 檢查文件格式
        file_extension = os.path.splitext(video_path)[1].lower()
        if file_extension not in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            return jsonify({'error': '不支援的視頻格式'}), 400
        
        # 獲取文件名
        filename = os.path.basename(video_path)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'path': video_path,
            'message': f'視頻路徑設置成功: {filename}'
        })
    except Exception as e:
        print(f"設置視頻路徑錯誤: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'設置失敗: {str(e)}'}), 500

@app.route('/api/set-highlights-path', methods=['POST'])
def set_highlights_path():
    """設置高亮 JSON 文件路徑 API"""
    try:
        data = request.get_json()
        if not data or 'path' not in data:
            return jsonify({'error': '沒有提供文件路徑'}), 400
        
        json_path = data['path']
        
        # 檢查文件是否存在
        if not os.path.exists(json_path):
            return jsonify({'error': f'文件不存在: {json_path}'}), 400
        
        # 檢查文件格式
        if not json_path.lower().endswith('.json'):
            return jsonify({'error': '只支援 JSON 格式文件'}), 400
        
        try:
            highlights = parse_highlights_json(json_path)
            filename = os.path.basename(json_path)
            
            return jsonify({
                'success': True,
                'filename': filename,
                'path': json_path,
                'highlights': highlights,
                'message': f'找到 {len(highlights)} 個高亮片段'
            })
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"設置高亮路徑錯誤: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'設置失敗: {str(e)}'}), 500

@app.route('/api/list-s3-videos', methods=['GET'])
def list_s3_videos():
    """列出 S3 bucket 中的視頻文件"""
    try:
        if s3_client is None:
            return jsonify({'error': 'S3 客戶端未初始化，請檢查 AWS 憑證'}), 500
            
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME)
        
        videos = []
        if 'Contents' in response:
            for obj in response['Contents']:
                key = obj['Key']
                file_extension = key.split('.')[-1].lower()
                
                # 只包含視頻文件
                if file_extension in ALLOWED_VIDEO_EXTENSIONS:
                    videos.append({
                        'key': key,
                        'name': key.split('/')[-1],  # 只顯示文件名
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat(),
                        'url': f"https://{S3_BUCKET_NAME}.s3.{S3_REGION}.amazonaws.com/{key}"
                    })
        
        # 按修改時間排序，最新的在前面
        videos.sort(key=lambda x: x['last_modified'], reverse=True)
        
        return jsonify({
            'success': True,
            'videos': videos,
            'count': len(videos)
        })
        
    except ClientError as e:
        print(f"S3 錯誤: {str(e)}")
        return jsonify({'error': f'無法訪問 S3: {str(e)}'}), 500
    except Exception as e:
        print(f"列出 S3 視頻錯誤: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'列出視頻失敗: {str(e)}'}), 500

@app.route('/api/download-s3-video', methods=['POST'])
def download_s3_video():
    """從 S3 下載視頻文件到本地"""
    try:
        if s3_client is None:
            return jsonify({'error': 'S3 客戶端未初始化，請檢查 AWS 憑證'}), 500
            
        data = request.get_json()
        if not data or 'key' not in data:
            return jsonify({'error': '沒有提供 S3 key'}), 400
        
        s3_key = data['key']
        
        # 創建本地文件路徑 - 下載到項目的 downloads 目錄
        local_filename = s3_key.split('/')[-1]
        downloads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'downloads')
        os.makedirs(downloads_dir, exist_ok=True)
        local_path = os.path.join(downloads_dir, local_filename)
        
        # 從 S3 下載文件
        s3_client.download_file(S3_BUCKET_NAME, s3_key, local_path)
        
        return jsonify({
            'success': True,
            'local_path': local_path,
            'filename': local_filename,
            'message': f'視頻已下載到本地: {local_filename}'
        })
        
    except ClientError as e:
        print(f"S3 下載錯誤: {str(e)}")
        return jsonify({'error': f'下載失敗: {str(e)}'}), 500
    except Exception as e:
        print(f"下載 S3 視頻錯誤: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'下載失敗: {str(e)}'}), 500

def run_transcription(video_path, language_code, task_id):
    """在後台運行轉錄任務"""
    try:
        # 更新任務狀態
        transcription_tasks[task_id] = {
            'status': 'running',
            'progress': 0,
            'message': '正在轉錄視頻...',
            'transcript': None,
            'error': None
        }
        
        # 確保語言代碼有效
        valid_languages = {'en', 'zh'}
        if language_code not in valid_languages:
            language_code = 'en'  # 默認使用英文
        
        # 構建命令
        cmd = [
            'uv', 'run', 'python', 'highlight_generator.py',
            '--input-file', video_path,
            '--transcribe-only',
            '--language-code', language_code
        ]
        
        print(f"執行轉錄命令: {' '.join(cmd)}")
        print(f"視頻路徑: {video_path}")
        print(f"語言代碼: {language_code}")
        
        # 運行命令
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        if result.returncode == 0:
            # 轉錄成功，讀取結果
            transcript_file = video_path.replace('.MP4', '.transcript.txt').replace('.mp4', '.transcript.txt')
            
            if os.path.exists(transcript_file):
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    transcript_content = f.read()
                
                transcription_tasks[task_id] = {
                    'status': 'completed',
                    'progress': 100,
                    'message': '轉錄完成',
                    'transcript': transcript_content,
                    'error': None
                }
            else:
                transcription_tasks[task_id] = {
                    'status': 'error',
                    'progress': 0,
                    'message': '轉錄文件未找到',
                    'transcript': None,
                    'error': '轉錄文件未找到'
                }
        else:
            # 轉錄失敗
            error_msg = result.stderr or result.stdout or '未知錯誤'
            transcription_tasks[task_id] = {
                'status': 'error',
                'progress': 0,
                'message': '轉錄失敗',
                'transcript': None,
                'error': error_msg
            }
            
    except Exception as e:
        print(f"轉錄任務錯誤: {str(e)}")
        traceback.print_exc()
        transcription_tasks[task_id] = {
            'status': 'error',
            'progress': 0,
            'message': '轉錄失敗',
            'transcript': None,
            'error': str(e)
        }

@app.route('/api/start-transcription', methods=['POST'])
def start_transcription():
    """啟動轉錄任務"""
    try:
        data = request.get_json()
        if not data or 'video_path' not in data:
            return jsonify({'error': '沒有提供視頻路徑'}), 400
        
        video_path = data['video_path']
        language_code = data.get('language_code', 'en')
        
        print(f"收到轉錄請求:")
        print(f"  視頻路徑: {video_path}")
        print(f"  語言代碼: {language_code}")
        
        # 檢查文件是否存在
        if not os.path.exists(video_path):
            return jsonify({'error': f'視頻文件不存在: {video_path}'}), 400
        
        # 生成任務 ID
        task_id = f"transcribe_{abs(hash(video_path))}"
        
        # 檢查是否已有進行中的任務
        if task_id in transcription_tasks and transcription_tasks[task_id]['status'] == 'running':
            return jsonify({'error': '轉錄任務已在進行中'}), 400
        
        # 在後台啟動轉錄任務
        thread = threading.Thread(
            target=run_transcription,
            args=(video_path, language_code, task_id)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': '轉錄任務已啟動'
        })
        
    except Exception as e:
        print(f"啟動轉錄錯誤: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'啟動失敗: {str(e)}'}), 500

@app.route('/api/transcription-status/<task_id>', methods=['GET'])
def get_transcription_status(task_id):
    """獲取轉錄任務狀態"""
    try:
        if task_id not in transcription_tasks:
            return jsonify({'error': '任務不存在'}), 404
        
        task = transcription_tasks[task_id]
        return jsonify({
            'task_id': task_id,
            'status': task['status'],
            'progress': task['progress'],
            'message': task['message'],
            'transcript': task['transcript'],
            'error': task['error']
        })
        
    except Exception as e:
        print(f"獲取轉錄狀態錯誤: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'獲取狀態失敗: {str(e)}'}), 500

# 簡化版本不需要這個端點，因為前端直接使用本地文件路徑
# @app.route('/api/video/<filename>')
# def serve_video(filename):
#     """提供視頻文件"""
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/export-highlight', methods=['POST'])
def export_highlight():
    """導出單個高亮片段"""
    data = request.get_json()
    
    if not data or 'title' not in data or 'start' not in data or 'end' not in data:
        return jsonify({'error': '缺少必要參數'}), 400
    
    # 格式化時間
    start_time = f"{int(data['start']//60):02d}:{int(data['start']%60):02d}"
    end_time = f"{int(data['end']//60):02d}:{int(data['end']%60):02d}"
    
    # 創建文字內容
    content = f"""高亮片段摘要

標題: {data['title']}
時間: {start_time} - {end_time}
時長: {data['end'] - data['start']:.0f} 秒

生成時間: {data.get('timestamp', '未知')}
來源: 教會事工 AI 高亮摘要工具
"""
    
    # 創建臨時文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    
    # 清理文件名
    safe_title = "".join(c for c in data['title'] if c.isalnum() or c in (' ', '-', '_')).strip()
    filename = f"highlight_{start_time.replace(':', '_')}-{end_time.replace(':', '_')}_{safe_title[:20]}.txt"
    
    return send_file(tmp_path, as_attachment=True, download_name=filename, mimetype='text/plain')

@app.route('/api/export-all', methods=['POST'])
def export_all():
    """導出所有高亮片段"""
    data = request.get_json()
    
    if not data or 'highlights' not in data:
        return jsonify({'error': '沒有高亮片段數據'}), 400
    
    highlights = data['highlights']
    
    # 創建合併內容
    content = "教會事工 AI 高亮摘要報告\n"
    content += "=" * 40 + "\n\n"
    content += f"總共找到 {len(highlights)} 個精彩片段\n\n"
    
    for i, highlight in enumerate(highlights, 1):
        start_time = f"{int(highlight['start']//60):02d}:{int(highlight['start']%60):02d}"
        end_time = f"{int(highlight['end']//60):02d}:{int(highlight['end']%60):02d}"
        duration = highlight['end'] - highlight['start']
        
        content += f"{i}. {highlight['title']}\n"
        content += f"   時間: {start_time} - {end_time} (時長: {duration:.0f}秒)\n"
        if 'score' in highlight:
            content += f"   評分: {highlight['score']:.1f}\n"
        content += "\n"
    
    content += f"\n生成時間: {data.get('timestamp', '未知')}\n"
    content += "來源: 教會事工 AI 高亮摘要工具\n"
    
    # 創建臨時文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    
    filename = f"all_highlights_{len(highlights)}_segments.txt"
    
    return send_file(tmp_path, as_attachment=True, download_name=filename, mimetype='text/plain')

@app.route('/api/health')
def health_check():
    """健康檢查端點"""
    return jsonify({'status': 'healthy', 'message': '教會事工 AI 高亮摘要工具 API 運行正常'})

if __name__ == '__main__':
    print("🚀 啟動教會事工 AI 高亮摘要工具 API 服務")
    print("📡 API 地址: http://localhost:5001")
    print("🎬 前端地址: http://localhost:3000 (需要另外啟動)")
    print("⚡ 使用 Ctrl+C 停止服務")
    app.run(debug=True, host='0.0.0.0', port=5001)