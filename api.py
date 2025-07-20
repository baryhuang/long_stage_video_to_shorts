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
import google.generativeai as genai
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tempfile
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# 載入環境變量
load_dotenv()

# 配置 Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

app = Flask(__name__)
CORS(app)  # 允許跨域請求

# 配置
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'm4a', 'aac', 'ogg'}
ALLOWED_MEDIA_EXTENSIONS = ALLOWED_VIDEO_EXTENSIONS | ALLOWED_AUDIO_EXTENSIONS
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
        allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.mp3', '.wav', '.m4a', '.aac', '.ogg']
        if file_extension not in allowed_extensions:
            return jsonify({'error': '不支援的媒體格式'}), 400
        
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

def generate_highlights_with_gemini(transcript_json_file, transcript_text):
    """使用 Gemini Pro 2.5 分析轉錄內容並生成精華片段"""
    try:
        if not GOOGLE_API_KEY:
            print("未設置 Google API Key，跳過精華片段生成")
            return None
            
        # 確保 JSON 轉錄文件存在
        if not os.path.exists(transcript_json_file):
            print(f"轉錄 JSON 文件不存在: {transcript_json_file}")
            return None
        
        # 導入 Google Generative AI library
        from google import genai
        from google.genai import types
        
        # 初始化 Gemini client
        client = genai.Client(api_key=GOOGLE_API_KEY)
        
        # 定義 JSON schema 用於結構化輸出
        json_schema = {
            "name": "extract_highlights",
            "description": "從教會講道視頻轉錄中提取精華片段",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "highlights": {
                        "type": "ARRAY",
                        "description": "精華片段列表",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "start": {
                                    "type": "NUMBER",
                                    "description": "片段開始時間(秒)"
                                },
                                "end": {
                                    "type": "NUMBER",
                                    "description": "片段結束時間(秒)"
                                },
                                "title": {
                                    "type": "STRING",
                                    "description": "片段標題(繁體中文)"
                                },
                                "content": {
                                    "type": "STRING",
                                    "description": "片段內容摘要"
                                },
                                "score": {
                                    "type": "NUMBER",
                                    "description": "評分(0-100)"
                                },
                                "reason": {
                                    "type": "STRING",
                                    "description": "選擇理由"
                                }
                            },
                            "required": ["start", "end", "title", "content", "score", "reason"]
                        }
                    }
                },
                "required": ["highlights"]
            }
        }
        
        # 構建提示詞
        prompt = f"""
你是一個專業的教會事工視頻編輯助手。請分析以下講道/教會視頻的轉錄內容，找出3-5個最精華的片段。

轉錄內容：
{transcript_text}

選擇標準：
1. 包含重要的神學概念或教義
2. 有感人的見證或故事
3. 實用的生活應用建議
4. 強有力的金句或重點
5. 會眾回應熱烈的部分

每個片段應該：
- 長度在30秒到2分鐘之間
- 有完整的思想表達
- 適合單獨分享
- 標題要吸引人且准確

請使用提供的函數回傳結構化的精華片段數據。
"""
        
        # 配置工具
        tools = types.Tool(function_declarations=[json_schema])
        config = types.GenerateContentConfig(
            temperature=0.4,
            tools=[tools]
        )
        
        # 發送請求
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[prompt],
            config=config
        )
        
        # 解析回應
        if not response.candidates:
            print("Gemini 沒有返回候選回應")
            return None
            
        candidate = response.candidates[0]
        if (candidate.content and
            candidate.content.parts and
            len(candidate.content.parts) > 0 and
            candidate.content.parts[0].function_call):
            
            function_call = candidate.content.parts[0].function_call
            
            # 如果 args 已經是 dict，直接使用
            if isinstance(function_call.args, dict):
                highlights_data = function_call.args
            else:
                highlights_data = json.loads(function_call.args)
            
            # 保存到文件
            highlights_file = transcript_json_file.replace('.transcript.json', '.highlights.json')
            with open(highlights_file, 'w', encoding='utf-8') as f:
                json.dump(highlights_data, f, ensure_ascii=False, indent=2)
            
            print(f"精華片段已保存到: {highlights_file}")
            return highlights_data
            
        else:
            print("Gemini 沒有返回函數調用")
            if candidate.content and candidate.content.parts:
                print(f"回應內容: {response.text}")
            return None
            
    except Exception as e:
        print(f"生成精華片段時發生錯誤: {str(e)}")
        traceback.print_exc()
        return None

@app.route('/api/list-s3-videos', methods=['GET'])
def list_s3_videos():
    """列出 S3 bucket 中的視頻和音頻文件"""
    try:
        if s3_client is None:
            return jsonify({'error': 'S3 客戶端未初始化，請檢查 AWS 憑證'}), 500
            
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME)
        
        videos = []
        if 'Contents' in response:
            for obj in response['Contents']:
                key = obj['Key']
                file_extension = key.split('.')[-1].lower()
                
                # 包含視頻和音頻文件
                if file_extension in ALLOWED_MEDIA_EXTENSIONS:
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
        
        # 檢查文件是否已存在
        if os.path.exists(local_path):
            # 獲取本地文件和 S3 文件的大小和修改時間進行比較
            try:
                local_size = os.path.getsize(local_path)
                local_mtime = os.path.getmtime(local_path)
                
                s3_response = s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
                s3_size = s3_response['ContentLength']
                s3_mtime = s3_response['LastModified'].timestamp()
                
                # 如果大小相同且本地文件不是更舊的，跳過下載
                if local_size == s3_size and local_mtime >= s3_mtime:
                    print(f"文件已存在且為最新版本，跳過下載: {local_filename}")
                    return jsonify({
                        'success': True,
                        'local_path': local_path,
                        'filename': local_filename,
                        'message': f'文件已存在且為最新版本，無需下載: {local_filename}'
                    })
                else:
                    print(f"文件已存在但需要更新，重新下載: {local_filename}")
            except Exception as e:
                print(f"檢查文件狀態時發生錯誤: {e}，重新下載")
        
        # 從 S3 下載文件
        print(f"正在從 S3 下載文件: {s3_key} -> {local_path}")
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

def run_transcription(video_path, language_code, task_id, force_overwrite=False):
    """在後台運行轉錄任務"""
    try:
        # 檢查是否已有轉錄文件 - 支援所有媒體格式
        import os
        base_path = os.path.splitext(video_path)[0]  # 移除任何擴展名
        transcript_file = f"{base_path}.transcript.txt"
        transcript_json_file = f"{base_path}.transcript.json"
        
        if not force_overwrite and os.path.exists(transcript_file) and os.path.exists(transcript_json_file):
            # 讀取現有的轉錄文件
            with open(transcript_file, 'r', encoding='utf-8') as f:
                existing_transcript = f.read()
            
            transcription_tasks[task_id] = {
                'status': 'completed',
                'progress': 100,
                'message': '使用現有轉錄文件',
                'transcript': existing_transcript,
                'error': None,
                'highlights': None
            }
            print(f"使用現有轉錄文件: {transcript_file}")
            return
        
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
        
        # 如果需要強制覆蓋，添加參數
        if force_overwrite:
            cmd.append('--force-overwrite')
        
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
            transcript_json_file = video_path.replace('.MP4', '.transcript.json').replace('.mp4', '.transcript.json')
            
            if os.path.exists(transcript_file):
                try:
                    with open(transcript_file, 'r', encoding='utf-8') as f:
                        transcript_content = f.read()
                except UnicodeDecodeError:
                    # Try with different encodings if UTF-8 fails
                    try:
                        with open(transcript_file, 'r', encoding='utf-8-sig') as f:
                            transcript_content = f.read()
                    except UnicodeDecodeError:
                        try:
                            with open(transcript_file, 'r', encoding='gbk') as f:
                                transcript_content = f.read()
                        except UnicodeDecodeError:
                            with open(transcript_file, 'rb') as f:
                                raw_data = f.read()
                                transcript_content = raw_data.decode('utf-8', errors='replace')
                
                # 轉錄完成，不自動生成精華片段
                transcription_tasks[task_id] = {
                    'status': 'completed',
                    'progress': 100,
                    'message': '轉錄完成',
                    'transcript': transcript_content,
                    'error': None,
                    'highlights': None
                }
            else:
                transcription_tasks[task_id] = {
                    'status': 'error',
                    'progress': 0,
                    'message': '轉錄文件未找到',
                    'transcript': None,
                    'error': '轉錄文件未找到',
                    'highlights': None
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
        force_overwrite = data.get('force_overwrite', False)
        
        print(f"收到轉錄請求:")
        print(f"  視頻路徑: {video_path}")
        print(f"  語言代碼: {language_code}")
        print(f"  強制覆蓋: {force_overwrite}")
        
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
            args=(video_path, language_code, task_id, force_overwrite)
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

@app.route('/api/generate-highlights', methods=['POST'])
def generate_highlights():
    """手動生成精華片段"""
    try:
        data = request.get_json()
        if not data or 'video_path' not in data:
            return jsonify({'error': '沒有提供視頻路徑'}), 400
        
        video_path = data['video_path']
        
        # 構建轉錄文件路徑
        transcript_file = video_path.replace('.MP4', '.transcript.txt').replace('.mp4', '.transcript.txt')
        transcript_json_file = video_path.replace('.MP4', '.transcript.json').replace('.mp4', '.transcript.json')
        
        # 檢查轉錄文件是否存在
        if not os.path.exists(transcript_file):
            return jsonify({'error': '轉錄文件不存在，請先完成轉錄'}), 400
        
        # 讀取轉錄內容
        with open(transcript_file, 'r', encoding='utf-8') as f:
            transcript_content = f.read()
        
        # 生成精華片段
        highlights = generate_highlights_with_gemini(transcript_json_file, transcript_content)
        
        if highlights:
            # 確保回傳的格式正確
            highlights_list = highlights.get('highlights', []) if isinstance(highlights, dict) else highlights
            
            return jsonify({
                'success': True,
                'highlights': highlights_list,
                'message': f'成功生成 {len(highlights_list)} 個精華片段'
            })
        else:
            return jsonify({'error': '精華片段生成失敗，請檢查 Google API Key 設置'}), 500
        
    except Exception as e:
        print(f"生成精華片段錯誤: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'生成失敗: {str(e)}'}), 500

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
            'error': task['error'],
            'highlights': task.get('highlights', None)
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