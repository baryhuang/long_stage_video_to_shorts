#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
教會事工 AI 高亮摘要工具 WebApp
極簡設計，用於上傳視頻和高亮 JSON，播放片段，導出摘要
"""

import os
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
import tempfile
from datetime import datetime

app = Flask(__name__)

# 配置上傳文件夾和允許的文件類型
UPLOAD_FOLDER = 'uploads'
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'm4a', 'aac', 'ogg'}
ALLOWED_MEDIA_EXTENSIONS = ALLOWED_VIDEO_EXTENSIONS | ALLOWED_AUDIO_EXTENSIONS
ALLOWED_JSON_EXTENSIONS = {'json'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB 上傳限制

# 確保上傳目錄存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename, allowed_extensions):
    """檢查文件擴展名是否允許"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def parse_highlights_json(json_path):
    """解析 highlights JSON 文件，轉換為前端需要的格式"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        highlights = []
        
        # 檢查是否是標準的 HighlightClip 格式
        if 'highlights' in data:
            for h in data['highlights']:
                highlights.append({
                    'start': h['start'],
                    'end': h['end'],
                    'title': h['title'],
                    'score': h.get('score', 0)
                })
        # 檢查是否是 segments 格式  
        elif 'segments' in data:
            for seg in data['segments']:
                highlights.append({
                    'start': seg['start_time'],
                    'end': seg['end_time'], 
                    'title': seg.get('content', '摘要片段'),
                    'score': 80.0
                })
        # 直接是數組格式
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

@app.route('/')
def index():
    """主頁面"""
    return render_template('index.html')

@app.route('/upload-video', methods=['POST'])
def upload_video():
    """上傳視頻文件"""
    if 'video' not in request.files:
        return jsonify({'error': '沒有選擇視頻文件'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': '沒有選擇文件'}), 400
    
    if file and allowed_file(file.filename, ALLOWED_MEDIA_EXTENSIONS):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 檢查相關文件是否已存在
        base_path = os.path.splitext(filepath)[0]
        
        # 檢查轉錄文件
        transcript_txt = f"{base_path}.transcript.txt"
        transcript_json = f"{base_path}.transcript.json"
        has_transcript = os.path.exists(transcript_txt) or os.path.exists(transcript_json)
        
        # 檢查狀態文件
        segments_state = f"{base_path}_segments.state.json"
        titles_state = f"{base_path}_titles.state.json"
        has_segments = os.path.exists(segments_state)
        has_titles = os.path.exists(titles_state)
        
        # 載入現有數據
        existing_data = {}
        
        if has_transcript and os.path.exists(transcript_txt):
            try:
                with open(transcript_txt, 'r', encoding='utf-8') as f:
                    existing_data['transcript'] = f.read()
            except UnicodeDecodeError:
                try:
                    with open(transcript_txt, 'r', encoding='utf-8-sig') as f:
                        existing_data['transcript'] = f.read()
                except UnicodeDecodeError:
                    try:
                        with open(transcript_txt, 'r', encoding='gbk') as f:
                            existing_data['transcript'] = f.read()
                    except UnicodeDecodeError:
                        with open(transcript_txt, 'rb') as f:
                            raw_data = f.read()
                            existing_data['transcript'] = raw_data.decode('utf-8', errors='replace')
        
        if has_segments and os.path.exists(segments_state):
            try:
                import json
                with open(segments_state, 'r', encoding='utf-8') as f:
                    existing_data['segments'] = json.load(f)
            except:
                pass
                
        if has_titles and os.path.exists(titles_state):
            try:
                import json
                with open(titles_state, 'r', encoding='utf-8') as f:
                    existing_data['titles'] = json.load(f)
            except:
                pass
        
        return jsonify({
            'success': True,
            'filename': filename,
            'message': f'媒體上傳成功: {filename}',
            'existing_files': {
                'has_transcript': has_transcript,
                'has_segments': has_segments, 
                'has_titles': has_titles
            },
            'existing_data': existing_data
        })
    else:
        return jsonify({'error': '不支援的媒體格式'}), 400

@app.route('/upload-highlights', methods=['POST'])
def upload_highlights():
    """上傳高亮 JSON 文件"""
    if 'highlights' not in request.files:
        return jsonify({'error': '沒有選擇 JSON 文件'}), 400
    
    file = request.files['highlights']
    if file.filename == '':
        return jsonify({'error': '沒有選擇文件'}), 400
    
    if file and allowed_file(file.filename, ALLOWED_JSON_EXTENSIONS):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # 解析 JSON 並返回高亮片段
            highlights = parse_highlights_json(filepath)
            
            return jsonify({
                'success': True,
                'filename': filename,
                'highlights': highlights,
                'message': f'找到 {len(highlights)} 個高亮片段'
            })
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
    else:
        return jsonify({'error': '只支援 JSON 格式文件'}), 400

@app.route('/video/<filename>')
def serve_video(filename):
    """提供視頻和音頻文件服務"""
    from flask import Response
    import mimetypes
    
    # 設置正確的 MIME 類型
    mimetype, _ = mimetypes.guess_type(filename)
    if not mimetype:
        # 手動設置常見音頻格式的 MIME 類型
        extension = filename.split('.')[-1].lower()
        if extension == 'm4a':
            mimetype = 'audio/mp4'
        elif extension == 'mp3':
            mimetype = 'audio/mpeg'
        elif extension == 'wav':
            mimetype = 'audio/wav'
        elif extension == 'aac':
            mimetype = 'audio/aac'
        elif extension == 'ogg':
            mimetype = 'audio/ogg'
    
    # 首先嘗試從 uploads 目錄提供文件
    uploads_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(uploads_path):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename, mimetype=mimetype)
    
    # 如果 uploads 中沒有，嘗試從 downloads 目錄提供文件
    downloads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'downloads')
    downloads_path = os.path.join(downloads_dir, filename)
    if os.path.exists(downloads_path):
        return send_from_directory(downloads_dir, filename, mimetype=mimetype)
    
    # 文件不存在
    from flask import abort
    abort(404)

@app.route('/export-highlight', methods=['POST'])
def export_highlight():
    """導出單個高亮片段文字"""
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

@app.route('/export-all', methods=['POST'])
def export_all():
    """導出所有高亮片段摘要"""
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

@app.route('/export-video-segment', methods=['POST'])
def export_video_segment():
    """導出單個視頻片段"""
    data = request.get_json()
    
    if not data or 'video_filename' not in data or 'highlight' not in data:
        return jsonify({'error': '缺少必要參數'}), 400
    
    video_filename = data['video_filename']
    highlight = data['highlight']
    
    try:
        # 獲取視頻文件路徑
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        if not os.path.exists(video_path):
            return jsonify({'error': '視頻文件不存在'}), 404
        
        # 創建導出文件夾
        base_name = os.path.splitext(video_filename)[0]
        export_folder = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_name}_highlights")
        os.makedirs(export_folder, exist_ok=True)
        
        # 生成輸出文件名
        start_time_str = f"{int(highlight['start']//60):02d}m{int(highlight['start']%60):02d}s"
        end_time_str = f"{int(highlight['end']//60):02d}m{int(highlight['end']%60):02d}s"
        safe_title = "".join(c for c in highlight['title'] if c.isalnum() or c in (' ', '-', '_')).strip()[:30]
        output_filename = f"{start_time_str}-{end_time_str}_{safe_title}.mp4"
        output_path = os.path.join(export_folder, output_filename)
        
        # 使用 ffmpeg 提取視頻片段
        import subprocess
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-ss', str(highlight['start']),
            '-t', str(highlight['end'] - highlight['start']),
            '-c', 'copy',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return jsonify({
                'success': True,
                'message': f'視頻片段已導出: {output_filename}',
                'output_path': output_path,
                'export_folder': export_folder
            })
        else:
            return jsonify({'error': f'導出失敗: {result.stderr}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'處理錯誤: {str(e)}'}), 500

@app.route('/export-all-video-segments', methods=['POST'])
def export_all_video_segments():
    """導出所有視頻片段到文件夾"""
    data = request.get_json()
    
    if not data or 'video_filename' not in data or 'highlights' not in data:
        return jsonify({'error': '缺少必要參數'}), 400
    
    video_filename = data['video_filename']
    highlights = data['highlights']
    
    try:
        # 獲取視頻文件路徑
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        if not os.path.exists(video_path):
            return jsonify({'error': '視頻文件不存在'}), 404
        
        # 創建導出文件夾（使用時間戳避免覆蓋）
        base_name = os.path.splitext(video_filename)[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_folder = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_name}_highlights_{timestamp}")
        os.makedirs(export_folder, exist_ok=True)
        
        # 保存 highlights.json 到導出文件夾
        highlights_json_path = os.path.join(export_folder, 'highlights.json')
        with open(highlights_json_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(highlights, f, ensure_ascii=False, indent=2)
        
        successful_exports = []
        failed_exports = []
        
        # 導出每個視頻片段
        for i, highlight in enumerate(highlights, 1):
            try:
                start_time_str = f"{int(highlight['start']//60):02d}m{int(highlight['start']%60):02d}s"
                end_time_str = f"{int(highlight['end']//60):02d}m{int(highlight['end']%60):02d}s"
                safe_title = "".join(c for c in highlight['title'] if c.isalnum() or c in (' ', '-', '_')).strip()[:30]
                output_filename = f"{i:02d}_{start_time_str}-{end_time_str}_{safe_title}.mp4"
                output_path = os.path.join(export_folder, output_filename)
                
                # 使用 ffmpeg 提取視頻片段
                import subprocess
                cmd = [
                    'ffmpeg', '-y',
                    '-i', video_path,
                    '-ss', str(highlight['start']),
                    '-t', str(highlight['end'] - highlight['start']),
                    '-c', 'copy',
                    output_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    successful_exports.append(output_filename)
                else:
                    failed_exports.append(f"{output_filename}: {result.stderr}")
                    
            except Exception as e:
                failed_exports.append(f"片段 {i}: {str(e)}")
        
        return jsonify({
            'success': True,
            'message': f'批量導出完成: {len(successful_exports)} 成功, {len(failed_exports)} 失敗',
            'successful_exports': successful_exports,
            'failed_exports': failed_exports,
            'export_folder': export_folder,
            'highlights_json': highlights_json_path
        })
        
    except Exception as e:
        return jsonify({'error': f'處理錯誤: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)