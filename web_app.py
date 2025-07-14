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

app = Flask(__name__)

# 配置上傳文件夾和允許的文件類型
UPLOAD_FOLDER = 'uploads'
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
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
    
    if file and allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'message': f'視頻上傳成功: {filename}'
        })
    else:
        return jsonify({'error': '不支援的視頻格式'}), 400

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
    """提供視頻文件服務"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)