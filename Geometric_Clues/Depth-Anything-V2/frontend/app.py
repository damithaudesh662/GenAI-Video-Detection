import os
import sys
from flask import Flask, render_template, request, jsonify, send_from_directory
from pathlib import Path
import shutil
import random

# Link to the backend logic
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from backend.processor import process_single_video, process_batch_videos

app = Flask(__name__)

# Config
UPLOAD_FOLDER = ROOT / "frontend" / "uploads"
RESULTS_FOLDER = ROOT / "frontend" / "results"
MODEL_PATH = ROOT / "trained_detection_models" / "best_r3d18_depthmaps_full.pt"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save video
    video_path = UPLOAD_FOLDER / file.filename
    file.save(str(video_path))

    # Process
    try:
        result = process_single_video(video_path, MODEL_PATH, RESULTS_FOLDER)
        
        # Make the result image path relative to the static folder for the UI to read
        img_name = Path(result['heatmap_path']).name
        # Copy to static for easier serving or use a route
        static_res_path = RESULTS_FOLDER / img_name
        
        return jsonify({
            "status": "success",
            "prediction": result['prediction'],
            "confidence": result['confidence'],
            "image_url": f"/results/{img_name}"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        # Cleanup upload
        if video_path.exists():
            os.remove(video_path)

@app.route('/batch', methods=['POST'])
def batch_process():
    if 'videos' not in request.files:
        return jsonify({"error": "No videos uploaded"}), 400
    
    files = request.files.getlist('videos')
    if not files or files[0].filename == '':
        return jsonify({"error": "No files selected"}), 400

    # Create a temporary folder for this batch
    batch_id = f"batch_{random.randint(0, 10000)}"
    batch_upload_dir = UPLOAD_FOLDER / batch_id
    batch_upload_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Save all videos (using only filename to avoid nested dir issues from webkitdirectory)
        for file in files:
            filename = Path(file.filename).name
            file.save(str(batch_upload_dir / filename))

        # Process Batch
        results = process_batch_videos(batch_upload_dir, MODEL_PATH, RESULTS_FOLDER)
        
        # Format results for UI (relative paths)
        formatted_results = []
        for res in results:
            if 'error' in res:
                formatted_results.append(res)
                continue
                
            img_name = Path(res['heatmap_path']).name
            formatted_results.append({
                "video": res['video'],
                "prediction": res['prediction'],
                "confidence": res['confidence'],
                "image_url": f"/results/{img_name}"
            })

        return jsonify({
            "status": "success",
            "results": formatted_results
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        # Cleanup batch upload folder
        if batch_upload_dir.exists():
            shutil.rmtree(batch_upload_dir)

@app.route('/results/<path:filename>')
def serve_results(filename):
    return send_from_directory(RESULTS_FOLDER, filename)

@app.route('/clear', methods=['POST'])
def clear_results():
    try:
        # Clear the results folder
        for f in RESULTS_FOLDER.iterdir():
            if f.is_file():
                os.remove(f)
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
