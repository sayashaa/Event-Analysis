import os
import csv
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from detection import DetectionRunner
from reassign import reassign_ids

# === Paths ===
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, 'data')
OUTPUT_DIR = os.path.join(APP_DIR, 'output')
VIDEOS_DIR = os.path.join(APP_DIR, 'videos')
REFERENCE_DIR = os.path.join(DATA_DIR, 'reference_frames')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REFERENCE_DIR, exist_ok=True)

# === Flask app ===
app = Flask(__name__)

# === Global Detection Runner ===
runner = DetectionRunner(
    data_dir=DATA_DIR,
    crops_dir=os.path.join(DATA_DIR, 'crops'),
    output_video=os.path.join(OUTPUT_DIR, 'detected_preview.mp4'),
)


# -------------------------------
# Routes
# -------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.post('/start_detection')
def start_detection():
    """Start the detection thread for the given video path."""
    video_path = request.json.get('video_path', '').strip()
    if not video_path:
        return jsonify({'ok': False, 'msg': 'Video path is required.'}), 400

    # If a relative path is provided, look under ./videos
    if not os.path.isabs(video_path):
        cand = os.path.join(VIDEOS_DIR, video_path)
        if os.path.exists(cand):
            video_path = cand

    if not os.path.exists(video_path):
        return jsonify({'ok': False, 'msg': f'Video not found: {video_path}'}), 404

    # Start/Restart runner
    if runner.is_running:
        runner.stop()
    runner.start(video_path)
    return jsonify({'ok': True, 'msg': 'Detection started.'})


@app.post('/api/pause')
def pause_detection():
    """Pause the detection thread."""
    if runner.is_running:
        runner.pause()
        return jsonify({'ok': True, 'msg': 'Detection paused.'})
    return jsonify({'ok': False, 'msg': 'No detection running.'}), 400


@app.post('/api/resume')
def resume_detection():
    """Resume the detection thread."""
    if runner.is_running:
        runner.resume()
        return jsonify({'ok': True, 'msg': 'Detection resumed.'})
    return jsonify({'ok': False, 'msg': 'No detection running.'}), 400


@app.post('/api/stop')
def stop_detection():
    """Completely stop the detection thread."""
    if runner.is_running:
        runner.stop()
        return jsonify({'ok': True, 'msg': 'Detection stopped.'})
    return jsonify({'ok': False, 'msg': 'No detection running.'}), 400


@app.get('/video_feed')
def video_feed():
    """Stream the processed video frames as MJPEG."""
    def gen():
        for frame in runner.frame_generator():
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.get('/api/latest_rows')
def latest_rows():
    """
    Return rows from the current video's movement_log_<video>.csv.
    - If ?after=N is provided, return rows with frame > N.
    - Otherwise, return all rows.
    """
    if not runner.current_log_name:
        return jsonify({'ok': True, 'rows': []})

    log_path = os.path.join(DATA_DIR, f'movement_log_{runner.current_log_name}.csv')
    if not os.path.exists(log_path):
        return jsonify({'ok': True, 'rows': []})

    try:
        after = request.args.get('after', None)
        after_val = int(after) if after is not None else None

        rows = []
        with open(log_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                frame_num = int(r['frame'])
                if after_val is None or frame_num > after_val:
                    rows.append({
                        'frame': r['frame'],
                        'id': r['id'],
                        'x': r['x'],
                        'y': r['y'],
                    })

        return jsonify({'ok': True, 'rows': rows})

    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e)}), 500


@app.post('/api/reassign')
def api_reassign():
    """Reassign wrong_id to correct_id in current video's movement log."""
    if not runner.current_log_name:
        return jsonify({'ok': False, 'msg': 'No active video log to modify.'}), 400

    data = request.get_json()
    wrong_id = str(data.get('wrong_id', '')).strip()
    correct_id = str(data.get('correct_id', '')).strip()

    if not wrong_id or not correct_id:
        return jsonify({'ok': False, 'msg': 'Both wrong_id and correct_id are required.'}), 400

    src_csv = os.path.join(DATA_DIR, f'movement_log_{runner.current_log_name}.csv')
    dst_csv = os.path.join(DATA_DIR, f'movement_log_{runner.current_log_name}_final.csv')

    if not os.path.exists(src_csv):
        return jsonify({'ok': False, 'msg': 'Movement log file not found.'}), 404

    changed_rows = reassign_ids(src_csv, dst_csv, wrong_id, correct_id)

    if runner.is_running:
        runner.reassign_id_in_memory(wrong_id, correct_id)

    return jsonify({'ok': True, 'msg': f'Reassigned {changed_rows} rows from {wrong_id} to {correct_id}.'})


@app.post('/api/delete')
def api_delete():
    """Delete ID occurrences in current video's movement log."""
    if not runner.current_log_name:
        return jsonify({'ok': False, 'msg': 'No active video log to modify.'}), 400

    data = request.get_json()
    target_id = str(data.get('id', '')).strip()
    if not target_id:
        return jsonify({'ok': False, 'msg': 'ID is required.'}), 400

    src_csv = os.path.join(DATA_DIR, f'movement_log_{runner.current_log_name}.csv')
    dst_csv = os.path.join(DATA_DIR, f'movement_log_{runner.current_log_name}_final.csv')

    if not os.path.exists(src_csv):
        return jsonify({'ok': False, 'msg': 'Movement log file not found.'}), 404

    is_unknown = target_id.startswith('unknown_')
    changed_rows = 0

    with open(src_csv, 'r', encoding='utf-8') as fin, open(dst_csv, 'w', newline='', encoding='utf-8') as fout:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            if not (is_unknown and row['id'] == target_id):
                writer.writerow(row)
            else:
                changed_rows += 1

    if runner.is_running:
        runner.delete_id_in_memory(target_id, delete_history=is_unknown)

    return jsonify({'ok': True, 'msg': f'Delete request for {target_id} completed ({changed_rows} rows affected).'})



@app.get('/api/reference_frames')
def api_reference_frames():
    """
    Return list of reference frame image URLs for the current video.
    These are saved every 100 frames during detection.
    """
    if not runner.current_log_name:
        return jsonify({'ok': True, 'frames': []})

    video_prefix = runner.current_log_name + "_ref_"
    images = [
        f for f in os.listdir(REFERENCE_DIR)
        if f.startswith(video_prefix) and f.endswith('.jpg')
    ]
    # Sort by frame number in filename
    def extract_frame_num(filename):
        try:
            return int(filename.split('_ref_')[-1].split('.jpg')[0])
        except:
            return 0

    images.sort(key=extract_frame_num)
    urls = [f"/reference_frames/{img}" for img in images]
    return jsonify({'ok': True, 'frames': urls})


@app.get('/reference_frames/<path:filename>')
def serve_reference_frame(filename):
    """Serve individual reference frame images."""
    return send_from_directory(REFERENCE_DIR, filename)


@app.get('/download/<path:filename>')
def download_file(filename):
    """Allow downloading generated CSVs like movement_log_<video>.csv."""
    return send_from_directory(DATA_DIR, filename, as_attachment=True)


# -------------------------------
# Main entry
# -------------------------------
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
