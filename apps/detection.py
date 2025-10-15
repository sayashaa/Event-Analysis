import os
import cv2
import time
import math
import csv
import threading
import queue
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


class DetectionRunner:
    def __init__(self, data_dir, crops_dir, output_video):
        self.data_dir = data_dir
        self.crops_dir = crops_dir
        self.output_video = output_video
        os.makedirs(self.crops_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.output_video), exist_ok=True)

        # Config
        self.distance_threshold = 30
        self.TEMPLATE_THRESHOLD = 0.1
        self.MIN_HEIGHT = 50
        self.NEIGHBOR_THRESHOLD = 50
        self.CONFIDENCE = 0.4
        self.IOU_THRESH = 0.6
        self.IMG_SIZE = 928
        self.MAX_DET = 300
        self.DEVICE = 'cpu'  # change to 'cuda' if GPU available
        self.BORDER = 50

        # State
        self._thread = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._frame_queue = queue.Queue(maxsize=5)
        self.is_running = False
        self.current_log_name = None
        self.reference_dir = os.path.join(self.data_dir, 'reference_frames')
        os.makedirs(self.reference_dir, exist_ok=True)

        # Mapping and template storage
        self._id_mapping = {}
        self._template_coords = {}
        self._saved_crops = set()   # <-- track saved crops to avoid duplicates

        # Models
        self.yolo = YOLO('yolo11n.pt')
        self.tracker = DeepSort(
            max_age=90,
            n_init=5,
            max_iou_distance=0.9,
            nn_budget=None,
            embedder='mobilenet',
            half=True,
            bgr=True,
        )

    # ----------------------------
    # Control functions
    # ----------------------------
    def start(self, video_path):
        if self.is_running:
            return
        self.is_running = True
        self._stop_event.clear()
        self._pause_event.clear()
        self._thread = threading.Thread(target=self._run, args=(video_path,), daemon=True)
        self._thread.start()

    def stop(self):
        if not self.is_running:
            return
        self._stop_event.set()
        self._pause_event.clear()
        if self._thread:
            self._thread.join(timeout=2)
        self.is_running = False

    def pause(self):
        if self.is_running:
            self._pause_event.set()

    def resume(self):
        if self.is_running:
            self._pause_event.clear()

    def frame_generator(self):
        while self.is_running:
            try:
                frame = self._frame_queue.get(timeout=1)
                yield frame
            except queue.Empty:
                continue

    # ----------------------------
    # Reassign & Delete in memory
    # ----------------------------
    def reassign_id_in_memory(self, wrong_id, correct_id):
        wrong_key = int(wrong_id) if str(wrong_id).isdigit() else wrong_id
        correct_key = int(correct_id) if str(correct_id).isdigit() else correct_id

        for ds_id, mapped_id in list(self._id_mapping.items()):
            if mapped_id == wrong_key:
                self._id_mapping[ds_id] = correct_key

        if wrong_key in self._template_coords:
            self._template_coords[correct_key] = self._template_coords[wrong_key]
            del self._template_coords[wrong_key]

    def delete_id_in_memory(self, target_id, delete_history=False):
        target_key = int(target_id) if str(target_id).isdigit() else target_id

        for ds_id, mapped_id in list(self._id_mapping.items()):
            if mapped_id == target_key:
                del self._id_mapping[ds_id]

        if delete_history and target_key in self._template_coords:
            del self._template_coords[target_key]

    # ----------------------------
    # Reference frame saving
    # ----------------------------
    def _save_reference_frame(self, frame, frame_idx, confirmed, id_mapping, video_name):
        annotated = frame.copy()
        for t in confirmed:
            ds_id = int(t.track_id)
            fid = id_mapping.get(ds_id, 'unknown')
            x1, y1, x2, y2 = map(int, t.to_ltrb())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            color = (0, 255, 0) if not str(fid).startswith('unknown') else (0, 0, 255)
            cv2.circle(annotated, (cx, cy), 5, color, -1)
            cv2.putText(annotated, f"ID {fid}", (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        ref_name = f"{video_name}_ref_{frame_idx}.jpg"
        ref_path = os.path.join(self.reference_dir, ref_name)
        cv2.imwrite(ref_path, annotated)

    # ----------------------------
    # Crop saving
    # ----------------------------
    def _save_crop(self, frame, bbox, assigned_id, frame_idx, video_name):
        """
        Save crop for the assigned ID only once.
        """
        crop_name = f"{video_name}_id-{assigned_id}_frame-{frame_idx}.jpg"
        crop_path = os.path.join(self.crops_dir, crop_name)

        if crop_name in self._saved_crops:
            return

        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return

        cv2.imwrite(crop_path, crop)
        self._saved_crops.add(crop_name)

    # ----------------------------
    # Main Detection Loop
    # ----------------------------
    def _run(self, video_path):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        log_filename = f"movement_log_{video_name}.csv"
        log_path = os.path.join(self.data_dir, log_filename)
        self.current_log_name = video_name

        # Clear old reference frames for this video
        for f in os.listdir(self.reference_dir):
            if f.startswith(video_name + "_ref_"):
                os.remove(os.path.join(self.reference_dir, f))

        self._id_mapping = {}
        self._template_coords = {}
        self._saved_crops.clear()
        id_mapping = self._id_mapping
        template_coords = self._template_coords

        max_id = 0
        unknown_counter = 0
        first_frame = True
        N_previous = 0

        with open(log_path, 'w', newline='', encoding='utf-8') as log_file:
            log_writer = csv.writer(log_file)
            log_writer.writerow(['frame', 'id', 'x', 'y'])

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_video, fourcc, fps, (frame_w, frame_h))

            frame_idx = 0

            while cap.isOpened() and not self._stop_event.is_set():
                while self._pause_event.is_set() and not self._stop_event.is_set():
                    time.sleep(0.1)

                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                # --- YOLO Detection ---
                result = self.yolo.predict(
                    source=frame,
                    classes=[0],
                    conf=self.CONFIDENCE,
                    iou=self.IOU_THRESH,
                    imgsz=self.IMG_SIZE,
                    max_det=self.MAX_DET,
                    device=self.DEVICE,
                    verbose=False,
                )[0]

                detections = []
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    clss = result.boxes.cls.cpu().numpy()
                    for box, score, cls in zip(boxes, confs, clss):
                        if int(cls) != 0:
                            continue
                        x1, y1, x2, y2 = map(int, box)
                        if (y2 - y1) < self.MIN_HEIGHT:
                            continue
                        detections.append(([x1, y1, x2 - x1, y2 - y1], float(score), 'person'))

                tracks = self.tracker.update_tracks(detections, frame=frame)
                confirmed = [t for t in tracks if t.is_confirmed()]
                current_count = len(confirmed)

                # --- First frame init ---
                if first_frame and current_count > 0:
                    first_frame = False
                    N_previous = current_count
                    for t in confirmed:
                        ds_id = int(t.track_id)
                        x1, y1, x2, y2 = map(int, t.to_ltrb())
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        id_mapping[ds_id] = ds_id
                        template_coords[ds_id] = (cx, cy)
                        max_id = max(max_id, ds_id)
                        self._save_crop(frame, (x1, y1, x2, y2), ds_id, frame_idx, video_name)
                else:
                    prev_ds = set(id_mapping.keys())
                    curr_ds = set(int(t.track_id) for t in confirmed)
                    for lost in prev_ds - curr_ds:
                        fid = id_mapping.get(lost)
                        pos = template_coords.get(fid)
                        if pos and (pos[0] < self.BORDER or pos[0] > frame_w - self.BORDER):
                            template_coords.pop(fid, None)
                            id_mapping.pop(lost, None)

                    def try_matching(cx, cy):
                        if not template_coords:
                            return None
                        best_key, best_val = min(template_coords.items(), key=lambda kv: euclidean(kv[1], (cx, cy)))
                        if euclidean(best_val, (cx, cy)) >= self.distance_threshold:
                            return None
                        return best_key

                    for t in confirmed:
                        ds_id = int(t.track_id)
                        x1, y1, x2, y2 = map(int, t.to_ltrb())
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                        if ds_id in id_mapping:
                            if isinstance(id_mapping[ds_id], int):
                                template_coords[id_mapping[ds_id]] = (cx, cy)
                        else:
                            assigned = try_matching(cx, cy)
                            if assigned is None:
                                if cx < self.BORDER or cx > frame_w - self.BORDER:
                                    max_id += 1
                                    assigned = max_id
                                else:
                                    unknown_counter += 1
                                    assigned = f"unknown_{unknown_counter}"
                            id_mapping[ds_id] = assigned
                            if isinstance(assigned, int):
                                template_coords[assigned] = (cx, cy)
                                self._save_crop(frame, (x1, y1, x2, y2), assigned, frame_idx, video_name)

                    N_previous = current_count

                # --- Log ---
                for t in confirmed:
                    ds_id = int(t.track_id)
                    fid = id_mapping.get(ds_id, 'unknown')
                    x1, y1, x2, y2 = map(int, t.to_ltrb())
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    log_writer.writerow([frame_idx, fid, cx, cy])

                # --- Save reference frames every 100 frames ---
                if frame_idx % 100 == 0 and current_count > 0:
                    self._save_reference_frame(frame, frame_idx, confirmed, id_mapping, video_name)

                # --- Overlay for preview ---
                cv2.line(frame, (self.BORDER, 0), (self.BORDER, frame_h), (255, 255, 0), 2)
                cv2.line(frame, (frame_w - self.BORDER, 0), (frame_w - self.BORDER, frame_h), (255, 255, 0), 2)
                for t in confirmed:
                    ds_id = int(t.track_id)
                    fid = id_mapping.get(ds_id, 'unknown')
                    x1, y1, x2, y2 = map(int, t.to_ltrb())
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    color = (0, 0, 255) if str(fid).startswith('unknown') else (0, 255, 0)
                    cv2.circle(frame, (cx, cy), 5, color, -1)
                    cv2.putText(frame, f"ID {fid}", (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                out.write(frame)
                ret2, jpg = cv2.imencode('.jpg', frame)
                if ret2:
                    if self._frame_queue.full():
                        self._frame_queue.get_nowait()
                    self._frame_queue.put_nowait(jpg.tobytes())

                time.sleep(max(0, 1.0 / fps))

            cap.release()
            out.release()

        self.is_running = False
