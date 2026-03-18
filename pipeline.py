import concurrent.futures
import logging
import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np

log = logging.getLogger(__name__)

UNKNOWN_COOLDOWN = 3  # seconds between saving unknown crops per camera


def apply_clahe(frame_bgr):
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _crop_face(frame, bbox, pad=20):
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return frame[y1:y2, x1:x2]


class CameraPipeline(threading.Thread):
    def __init__(self, camera_name, cfg, recognizer, face_db, mqtt_client,
                 snapshot_dir, cooldown_sec, stop_event):
        super().__init__(name=f"cam-{camera_name}", daemon=True)
        self.camera_name = camera_name
        self.rtsp = cfg["rtsp"]
        self.rtsp_hd = cfg.get("rtsp_hd")
        self.rotation = cfg.get("rotation", 0)
        self.fps_process = cfg.get("fps_process", 2)
        # minimum face width in pixels to attempt recognition/save
        # faces smaller than this are completely ignored (distant/blurry)
        self.min_face_w = cfg.get("min_face_w", 0)
        self.max_pitch = cfg.get("max_pitch", 35)  # max head pitch for unknown saves
        self.recognizer = recognizer
        self.face_db = face_db
        self.mqtt = mqtt_client
        self.snapshot_dir = snapshot_dir
        self.unknown_dir = os.path.join(os.path.dirname(snapshot_dir), "unknown")
        self.cooldown_sec = cooldown_sec
        self.stop_event = stop_event
        self._last_seen = {}        # name -> datetime (cooldown start)
        self._best = {}             # name -> {score, sim, frame, face, hd_future, deadline, sent}
        self._last_unknown = 0.0   # monotonic (unknown saves)
        self._best_shot_window = 4.0  # seconds to collect best frame
        self._last_frame = None      # most recent rotated frame from run() loop
        # HD grabs run in a background thread so they never block the detection loop
        self._hd_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1,
                                                                   thread_name_prefix=f"hd-{camera_name}")

    def run(self):
        log.info("[%s] starting %s", self.camera_name, self.rtsp)
        frame_interval = 1.0 / self.fps_process
        # force TCP transport to avoid HEVC/H.265 packet loss over UDP
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        while not self.stop_event.is_set():
            cap = cv2.VideoCapture(self.rtsp, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                log.error("[%s] cannot open stream, retrying in 10s", self.camera_name)
                self.stop_event.wait(10)
                continue
            log.info("[%s] stream opened", self.camera_name)
            last_process = 0.0
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    log.warning("[%s] read failed, reconnecting", self.camera_name)
                    break
                now = time.monotonic()
                if now - last_process < frame_interval:
                    continue
                last_process = now
                self._last_frame = self._rotate(frame)  # store rotated for trigger()
                try:
                    self._process(frame)
                except Exception:
                    log.exception("[%s] _process crashed", self.camera_name)
            cap.release()
            if not self.stop_event.is_set():
                self.stop_event.wait(3)
        log.info("[%s] stopped", self.camera_name)

    def _rotate(self, frame):
        r = self.rotation
        if r == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if r == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if r == 270 or r == -90:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def _process(self, frame):
        frame = self._rotate(frame)
        enhanced = apply_clahe(frame)
        faces = self.recognizer.get_faces(enhanced)
        if not faces:
            return

        now_dt = datetime.now()
        now_mono = time.monotonic()

        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            face_w = x2 - x1

            # skip distant/blurry faces entirely — no recognition, no HD grab
            if self.min_face_w and face_w < self.min_face_w:
                log.info("[%s] SKIP (too small): w=%d < min=%d det=%.2f",
                         self.camera_name, face_w, self.min_face_w, face["det_score"])
                continue

            name, score = self.face_db.identify(face["embedding"])
            log.info("[%s] det=%.2f w=%d → %s %.3f",
                     self.camera_name, face["det_score"], face_w, name, score)

            if name == "unknown":
                pose = face.get("pose")
                frontal = (pose is None or
                           (abs(pose[0]) < 30 and abs(pose[1]) < self.max_pitch))
                if (frontal
                        and face_w >= 80
                        and now_mono - self._last_unknown >= UNKNOWN_COOLDOWN
                        and face["det_score"] >= 0.45):
                    # async HD grab + save so loop is not blocked
                    self._hd_executor.submit(self._grab_and_save_unknown, frame, face)
                    self._last_unknown = now_mono
                else:
                    yaw = round(pose[0], 1) if pose else None
                    pitch = round(pose[1], 1) if pose else None
                    log.info("[%s] unknown SKIP: w=%d frontal=%s det=%.2f yaw=%s pitch=%s",
                             self.camera_name, face_w, frontal, face["det_score"], yaw, pitch)
                continue

            last = self._last_seen.get(name)
            if last and (now_dt - last).total_seconds() < self.cooldown_sec:
                # within cooldown — update best shot if window still open
                best = self._best.get(name)
                if best and now_mono < best["deadline"] and face["det_score"] > best["score"]:
                    best["score"] = face["det_score"]
                    best["frame"] = frame
                    best["face"] = face
                continue

            # first detection after cooldown — start best-shot window
            # HD grab is submitted to background thread immediately so loop stays fast
            self._last_seen[name] = now_dt
            hd_future = self._hd_executor.submit(self._grab_hd_frame)
            self._best[name] = {
                "score": face["det_score"],
                "sim": score,
                "frame": frame,
                "face": face,
                "hd_future": hd_future,
                "deadline": now_mono + self._best_shot_window,
                "sent": False,
            }

        # flush best shots whose window has closed
        for name, best in list(self._best.items()):
            if not best["sent"] and now_mono >= best["deadline"]:
                best["sent"] = True
                hd_frame = None
                try:
                    hd_frame = best["hd_future"].result(timeout=2.0)
                except Exception:
                    pass
                if hd_frame is not None:
                    snapshot = self._draw_hd(hd_frame, name, best["sim"])
                else:
                    snapshot = self._draw(best["frame"], best["face"], name, best["sim"])
                self._save_snapshot(snapshot, name)
                self.mqtt.publish(self.camera_name, name, best["sim"], snapshot)
                log.info("[%s] RECOGNIZED %s det=%.2f sim=%.3f (best shot, hd=%s)",
                         self.camera_name, name, best["score"], best["sim"],
                         hd_frame is not None)

    def trigger(self):
        """Force immediate recognition — grabs fresh frames from stream, no stale cache."""
        log.info("[%s] triggered by external event", self.camera_name)
        try:
            cap = cv2.VideoCapture(self.rtsp, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            found = False
            # grab up to 10 fresh frames, process each until face found
            for i in range(10):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = self._rotate(frame)
                cv2.imwrite(f"/tmp/trigger_{self.camera_name}.jpg", frame)
                faces = self.recognizer.get_faces(apply_clahe(frame))
                log.info("[%s] trigger frame %d: %d faces", self.camera_name, i, len(faces))
                if faces:
                    self._process_raw(frame)
                    found = True
                    break
            cap.release()
            if not found:
                log.info("[%s] trigger: no faces in any frame", self.camera_name)
        except Exception:
            log.exception("[%s] trigger crashed", self.camera_name)

    def _process_raw(self, frame):
        """Run recognition on an already-rotated frame (used by trigger)."""
        enhanced = apply_clahe(frame)
        faces = self.recognizer.get_faces(enhanced)
        if not faces:
            log.info("[%s] trigger: no faces detected", self.camera_name)
            return
        now_dt = datetime.now()
        now_mono = time.monotonic()
        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            face_w = x2 - x1
            name, score = self.face_db.identify(face["embedding"])
            log.info("[%s] trigger det=%.2f w=%d → %s %.3f",
                     self.camera_name, face["det_score"], face_w, name, score)
            if name == "unknown":
                pose = face.get("pose")
                frontal = (pose is None or (abs(pose[0]) < 30 and abs(pose[1]) < 35))
                if frontal and face_w >= 80 and face["det_score"] >= 0.5:
                    self._hd_executor.submit(self._grab_and_save_unknown, frame, face)
                continue
            last = self._last_seen.get(name)
            if last and (now_dt - last).total_seconds() < self.cooldown_sec:
                log.info("[%s] trigger: %s in cooldown", self.camera_name, name)
                continue
            self._last_seen[name] = now_dt
            snapshot = self._draw_hd(frame, name, score)
            self._save_snapshot(snapshot, name)
            self.mqtt.publish(self.camera_name, name, score, snapshot)
            log.info("[%s] trigger RECOGNIZED %s sim=%.3f", self.camera_name, name, score)

    def _grab_hd_frame(self):
        """Open HD stream in background, grab one frame, return rotated (or None)."""
        if not self.rtsp_hd:
            return None
        try:
            cap = cv2.VideoCapture(self.rtsp_hd, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            frame = None
            for _ in range(10):
                ret, f = cap.read()
                if ret:
                    frame = f
                    break
            cap.release()
            if frame is not None:
                return self._rotate(frame)
        except Exception as e:
            log.debug("[%s] HD grab failed: %s", self.camera_name, e)
        return None

    def _grab_and_save_unknown(self, low_res_frame, face):
        """Grab HD frame in background and save unknown crop."""
        hd = self._grab_hd_frame()
        if hd is not None:
            self._save_unknown(hd, face, use_full=True)
        else:
            self._save_unknown(low_res_frame, face, use_full=False)

    def _draw(self, frame, face, name, score):
        out = frame.copy()
        x1, y1, x2, y2 = face["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{name} {score:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return out

    def _draw_hd(self, frame, name, score):
        """Overlay name/score in corner of HD frame (no bbox — coords don't match)."""
        out = frame.copy()
        h = out.shape[0]
        label = f"{name} {score:.2f}"
        cv2.putText(out, label, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        return out

    def _save_snapshot(self, frame, name):
        day_dir = os.path.join(self.snapshot_dir, datetime.now().strftime("%Y-%m-%d"))
        os.makedirs(day_dir, exist_ok=True)
        ts = datetime.now().strftime("%H%M%S")
        path = os.path.join(day_dir, f"{self.camera_name}_{name}_{ts}.jpg")
        cv2.imwrite(path, frame)

    def _save_unknown(self, frame, face, use_full=False):
        if use_full:
            out = frame
        else:
            out = _crop_face(frame, face["bbox"])
            if out.size == 0:
                return
        os.makedirs(self.unknown_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.unknown_dir, f"{self.camera_name}_{ts}.jpg")
        cv2.imwrite(path, out)
        log.info("[%s] saved unknown → %s (hd=%s)", self.camera_name, path, use_full)
