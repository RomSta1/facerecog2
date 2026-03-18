import logging
import os
import threading
import time

import cv2
import mediapipe as mp

log = logging.getLogger(__name__)

_mp_hands = mp.solutions.hands


def _is_thumbs_up(hand_landmarks):
    """Return True if hand shows thumbs-up gesture.

    Works for both frontal and top-down cameras:
    - Frontal: thumb tip is above IP joint in Y axis
    - Top-down: thumb tip is closer to camera (lower Z) than wrist
    Either condition suffices so the gesture fires regardless of camera angle.
    """
    lm = hand_landmarks.landmark
    # Frontal view: thumb tip above IP joint in image Y
    thumb_up_y = lm[4].y < lm[3].y < lm[2].y
    # Top-down view: thumb tip significantly closer to camera than wrist
    thumb_up_z = (lm[0].z - lm[4].z) > 0.08
    thumb_up = thumb_up_y or thumb_up_z
    # All other fingers curled: tip below (or same depth as) PIP joint
    fingers_curled = all(
        lm[tip].y > lm[pip].y or lm[tip].z > lm[pip].z - 0.02
        for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]
    )
    return thumb_up and fingers_curled


class GesturePipeline(threading.Thread):
    def __init__(self, camera_name, rtsp, rotation, fps_process,
                 cooldown_sec, stop_event, on_thumbs_up):
        super().__init__(name=f"gesture-{camera_name}", daemon=True)
        self.camera_name = camera_name
        self.rtsp = rtsp
        self.rotation = rotation
        self.fps_process = fps_process
        self.cooldown_sec = cooldown_sec
        self.stop_event = stop_event
        self.on_thumbs_up = on_thumbs_up
        self._last_trigger = 0.0

    def run(self):
        log.info("[gesture-%s] starting", self.camera_name)
        frame_interval = 1.0 / self.fps_process
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        hands = _mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        while not self.stop_event.is_set():
            cap = cv2.VideoCapture(self.rtsp, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                log.error("[gesture-%s] cannot open stream, retrying in 10s", self.camera_name)
                self.stop_event.wait(10)
                continue
            last_process = 0.0
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    log.warning("[gesture-%s] read failed, reconnecting", self.camera_name)
                    break
                now = time.monotonic()
                if now - last_process < frame_interval:
                    continue
                last_process = now
                frame = self._rotate(frame)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                if results.multi_hand_landmarks:
                    for hand_lm in results.multi_hand_landmarks:
                        if _is_thumbs_up(hand_lm):
                            if now - self._last_trigger >= self.cooldown_sec:
                                log.info("[gesture-%s] thumbs up → trigger", self.camera_name)
                                self._last_trigger = now
                                try:
                                    self.on_thumbs_up()
                                except Exception:
                                    log.exception("[gesture-%s] callback failed", self.camera_name)
                            break
            cap.release()
            if not self.stop_event.is_set():
                self.stop_event.wait(3)
        hands.close()
        log.info("[gesture-%s] stopped", self.camera_name)

    def _rotate(self, frame):
        r = self.rotation
        if r == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if r == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if r == 270 or r == -90:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame
