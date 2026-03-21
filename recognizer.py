import threading

import numpy as np
import insightface
from insightface.app import FaceAnalysis


class Recognizer:
    def __init__(self, det_score_min=0.3, det_size=320):
        self.det_score_min = det_score_min
        self._lock = threading.Lock()  # FIX: InsightFace is not thread-safe across cameras
        # OpenVINO on CPU is faster than iGPU on N100 (AVX2/VNNI optimized)
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["OpenVINOExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=(det_size, det_size))
        # warmup to avoid 7s latency on first real frame
        self.app.get(np.zeros((480, 640, 3), dtype=np.uint8))

    def get_faces(self, frame_bgr):
        """Return list of face dicts with bbox, embedding, det_score, pose."""
        with self._lock:  # FIX: serialize calls from multiple camera threads
            rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
            faces = self.app.get(rgb)
        result = []
        for f in faces:
            if f.det_score < self.det_score_min:
                continue
            pose = f.pose.tolist() if f.pose is not None else None
            result.append({
                "bbox": f.bbox.astype(int).tolist(),
                "embedding": f.normed_embedding,
                "det_score": float(f.det_score),
                "pose": pose,
            })
        return result
