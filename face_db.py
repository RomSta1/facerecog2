import logging
import os
import pickle
import threading

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

log = logging.getLogger(__name__)

_ROTATIONS = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]


def _make_enroll_app():
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def _ensure_min_size(img, min_side=224):
    """Upscale image if too small for face detection."""
    h, w = img.shape[:2]
    if min(h, w) < min_side:
        scale = min_side / min(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    return img


def _best_face(faces_app, img_bgr):
    img_bgr = _ensure_min_size(img_bgr)
    for rot in _ROTATIONS:
        frame = cv2.rotate(img_bgr, rot) if rot is not None else img_bgr
        faces = faces_app.get(np.ascontiguousarray(frame[:, :, ::-1]))
        faces = [f for f in faces if f.det_score >= 0.2]
        if faces:
            return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return None


class FaceDB:
    def __init__(self, faces_dir, cache_path, similarity_threshold, unknown_threshold):
        self.faces_dir = faces_dir
        self.cache_path = cache_path
        self.similarity_threshold = similarity_threshold
        self.unknown_threshold = unknown_threshold
        self._lock = threading.RLock()
        self._embeddings = {}  # name -> list of np.ndarray
        self._load()

    # ── public API ──────────────────────────────────────────────────────────

    def identify(self, embedding):
        """Return (name, score). name='unknown' if below threshold."""
        with self._lock:
            best_name, best_score = "unknown", 0.0
            for name, embs in self._embeddings.items():
                for e in embs:
                    s = float(np.dot(embedding, e))
                    if s > best_score:
                        best_score = s
                        best_name = name
            if best_score >= self.similarity_threshold:
                return best_name, best_score
            if best_score >= self.unknown_threshold:
                return "uncertain", best_score
            return "unknown", best_score

    def enroll(self, name, img_bgr):
        """Add one image to person. Returns total embedding count."""
        enroll_app = _make_enroll_app()
        face = _best_face(enroll_app, img_bgr)
        if face is None:
            log.warning("enroll(%s): no face detected", name)
            return 0
        person_dir = os.path.join(self.faces_dir, name)
        os.makedirs(person_dir, exist_ok=True)
        idx = len(os.listdir(person_dir))
        cv2.imwrite(os.path.join(person_dir, f"{idx:04d}.jpg"), img_bgr)
        with self._lock:
            self._embeddings.setdefault(name, []).append(face.normed_embedding)
            self._save_cache()
            return len(self._embeddings[name])

    def delete_person(self, name):
        with self._lock:
            if name not in self._embeddings:
                return False
            del self._embeddings[name]
            self._save_cache()
        import shutil
        person_dir = os.path.join(self.faces_dir, name)
        if os.path.isdir(person_dir):
            shutil.rmtree(person_dir)
        return True

    def list_persons(self):
        with self._lock:
            return {name: len(embs) for name, embs in self._embeddings.items()}

    def photo_count(self, name):
        with self._lock:
            return len(self._embeddings.get(name, []))

    def rebuild_person(self, name):
        """Re-scan one person's folder and update their embeddings in cache."""
        enroll_app = _make_enroll_app()
        person_dir = os.path.join(self.faces_dir, name)
        embs = []
        if os.path.isdir(person_dir):
            for fname in sorted(os.listdir(person_dir)):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img = cv2.imread(os.path.join(person_dir, fname))
                if img is None:
                    continue
                face = _best_face(enroll_app, img)
                if face is not None:
                    embs.append(face.normed_embedding)
        with self._lock:
            if embs:
                self._embeddings[name] = embs
            elif name in self._embeddings:
                del self._embeddings[name]
            self._save_cache()
        log.info("rebuild_person(%s): %d embeddings", name, len(embs))

    def reload(self):
        with self._lock:
            self._load()

    # ── internal ────────────────────────────────────────────────────────────

    def _load(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    self._embeddings = pickle.load(f)
                log.info("FaceDB loaded from cache: %s persons", len(self._embeddings))
                return
            except Exception as e:
                log.warning("Cache load failed (%s), rebuilding from disk", e)
        self._rebuild()

    def _rebuild(self):
        log.info("FaceDB: rebuilding from %s", self.faces_dir)
        enroll_app = _make_enroll_app()
        embeddings = {}
        if not os.path.isdir(self.faces_dir):
            os.makedirs(self.faces_dir, exist_ok=True)
        for person in sorted(os.listdir(self.faces_dir)):
            person_dir = os.path.join(self.faces_dir, person)
            if not os.path.isdir(person_dir):
                continue
            embs = []
            for fname in sorted(os.listdir(person_dir)):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img = cv2.imread(os.path.join(person_dir, fname))
                if img is None:
                    continue
                face = _best_face(enroll_app, img)
                if face is None:
                    log.debug("FaceDB: no face in %s/%s", person, fname)
                    continue
                embs.append(face.normed_embedding)
            if embs:
                embeddings[person] = embs
                log.info("FaceDB: %s → %d embeddings", person, len(embs))
            else:
                log.warning("FaceDB: %s → 0 embeddings (skipped)", person)
        self._embeddings = embeddings
        self._save_cache()
        log.info("FaceDB: rebuild done, %d persons", len(embeddings))

    def _save_cache(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump(self._embeddings, f)
