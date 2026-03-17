import logging
import os
import signal
import threading
import time

import yaml

from face_db import FaceDB
from recognizer import Recognizer
from pipeline import CameraPipeline
from mqtt_client import MQTTClient
from api import run_api

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("main")

CONFIG_PATH = os.environ.get("FR2_CONFIG", "/opt/facerecog2/app/config.yml")


def _cleanup_snapshots(snapshot_dir, keep_hours, stop_event):
    """Delete snapshot files older than keep_hours. Runs every hour."""
    while not stop_event.is_set():
        try:
            cutoff = time.time() - keep_hours * 3600
            removed = 0
            if os.path.isdir(snapshot_dir):
                for day in os.listdir(snapshot_dir):
                    day_dir = os.path.join(snapshot_dir, day)
                    if not os.path.isdir(day_dir):
                        continue
                    for fname in os.listdir(day_dir):
                        fpath = os.path.join(day_dir, fname)
                        if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                            os.remove(fpath)
                            removed += 1
                    try:
                        os.rmdir(day_dir)
                    except OSError:
                        pass
            if removed:
                log.info("Snapshot cleanup: removed %d files older than %dh", removed, keep_hours)
        except Exception as e:
            log.warning("Snapshot cleanup error: %s", e)
        stop_event.wait(3600)


def _make_pipeline(cam_name, cam_cfg, recognizer, face_db, mqtt, snap_path,
                   cooldown_sec, stop_event):
    t = CameraPipeline(
        camera_name=cam_name,
        cfg=cam_cfg,
        recognizer=recognizer,
        face_db=face_db,
        mqtt_client=mqtt,
        snapshot_dir=snap_path,
        cooldown_sec=cooldown_sec,
        stop_event=stop_event,
    )
    t.start()
    return t


def _watchdog(cameras_cfg, recognizer, face_db, mqtt, snap_path,
              cooldown_sec, stop_event, threads):
    """Restart any camera thread that has died."""
    while not stop_event.is_set():
        stop_event.wait(30)
        if stop_event.is_set():
            break
        for cam_name, cam_cfg in cameras_cfg.items():
            t = threads.get(cam_name)
            if t is None or not t.is_alive():
                log.error("[%s] thread dead — restarting", cam_name)
                new_t = _make_pipeline(cam_name, cam_cfg, recognizer, face_db,
                                       mqtt, snap_path, cooldown_sec, stop_event)
                threads[cam_name] = new_t


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    rec_cfg = cfg["recognition"]
    snap_cfg = cfg["snapshots"]
    db_cfg = cfg["face_db"]
    api_cfg = cfg["api"]
    mqtt_cfg = cfg["mqtt"]

    os.makedirs(snap_cfg["path"], exist_ok=True)
    os.makedirs(os.path.dirname(db_cfg["cache"]), exist_ok=True)

    face_db = FaceDB(
        faces_dir=db_cfg["path"],
        cache_path=db_cfg["cache"],
        similarity_threshold=rec_cfg["similarity_threshold"],
        unknown_threshold=rec_cfg["unknown_threshold"],
    )

    recognizer = Recognizer(
        det_score_min=rec_cfg["det_score_min"],
        det_size=rec_cfg["det_size"],
    )

    mqtt = MQTTClient(mqtt_cfg)

    # cameras dict is passed to run_api so /trigger endpoint can reach pipelines
    cameras = {}
    run_api(face_db, api_cfg["host"], api_cfg["port"], snap_cfg["path"], cameras)

    stop_event = threading.Event()

    keep_hours = snap_cfg.get("keep_hours", 24)
    cleanup_t = threading.Thread(
        target=_cleanup_snapshots,
        args=(snap_cfg["path"], keep_hours, stop_event),
        daemon=True, name="cleanup",
    )
    cleanup_t.start()

    threads = {}
    for cam_name, cam_cfg in cfg["cameras"].items():
        threads[cam_name] = _make_pipeline(
            cam_name, cam_cfg, recognizer, face_db, mqtt,
            snap_cfg["path"], rec_cfg["cooldown_sec"], stop_event,
        )
        cameras[cam_name] = threads[cam_name]  # expose to /trigger endpoint

    watchdog_t = threading.Thread(
        target=_watchdog,
        args=(cfg["cameras"], recognizer, face_db, mqtt,
              snap_cfg["path"], rec_cfg["cooldown_sec"], stop_event, threads),
        daemon=True, name="watchdog",
    )
    watchdog_t.start()

    def _shutdown(sig, frame):
        log.info("Shutting down (signal %d)...", sig)
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    stop_event.wait()
    for t in threads.values():
        t.join(timeout=5)
    log.info("Stopped")
    os._exit(0)


if __name__ == "__main__":
    main()
