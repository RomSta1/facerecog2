import base64
import json
import logging
import threading
import time

import cv2
import paho.mqtt.client as mqtt

log = logging.getLogger(__name__)


class MQTTClient:
    def __init__(self, cfg):
        self.topic = cfg["topic"]
        self._client = mqtt.Client()
        if cfg.get("user"):
            self._client.username_pw_set(cfg["user"], cfg.get("password", ""))
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._connected = False
        self._lock = threading.Lock()
        self._host = cfg["host"]
        self._port = int(cfg.get("port", 1883))
        self._connect()

    def _connect(self):
        try:
            self._client.connect(self._host, self._port, keepalive=60)
            self._client.loop_start()
        except Exception as e:
            log.error("MQTT connect error: %s", e)

    def _on_connect(self, client, userdata, flags, rc):
        self._connected = rc == 0
        if rc == 0:
            log.info("MQTT connected")
        else:
            log.error("MQTT connect failed rc=%d", rc)

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        log.warning("MQTT disconnected rc=%d, reconnecting...", rc)
        time.sleep(5)
        self._connect()

    def publish(self, camera, name, score, snapshot_bgr):
        """Publish recognition result with JPEG snapshot."""
        _, buf = cv2.imencode(".jpg", snapshot_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        jpg_bytes = buf.tobytes()
        img_b64 = base64.b64encode(jpg_bytes).decode()
        payload = json.dumps({
            "camera": camera,
            "name": name,
            "score": round(score, 3),
            "snapshot": img_b64,
        })
        with self._lock:
            if self._connected:
                # JSON result (existing)
                self._client.publish(self.topic, payload, qos=1)
                # raw JPEG for HA mqtt camera — facerecog2/<camera>/image
                img_topic = f"{self.topic.rsplit('/', 1)[0]}/{camera}/image"
                self._client.publish(img_topic, jpg_bytes, qos=0, retain=True)
            else:
                log.warning("MQTT not connected, dropping message")
