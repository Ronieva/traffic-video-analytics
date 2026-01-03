from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from ultralytics import YOLO

from .tracker import IoUTracker
from .metrics import PerfMeter

class VideoAnalyticsPipeline:
    def __init__(
        self,
        model: str = "yolov8n.pt",
        device: str = "cpu",
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        classes: list[int] | None = None,
        tracker_iou_th: float = 0.3,
        tracker_max_age: int = 30,
    ) -> None:
        self.model = YOLO(model)
        self.device = device
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.classes = classes
        self.tracker = IoUTracker(iou_th=tracker_iou_th, max_age=tracker_max_age)
        self.perf = PerfMeter(window=60)

        # class names from Ultralytics
        self.names = self.model.model.names if hasattr(self.model, "model") else self.model.names

    def infer(self, frame_bgr: np.ndarray):
        self.perf.tic()
        results = self.model.predict(
            source=frame_bgr,
            device=self.device,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            verbose=False,
        )
        lat = self.perf.toc()

        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            det_xyxy = np.zeros((0,4), dtype=np.float32)
            det_cls = np.zeros((0,), dtype=np.int32)
            det_conf = np.zeros((0,), dtype=np.float32)
        else:
            det_xyxy = r0.boxes.xyxy.cpu().numpy().astype(np.float32)
            det_cls = r0.boxes.cls.cpu().numpy().astype(np.int32)
            det_conf = r0.boxes.conf.cpu().numpy().astype(np.float32)

        tracks = self.tracker.update(det_xyxy, det_cls, det_conf)

        counts: dict[str,int] = {}
        for t in tracks:
            name = self.names[int(t.cls)] if isinstance(self.names, dict) else str(t.cls)
            counts[name] = counts.get(name, 0) + 1

        return tracks, counts, lat, self.perf.fps

    def track_to_dict(self, frame_idx: int, tracks: list) -> dict:
        out = {"frame": frame_idx, "tracks": []}
        for t in tracks:
            out["tracks"].append({
                "id": int(t.tid),
                "cls": int(t.cls),
                "label": (self.names[int(t.cls)] if isinstance(self.names, dict) else str(t.cls)),
                "conf": float(t.conf),
                "bbox_xyxy": [float(x) for x in t.bbox.tolist()],
                "age": int(t.age),
                "hits": int(t.hits),
            })
        return out

    def save_json(self, data: list[dict], path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

