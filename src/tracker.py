from __future__ import annotations
from dataclasses import dataclass
import numpy as np

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    # a, b: [x1,y1,x2,y2]
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0

@dataclass
class Track:
    tid: int
    bbox: np.ndarray  # xyxy
    cls: int
    conf: float
    age: int = 0       # frames since last matched
    hits: int = 1

class IoUTracker:
    """
    Simple online tracker:
      - matches detections to existing tracks by IoU (greedy)
      - creates new tracks for unmatched detections
      - removes tracks that exceed max_age
    """
    def __init__(self, iou_th: float = 0.3, max_age: int = 30) -> None:
        self.iou_th = float(iou_th)
        self.max_age = int(max_age)
        self._next_id = 1
        self.tracks: list[Track] = []

    def update(self, dets_xyxy: np.ndarray, dets_cls: np.ndarray, dets_conf: np.ndarray) -> list[Track]:
        # age existing tracks
        for t in self.tracks:
            t.age += 1

        used_det = np.zeros((len(dets_xyxy),), dtype=bool)

        # greedy matching by IoU
        for t in self.tracks:
            best_iou = 0.0
            best_j = -1
            for j, bb in enumerate(dets_xyxy):
                if used_det[j]:
                    continue
                # optional: enforce same class for stability
                if int(dets_cls[j]) != int(t.cls):
                    continue
                val = iou_xyxy(t.bbox, bb)
                if val > best_iou:
                    best_iou = val
                    best_j = j

            if best_j >= 0 and best_iou >= self.iou_th:
                t.bbox = dets_xyxy[best_j].copy()
                t.cls = int(dets_cls[best_j])
                t.conf = float(dets_conf[best_j])
                t.age = 0
                t.hits += 1
                used_det[best_j] = True

        # create new tracks
        for j in range(len(dets_xyxy)):
            if used_det[j]:
                continue
            self.tracks.append(
                Track(
                    tid=self._next_id,
                    bbox=dets_xyxy[j].copy(),
                    cls=int(dets_cls[j]),
                    conf=float(dets_conf[j]),
                )
            )
            self._next_id += 1

        # remove old tracks
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]
        return self.tracks

