from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np

Point = Tuple[int, int]

def _side_of_line(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """
    Returns the signed area (cross product) to know which side of line AB point P lies on.
    Positive/negative indicates side; near 0 means on the line.
    """
    return float((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]))

@dataclass
class LineCounterConfig:
    a: Point = (100, 380)     # line start (x,y)
    b: Point = (860, 380)     # line end   (x,y)
    min_hits: int = 2         # require track stability before counting
    cooldown_frames: int = 30 # prevent double count for same ID

class LineCounter:
    def __init__(self, cfg: LineCounterConfig) -> None:
        self.cfg = cfg
        self.a = np.array(cfg.a, dtype=np.float32)
        self.b = np.array(cfg.b, dtype=np.float32)

        # track_id -> last side sign
        self.last_side: Dict[int, float] = {}
        # track_id -> last frame index counted
        self.last_count_frame: Dict[int, int] = {}

        self.count_total = 0
        self.count_by_label: Dict[str, int] = {}

    def update(self, frame_idx: int, tracks: list) -> Dict[str, int]:
    # asegúrate de tener este diccionario
      if not hasattr(self, "last_centroid"):
          self.last_centroid = {}

      for t in tracks:
          if getattr(t, "hits", 1) < self.cfg.min_hits:
              continue

          tid = int(t.tid)
          x1, y1, x2, y2 = [float(v) for v in t.bbox]
          cx = 0.5 * (x1 + x2)
          cy = 0.5 * (y1 + y2)
          p = np.array([cx, cy], dtype=np.float32)

        # --- (A) centroid anterior SIEMPRE disponible ---
          prev_p = self.last_centroid.get(tid, None)
          self.last_centroid[tid] = p.copy()  # guardamos para el siguiente frame

        # lado respecto a la línea
          side = _side_of_line(p, self.a, self.b)

          if tid not in self.last_side:
              self.last_side[tid] = side
              continue

          prev_side = self.last_side[tid]
          self.last_side[tid] = side

        # crossing if sign changed
          if prev_side == 0 or side == 0:
              continue

          crossed = (prev_side > 0 and side < 0) or (prev_side < 0 and side > 0)
          if not crossed:
              continue

        # si aún no tenemos centroide previo, no podemos sacar dirección
          if prev_p is None:
              continue

          dy = float(p[1] - prev_p[1])

        # IN: se mueve hacia abajo (y aumenta). OUT: hacia arriba (y disminuye)
          direction = "IN" if dy > 0 else "OUT"

        # cooldown por (id, dirección) para no duplicar
          last_f = self.last_count_frame.get((tid, direction), -10**9)
          if frame_idx - last_f < self.cfg.cooldown_frames:
              continue

          self.last_count_frame[(tid, direction)] = frame_idx
          self.count_total += 1
          self.count_by_label[direction] = self.count_by_label.get(direction, 0) + 1

      return self.count_by_label


    def count_for_label(self, label: str, inc: int = 1) -> None:
        self.count_by_label[label] = self.count_by_label.get(label, 0) + inc
