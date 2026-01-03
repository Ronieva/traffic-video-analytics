from __future__ import annotations
import cv2
import numpy as np

def draw_hud(frame: np.ndarray, fps: float, lat_ms: float, counts: dict[str,int]) -> np.ndarray:
    hud = frame.copy()
    y = 25
    cv2.putText(hud, f"FPS: {fps:.1f} | Latency: {lat_ms:.1f} ms", (10, y),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    y += 25
    if counts:
        txt = " | ".join([f"{k}: {v}" for k,v in counts.items()])
        cv2.putText(hud, f"Counts: {txt}", (10, y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    return hud

def draw_tracks(frame: np.ndarray, tracks: list, names: dict[int,str]) -> np.ndarray:
    out = frame
    for t in tracks:
        x1,y1,x2,y2 = [int(v) for v in t.bbox]
        label = names.get(int(t.cls), str(t.cls))
        text = f"{label} #{t.tid} {t.conf:.2f}"
        cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(out, text, (x1, max(0,y1-7)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
    return out

def draw_line(frame: np.ndarray, a: tuple[int,int], b: tuple[int,int]) -> np.ndarray:
    cv2.line(frame, a, b, (0, 0, 255), 1)
    return frame

def draw_counter(frame: np.ndarray, total: int, in_count: int, out_count: int) -> np.ndarray:
    cv2.putText(frame, f"LINE COUNT: {total} | IN: {in_count} | OUT: {out_count}", (10, 80),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,0), 2, cv2.LINE_AA)
    return frame


