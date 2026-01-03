from __future__ import annotations
import argparse
from pathlib import Path
import cv2
from tqdm import tqdm

from src.pipeline import VideoAnalyticsPipeline
from src.overlay import draw_tracks, draw_hud, draw_line, draw_counter
from src.line_counter import LineCounter, LineCounterConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, required=True, help="Video path or webcam index (e.g., 0)")
    p.add_argument("--model", type=str, default="yolov8n.pt")
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--classes", type=int, nargs="*", default=None, help="Filter classes by id (COCO)")
    p.add_argument("--save_video", action="store_true")
    p.add_argument("--save_json", action="store_true")
    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--show", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # source: webcam index or path
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    fps_in = fps_in if fps_in and fps_in > 0 else 30.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    pipeline = VideoAnalyticsPipeline(
        model=args.model,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        classes=args.classes,
    )
    # Line for 960x540 traffic video: tweak if needed
    line_cfg = LineCounterConfig(a=(100, 380), b=(860, 380), min_hits=2, cooldown_frames=30)
    line_counter = LineCounter(line_cfg)

    writer = None
    out_mp4 = out_dir / "out.mp4"
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_mp4), fourcc, fps_in, (w, h))

    json_rows = []
    pbar = tqdm(total=nframes if nframes > 0 else None, desc="Processing")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        tracks, counts, lat_ms, fps = pipeline.infer(frame)
        # update line counter
        prev_total = line_counter.count_total
        line_counter.update(frame_idx, tracks)

        # increment per-label count only when total changed
        if line_counter.count_total > prev_total:
          # use the label of the track(s) that crossed is not tracked here; keep it simple for MVP
          pass

        vis = draw_tracks(frame, tracks, pipeline.names if isinstance(pipeline.names, dict) else {})
        vis = draw_hud(vis, fps=fps, lat_ms=lat_ms, counts=counts)
        vis = draw_line(vis, line_cfg.a, line_cfg.b)
        in_count = line_counter.count_by_label.get("IN", 0)
        out_count = line_counter.count_by_label.get("OUT", 0)
        vis = draw_counter(vis, line_counter.count_total, in_count, out_count)


        if args.save_video and writer is not None:
            writer.write(vis)

        if args.save_json:
            json_rows.append(pipeline.track_to_dict(frame_idx, tracks))

        if args.show:
            cv2.imshow("video-analytics", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    if args.save_json:
        out_json = out_dir / "out.json"
        pipeline.save_json(json_rows, out_json)
        print(f"[OK] Saved JSON: {out_json}")

    if args.save_video:
        print(f"[OK] Saved video: {out_mp4}")

if __name__ == "__main__":
    main()

