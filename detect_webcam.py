"""
Real-time helmet detection from a webcam or video file.

  python detect_webcam.py                    # default webcam
  python detect_webcam.py --source video.mp4
  python detect_webcam.py --save-violations  # saves snapshots of riders without helmets
Press Q to quit.
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

from helmet_utils import draw_detections

p = argparse.ArgumentParser()
p.add_argument("--weights", default="models/helmet_best.pt")
p.add_argument("--source", default="0", help="webcam index (0, 1, ...) or video path")
p.add_argument("--conf", type=float, default=0.4)
p.add_argument("--save-violations", action="store_true")
args = p.parse_args()

if not Path(args.weights).exists():
    raise SystemExit(f"Model not found: {args.weights}. Train it first (see README).")

model = YOLO(args.weights)
cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
if not cap.isOpened():
    raise SystemExit(f"Could not open source {args.source}")

out_dir = Path("violations")
out_dir.mkdir(exist_ok=True)
last_save = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    start = time.time()
    result = model(frame, conf=args.conf, verbose=False)[0]
    frame, safe, viol = draw_detections(frame, result)
    fps = 1 / max(time.time() - start, 1e-6)

    cv2.rectangle(frame, (0, 0), (330, 34), (30, 30, 30), -1)
    cv2.putText(frame, f"FPS {fps:4.1f} | Helmet {safe} | No helmet {viol}", (8, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    if args.save_violations and viol and time.time() - last_save > 2:
        name = out_dir / f"violation_{datetime.now():%Y%m%d_%H%M%S}.jpg"
        cv2.imwrite(str(name), frame)
        last_save = time.time()
        print("Saved", name)

    cv2.imshow("Helmet Detection (Q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
