"""Shared helpers: decide which classes are violations and draw boxes."""
import cv2

GREEN = (0, 200, 0)
RED = (0, 0, 255)


def is_violation(class_name: str) -> bool:
    """True for 'Without Helmet' / 'no_helmet' style class names."""
    n = class_name.lower().replace("_", " ").replace("-", " ")
    return "without" in n or n.startswith("no ")


def draw_detections(frame, result):
    """Draw YOLO boxes on a BGR frame. Returns (frame, safe_count, violation_count)."""
    safe = violations = 0
    for box in result.boxes:
        name = result.names[int(box.cls)]
        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bad = is_violation(name)
        violations += bad
        safe += not bad
        color = RED if bad else GREEN
        label = f"{name} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame, safe, violations
