import math
import cv2 as cv
import time
from collections import deque

tracker = cv.TrackerMIL_create()

cap = cv.VideoCapture(0)
success, frame = cap.read()
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

border_left = 220
border_right = 420
border_top = 140
border_bottom = 340

trajectory_x = deque([0] * 2, maxlen=2)
trajectory_y = deque([0] * 2, maxlen=2)
alpha = 0.325
speed_x = speed_y = speed = 0
last_time = time.time()

bbox = cv.selectROI("Tracking", frame, False)
tracker.init(frame, bbox)

while True:
    current_time = time.time()
    dt = current_time - last_time
    last_time = current_time

    success, frame = cap.read()
    if not success: break

    ok, bbox = tracker.update(frame)

    if ok:
        x, y, w, h = map(int, bbox)
        cx, cy = x + w // 2, y + h // 2

        trajectory_x.append(cx)
        trajectory_y.append(cy)

        point_x = trajectory_x[-1] - trajectory_x[0]
        point_y = trajectory_y[-1] - trajectory_y[0]

        is_inside = (border_left <= cx <= border_right and border_top <= cy <= border_bottom)
        rect_color = (0, 255, 0) if is_inside else (0, 0, 255)

        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv.line(frame, (cx, cy), (cx + point_x * 5, cy + point_y * 5), (0, 255, 0), 2)

        dist_x = 0 if is_inside else (border_left - cx if cx < border_left else cx - border_right)
        dist_y = 0 if is_inside else (border_top - cy if cy < border_top else cy - border_bottom)

        cv.putText(frame, f"Dist X: {dist_x}", (20, 50), 1, 1.5, rect_color, 2)
        cv.putText(frame, f"Dist Y: {dist_y}", (20, 80), 1, 1.5, rect_color, 2)
    else:
        cv.putText(frame, "LOST", (width // 2 - 50, height // 2), 1, 3, (0, 0, 255), 3)
        rect_color = (0, 0, 255)

    cv.rectangle(frame, (border_left, border_top), (border_right, border_bottom), rect_color, 2)

    cv.imshow("Tracking", frame)

    if cv.waitKey(1) & 0xFF == 27: break

cap.release()
cv.destroyAllWindows()