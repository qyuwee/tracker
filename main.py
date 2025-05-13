import math
import cv2 as cv
from collections import deque
import time

tracker = cv.TrackerCSRT.create()

# frame_count = 0
# start_time = time.time()
fps = 11  # ВВЕДИТЕ СЮДА ПРИМЕРНОЕ ЗНАЧЕНИЕ ВАШЕГО ФПС. КАК ЕГО УЗНАТЬ ЧИТАЙТЕ В README
trajectory_x = deque([0] * 2, maxlen=2)
trajectory_y = deque([0] * 2, maxlen=2)
alpha = 0.325  # коэффицент сглаживания рассчёта скорости движения объекта. Чем он >, тем быстрее меняется скорость
speed, speed_x, speed_y = 0, 0, 0
dist_x, dist_y = 0, 0
border_left = 220
border_right = 420
border_top = 140
border_bottom = 340

cap = cv.VideoCapture(0)
_, frame = cap.read()

width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))


def drawRectangle(frame, bbox):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    trajectory_x.append(p1[0] + (p2[0] - p1[0]) // 2)
    trajectory_y.append(p1[1] + (p2[1] - p1[1]) // 2)
    cv.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)


def calculateSpeed(frame, trajectory_x, trajectory_y):
    global speed_x, speed_y, speed

    a = abs(trajectory_x[1] - trajectory_x[0]) * fps  # Текущая скорость X
    b = abs(trajectory_y[1] - trajectory_y[0]) * fps  # Текущая скорость Y
    c = int(math.sqrt(a ** 2 + b ** 2))  # Текущая общая скорость

    # EMA (Экспоненциальное скользящее среднее), вместо очередей (так называемая оптимизация)
    speed_x = int(alpha * a + (1 - alpha) * speed_x)
    speed_y = int(alpha * b + (1 - alpha) * speed_y)
    speed = int(alpha * c + (1 - alpha) * speed)

    cv.putText(frame, str(speed) + 'pixel/sec', (width-120, 60),
               1, 1, (0, 255, 0))
    cv.putText(frame, str(speed_x) + 'Vx pixel/sec', (width-120, 80),
               1, 1, (0, 255, 0))
    cv.putText(frame, str(speed_y) + 'Vy pixel/sec', (width-120, 100),
               1, 1, (0, 255, 0))


def check_borders(cx, cy):
    left = right = top = bottom = False
    dist_x = dist_y = 0

    if cx < border_left:
        left = True
        dist_x = border_left - cx
    elif cx > border_right:
        right = True
        dist_x = cx - border_right

    if cy < border_top:
        top = True
        dist_y = border_top - cy
    elif cy > border_bottom:
        bottom = True
        dist_y = cy - border_bottom

    return left, right, top, bottom, dist_x, dist_y


bbox = cv.selectROI(frame, False)
p1 = (int(bbox[0]), int(bbox[1]))
p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
cv.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
cv.imshow("Tracking", frame)
cv.waitKey(100)

tracker.init(frame, bbox)

left, top, right, bottom = False, False, False, False

while True:
    # frame_start = time.time()

    ok, bbox = tracker.update(frame)
    _, frame = cap.read()

    # frame_count += 1
    # if frame_count >= 10:
    #     fps = round(frame_count / (time.time() - start_time), 1)
    #     start_time = time.time()
    #     frame_count = 0
    # cv.putText(frame, "FPS : " + str(int(fps)), (80, 100), 1, 1, (0, 255, 0))

    x, y, w, h = map(int, bbox)
    cx, cy = x + w // 2, y + h // 2  # Центр объекта

    if ok:
        drawRectangle(frame, bbox)
        calculateSpeed(frame, trajectory_x, trajectory_y)

    point_x = trajectory_x[-1] - trajectory_x[0]
    point_y = trajectory_y[-1] - trajectory_y[0]

    cv.line(frame, (70, 70), (70+point_x, 70+point_y), (0, 255, 0), 2)  # Направление движения

    left, right, top, bottom, dist_x, dist_y = check_borders(cx, cy)

    is_inside = (border_left <= cx <= border_right and border_top <= cy <= border_bottom)

    rect_color = (0, 255, 0) if is_inside else (0, 0, 255)
    cv.rectangle(frame, (width // 2 - 100, height // 2 - 100), (width // 2 + 100, height // 2 + 100), rect_color, 2)

    if is_inside:
        dist_x = dist_y = 0

    cv.imshow("Tracking", frame)

    if cv.waitKey(30) & 0xFF == 27:
        break

cv.destroyAllWindows()
