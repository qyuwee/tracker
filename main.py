import math
import cv2 as cv
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

tracker = cv.TrackerCSRT.create()

trajectory_x = deque([0] * 5, maxlen=2)
trajectory_y = deque([0] * 5, maxlen=2)
speed = deque([0] * 10, maxlen=15)
speed_x = deque([0] * 10, maxlen=15)
speed_y = deque([0] * 10, maxlen=15)
# 15 - примерное количество фпс у меня на камере, поэтому для того чтобы считать скорость в секунду
# буду каждые 15 тиков считать. Фпс можно посмотреть спомощью пары функций и таймера
# PS: У меня их примерно 15 оказывается
# пояснение к трекеру: я исользую метод, который автоматически отслеживает выбранный с помощью функции ROI объект
# Он не точен и со времен теряет объект из поля отслеживания, либо границы немного слетают. В идеале писать такой трекер
# самому, но для этого необходимо предворительно знать объект и его специфику
# (например, что нам нужен самый большой объект красного цвета, такое написать самому вполне легко).
# Иначе возможно пользование только таким трекером.

cap = cv.VideoCapture(0)
_, frame = cap.read()

width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))


def drawRectangle(frame, bbox):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    trajectory_x.append(p1[0] + (p2[0] - p1[0]) // 2)
    trajectory_y.append(p1[1] + (p2[1] - p1[1]) // 2)
    # cv.circle(frame, (p1[0] + (p2[0] - p1[0]) // 2, p1[1] + (p2[1] - p1[1]) // 2), 2, (255, 0, 0), 1)
    cv.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)


def calculateSpeed(frame, trajectory_x, trajectory_y):
    a = abs(trajectory_x[1] - trajectory_x[0])
    speed_x.append(a)
    b = abs(trajectory_y[1] - trajectory_y[0])
    speed_y.append(b)
    speed.append(int(math.sqrt(a**2 + b**2)))
    cv.putText(frame, str(sum(speed)) + 'pixel/sec', (int(cap.get(cv.CAP_PROP_FRAME_WIDTH))-120, 60),
               1, 1, (0, 255, 0))
    cv.putText(frame, str(sum(speed_x)) + 'Vx pixel/sec', (int(cap.get(cv.CAP_PROP_FRAME_WIDTH))-120, 80),
               1, 1, (0, 255, 0))
    cv.putText(frame, str(sum(speed_y)) + 'Vy pixel/sec', (int(cap.get(cv.CAP_PROP_FRAME_WIDTH))-120, 100),
               1, 1, (0, 255, 0))


def displayRectangle(frame, bbox):
    plt.figure(figsize=(20, 10))
    frameCopy = frame.copy()
    drawRectangle(frameCopy, bbox)
    frameCopy = cv.cvtColor(frameCopy, cv.COLOR_RGB2BGR)
    plt.imshow(frameCopy)
    plt.axis("off")


bbox = cv.selectROI(frame, False)
displayRectangle(frame, bbox)
tracker.init(frame, bbox)

left, top, right, bottom = False, False, False, False

while True:
    ok, bbox = tracker.update(frame)
    _, frame = cap.read()

    x, y, w, h = map(int, bbox)
    cx, cy = x + w // 2, y + h // 2  # Центр объекта

    if ok:
        drawRectangle(frame, bbox)
        calculateSpeed(frame, trajectory_x, trajectory_y)

    point_x = trajectory_x[-1] - trajectory_x[0]
    point_y = trajectory_y[-1] - trajectory_y[0]

    cv.line(frame, (70, 70), (70+point_x, 70+point_y), (0, 255, 0), 2)  # Направление движения

    if cx < 220:
        left = True
    else:
        left = False
    if cx > 420:
        right = True
    else:
        right = False
    if cy < 140:
        top = True
    else:
        top = False
    if cy > 340:
        bottom = True
    else:
        bottom = False

    if 220 < cx < 420 and 140 < cy < 340:  # Ограничивайющий прямоугольник
        cv.rectangle(frame, (width//2+100, height//2+100), (width//2-100, height//2-100), (0, 255, 0), 2)
    else:
        cv.rectangle(frame, (width//2+100, height//2+100), (width//2-100, height//2-100), (0, 0, 255), 2)

    cv.imshow("Tracking", frame)

    if cv.waitKey(30) & 0xFF == 27:
        break

cv.destroyAllWindows()
