import cv2
import numpy as np
import time
import threading
from ultralytics import YOLO
from dataclasses import dataclass
from collections import deque
from typing import List, Tuple


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    confidence: float
    center: Tuple[int, int]
    is_predicted: bool = False


class CameraStream:
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
            else:
                self.stopped = True

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()


class TrackedObject:
    def __init__(self, detection: Detection, object_id: int):
        self.id = object_id
        self.color = [(0, 255, 0), (0, 255, 255), (255, 0, 0), (255, 0, 255)][object_id % 4]
        self.history = deque(maxlen=15)
        self.missed_frames = 0

        self.smooth_bbox = np.array(detection.bbox, dtype=float)

        self.last_det = detection

    def update(self, det: Detection):
        alpha = 0.15
        deadzone = 2.0

        # 1. Превращаем входные координаты в массив NumPy для расчетов
        new_bbox_arr = np.array(det.bbox, dtype=float)

        # 2. Убеждаемся, что self.smooth_bbox тоже массив (на случай, если это не так)
        if not isinstance(self.smooth_bbox, np.ndarray):
            self.smooth_bbox = np.array(self.smooth_bbox, dtype=float)

        # 3. Считаем разницу
        diff = np.abs(new_bbox_arr - self.smooth_bbox).max()

        if diff > deadzone:
            # Теперь эта строка будет работать без ошибок
            self.smooth_bbox = alpha * new_bbox_arr + (1 - alpha) * self.smooth_bbox

        # 4. Превращаем обратно в целые числа для отрисовки
        final_bbox = tuple(map(int, self.smooth_bbox))

        # 5. Обновляем данные в объекте
        self.last_det = Detection(
            bbox=final_bbox,
            confidence=det.confidence,
            center=((final_bbox[0] + final_bbox[2]) // 2, (final_bbox[1] + final_bbox[3]) // 2),
            is_predicted=det.is_predicted
        )

        if not det.is_predicted:
            self.history.append(self.last_det.center)
            self.missed_frames = 0

    def predict(self):
        self.missed_frames += 1
        if len(self.history) < 2: return self.last_det

        h = list(self.history)
        dx = (h[-1][0] - h[0][0]) // len(h)
        dy = (h[-1][1] - h[0][1]) // len(h)

        # Экстраполяция координат при потере объекта
        p_bbox = (self.smooth_bbox[0] + dx, self.smooth_bbox[1] + dy,
                  self.smooth_bbox[2] + dx, self.smooth_bbox[3] + dy)

        return Detection(tuple(map(int, p_bbox)), self.last_det.confidence * 0.9,
                         (int(p_bbox[0] + p_bbox[2]) // 2, int(p_bbox[1] + p_bbox[3]) // 2), True)


class UniversalTracker:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.stream = CameraStream().start()
        self.tracked_objs = {}
        self.next_id = 0
        self.click_point = None
        self.conf_threshold = 0.3
        self.show_ui = True
        self.fps = 0

    def _iou(self, b1, b2):
        xA, yA, xB, yB = max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        return inter / float((b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) - inter + 1e-6)

    def run(self):
        cv2.namedWindow("Tracker")
        cv2.setMouseCallback("Tracker", lambda e, x, y, f, p: setattr(self, 'click_point',
                                                                      (x, y)) if e == cv2.EVENT_LBUTTONDOWN else None)

        f_idx = 0
        while True:
            t_start = time.time()
            frame = self.stream.read()
            if frame is None: break
            f_idx += 1

            # Детекция (на Raspberry Pi можно поставить f_idx % 2 == 0 для ускорения)
            detections = []
            results = self.model(frame, conf=self.conf_threshold, imgsz=320, verbose=False, iou=0.3, agnostic_nms=True)
            for r in results[0].boxes:
                if int(r.cls[0]) == 0:
                    b = r.xyxy[0].cpu().numpy().astype(int)
                    detections.append(Detection(tuple(b), float(r.conf[0]), ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)))

            # Выбор объекта кликом
            if self.click_point:
                for det in detections:
                    if det.bbox[0] < self.click_point[0] < det.bbox[2] and det.bbox[1] < self.click_point[1] < det.bbox[
                        3]:
                        # Если уже следим - удаляем, иначе - добавляем
                        exists = [i for i, o in self.tracked_objs.items() if self._iou(o.last_det.bbox, det.bbox) > 0.5]
                        if exists:
                            for i in exists: del self.tracked_objs[i]
                        else:
                            self.tracked_objs[self.next_id] = TrackedObject(det, self.next_id)
                            self.next_id += 1
                self.click_point = None

            # Обновление трекинга
            for obj_id, obj in list(self.tracked_objs.items()):
                match = next((d for d in detections if self._iou(obj.last_det.bbox, d.bbox) > 0.4), None)
                obj.update(match if match else obj.predict())
                if obj.missed_frames > 15: del self.tracked_objs[obj_id]

            # Отрисовка
            if self.show_ui:
                # 1. Сначала определяем список боксов, которые уже "заняты" трекером
                occupied_bboxes = [obj.last_det.bbox for obj in self.tracked_objs.values()]

                # 2. Рисуем ТОЛЬКО ТЕ детекции, которые НЕ совпадают с отслеживаемыми
                for det in detections:
                    is_duplicate = False
                    for occupied in occupied_bboxes:
                        # Если сырая рамка совпадает с треком более чем на 50%, считаем её дублем
                        if self._iou(det.bbox, occupied) > 0.5:
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        # Рисуем только свободные объекты тусклым цветом
                        x1, y1, x2, y2 = det.bbox
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 150, 150), 1)

                # 3. Рисуем выбранные объекты (наши основные плавные рамки)
                for obj in self.tracked_objs.values():
                    b = obj.last_det.bbox
                    color = obj.color

                    # 1. Рисуем основную рамку
                    cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), color, 2)
                    cv2.putText(frame, f"ID {obj.id}", (b[0], b[1] - 10), 0, 0.6, color, 2)

                    # 2. Логика отрисовки вектора (исправление NameError)
                    if len(obj.history) >= 5:
                        c = obj.last_det.center  # Текущий центр (определяем 'c')
                        pc = obj.history[0]  # Центр 5 кадров назад (определяем 'pc')

                        # Рассчитываем конечную точку вектора
                        end_point = (c[0] + (c[0] - pc[0]) * 3,
                                     c[1] + (c[1] - pc[1]) * 3)

                        cv2.arrowedLine(frame, c, end_point, color, 2, tipLength=0.3)

                # Тонкие рамки для всех людей в кадре (невыбранных)
                for det in detections:
                    if not any(self._iou(det.bbox, o.last_det.bbox) > 0.6 for o in self.tracked_objs.values()):
                        cv2.rectangle(frame, (det.bbox[0], det.bbox[1]), (det.bbox[2], det.bbox[3]), (0, 150, 150), 1)

                cv2.putText(frame, f"FPS: {self.fps:.1f} | Conf: {self.conf_threshold:.2f}", (10, 30), 0, 0.7,
                            (0, 255, 0), 2)

            cv2.imshow("Tracker", frame)

            # Управление клавишами
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                self.show_ui = True
            elif key == ord('2'):
                self.show_ui = False
            elif key == ord('+') or key == ord('='):
                self.conf_threshold = min(0.95, self.conf_threshold + 0.05)
            elif key == ord('-'):
                self.conf_threshold = max(0.05, self.conf_threshold - 0.05)
            elif key == ord('c'):
                self.tracked_objs.clear()

            self.fps = 1.0 / (time.time() - t_start)

        self.stream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Для Windows: r'C:\path\to\model.pt'
    # Для Raspberry: 'best_openvino' (папка после экспорта)
    MODEL_PATH = r'C:\dev\materials\opencv\training_scripts\runs\detect\final_model\weights\best.pt'
    tracker = UniversalTracker(MODEL_PATH)
    tracker.run()