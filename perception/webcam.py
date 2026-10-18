# -*- coding: utf-8 -*-
"""Веб-камера ноутбука: захват в фоне, реальный инференс YOLO (если доступен),
запись датасета (кадры + псевдоразметка YOLO-формата) для дообучения моделей.

Всё устойчиво к отсутствию cv2/ultralytics/камеры — панель покажет причину.
"""
import os
import threading
import time

import config


class WebcamSource:
    """Фоновый потоковый захват OpenCV VideoCapture."""

    def __init__(self, index=None):
        index = config.WEBCAM_INDEX if index is None else index
        self.err = None
        self.ok = False
        self._lock = threading.Lock()
        self._stop = False
        self._frame = None
        self.cap = None
        try:
            import cv2
        except ImportError:
            self.err = "нет пакета opencv-python (pip install opencv-python)"
            return
        self.cv2 = cv2
        try:
            cap = cv2.VideoCapture(index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.WEBCAM_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.WEBCAM_H)
            if not cap.isOpened():
                self.err = f"камера {index} не открылась"
                cap.release()
                return
            self.cap = cap
        except Exception as e:  # noqa: BLE001
            self.err = f"ошибка камеры: {e}"
            return
        self.ok = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop:
            try:
                ret, frame = self.cap.read()
            except Exception:  # noqa: BLE001
                ret, frame = False, None
            with self._lock:
                self._frame = frame if ret else None
            time.sleep(0.004)

    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def close(self):
        self._stop = True
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:  # noqa: BLE001
                pass
        self.ok = False


class RealYOLO:
    """Ленивая загрузка ultralytics + детекция на кадре веб-камеры."""

    def __init__(self):
        self.available = False
        self.err = None
        self.model = None
        self.loading = False

    def ensure_loaded(self):
        if self.available or self.err or self.loading:
            return
        self.loading = True
        try:
            from ultralytics import YOLO
            self.model = YOLO(config.YOLO_WEIGHTS["detect"])
            self.available = True
        except ImportError:
            self.err = "нет пакета ultralytics (pip install ultralytics)"
        except Exception as e:  # noqa: BLE001
            self.err = f"модель недоступна: {e}"
        self.loading = False

    def infer(self, frame_bgr):
        """→ (кадр с разметкой BGR, строки YOLO-псевдоразметки) | (None, [])."""
        if not self.available:
            return None, []
        try:
            res = self.model.predict(frame_bgr, verbose=False)[0]
        except Exception as e:  # noqa: BLE001
            self.err = f"ошибка инференса: {e}"
            return None, []
        ann = res.plot()
        lines = []
        names = res.names if hasattr(res, "names") else {}
        h, w = frame_bgr.shape[:2]
        boxes = getattr(res, "boxes", None)
        if boxes is not None:
            for b in boxes:
                try:
                    cx, cy, bw, bh = b.xywhn[0].tolist()
                    cid = int(b.cls[0])
                    lines.append(f"{cid} {cx:.5f} {cy:.5f} {bw:.5f} {bh:.5f}")
                except Exception:  # noqa: BLE001
                    continue
        return ann, lines


class DatasetRecorder:
    """Запись датасета: images/train + labels/train (YOLO-формат) + data.yaml."""

    def __init__(self):
        self.active = False
        self.count = 0
        self._timer = 0.0
        self.img_dir = os.path.join(config.DATASET_DIR, "images", "train")
        self.lbl_dir = os.path.join(config.DATASET_DIR, "labels", "train")
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.lbl_dir, exist_ok=True)

    def _paths(self, name):
        return (os.path.join(self.img_dir, name + ".jpg"),
                os.path.join(self.lbl_dir, name + ".txt"))

    def save_pygame_surface(self, surface, yolo_lines, tag="sim"):
        """Кадр из симуляции (бортовая камера) + псевдоразметка."""
        import pygame
        self.count += 1
        name = f"{tag}_{self.count:06d}"
        img_path, lbl_path = self._paths(name)
        try:
            pygame.image.save(surface, img_path)
        except Exception:  # noqa: BLE001
            self.count -= 1
            return None
        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))
        return img_path

    def save_cv2_frame(self, frame_bgr, yolo_lines):
        """Кадр веб-камеры + псевдоразметка от реальной модели."""
        self.count += 1
        name = f"cam_{self.count:06d}"
        img_path, lbl_path = self._paths(name)
        try:
            import cv2
            cv2.imwrite(img_path, frame_bgr)
        except Exception:  # noqa: BLE001
            self.count -= 1
            return None
        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))
        return img_path

    def ensure_data_yaml(self):
        path = os.path.join(config.DATASET_DIR, "data.yaml")
        if not os.path.exists(path):
            names = "\n".join(f"  {i}: {n}" for i, n in enumerate(config.CLASSES))
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"path: {config.DATASET_DIR}\n"
                        f"train: images/train\nval: images/train\nnames:\n{names}\n")
        return path
