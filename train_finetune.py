# -*- coding: utf-8 -*-
"""Дообучение (fine-tuning) YOLO на датасете, собранном в симуляции и с веб-камеры.

Запуск:
    python train_finetune.py                          # yolov8n, 30 эпох
    python train_finetune.py --model yolov8s.pt --epochs 50
Результат:
    runs/detect/<name>/weights/best.pt — скопируйте в проект и подключайте:
    python -c "from ultralytics import YOLO; YOLO('best.pt').predict('frame.jpg', save=True)"
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402


def ensure_data_yaml(dataset_dir):
    path = os.path.join(dataset_dir, "data.yaml")
    if not os.path.exists(path):
        names = "\n".join(f"  {i}: {n}" for i, n in enumerate(config.CLASSES))
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"path: {os.path.abspath(dataset_dir)}\n"
                    f"train: images/train\nval: images/train\nnames:\n{names}\n")
        print(f"[data] создан {path}")
    return path


def main():
    ap = argparse.ArgumentParser(description="Дообучение YOLO на датасете трамвайчика")
    ap.add_argument("--data", default=config.DATASET_DIR, help="каталог датасета")
    ap.add_argument("--model", default=config.YOLO_WEIGHTS["detect"],
                    help="базовые веса (yolov8n.pt и т.п.)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="tramchik_finetune")
    args = ap.parse_args()

    n_img = 0
    img_dir = os.path.join(args.data, "images", "train")
    if os.path.isdir(img_dir):
        n_img = len([f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))])
    print(f"[data] кадров в датасете: {n_img} ({img_dir})")
    if n_img == 0:
        print("[!] Датасет пуст. Соберите кадры: запустите main.py, откройте веб-камеру (V) "
              "и включите запись (O), либо записывайте бортовую камеру симуляции (O).")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[!] Нет пакета ultralytics: pip install ultralytics")
        sys.exit(1)

    data_yaml = ensure_data_yaml(args.data)
    print(f"[train] база: {args.model} | эпох: {args.epochs} | imgsz: {args.imgsz}")
    model = YOLO(args.model)
    model.train(data=data_yaml, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, name=args.name)
    best = os.path.join("runs", "detect", args.name, "weights", "best.pt")
    print(f"[done] Лучшая модель: {best}")
    print("       Используйте её в своих скриптах: YOLO(best).predict(frame)")


if __name__ == "__main__":
    main()
