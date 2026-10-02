"""
Helmet Detection - download dataset, convert to YOLO format, train YOLOv8n.

Run this on Google Colab with a GPU (see README). Needs kaggle.json in this
folder or in ~/.kaggle/. Set EPOCHS env variable to change epochs (default 50).
"""
import os
import random
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

DATASET = "andrewmvd/helmet-detection"   # Kaggle: PASCAL VOC annotations
RAW = Path("data/raw")
OUT = Path("data/helmet")
EPOCHS = int(os.environ.get("EPOCHS", 50))
IMG_EXTS = {".png", ".jpg", ".jpeg"}


def setup_kaggle():
    kdir = Path.home() / ".kaggle"
    kdir.mkdir(exist_ok=True)
    target = kdir / "kaggle.json"
    if not target.exists() and Path("kaggle.json").exists():
        shutil.copy("kaggle.json", target)
    if not target.exists():
        raise SystemExit("kaggle.json not found. Put it in this folder and run again.")
    os.chmod(target, 0o600)


def download():
    if RAW.exists() and any(RAW.rglob("*.xml")):
        print("Dataset already downloaded.")
        return
    RAW.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET} ...")
    subprocess.run(["kaggle", "datasets", "download", "-d", DATASET,
                    "-p", str(RAW), "--unzip"], check=True)


def convert():
    """PASCAL VOC XML -> YOLO txt, with an 80/20 train/val split."""
    xmls = sorted(RAW.rglob("*.xml"))
    images = {p.stem: p for p in RAW.rglob("*") if p.suffix.lower() in IMG_EXTS}
    classes = sorted({o.findtext("name").strip()
                      for x in xmls for o in ET.parse(x).findall("object")})
    print("Classes found:", classes)

    pairs = []
    for x in xmls:
        root = ET.parse(x).getroot()
        img = images.get(Path(root.findtext("filename", "")).stem) or images.get(x.stem)
        if img is None:
            continue
        w = float(root.findtext("size/width") or 0)
        h = float(root.findtext("size/height") or 0)
        if w <= 0 or h <= 0:
            w, h = Image.open(img).size
        lines = []
        for o in root.findall("object"):
            c = classes.index(o.findtext("name").strip())
            b = o.find("bndbox")
            x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
            y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            lines.append(f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                         f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        pairs.append((img, lines))

    random.seed(42)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * 0.2))
    if OUT.exists():
        shutil.rmtree(OUT)
    for split, items in (("val", pairs[:n_val]), ("train", pairs[n_val:])):
        (OUT / "images" / split).mkdir(parents=True)
        (OUT / "labels" / split).mkdir(parents=True)
        for img, lines in items:
            shutil.copy(img, OUT / "images" / split / img.name)
            (OUT / "labels" / split / f"{img.stem}.txt").write_text("\n".join(lines))
    print(f"Train: {len(pairs) - n_val} images, Val: {n_val} images")

    yaml = OUT / "data.yaml"
    names = "\n".join(f"  {i}: '{n}'" for i, n in enumerate(classes))
    yaml.write_text(f"path: {OUT.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    return yaml


def train(yaml):
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")   # small + fast, runs in real time on a laptop CPU
    model.train(data=str(yaml), epochs=EPOCHS, imgsz=640, batch=16,
                project="runs", name="helmet", exist_ok=True)
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    Path("models").mkdir(exist_ok=True)
    shutil.copy(best, "models/helmet_best.pt")
    print("\nSaved trained model to models/helmet_best.pt")
    print(f"Training graphs and metrics are in {model.trainer.save_dir}")


if __name__ == "__main__":
    setup_kaggle()
    download()
    train(convert())
