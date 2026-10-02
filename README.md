# Helmet Detection for Two-Wheeler Riders (YOLOv8)

Real-time deep learning system that detects motorcycle riders **with** and **without**
helmets from a webcam, image, or video, and saves snapshots of violations.

**Tech:** Python, YOLOv8n (Ultralytics), OpenCV, Streamlit
**Dataset:** Kaggle "Helmet Detection" (andrewmvd/helmet-detection), about 760 images

## Folder contents

| File | What it does |
|---|---|
| `setup_and_train.py` | Downloads dataset, converts VOC XML labels to YOLO format, trains the model |
| `detect_webcam.py` | Real-time detection from webcam/video with FPS and violation counter |
| `app.py` | Streamlit web demo (image, camera snapshot, video) |
| `helmet_utils.py` | Shared drawing and violation logic |

## Step 1: Train on Google Colab (free GPU, about 30 min)

1. Zip this folder and open colab.research.google.com → **New notebook**.
2. **Runtime → Change runtime type → T4 GPU → Save**.
3. Click the folder icon on the left and upload the zip and your `kaggle.json`.
4. Run these cells:

```
!unzip -q helmet-detection.zip -d .
%cd helmet-detection
!cp ../kaggle.json .
!pip install -q ultralytics kaggle
!python setup_and_train.py
```

```
from google.colab import files
files.download("models/helmet_best.pt")
!zip -qr results.zip runs/helmet
files.download("results.zip")
```

`results.zip` contains the graphs for your report: `results.png` (loss and mAP curves),
`confusion_matrix.png`, `PR_curve.png`, and sample predictions `val_batch0_pred.jpg`.

## Step 2: Run on your laptop (Windows)

Put `helmet_best.pt` into a `models` folder inside this project, then in Command Prompt:

```
cd path\to\helmet-detection
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python detect_webcam.py --save-violations
streamlit run app.py
```

YOLOv8n runs at roughly 8–20 FPS on a normal laptop CPU, which is enough for a live demo.

## 5-day plan

| Day | Task |
|---|---|
| 1 | Set up, read about YOLO, run training on Colab |
| 2 | Test webcam and Streamlit demo, collect your own test images/videos |
| 3 | Tune confidence, try more epochs (`EPOCHS=80 python setup_and_train.py`), note metrics |
| 4 | Write report, make slides with graphs from `results.zip` |
| 5 | Rehearse demo and viva questions |

## Report outline

1. Introduction: road accidents and helmet-law enforcement in India
2. Literature review: R-CNN vs SSD vs YOLO
3. Dataset: source, classes, train/val split, annotation conversion
4. Methodology: YOLOv8 architecture (backbone, neck, head), transfer learning from COCO
5. Implementation: training settings (640px, 50 epochs, batch 16), tools
6. Results: precision, recall, mAP@0.5, mAP@0.5:0.95, confusion matrix, sample outputs
7. Limitations and future work: night/rain images, occlusion, combine with number plate recognition to auto-issue challans

## Likely viva questions

- **Why YOLO?** One-stage detector: predicts boxes and classes in a single pass, so it's fast enough for real time.
- **What is mAP?** Mean Average Precision, the area under the precision-recall curve averaged over classes; mAP@0.5 counts a detection as correct if IoU ≥ 0.5.
- **What is IoU?** Overlap area divided by union area of predicted and true boxes.
- **What is transfer learning here?** We start from weights pretrained on COCO (80 classes) and fine-tune on helmet data, so we need far fewer images.
- **What is NMS?** Non-Maximum Suppression removes duplicate overlapping boxes for the same object.
- **Why YOLOv8n?** The "nano" model has about 3M parameters, so it runs on a CPU; bigger variants are more accurate but slower.
