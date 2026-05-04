# rPPG Heart Rate Estimation

This project uses remote photoplethysmography (rPPG) to estimate heart rate from
face video without physical contact.

Instead of body sensors, it uses a normal camera. The camera records small skin
color changes caused by blood flow. These changes are converted into a
one-dimensional physiological signal, and heart rate is estimated from it.

## PhysNet Architecture

![PhysNet architecture](assets/physnet_architecture.png)

Fig. 1. PhysNet architecture with ROI patches.

## Features

- Extracts multi-ROI face patches with MediaPipe.
- Prepares `.npz` windows for training.
- Uses subject-level train/validation split.
- Trains `baseline` CNN and `physnet` models.
- Supports `negpearson` and `cnn` loss functions.
- Evaluates HR with MAE, RMSE, scatter plot, and Bland-Altman plot.
- Runs realtime POS/CHROM baseline with a webcam.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Structure

```text
rPPG-Detection/
  assets/
  data/
  models/
    baseline.py
    chrom.py
    loss.py
    physnet.py
    pos.py
  src/
    config.py
    dataset.py
    face_detector.py
    preprocessing.py
    test.py
    train.py
    utils.py
    video.py
    visualization.py
  main.py
  requirements.txt
  README.md
```
