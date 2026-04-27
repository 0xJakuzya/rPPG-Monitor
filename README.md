# rPPG-Detection

Real-time remote photoplethysmography (rPPG) — heart rate measurement from a regular webcam by analyzing subtle color changes in facial skin caused by blood flow, without any contact sensors.

![Demo](assets/me.png)

## How it works

1. **Face detection** — MediaPipe FaceLandmarker detects 478 facial landmarks per frame
2. **ROI extraction** — three regions are masked: forehead, left cheek, right cheek
3. **Signal extraction** — mean RGB is averaged across all three ROIs per frame, buffered over a 10-second sliding window
4. **Detrending** — slow baseline drift removed  sparse matrix method
5. **Bandpass filtering** — Chebyshev Type II filter 
6. **HR estimation** — FFT peak detection in the 40–180 BPM band

## Stack

| Component | Technology |
|---|---|
| Face detection | [MediaPipe FaceLandmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) |
| Video capture | OpenCV |
| Signal filtering | Detrend  + Chebyshev Type II bandpass |
| Signal processing | NumPy, SciPy (sparse matrix, FFT) |

## Project structure

```
rPPG-Detection/
├── main.py                  # Entry point
├── src/
│   ├── pipeline.py          # Main real-time loop
│   ├── face_detector.py     # MediaPipe wrapper + ROI masking
│   ├── video_capture.py     # Threaded camera capture
│   ├── processing.py        # Detrend, Chebyshev bandpass, HR estimation
│   ├── visualization.py     # BVP plot, ROI overlay
│   └── config.py            # All parameters (camera, filters, ROI indices)
├── models/
│   ├── pos.py               # POS algorithm 
│   └── chrom.py             # CHROM algorithm 
└── assets/
    └── me.png
```

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

## References

- Tarvainen, M. et al. (2002). *An advanced detrending method with application to HRV analysis*. IEEE TBME.
- Wang, W. et al. (2017). *Algorithmic Principles of Remote PPG*. IEEE TBME.
- De Haan, G. & Jeanne, V. (2013). *Robust Pulse Rate From Chrominance-Based rPPG*. IEEE TBME.
- Egorov, K. et al. (2025). *Gaze into the Heart: A Multi-View Video Dataset for rPPG*. ACM MM '25.
