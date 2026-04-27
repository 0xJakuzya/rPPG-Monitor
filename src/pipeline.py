from collections import deque
import cv2
import numpy as np
from src import config
from src.face_detector import FaceDetector
from src.video_capture import VideoCapture
from src.visualization import draw_roi, roi_to_mask
from src.processing import extract_mean_rgb, estimate_hr
from src.utils import make_algorithm


def run_pipeline():
    cap = VideoCapture()
    cap.start()
    detector = FaceDetector()
    fps = float(config.FPS_TARGET)
    algo = make_algorithm(fps)
    rgb_buf = deque(maxlen=int(config.BUFFER_SEC * fps))
    bvp_signal = np.array([], dtype=np.float32)
    hr = None
    hr_ema = None
    hr_history = deque(maxlen=15)
    while True:
        frame = cap.read()
        if frame is None:
            continue
        display = frame.copy()
        landmarks = detector.get_landmarks(frame)

        if landmarks is not None:
            left_cheek, right_cheek, forehead = detector.get_roi(frame, landmarks)
            detector.draw_landmarks(display, landmarks)

            display = draw_roi(display, roi_to_mask(forehead), (255, 200, 0))
            display = draw_roi(display, roi_to_mask(left_cheek), (0, 180, 255))
            display = draw_roi(display, roi_to_mask(right_cheek), (0, 180, 255))

            rois_weights = [(forehead, 0.5), (left_cheek, 0.25), (right_cheek, 0.25)]
            weighted, total_w = np.zeros(3), 0.0
            for roi, w in rois_weights:
                sample = extract_mean_rgb(frame, roi)
                if sample is not None and sample.sum() > 0:
                    weighted += sample * w
                    total_w += w
            rgb_val = weighted / total_w if total_w > 0 else None
            if rgb_val is not None:
                rgb_buf.append([rgb_val[2], rgb_val[1], rgb_val[0]])
            if len(rgb_buf) >= int(fps * 2):
                bvp_signal = algo.run(np.array(rgb_buf, dtype=np.float32))
                result = estimate_hr(bvp_signal, fps)
                if result is not None:
                    hr_raw = result
                    hr_ema = hr_raw if hr_ema is None else 0.15 * hr_raw + 0.85 * hr_ema
                    hr_history.append(hr_ema)
                hr = round(float(np.median(hr_history))) if hr_history else None
            else:
                hr = None
            status, color_status = "DETECTED", (0, 255, 0)
        else:
            rgb_buf.clear()
            bvp_signal = np.array([], dtype=np.float32)
            hr = None
            hr_ema = None
            hr_history.clear()
            status, color_status = "NO FACE", (0, 0, 255)

        hr_text = f"HR: {hr} BPM" if hr is not None else "HR: --"
        cv2.putText(display, hr_text, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(display, f"[{config.RPPG_METHOD}] {status}",
                    (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE,
                    color_status, config.FONT_THICKNESS)

        cv2.imshow(config.WINDOW_NAME, display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.stop()
    detector.close()
    cv2.destroyAllWindows()
