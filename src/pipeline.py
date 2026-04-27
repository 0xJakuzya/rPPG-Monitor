from collections import deque
import cv2
import numpy as np
from src import config
from src.face_detector import FaceDetector
from src.video_capture import VideoCapture
from src.visualization import bvp_plot, draw_roi, roi_to_mask
from src.processing import extract_mean_rgb, estimate_hr, process_bvp

def run_pipeline():
    cap = VideoCapture()
    cap.start()
    detector = FaceDetector()
    fps = float(config.FPS_TARGET)
    rgb_buf = deque(maxlen=int(config.BUFFER_SEC * fps))
    bvp_signal = np.array([], dtype=np.float32)
    hr = None
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

            samples = [extract_mean_rgb(frame, roi) for roi in (forehead, left_cheek, right_cheek)]
            samples = [sample for sample in samples if sample is not None]
            rgb_val = np.mean(samples, axis=0) if samples else None

            if rgb_val is not None:
                rgb_buf.append([rgb_val[2], rgb_val[1], rgb_val[0]])

            if len(rgb_buf) >= int(fps * 2):
                bvp_signal = process_bvp(np.array(rgb_buf, dtype=np.float32), fps)
                hr = estimate_hr(bvp_signal, fps)
            else:
                hr = None

            status, color_status = "DETECTED", (0, 255, 0)
        else:
            hr = None
            status, color_status = "NO FACE", (0, 0, 255)

        cv2.putText(
            display,
            f"[{config.RPPG_METHOD}] {status}",
            (10, display.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.FONT_SCALE,
            color_status,
            config.FONT_THICKNESS,
        )

        plot = bvp_plot(bvp_signal, display.shape[1], config.PLOT_H, hr)
        cv2.imshow(config.WINDOW_NAME, np.vstack([display, plot]))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.stop()
    detector.close()
    cv2.destroyAllWindows()
