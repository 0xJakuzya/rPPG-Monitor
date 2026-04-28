from collections import deque
import cv2
import numpy as np
import torch
from src import config
from src.face_detector import FaceDetector
from src.video_capture import VideoCapture
from src.visualization import draw_roi, roi_to_mask
from src.processing import extract_rois_rgb, estimate_hr, load_physnet, physnet_bvp
from src.utils import make_algorithm

def run_pipeline():
    cap = VideoCapture()
    cap.start()
    detector = FaceDetector()
    fps = float(config.FPS_TARGET)

    use_physnet = config.RPPG_METHOD.upper() == "PHYSNET"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    physnet_model = load_physnet(device) if use_physnet else None
    algo = None if use_physnet else make_algorithm(fps)
    method_label = config.RPPG_METHOD

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

            rgb_val = extract_rois_rgb(frame, forehead, left_cheek, right_cheek)
            
            if rgb_val is not None:
                rgb_buf.append(rgb_val.tolist())

            if len(rgb_buf) >= int(fps * 2):
                buf_arr = np.array(rgb_buf, dtype=np.float32)
                if use_physnet:
                    bvp_signal = physnet_bvp(physnet_model, buf_arr, device)
                else:
                    bvp_signal = algo.run(buf_arr)
                result = estimate_hr(bvp_signal, fps)
                hr = round(float(result)) if result is not None else None
            else:
                hr = None
            status, color_status = "DETECTED", (0, 255, 0)
        else:
            rgb_buf.clear()
            bvp_signal = np.array([], dtype=np.float32)
            hr = None
            status, color_status = "NO FACE", (0, 0, 255)

        hr_text = f"HR: {hr} BPM" if hr is not None else "HR: --"
        cv2.putText(display, hr_text, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(display, f"[{method_label}] {status}",
                    (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE,
                    color_status, config.FONT_THICKNESS)

        cv2.imshow(config.WINDOW_NAME, display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.stop()
    detector.close()
    cv2.destroyAllWindows()
