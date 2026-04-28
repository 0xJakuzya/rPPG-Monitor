import scipy.signal as scipy_signal
import scipy.signal
import scipy.sparse
import numpy as np
import torch
import cv2
from src import config

def load_ppg_sync(path: str) -> tuple[np.ndarray, float]:
    data = np.loadtxt(path)
    vals = data[:, 0].astype(np.float32)
    total_time = data[:, 1].sum()
    fps = len(vals) / total_time if total_time > 0 else 100.0
    return vals, fps

def resample_ppg(ppg: np.ndarray, ppg_fps: float,
                  video_fps: float, n_frames: int) -> np.ndarray:
    resampled = scipy_signal.resample(ppg, int(len(ppg) / ppg_fps * video_fps))
    if len(resampled) >= n_frames:
        return resampled[:n_frames].astype(np.float32)
    pad = np.zeros(n_frames - len(resampled), dtype=np.float32)
    return np.concatenate([resampled, pad]).astype(np.float32)

def load_physnet(device: torch.device):
    from models.physnet import PhysNet
    model = PhysNet().to(device)
    model.load_state_dict(torch.load(config.PHYSNET_MODEL_PATH, map_location=device))
    model.eval()
    return model

def physnet_bvp(model, rgb_buf: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(rgb_buf).unsqueeze(0).to(device)  
    with torch.no_grad():
        out = model(x)                                       
    sig = out[0].cpu().numpy().astype(np.float32)
    sig -= sig.mean()
    std = sig.std()
    if std > 1e-6:
        sig /= std
    return sig

def extract_mean_rgb(frame: np.ndarray, roi: np.ndarray) -> np.ndarray | None:
    mask = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    pixels = frame[mask > 0]
    if len(pixels) == 0:
        return None
    return pixels.mean(axis=0)

def extract_rois_rgb(frame: np.ndarray, forehead: np.ndarray,
                     left_cheek: np.ndarray, right_cheek: np.ndarray) -> np.ndarray | None:
    rois_weights = [(forehead, 0.5), (left_cheek, 0.25), (right_cheek, 0.25)]
    weighted, total_w = np.zeros(3), 0.0
    for roi, w in rois_weights:
        sample = extract_mean_rgb(frame, roi)
        if sample is not None:
            weighted += sample * w
            total_w += w
    if total_w == 0:
        return None
    rgb_val = weighted / total_w
    return np.array([rgb_val[2], rgb_val[1], rgb_val[0]], dtype=np.float32)

def detrend(sig: np.ndarray, lam: float = config.DETREND_LAMBDA) -> np.ndarray:
    n = len(sig)
    H = np.eye(n)
    ones = np.ones(n)
    D = scipy.sparse.spdiags(
        np.array([ones, -2 * ones, ones]), [0, 1, 2], n - 2, n
    ).toarray()
    return (H - np.linalg.inv(H + lam ** 2 * D.T @ D)) @ sig.astype(np.float64)

def bandpass_filter(sig: np.ndarray, fps: float, lo: float, hi: float) -> np.ndarray:
    nyq = fps / 2.0
    sig64 = sig.astype(np.float64)
    if config.FILTER_TYPE == "chebyshev2":
        b, a = scipy.signal.cheby2(
            config.CHEBY_ORDER, config.CHEBY_RS, [lo / nyq, hi / nyq], btype="bandpass"
        )
    else:
        b, a = scipy.signal.butter(1, [lo / nyq, hi / nyq], btype="bandpass")
    return scipy.signal.filtfilt(b, a, sig64).astype(np.float32)

def process_bvp(rgb_buf: np.ndarray, fps: float) -> np.ndarray:
    sig = rgb_buf[:, 1].astype(np.float64)
    sig = detrend(sig)
    if len(sig) >= int(fps * 2):
        sig = bandpass_filter(sig, fps, config.CHEBY_LO, config.CHEBY_HI)
    sig -= sig.mean()
    std = sig.std()
    if std > 1e-6:
        sig /= std
    return sig.astype(np.float32)

def estimate_hr(bvp: np.ndarray, fps: float) -> float | None:
    if len(bvp) < int(fps * 2):
        return None
    freqs = np.fft.rfftfreq(len(bvp), d=1.0 / fps)
    power = np.abs(np.fft.rfft(bvp)) ** 2
    mask = (freqs >= config.HR_LO_HZ) & (freqs <= config.HR_HI_HZ)
    if not mask.any():
        return None
    return float(freqs[mask][np.argmax(power[mask])] * 60.0)

