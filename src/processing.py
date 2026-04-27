import scipy.signal
import scipy.sparse
import scipy.sparse.linalg
import numpy as np
import cv2
from src import config

def extract_mean_rgb(frame: np.ndarray, roi: np.ndarray) -> np.ndarray | None:
    mask = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    pixels = frame[mask > 0]
    if len(pixels) == 0:
        return None
    return pixels.mean(axis=0)

def detrend(sig: np.ndarray) -> np.ndarray:
    """Remove slow baseline drift"""
    n = len(sig)
    I = scipy.sparse.eye(n, format="csc")
    ones = np.ones(n)
    D = scipy.sparse.spdiags(
        np.array([ones, -2 * ones, ones]),
        [0, 1, 2],
        n - 2, n,
    ).tocsc()
    detrended = (I - scipy.sparse.linalg.inv(I + config.DETREND_LAMBDA ** 2 * D.T @ D)) @ sig
    return np.asarray(detrended).ravel().astype(np.float32)

def chebyshev_bandpass(sig: np.ndarray, fps: float) -> np.ndarray:
    """Chebyshev Type II bandpass filter"""
    nyq = fps / 2.0
    b, a = scipy.signal.cheby2(config.CHEBY_ORDER, config.CHEBY_RS, [config.CHEBY_LO / nyq, config.CHEBY_HI / nyq], btype="bandpass",)
    return scipy.signal.filtfilt(b, a, sig.astype(np.float64)).astype(np.float32)

def process_bvp(rgb_buf: np.ndarray, fps: float) -> np.ndarray:
    sig = rgb_buf[:, 1].astype(np.float64)
    sig = detrend(sig)
    if len(sig) >= int(fps * 2):
        sig = chebyshev_bandpass(sig, fps)
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
