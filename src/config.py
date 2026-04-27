# ── Algorithm 
RPPG_METHOD: str = "POS"    

# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_INDEX: int = 0         
FRAME_WIDTH: int = 640
FRAME_HEIGHT: int = 480
FPS_TARGET: int = 30

# Window
WINDOW_NAME: str = "rPPG"
PLOT_H: int = 80               

# Overlay text 
FONT_SCALE: float = 0.55
FONT_COLOR: tuple[int, int, int] = (180, 180, 180)
FONT_THICKNESS: int = 1

#  Signal buffer 
BUFFER_SEC: int = 10      

#  Detrend filter 
DETREND_LAMBDA: float = 100.0 

# ── Chebyshev Type II bandpass
CHEBY_LO: float = 0.7          
CHEBY_HI: float = 3.5          
CHEBY_ORDER: int = 2           
CHEBY_RS: float = 40.0        

# ── HR estimation 
HR_LO_HZ: float = 0.67         
HR_HI_HZ: float = 3.0         

# MediaPipe face model 
FACE_MODEL_PATH: str = "face_landmarker.task"
FACE_MAX_NUM: int = 1
FACE_MIN_DETECTION_CONFIDENCE: float = 0.5
FACE_MIN_TRACKING_CONFIDENCE: float = 0.5

# ── ROI landmark indices 
LEFT_CHEEK_IDX: list[int] = [50, 101, 118, 117, 116, 123, 147, 213, 192, 214]
RIGHT_CHEEK_IDX: list[int] = [280, 330, 347, 346, 345, 352, 376, 433, 416, 434]
FOREHEAD_IDX: list[int] = [
    10, 109, 67, 103, 54, 21, 162, 127,
    55, 65, 52, 53, 46, 124, 35, 31,
    228, 229, 230, 231, 232, 233, 244,
    245, 122, 6, 351, 465, 464, 463,
    462, 461, 460, 459, 458, 309, 261,
    291, 285, 295, 282, 283, 276, 356,
    389, 251, 284, 332, 297, 338, 10,
]
