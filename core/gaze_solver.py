import math
import time
import numpy as np

def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

class GazeSolver:
    def __init__(self, alpha=0.3):
        self.alpha = alpha  # EMA 平滑係數
        self.cursor_x = 0
        self.cursor_y = 0
        self.is_initialized = False
        
        self.blink_threshold = 0.22
        self.is_blinking = False
        self.last_blink_time = 0

    def update(self, face_landmarks, W, H):
        """
        傳入 FaceMesh landmarks，回傳 (x, y, is_blink)
        """
        lm = face_landmarks.landmark
        
        # ── 1. 眨眼偵測 (EAR) ──
        # 左眼 (對向使用者的右邊)
        l_ear = distance(lm[159], lm[145]) / (distance(lm[33], lm[133]) + 1e-6)
        # 右眼
        r_ear = distance(lm[386], lm[374]) / (distance(lm[362], lm[263]) + 1e-6)
        
        ear = (l_ear + r_ear) / 2.0
        blink = ear < self.blink_threshold
        
        # Debounce blink
        blink_trigger = False
        if blink and not self.is_blinking:
            if time.time() - self.last_blink_time > 0.5:
                blink_trigger = True
                self.last_blink_time = time.time()
        self.is_blinking = blink

        # ── 2. 視線游標位置 (混合鼻尖與瞳孔偏移) ──
        # 鼻尖
        nose_x, nose_y = lm[1].x, lm[1].y
        
        # 瞳孔中心
        l_iris = lm[468]
        r_iris = lm[473]
        iris_x = (l_iris.x + r_iris.x) / 2.0
        iris_y = (l_iris.y + r_iris.y) / 2.0
        
        # 將小範圍的頭部移動映射到全螢幕
        # 假設使用者頭部在畫面中間 0.3 ~ 0.7 之間移動
        mapped_x = (nose_x - 0.3) / 0.4
        mapped_y = (nose_y - 0.3) / 0.4
        
        # 加上眼球偏移量 (放大 5 倍讓眼動更明顯)
        eye_offset_x = (iris_x - nose_x) * 5
        eye_offset_y = (iris_y - nose_y) * 5
        
        raw_x = (mapped_x + eye_offset_x) * W
        raw_y = (mapped_y + eye_offset_y) * H
        
        # 邊界限制
        raw_x = max(0, min(W, raw_x))
        raw_y = max(0, min(H, raw_y))
        
        # EMA 平滑
        if not self.is_initialized:
            self.cursor_x = raw_x
            self.cursor_y = raw_y
            self.is_initialized = True
        else:
            self.cursor_x = self.cursor_x * (1 - self.alpha) + raw_x * self.alpha
            self.cursor_y = self.cursor_y * (1 - self.alpha) + raw_y * self.alpha

        return int(self.cursor_x), int(self.cursor_y), blink_trigger, ear
