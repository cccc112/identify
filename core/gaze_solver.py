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

        # ── 2. 純眼球追蹤 (Pure Eye Tracking) ──
        # 左眼外角 33, 內角 133 (在畫面上因鏡像可能是反的)
        l_inner = lm[133]
        l_outer = lm[33]
        l_iris  = lm[468]
        
        # 右眼內角 362, 外角 263
        r_inner = lm[362]
        r_outer = lm[263]
        r_iris  = lm[473]

        # 計算瞳孔在眼眶中的相對 X, Y 比例
        def get_ratio(inner, outer, iris):
            w = outer.x - inner.x
            h = outer.y - inner.y
            rx = (iris.x - inner.x) / (w + 1e-6)
            ry = (iris.y - inner.y) / (h + 1e-6)
            return rx, ry
            
        lx, ly = get_ratio(l_inner, l_outer, l_iris)
        rx, ry = get_ratio(r_inner, r_outer, r_iris)
        
        # 平均雙眼的比例
        avg_x = (lx + rx) / 2.0
        
        # 上下移動用額頭與鼻尖的相對位置輔助，因為眼皮會干擾 Y 軸
        avg_y = (l_iris.y + r_iris.y)/2.0 - lm[1].y
        
        # 映射到螢幕 (根據實測，瞳孔比例大約在 0.3 ~ 0.7 之間變動)
        # X 軸放大
        raw_x = (avg_x - 0.4) * 3.0 * W
        # Y 軸放大
        raw_y = (avg_y + 0.05) * 15.0 * H
        
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
