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
        self.use_eye_only = False

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

        if self.use_eye_only:
            # ── 2. 純眼球追蹤 (Pure Eye Tracking) ──
            # 左眼外角 33, 內角 133
            # 右眼內角 362, 外角 263
            def get_ratio(inner, outer, iris):
                w = outer.x - inner.x
                rx = (iris.x - inner.x) / (w + 1e-6)
                return rx
                
            lx = get_ratio(lm[133], lm[33], lm[468])
            rx = get_ratio(lm[362], lm[263], lm[473])
            avg_x = (lx + rx) / 2.0
            
            # Y軸用眼眶上下緣: 上 159, 下 145
            def get_y_ratio(top, bottom, iris):
                h = bottom.y - top.y
                ry = (iris.y - top.y) / (h + 1e-6)
                return ry
            ly = get_y_ratio(lm[159], lm[145], lm[468])
            ry_y = get_y_ratio(lm[386], lm[374], lm[473])
            avg_y = (ly + ry_y) / 2.0
            
            # 瞳孔在眼眶內的比例，置中時大約是 0.5
            # 針對左右不好移動的問題，我們以 0.5 為中心，進行放大 (Sensitivity)
            # multiplier 越大，越容易碰到螢幕邊緣
            SENSITIVITY_X = 4.0
            SENSITIVITY_Y = 3.5
            mapped_x = 0.5 + (avg_x - 0.5) * SENSITIVITY_X
            mapped_y = 0.5 + (avg_y - 0.45) * SENSITIVITY_Y # Y中心稍微偏上
            
            # Y軸經常會感覺相反（往上變往下），如果發現反向，將它反轉：
            mapped_y = 1.0 - mapped_y
            
            raw_x = mapped_x * W
            raw_y = mapped_y * H
        else:
            # ── 2. 高穩定度頭部/視線追蹤 (Head Pointer) ──
            nose_x = lm[1].x
            nose_y = lm[1].y
            
            # 放大使用者的微小頭部動作，映射到全螢幕
            # 之前範圍太大導致要動很多，現在縮小範圍，提高靈敏度
            mapped_x = (nose_x - 0.4) / 0.2
            mapped_y = (nose_y - 0.4) / 0.2
            raw_x = mapped_x * W
            raw_y = mapped_y * H
        
        # 邊界限制
        raw_x = max(0, min(W, raw_x))
        raw_y = max(0, min(H, raw_y))
        
        # EMA 平滑
        if not self.is_initialized:
            self.cursor_x = raw_x
            self.cursor_y = raw_y
            self.is_initialized = True
        else:
            # 純眼球追蹤容易抖動，套用更強的平滑 (alpha 變小)
            current_alpha = 0.1 if self.use_eye_only else self.alpha
            self.cursor_x = self.cursor_x * (1 - current_alpha) + raw_x * current_alpha
            self.cursor_y = self.cursor_y * (1 - current_alpha) + raw_y * current_alpha

        return int(self.cursor_x), int(self.cursor_y), blink_trigger, ear
