"""
HandTracker — 使用 MediaPipe Legacy API (mp.solutions.hands)
舊版 API 有內建的時序追蹤（Kalman filter），在即時攝影機場景下比 Tasks API 穩定很多。
"""

import cv2
import numpy as np
import mediapipe as mp


class _LegacyResults:
    """
    把 Legacy API 的結果包裝成與原本程式碼相容的格式。
    原本程式碼：results.hand_landmarks[0][8].x
    Legacy API：results.multi_hand_landmarks[0].landmark[8].x
    透過這個包裝類，後面的程式碼不需要改。
    """
    def __init__(self, multi_hand_landmarks):
        if multi_hand_landmarks:
            # 每個 hand 變成一個 list，內容是 landmark 物件（有 .x .y .z 屬性）
            self.hand_landmarks = [lm.landmark for lm in multi_hand_landmarks]
        else:
            self.hand_landmarks = []


class HandTracker:
    def __init__(self,
                 min_hand_detection_confidence=0.5,
                 min_tracking_confidence=0.5,
                 max_num_hands=1,
                 **kwargs):   # 忽略舊的 model_path 參數，保持向下相容

        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles

        print("[進度] 正在初始化 MediaPipe Legacy Hands (穩定版)...")
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,        # 視訊模式：啟用跨幀追蹤
            max_num_hands=max_num_hands,    # 只追蹤一隻手，更穩定
            model_complexity=1,             # 0=快, 1=平衡(預設), 影響準確度
            min_detection_confidence=min_hand_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        print(f"[進度] MediaPipe Legacy Hands 初始化成功！"
              f"(det={min_hand_detection_confidence}, track={min_tracking_confidence})")

    def apply_lighting_optimization(self, image):
        """CLAHE 光線均衡化，提升暗處或逆光時的偵測率"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def process_frame(self, image, optimize_lighting=False):
        """處理單幀畫面，回傳 (_LegacyResults, image)"""
        if optimize_lighting:
            image = self.apply_lighting_optimization(image)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        raw = self._hands.process(rgb)
        rgb.flags.writeable = True

        results = _LegacyResults(raw.multi_hand_landmarks)
        return results, image

    def draw_landmarks(self, image, results):
        """在畫面上繪製骨架（直接用 mp.solutions.drawing_utils）"""
        # 因為 _LegacyResults 沒有存原始物件，需要再跑一次 process
        # 改為直接在外部存原始結果，這裡簡化為手動繪製
        if results.hand_landmarks:
            for lm_list in results.hand_landmarks:
                # 手動繪製連線（lm_list 是 landmark 的 list）
                for connection in self._mp_hands.HAND_CONNECTIONS:
                    p1 = lm_list[connection[0]]
                    p2 = lm_list[connection[1]]
                    x1 = int(p1.x * image.shape[1])
                    y1 = int(p1.y * image.shape[0])
                    x2 = int(p2.x * image.shape[1])
                    y2 = int(p2.y * image.shape[0])
                    cv2.line(image, (x1, y1), (x2, y2), (0, 200, 0), 1, cv2.LINE_AA)

                # 繪製關節點
                for lm in lm_list:
                    px = int(lm.x * image.shape[1])
                    py = int(lm.y * image.shape[0])
                    cv2.circle(image, (px, py), 3, (0, 80, 255), -1, cv2.LINE_AA)

        return image
