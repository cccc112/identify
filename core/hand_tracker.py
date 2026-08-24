"""
HandTracker — Tasks API with VIDEO running mode
VIDEO 模式有跨幀時序追蹤，穩定性接近 Legacy API，
且與目前安裝的新版 mediapipe (>=0.10) 完全相容。
"""

import cv2
import numpy as np
import os
import time
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode


class HandTracker:
    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )

    HAND_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (5,6),(6,7),(7,8),
        (9,10),(10,11),(11,12),
        (13,14),(14,15),(15,16),
        (17,18),(18,19),(19,20),
        (0,5),(5,9),(9,13),(13,17),(0,17),
    ]

    def __init__(self,
                 model_path="C:/hand/hand_landmarker.task",
                 min_hand_detection_confidence=0.5,
                 min_tracking_confidence=0.5,
                 max_num_hands=1,
                 **kwargs):

        self.model_path = model_path
        self._ensure_model()

        print("[進度] 正在初始化 MediaPipe Tasks API (VIDEO 模式)...")
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.VIDEO,          # 關鍵：VIDEO 模式有跨幀追蹤
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)
        self._start_ms = int(time.time() * 1000)     # 基準時間戳（毫秒）
        print("[進度] MediaPipe Tasks API (VIDEO 模式) 初始化成功！")

    def _ensure_model(self):
        if not os.path.exists(self.model_path):
            print("[進度] 正在下載 hand_landmarker.task ...")
            try:
                urllib.request.urlretrieve(self.MODEL_URL, self.model_path)
                print("[進度] 模型下載完成！")
            except Exception as e:
                print(f"[錯誤] 下載失敗: {e}")

    def apply_lighting_optimization(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    def process_frame(self, image, optimize_lighting=False):
        """
        處理單幀並回傳 (results, image)。
        results.hand_landmarks 是 list of list of landmark，
        用法：lm = results.hand_landmarks[0]，lm[8].x 是食指指尖 x。
        """
        if optimize_lighting:
            image = self.apply_lighting_optimization(image)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO 模式必須傳遞單調遞增的時間戳（毫秒）
        timestamp_ms = int(time.time() * 1000) - self._start_ms

        results = self._detector.detect_for_video(mp_image, timestamp_ms)
        return results, image

    def draw_landmarks(self, image, results):
        """在畫面上繪製骨架與關節點"""
        if not results.hand_landmarks:
            return image

        H, W = image.shape[:2]
        for lm_list in results.hand_landmarks:
            for c in self.HAND_CONNECTIONS:
                p1, p2 = lm_list[c[0]], lm_list[c[1]]
                cv2.line(image,
                         (int(p1.x * W), int(p1.y * H)),
                         (int(p2.x * W), int(p2.y * H)),
                         (0, 200, 0), 1, cv2.LINE_AA)
            for lm in lm_list:
                cv2.circle(image,
                           (int(lm.x * W), int(lm.y * H)),
                           3, (0, 80, 255), -1, cv2.LINE_AA)
        return image
