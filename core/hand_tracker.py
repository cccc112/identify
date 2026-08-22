import cv2
import numpy as np
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandTracker:
    def __init__(self, model_path="C:/hand/hand_landmarker.task", min_hand_detection_confidence=0.4, min_tracking_confidence=0.3):
        self.model_path = model_path
        self._ensure_model_exists()
        
        print("[進度] 正在初始化 MediaPipe Tasks API detector...")
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        self.HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), 
                                 (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16), 
                                 (17, 18), (18, 19), (19, 20), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)]
        print("[進度] MediaPipe 初始化成功！")

    def _ensure_model_exists(self):
        if not os.path.exists(self.model_path):
            print("[進度] 正在自動下載最新的 MediaPipe 手部追蹤模型...")
            try:
                urllib.request.urlretrieve(
                    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                    self.model_path
                )
                print("[進度] 模型下載完成！")
            except Exception as e:
                print(f"[錯誤] 下載 MediaPipe 模型失敗: {e}")

    def apply_lighting_optimization(self, image):
        """
        對攝影機畫面進行 CLAHE (限制對比度自適應直方圖均衡化)
        這可以提升在逆光或暗處時，MediaPipe 對手部的辨識度。
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        optimized_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return optimized_img

    def process_frame(self, image, optimize_lighting=False):
        """
        處理單幀畫面並回傳偵測結果。
        """
        if optimize_lighting:
            image = self.apply_lighting_optimization(image)
            
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        results = self.detector.detect(mp_image)
        return results, image

    def draw_landmarks(self, image, results):
        """
        在畫面上繪製骨架與關節點。
        """
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                for connection in self.HAND_CONNECTIONS:
                    p1 = hand_landmarks[connection[0]]
                    p2 = hand_landmarks[connection[1]]
                    x1, y1 = int(p1.x * image.shape[1]), int(p1.y * image.shape[0])
                    x2, y2 = int(p2.x * image.shape[1]), int(p2.y * image.shape[0])
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                
                for point in hand_landmarks:
                    px, py = int(point.x * image.shape[1]), int(point.y * image.shape[0])
                    cv2.circle(image, (px, py), 2, (0, 0, 255), -1)
        return image
