import cv2
import numpy as np
import os
import urllib.request
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode

class FaceTracker:
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.model_path = os.path.join(os.path.dirname(__file__), '..', 'face_landmarker.task')
        self._ensure_model_exists()
        self._start_ms = int(time.time() * 1000)

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def _ensure_model_exists(self):
        if not os.path.exists(self.model_path):
            print(f"[系統] 正在下載 Face Landmarker 模型 (這只需要一次)...")
            try:
                urllib.request.urlretrieve(self.MODEL_URL, self.model_path)
                print("[系統] 模型下載完成！")
            except Exception as e:
                print(f"[系統] 模型下載失敗: {e}")

    def process_frame(self, frame):
        """
        處理影像，回傳 results 物件。
        為了相容舊版 API 的寫法，我們會包裝回傳值。
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # 必須傳入 timestamp (ms)
        timestamp_ms = int(time.time() * 1000) - self._start_ms
        detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)
        
        # 為了跟 GazeSolver 相容，我們自建一個假的 results 物件
        class DummyResults:
            pass
        
        results = DummyResults()
        results.multi_face_landmarks = None
        
        if detection_result and detection_result.face_landmarks:
            # 建立一個相容於舊版的 landmark 列表
            class DummyLandmarkList:
                pass
            class DummyLandmark:
                def __init__(self, x, y, z):
                    self.x = x
                    self.y = y
                    self.z = z
            
            face = detection_result.face_landmarks[0]
            lm_list = DummyLandmarkList()
            lm_list.landmark = [DummyLandmark(lm.x, lm.y, lm.z) for lm in face]
            results.multi_face_landmarks = [lm_list]
            
        return results

    def close(self):
        if hasattr(self, 'detector'):
            self.detector.close()
