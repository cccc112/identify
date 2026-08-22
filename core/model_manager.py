import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import os

# 抑制 TensorFlow 啟動時繁複的警告與日誌輸出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TF_USE_LEGACY_KERAS"] = "1"

class ModelManager:
    def __init__(self, digit_model_path="C:/hand/best_model.h5", letter_model_path="C:/hand/augmented_model.h5"):
        self.digit_model_path = digit_model_path
        self.letter_model_path = letter_model_path
        
        self.digit_model = None
        self.letter_model = None
        
        # 延遲載入模型，避免啟動卡死
        self._load_models()
        
    def _load_models(self):
        print("[進度] 正在載入 TensorFlow 模型...")
        if os.path.exists(self.digit_model_path):
            self.digit_model = tf.keras.models.load_model(self.digit_model_path, compile=False)
            print("[進度] 數字模型載入成功！")
        else:
            print(f"[警告] 找不到數字模型: {self.digit_model_path}")
            
        if os.path.exists(self.letter_model_path):
            self.letter_model = tf.keras.models.load_model(self.letter_model_path, compile=False)
            print("[進度] 字母模型載入成功！")
        else:
            print(f"[警告] 找不到字母模型: {self.letter_model_path}")

    def preprocess_roi(self, roi, size=(28, 28)):
        """將切下來的筆跡區塊轉換為模型所需的格式"""
        # 模型訓練時可能是吃 BGR 或 Grayscale，根據原本邏輯，它先被轉成灰階再轉回 BGR?
        # 原版邏輯：cv2.cvtColor(cv2.resize(roi, (28, 28)), cv2.COLOR_GRAY2BGR) 再去 preprocess
        # 我們簡化為 28x28 灰階再擴展維度
        roi_resized = cv2.resize(roi, size)
        
        # 確認模型輸入維度，通常是 (1, 28, 28, 1)
        normalized = roi_resized / 255.0
        reshaped = normalized.reshape(1, 28, 28, 1)
        return reshaped

    def predict_canvas_content(self, drawing_layer, mode="digit"):
        """
        掃描畫布上的繪圖層，擷取有效筆跡並預測。
        mode 可以是 "digit" 或 "letter"
        """
        model = self.digit_model if mode == "digit" else self.letter_model
        if model is None:
            return []

        gray = cv2.cvtColor(drawing_layer, cv2.COLOR_BGR2GRAY)
        
        # 效能優化：如果畫布全空，直接返回
        if cv2.countNonZero(gray) == 0:
            return []
            
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bounding_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 50] # 過濾太小的雜訊
        
        # 合併相近的 Bounding Boxes (簡化版)
        merge_threshold = 70
        merged_boxes = []
        for box in bounding_boxes:
            x, y, w, h = box
            merged = False
            for i, m_box in enumerate(merged_boxes):
                mx, my, mw, mh = m_box
                box_center = (x + w//2, y + h//2)
                m_center = (mx + mw//2, my + mh//2)
                dist = np.sqrt((box_center[0] - m_center[0])**2 + (box_center[1] - m_center[1])**2)
                if dist < merge_threshold:
                    new_box = (
                        min(x, mx), min(y, my),
                        max(x+w, mx+mw) - min(x, mx),
                        max(y+h, my+mh) - min(y, my)
                    )
                    merged_boxes[i] = new_box
                    merged = True
                    break
            if not merged:
                merged_boxes.append(box)
                
        predictions = []
        for box in merged_boxes:
            x, y, w, h = box
            # 留一點 margin
            pad = 10
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(gray.shape[1], x + w + pad)
            y2 = min(gray.shape[0], y + h + pad)
            
            roi = gray[y1:y2, x1:x2]
            if roi.size == 0:
                continue
                
            input_tensor = self.preprocess_roi(roi)
            pred_probs = model.predict(input_tensor, verbose=0)
            pred_label = np.argmax(pred_probs)
            
            # 轉換標籤 (如果是字母模式)
            if mode == "letter":
                # 假設字母模型輸出 0~25 代表 A~Z
                # 若需要特別對齊字典，需在此調整
                pred_char = chr(pred_label + 65) 
            else:
                pred_char = str(pred_label)
                
            predictions.append(((x, y, w, h), pred_char))
            
        return predictions
