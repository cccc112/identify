import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import os

# 抑制 TensorFlow 啟動時繁複的警告與日誌輸出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TF_USE_LEGACY_KERAS"] = "1"

class ModelManager:
    def __init__(self, digit_model_path="C:/hand/best_model.h5", 
                 letter_model_path="C:/hand/augmented_model.h5",
                 symbol_model_path="C:/hand/symbol.h5"):
        self.digit_model_path = digit_model_path
        self.letter_model_path = letter_model_path
        self.symbol_model_path = symbol_model_path
        
        self.digit_model = None
        self.letter_model = None
        self.symbol_model = None
        
        self.letter_classes = ['A', 'B', 'C','D','E','F','G','H','I','J','K','L','M',
                               'N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
        self.symbol_classes = ['+', '-', '*', '/', '=','!', '(', ')', 'sqrt', 'pi', 
                               'sin', 'cos', 'tan', 'log', 'exp','!',',',"[","]",'{','}',
                               'alpha','ascii_124','beta','Delta','exists','forall',
                               'forward_slash','gamma','geq','gt','in','infty','int',
                               'lambda','ldots','leq','lt','mu','neq','phi','pm',
                               'prime','rightarrow','sigma','sum','theta','times']
        
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

        if os.path.exists(self.symbol_model_path):
            self.symbol_model = tf.keras.models.load_model(self.symbol_model_path, compile=False)
            print("[進度] 符號模型載入成功！")
        else:
            print(f"[警告] 找不到符號模型: {self.symbol_model_path}")

    def preprocess_roi(self, roi, mode):
        """將切下來的筆跡區塊轉換為對應模型所需的格式"""
        if mode == "digit":
            # 28x28x1 灰階
            roi_resized = cv2.resize(roi, (28, 28))
            normalized = roi_resized / 255.0
            reshaped = normalized.reshape(1, 28, 28, 1)
            return reshaped
        else:
            # letter 與 symbol 都是 64x64x3
            # 根據原版 preprocess_image: 灰階 -> 反轉(bitwise_not) -> resize -> 轉BGR -> 正規化
            # 原本的 roi 已經是灰階 (0是黑底, 白線條)
            roi_not = cv2.bitwise_not(roi) # 反轉變成白底黑線條 (與原始模型一致)
            roi_resized = cv2.resize(roi_not, (64, 64))
            roi_bgr = cv2.cvtColor(roi_resized, cv2.COLOR_GRAY2BGR)
            normalized = roi_bgr / 255.0
            reshaped = normalized.reshape(1, 64, 64, 3)
            return reshaped

    def predict_canvas_content(self, drawing_layer, mode="digit"):
        """
        掃描畫布上的繪圖層，擷取有效筆跡並預測。
        mode 可以是 "digit", "letter", "symbol"
        """
        if mode == "digit":
            model = self.digit_model
        elif mode == "letter":
            model = self.letter_model
        elif mode == "symbol":
            model = self.symbol_model
        else:
            return []
            
        if model is None:
            return []

        gray = cv2.cvtColor(drawing_layer, cv2.COLOR_BGR2GRAY)
        
        if cv2.countNonZero(gray) == 0:
            return []
            
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bounding_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 50]
        
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
                
        # 由左至右排序
        merged_boxes.sort(key=lambda b: b[0])
                
        MIN_CONFIDENCE = 0.55  # 低於此門檻的預測一律丟棄，避免噪點誤判

        predictions = []
        for box in merged_boxes:
            x, y, w, h = box
            # 過濾太小的框（可能是筆觸毛邊或噪點）
            if w * h < 200:
                continue

            pad = 20
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(gray.shape[1], x + w + pad)
            y2 = min(gray.shape[0], y + h + pad)

            roi = gray[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            input_tensor = self.preprocess_roi(roi, mode)
            pred_probs = model.predict(input_tensor, verbose=0)[0]
            pred_label = int(np.argmax(pred_probs))
            confidence = float(pred_probs[pred_label])

            # 信心值過濾
            if confidence < MIN_CONFIDENCE:
                continue

            if mode == "letter" and pred_label < len(self.letter_classes):
                pred_char = self.letter_classes[pred_label]
            elif mode == "symbol" and pred_label < len(self.symbol_classes):
                pred_char = f" {self.symbol_classes[pred_label]} "
            else:
                pred_char = str(pred_label)

            predictions.append(((x, y, w, h), pred_char, confidence))

        return predictions

    # ─────────────────────────────────────────────────────────────
    #  書寫順序辨識（主要入口）
    # ─────────────────────────────────────────────────────────────

    def predict_from_paths(self, paths, canvas_w, canvas_h,
                           line_thickness=7, mode="digit"):
        """
        依照書寫順序辨識。
        - 把空間上接近的筆畫合併成同一個字元群組
        - 群組順序 = 該群組第一筆畫的書寫時間順序
        - 對每個群組單獨渲染後送入模型
        回傳：[(bbox, char, confidence), ...]  依書寫順序排列
        """
        if not paths or mode not in ('digit', 'letter', 'symbol'):
            return []
        model = {'digit': self.digit_model,
                 'letter': self.letter_model,
                 'symbol': self.symbol_model}.get(mode)
        if model is None:
            return []

        MIN_CONFIDENCE = 0.55

        groups = self._group_strokes(paths)   # 使用預設 gap_threshold=30
        predictions = []

        for group_bbox, path_indices in groups:
            gx, gy, gw, gh = group_bbox
            if gw * gh < 200:
                continue

            # 只把該群組的筆畫渲染到獨立的灰階畫布
            mini = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
            for idx in path_indices:
                path = paths[idx]
                if len(path) > 1:
                    for i in range(1, len(path)):
                        cv2.line(mini, path[i-1], path[i],
                                 255, line_thickness, cv2.LINE_AA)
                        cv2.circle(mini, path[i],
                                   line_thickness // 2, 255, -1, cv2.LINE_AA)
                elif len(path) == 1:
                    cv2.circle(mini, path[0],
                               line_thickness // 2 + 1, 255, -1, cv2.LINE_AA)

            # 裁出含 padding 的 ROI
            pad = 20
            x1 = max(0, gx - pad)
            y1 = max(0, gy - pad)
            x2 = min(canvas_w, gx + gw + pad)
            y2 = min(canvas_h, gy + gh + pad)
            roi = mini[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            input_tensor = self.preprocess_roi(roi, mode)
            pred_probs   = model.predict(input_tensor, verbose=0)[0]
            pred_label   = int(np.argmax(pred_probs))
            confidence   = float(pred_probs[pred_label])

            if confidence < MIN_CONFIDENCE:
                continue

            if mode == "letter" and pred_label < len(self.letter_classes):
                pred_char = self.letter_classes[pred_label]
                # EMNIST 常見視覺混淆對後處理
                CONFUSION_PAIRS = {
                    'Q': 'q', 'O': 'o', 'P': 'p', 'C': 'c',
                    'S': 's', 'U': 'u', 'V': 'v', 'W': 'w',
                    'X': 'x', 'Z': 'z',
                }
                if pred_char in CONFUSION_PAIRS:
                    top2_label = int(np.argsort(pred_probs)[-2])
                    top2_char  = (self.letter_classes[top2_label]
                                  if top2_label < len(self.letter_classes) else None)
                    alt = CONFUSION_PAIRS[pred_char]
                    # top-2 是對應小寫且信心差距 < 35% → 改輸出小寫
                    if top2_char == alt and (confidence - float(pred_probs[top2_label])) < 0.35:
                        pred_char = alt
            elif mode == "symbol" and pred_label < len(self.symbol_classes):
                pred_char = f" {self.symbol_classes[pred_label]} "
            else:
                pred_char = str(pred_label)

            predictions.append(((gx, gy, gw, gh), pred_char, confidence))

        return predictions

    def _group_strokes(self, paths, gap_threshold=30):
        """
        將筆畫依空間鄰近性合併成字元群組，書寫順序由第一筆決定。
        gap_threshold: 兩個 bbox 邊緣間最大允許間距（像素）。
        用邊緣間距而非中心距離，避免大字元把遠處的字也吸進來。
        回傳 [(merged_bbox, [path_indices]), ...]
        """
        def path_bbox(path):
            if not path:
                return (0, 0, 1, 1)
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)
            return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))

        bboxes = [path_bbox(p) for p in paths]
        groups = []   # list of [[bbox], [path_indices]]

        for i, (x, y, w, h) in enumerate(bboxes):
            assigned = False

            for j in range(len(groups)):
                gx, gy, gw, gh = groups[j][0]
                # bbox 邊緣間距 (不重疊時才有值，重疊時為 0)
                gap_x = max(0, x - (gx + gw)) if x > gx else max(0, gx - (x + w))
                gap_y = max(0, y - (gy + gh)) if y > gy else max(0, gy - (y + h))
                if gap_x < gap_threshold and gap_y < gap_threshold:
                    nx1 = min(x, gx)
                    ny1 = min(y, gy)
                    nx2 = max(x + w, gx + gw)
                    ny2 = max(y + h, gy + gh)
                    groups[j][0] = (nx1, ny1, nx2 - nx1, ny2 - ny1)
                    groups[j][1].append(i)
                    assigned = True
                    break

            if not assigned:
                groups.append([(x, y, w, h), [i]])

        return groups
