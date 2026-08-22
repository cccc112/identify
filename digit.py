import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import time
from tensorflow.keras.preprocessing.image import img_to_array
import os
import urllib.request  # 用於自動下載 MediaPipe 模型
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 抑制 TensorFlow 啟動時繁複的警告與日誌輸出，保持終端機乾淨
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TF_USE_LEGACY_KERAS"] = "1"

def normalize_lighting(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final_img

def adaptive_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return thresh

def preprocess_image(image, size=(28, 28)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 將圖像轉換為灰度圖像
    image = cv2.resize(image, size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0 # 將圖像像素值歸一化到 0 到 1 之間
    return image

# 預處理圖像並預測
def make_prediction(image, model):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image) # 使用模型進行預測
    return np.argmax(prediction), np.max(prediction) # 返回預測的數字和概率

# 繪製畫布網格
def draw_grid(canvas, num_rows=4, num_cols=2, color=(200, 200, 200), thickness=1):
    height, width = canvas.shape[:2]
    row_height = height // num_rows
    col_width = width // num_cols
    # 繪製垂直線
    for x in range(0, width, col_width):
        cv2.line(canvas, (x, 0), (x, height), color, thickness)
    # 繪製水平線
    for y in range(0, height, row_height):
        cv2.line(canvas, (0, y), (width, y), color, thickness)

# 繪製手部追蹤網格
def draw_hand_tracking_grid(image, num_rows=4, num_cols=2, color=(200, 200, 200), thickness=1):
    height, width = image.shape[:2]
    row_height = height // num_rows
    col_width = width // num_cols
    for x in range(0, width, col_width):
        cv2.line(image, (x, 0), (x, height), color, thickness)
    for y in range(0, height, row_height):
        cv2.line(image, (0, y), (width, y), color, thickness)

# 繪製控制面板
def draw_control_panel(image, width, height):
    panel_height = 50
    panel_color = (240, 255, 255)
    button_color = (200, 200, 200)
    border_color = (255, 0, 0) # 藍色框線
    button_size = (100, 40)
    alpha = 0.7 # 半透明因子

    # 繪製控制面板背景
    cv2.rectangle(image, (0, 0), (width, panel_height), panel_color, 1)
    
    # 定義按鈕區域
    button_area_clear = (10, 5, 10 + button_size[0], panel_height - 5)
    button_area_save = (140, 5, 140 + button_size[0], panel_height - 5)
    button_area_delete = (270, 5, 270 + button_size[0], panel_height - 5)
    
    # 使用半透明的白色按鈕
    overlay = image.copy()
    for button_area in [button_area_clear]:
        cv2.rectangle(overlay, button_area[:2], button_area[2:], button_color, 1)
        cv2.rectangle(overlay, button_area[:2], button_area[2:], border_color, 2)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # 添加按鈕文字
    cv2.putText(image, 'Clear-c', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return button_area_clear, button_area_save, button_area_delete

# 計算點擊位置在哪個網格內
def get_grid_position(x, y, width, height, num_rows=4, num_cols=2):
    grid_width = width // num_cols
    grid_height = height // num_rows
    col = x // grid_width
    row = y // grid_height
    return row * num_cols + col

# 繪製網格計數資訊 (補全原本被截斷的實作)
def draw_grid_info(image, grid_counts, width, height, num_rows=4, num_cols=2):
    grid_width = width // num_cols
    grid_height = height // num_rows
    for i in range(num_rows):
        for j in range(num_cols):
            grid_index = i * num_cols + j
            # 將計數資訊疊加在右半邊（畫布區）對應網格中
            text_position = (width + j * grid_width + 10, i * grid_height + 25)
            text = f"G{grid_index+1}: {grid_counts[grid_index]}"
            cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

# 繪製手寫區域的邊界框
def draw_bounding_box(image, points, color=(0, 245, 255), thickness=2):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    if len(x_coords) > 0 and len(y_coords) > 0:
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

# 偵測形狀並預測
def detect_shapes_and_predict(drawing_layer, target_model, is_digit_mode):
    gray = cv2.cvtColor(drawing_layer, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kernel = np.ones((3,3),np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    merge_threshold = 70  # 合併重疊框的閾值
    
    merged_boxes = []
    for box in bounding_boxes:
        x, y, w, h = box
        merged = False
        for merged_box in merged_boxes:
            mx, my, mw, mh = merged_box
            box_center = (x + w // 2, y + h // 2)
            merged_center = (mx + mw // 2, my + mh // 2)
            distance = np.sqrt((box_center[0] - merged_center[0]) ** 2 + (box_center[1] - merged_center[1]) ** 2)
            if distance < merge_threshold:
                new_box = (
                    min(x, mx),
                    min(y, my),
                    max(x + w, mx + mw) - min(x, mx),
                    max(y + h, my + mh) - min(y, my)
                )
                merged_boxes.remove(merged_box)
                merged_boxes.append(new_box)
                merged = True
                break
        if not merged:
            merged_boxes.append(box)
    
    predictions = []
    for box in merged_boxes:
        x, y, w, h = box
        roi = gray[y:y+h, x:x+w]
        preprocessed_roi = preprocess_image(cv2.cvtColor(cv2.resize(roi, (28, 28)), cv2.COLOR_GRAY2BGR))
        
        prediction = target_model.predict(preprocessed_roi)
        predicted_label = np.argmax(prediction)
        predictions.append((box, predicted_label))
    
    predictions.sort(key=lambda x: (x[0][1], x[0][0]))
    return predictions

# 繪製識別出的數字視窗 (補全原本被截斷的實作)
def draw_recognized_digits_window(image, recognized_digits, canvas_width, canvas_height, num_digits=8):
    text_color = (255, 255, 255)
    font_scale = 0.5
    thickness = 1
    labels = ['Units', 'Tens', 'Hundreds', 'Thousands', 'Ten Thousands', 'Hundred Thousands', 'Millions', 'Ten Millions']
    
    for i in range(min(num_digits, len(recognized_digits))):
        digits = recognized_digits[i][-10:] # 只顯示最近 10 個數字
        text = f"{labels[i]}: {' '.join(map(str, digits))}"
        # 在右側畫布的最上方，依序繪製各網格的歷史識別紀錄
        cv2.putText(image, text, (canvas_width + 10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

def draw_recognized_text_window(image, recognized_texts, canvas_width, canvas_height, num_entries=8):
    window_width = 200
    window_height = 400
    text_image = np.zeros((window_height, window_width, 3), dtype=np.uint8)
    text_color = (255, 255, 255)
    font_scale = 0.7
    thickness = 2

    for i, text_entry in enumerate(recognized_texts):
        if i >= num_entries:
            break
        text = f'{i+1}: {text_entry}'
        cv2.putText(text_image, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
    
    combined_image = np.hstack((image, text_image))
    return combined_image

def hand_tracking():
    print("[進度] 正在延遲載入 TensorFlow 模型...")
    # 將載入模型移入函數內，避免 Python 3.13 全域執行緒死鎖
    digit_model = tf.keras.models.load_model("C:/hand/best_model.h5")
    symbol_model = tf.keras.models.load_model("C:/hand/symbol.h5")
    print("[進度] TensorFlow 模型載入成功！")

    # 確保 MediaPipe 手部模型檔案存在，若無則自動下載
    model_path = 'hand_landmarker.task'
    if not os.path.exists(model_path):
        print("[進度] 正在自動下載最新的 MediaPipe 手部追蹤模型...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            model_path
        )
        print("[進度] 模型下載完成！")

    print("[進度] 正在初始化 MediaPipe Tasks API detector...")
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.4,
        min_tracking_confidence=0.3
    )
    detector = vision.HandLandmarker.create_from_options(options)
    print("[進度] MediaPipe 初始化成功！")

    # 手部骨架連線關聯點
    HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), 
                        (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16), 
                        (17, 18), (18, 19), (19, 20), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)]

    alpha = 0.5   
    is_drawing_enabled = False  
    button_area_draw = (250, 230, 380, 270)  
    erase_start_time = None  
    is_erasing = False  

    cap = cv2.VideoCapture(0) # 打開攝影機
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 設置寬度
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # 設置高度

    points = []  
    paths = []  
    drawing = False  
    clear_canvas = False  
    hand_in_frame = False  
    recognized_digits = [[] for _ in range(8)]  
    grid_counts = [0] * 8  

    canvas_width = 640
    canvas_height = 480
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    drawing_layer = np.zeros_like(canvas)
    bounding_boxes_layer = np.zeros_like(canvas)  

    draw_grid(canvas, num_rows=4, num_cols=2, color=(200, 200, 200), thickness=1)
    
    crosshair_size = 20
    crosshair_color = (255, 255, 255)
    crosshair_thickness = 2
    erase_radius = 50
    erase_color = (0, 245, 255)
    line_thickness = 4

    is_digit_mode = True

    print("[系統] 正在啟動鏡頭視窗...")
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1) # 鏡像翻轉
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        results = detector.detect(mp_image)

        # 繪製網格和控制面板
        draw_hand_tracking_grid(image, num_rows=4, num_cols=2, color=(200, 200, 200), thickness=1)
        button_area_clear, button_area_save, button_area_delete = draw_control_panel(image, canvas_width, canvas_height)

        if not is_drawing_enabled:
            border_color = (255, 0, 0)
            overlay = image.copy()
            cv2.rectangle(overlay, button_area_draw[:2], button_area_draw[2:], (255, 255, 255), 1)
            cv2.rectangle(overlay, button_area_draw[:2], button_area_draw[2:], border_color, 2)
            cv2.putText(overlay, 'Start Draw', (button_area_draw[0] + 10, button_area_draw[1] + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        current_points = []

        if results.hand_landmarks:
            hand_in_frame = True
            if len(results.hand_landmarks) > 1:
                drawing = False
            else:
                drawing = True

            for hand_landmarks in results.hand_landmarks:
                # 繪製骨架連線
                for connection in HAND_CONNECTIONS:
                    p1 = hand_landmarks[connection[0]]
                    p2 = hand_landmarks[connection[1]]
                    x1, y1 = int(p1.x * image.shape[1]), int(p1.y * image.shape[0])
                    x2, y2 = int(p2.x * image.shape[1]), int(p2.y * image.shape[0])
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                
                # 繪製節點
                for point in hand_landmarks:
                    px, py = int(point.x * image.shape[1]), int(point.y * image.shape[0])
                    cv2.circle(image, (px, py), 2, (0, 0, 255), -1)

                index_finger_tip = hand_landmarks[8]
                middle_finger_tip = hand_landmarks[12]
                x_index, y_index = int(index_finger_tip.x * image.shape[1]), int(index_finger_tip.y * image.shape[0])
                x_middle, y_middle = int(middle_finger_tip.x * image.shape[1]), int(middle_finger_tip.y * image.shape[0])

                x_index = np.clip(x_index, 10, image.shape[1] - 10)
                y_index = np.clip(y_index, 10, image.shape[0] - 10)

                current_points.append((x_index, y_index))

                # 檢查是否按下了開始繪圖按鈕
                if not is_drawing_enabled and (button_area_draw[0] <= x_index <= button_area_draw[2] and 
                                               button_area_draw[1] <= y_index <= button_area_draw[3]):
                    is_drawing_enabled = True
                
                # 檢查是否碰觸到 Clear 按鈕
                if (button_area_clear[0] <= x_index <= button_area_clear[2] and 
                    button_area_clear[1] <= y_index <= button_area_clear[3]):
                    clear_canvas = True

                distance = np.sqrt((x_index - x_middle) ** 2 + (y_index - y_middle) ** 2)
                if distance < 40:
                    drawing = False
                    points = []  # 兩指靠攏，停止繪圖
                else:
                    drawing = True

                # 檢查是否為拳頭模式
                fist_state = True
                fingertips = [4, 8, 12, 16, 20]
                palm = hand_landmarks[0]
                for tip_idx in fingertips:
                    fingertip = hand_landmarks[tip_idx]
                    if np.linalg.norm(np.array([fingertip.x, fingertip.y]) - np.array([palm.x, palm.y])) > 0.1:
                        fist_state = False
                        break

                if fist_state:
                    mid_x, mid_y = (x_index + x_middle) // 2, (y_index + y_middle) // 2
                    cv2.circle(image, (mid_x, mid_y), erase_radius, erase_color, 2)

                    if erase_start_time is None:
                        erase_start_time = time.time()
                    is_erasing = True

                if clear_canvas:
                    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
                    drawing_layer = np.zeros_like(canvas)
                    bounding_boxes_layer = np.zeros_like(canvas)
                    draw_grid(canvas, num_rows=4, num_cols=2)
                    clear_canvas = False
        else:
            if hand_in_frame:
                predictions = detect_shapes_and_predict(drawing_layer, digit_model if is_digit_mode else symbol_model, is_digit_mode)
                bounding_boxes_layer = np.zeros_like(canvas)
                for (box, prediction) in predictions:
                    x, y, w, h = box
                    cv2.rectangle(bounding_boxes_layer, (x, y), (x + w, y + h), (255, 245, 0), 2)
                    prediction_text = str(prediction)
                    cv2.putText(bounding_boxes_layer, prediction_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if drawing and len(current_points) > 0 and not is_erasing and is_drawing_enabled:
            points.extend(current_points)

        if len(points) > 1 and not is_erasing:
            path = []
            for i in range(1, len(points)):
                path.append(points[i - 1])
                cv2.line(drawing_layer, points[i - 1], points[i], (240, 202, 166), line_thickness)
                cv2.line(image, points[i - 1], points[i], (240, 202, 166), line_thickness)
            paths.append(path)
            points = [points[-1]]
        
        if is_erasing:
            erase_mask = np.zeros_like(drawing_layer)
            cv2.circle(erase_mask, (mid_x, mid_y), erase_radius, (255, 255, 255), -1)
            drawing_layer = cv2.bitwise_and(drawing_layer, cv2.bitwise_not(erase_mask))
            new_paths = []
            for path in paths:
                new_path = [point for point in path if erase_mask[point[1], point[0]].sum() == 0]
                if new_path:
                    new_paths.append(new_path)
            paths = new_paths
            points = []
            is_erasing = False

        if len(paths) == 0:
            bounding_boxes_layer = np.zeros_like(canvas)

        combined_image = np.hstack((image, canvas + drawing_layer + bounding_boxes_layer))
        
        # 準備繪圖層以進行預測
        gray_layer = cv2.cvtColor(drawing_layer, cv2.COLOR_BGR2GRAY)
        
        # 效能優化：僅在畫布有內容時進行模型推論，避免無謂消耗 CPU 與卡死
        if cv2.countNonZero(gray_layer) > 0:
            resized_layer = cv2.resize(gray_layer, (28, 28))
            normalized_layer = resized_layer / 255.0
            reshaped_layer = normalized_layer.reshape(1, 28, 28, 1)
            prediction = np.argmax(digit_model.predict(reshaped_layer, verbose=0), axis=-1) if is_digit_mode else np.argmax(symbol_model.predict(reshaped_layer, verbose=0), axis=-1)
        else:
            prediction = [0]
            
        if len(current_points) > 0:
            x, y = current_points[-1]
            grid_index = get_grid_position(x, y, canvas_width, canvas_height)
            grid_counts[grid_index] += 1

            if len(points) > 1:
                recognized_digits[grid_index].append(prediction[0])
                draw_bounding_box(bounding_boxes_layer, points, color=(0, 255, 0), thickness=2)

            cv2.circle(image, (x, y), 5, (255, 255, 255), -1)

        draw_grid_info(combined_image, grid_counts, canvas_width, canvas_height)

        if len(current_points) > 0:
            x, y = current_points[-1]
            cv2.circle(image, (x, y), 5, crosshair_color, -1)
            cv2.line(image, (x - crosshair_size, y), (x + crosshair_size, y), crosshair_color, crosshair_thickness)
            cv2.line(image, (x, y - crosshair_size), (x, y + crosshair_size), crosshair_color, crosshair_thickness)

        draw_recognized_digits_window(combined_image, recognized_digits, canvas_width, canvas_height, num_digits=8)

        # 顯示最終整合視窗
        cv2.imshow('Hand Tracking', combined_image)

        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC 退出
            break
        elif key == ord('z') and paths:
            paths.pop()
            drawing_layer = np.zeros_like(canvas)
            for path in paths:
                for i in range(1, len(path)):
                    cv2.line(drawing_layer, path[i - 1], path[i], (240, 202, 166), line_thickness)
        
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    hand_tracking()