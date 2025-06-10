import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import mediapipe as mp
import time
from tensorflow.keras.preprocessing.image import img_to_array

# 載入訓練好的模型
letter_model = tf.keras.models.load_model("C:/hand/augmented_model.h5")
# 類別名稱映射
class_names = ['A', 'B', 'C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
recognition_area = (100, 100, 540, 380) # (x_min, y_min, x_max, y_max)

# --- 顏色定義 (可以自訂這些顏色) ---
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_GRAY = (100, 100, 100)
COLOR_LIGHT_GRAY = (200, 200, 200)
COLOR_DARK_GRAY = (50, 50, 50)
COLOR_YELLOW = (0, 255, 255)
COLOR_ORANGE = (0, 165, 255)

# UI 相關顏色
UI_BG_COLOR = COLOR_DARK_GRAY
BUTTON_COLOR_NORMAL = (90, 90, 90) # 更深的灰色
BUTTON_COLOR_HOVER = (120, 120, 120) # 鼠標懸停效果
TEXT_COLOR = COLOR_WHITE
ACCENT_COLOR = COLOR_ORANGE # 用於強調或準星
DRAWING_COLOR = COLOR_BLUE # 繪圖顏色

# --- 輔助函數 (不變) ---
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
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

def is_hand_in_recognition_area(hand_landmarks, area):
    if not hand_landmarks:
        return False
    # 使用食指尖端作為判斷點
    tip_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)
    tip_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)
    x_min, y_min, x_max, y_max = area
    return x_min < tip_x < x_max and y_min < tip_y < y_max

def preprocess_image(image_roi):
    # 將圖像轉換為灰度
    gray_roi = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
    # 反轉顏色（因為模型可能是在黑色背景上訓練的白色字母）
    inverted_roi = cv2.bitwise_not(gray_roi)
    # 調整大小以匹配模型輸入
    resized_roi = cv2.resize(inverted_roi, (64, 64), interpolation=cv2.INTER_AREA)
    # 擴展維度以匹配模型輸入 (batch_size, height, width, channels)
    normalized_roi = resized_roi / 255.0 # 歸一化到 0-1
    return normalized_roi.reshape(1, 64, 64, 1)

def detect_shapes_and_predict(drawing_layer, model, class_names):
    # 轉換為灰度並找到輪廓
    gray_layer = cv2.cvtColor(drawing_layer, cv2.COLOR_BGR2GRAY)
    # 使用大津法或自適應閾值處理以獲得清晰的二值圖像
    _, thresh = cv2.threshold(gray_layer, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # 閾值調整
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    predictions = []
    min_contour_area = 50 # 忽略太小的雜訊點

    # 合併靠近的邊界框的閾值（像素）
    merge_threshold = 30

    # 存儲候選框
    candidate_boxes = []
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            candidate_boxes.append((x, y, w, h))

    # 合併重疊或非常接近的邊界框
    merged_boxes = []
    while candidate_boxes:
        current_box = candidate_boxes.pop(0)
        x1, y1, w1, h1 = current_box
        x1_end, y1_end = x1 + w1, y1 + h1
        merged = False

        for i, (mx, my, mw, mh) in enumerate(merged_boxes):
            mx_end, my_end = mx + mw, my + mh

            # 檢查是否重疊或足夠接近以合併
            if not (x1_end < mx - merge_threshold or
                    mx_end < x1 - merge_threshold or
                    y1_end < my - merge_threshold or
                    my_end < y1 - merge_threshold):
                # 合併
                new_x = min(x1, mx)
                new_y = min(y1, my)
                new_w = max(x1_end, mx_end) - new_x
                new_h = max(y1_end, my_end) - new_y
                merged_boxes[i] = (new_x, new_y, new_w, new_h)
                merged = True
                break
        if not merged:
            merged_boxes.append(current_box)

    # 對合併後的邊界框進行預測
    for (x, y, w, h) in merged_boxes:
        # 從繪圖層中提取ROI
        roi = drawing_layer[max(0, y-5):min(drawing_layer.shape[0], y+h+5), max(0, x-5):min(drawing_layer.shape[1], x+w+5)] # 增加一些邊距
        if roi.size == 0: # 避免空ROI
            continue
        
        preprocessed_roi = preprocess_image(roi)
        prediction = model.predict(preprocessed_roi, verbose=0)[0]
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # 排除低置信度的預測
        if confidence > 0.8: # 調整置信度閾值
            predicted_char = class_names[predicted_class]
            predictions.append((predicted_char, confidence, (x, y, w, h)))

    return predictions

# --- 繪圖功能調整 ---

def draw_control_panel(image, current_hand_landmarks, buttons):
    # 面板背景
    panel_height = 80
    panel_y_start = image.shape[0] - panel_height
    cv2.rectangle(image, (0, panel_y_start), (image.shape[1], image.shape[0]), UI_BG_COLOR, -1)

    cursor_x, cursor_y = -1, -1
    if current_hand_landmarks:
        # 使用食指尖作為游標
        index_tip = current_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        cursor_x = int(index_tip.x * image.shape[1])
        cursor_y = int(index_tip.y * image.shape[0])

    for btn_name, btn_info in buttons.items():
        x, y, w, h = btn_info['rect']
        btn_center_x = x + w // 2
        btn_center_y = y + h // 2
        
        # 檢查鼠標是否懸停在按鈕上
        is_hovering = False
        if cursor_x != -1 and cursor_y != -1 and \
           x < cursor_x < x + w and y < cursor_y < y + h:
            is_hovering = True
            btn_color = BUTTON_COLOR_HOVER
        else:
            btn_color = BUTTON_COLOR_NORMAL

        # 繪製按鈕背景 (圓角效果)
        radius = 15 # 圓角半徑
        cv2.rectangle(image, (x + radius, y), (x + w - radius, y + h), btn_color, -1)
        cv2.rectangle(image, (x, y + radius), (x + w, y + h - radius), btn_color, -1)
        cv2.circle(image, (x + radius, y + radius), radius, btn_color, -1)
        cv2.circle(image, (x + w - radius, y + radius), radius, btn_color, -1)
        cv2.circle(image, (x + radius, y + h - radius), radius, btn_color, -1)
        cv2.circle(image, (x + w - radius, y + h - radius), radius, btn_color, -1)
        
        # 繪製按鈕邊框
        cv2.rectangle(image, (x + radius, y), (x + w - radius, y + h), COLOR_LIGHT_GRAY, 2)
        cv2.rectangle(image, (x, y + radius), (x + w, y + h - radius), COLOR_LIGHT_GRAY, 2)
        cv2.circle(image, (x + radius, y + radius), radius, COLOR_LIGHT_GRAY, 2)
        cv2.circle(image, (x + w - radius, y + radius), radius, COLOR_LIGHT_GRAY, 2)
        cv2.circle(image, (x + radius, y + h - radius), radius, COLOR_LIGHT_GRAY, 2)
        cv2.circle(image, (x + w - radius, y + h - radius), radius, COLOR_LIGHT_GRAY, 2)


        # 繪製按鈕文字
        text_size = cv2.getTextSize(btn_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        cv2.putText(image, btn_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)

    return image

def draw_bounding_box(image, predictions):
    bounding_boxes_layer = np.zeros_like(image) # 獨立的圖層來繪製邊界框
    for char, confidence, bbox in predictions:
        x, y, w, h = bbox
        # 繪製邊界框
        cv2.rectangle(bounding_boxes_layer, (x-5, y-5), (x+w+5, y+h+5), ACCENT_COLOR, 2) # 增加邊框寬度
        # 顯示文字
        text_label = f"{char} ({confidence:.2f})"
        font_scale = 0.8
        font_thickness = 2
        text_size = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_x = max(0, x - 5) # 調整文字位置，避免出界
        text_y = max(text_size[1] + 10, y - 10) # 調整文字位置，確保在框上方
        
        # 繪製文字背景（讓文字更清晰）
        cv2.rectangle(bounding_boxes_layer, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0], text_y + 5), ACCENT_COLOR, -1)
        cv2.putText(bounding_boxes_layer, text_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_BLACK, font_thickness)
    return bounding_boxes_layer


# --- 新增的字母結果顯示視窗優化 ---
def update_letter_window(predictions, total_width):
    # 建立一個背景層
    window_height = 100
    letter_window = np.zeros((window_height, total_width, 3), dtype=np.uint8)
    
    # 繪製背景
    cv2.rectangle(letter_window, (0,0), (total_width, window_height), UI_BG_COLOR, -1)
    
    # 如果有預測結果，則顯示
    if predictions:
        # 將所有預測到的字母按順序排列，並合併成一個字串
        sorted_predictions = sorted(predictions, key=lambda p: p[2][0]) # 按照X座標排序
        recognized_sequence = "".join([p[0] for p in sorted_predictions])

        font_scale = 1.5 # 放大字體
        font_thickness = 3 # 加粗字體
        
        # 計算文字大小
        text_size = cv2.getTextSize(recognized_sequence, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        
        # 計算文字位置，使其居中顯示
        text_x = (total_width - text_size[0]) // 2
        text_y = (window_height + text_size[1]) // 2
        
        cv2.putText(letter_window, recognized_sequence, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, ACCENT_COLOR, font_thickness) # 使用強調色
    else:
        # 顯示提示訊息
        tip_text = "等待您的手寫輸入..."
        font_scale = 0.8
        font_thickness = 1
        text_size = cv2.getTextSize(tip_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_x = (total_width - text_size[0]) // 2
        text_y = (window_height + text_size[1]) // 2
        cv2.putText(letter_window, tip_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_LIGHT_GRAY, font_thickness)

    return letter_window


# --- 主程式碼 ---

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6)

cap = cv2.VideoCapture(0)

# 初始化畫布和繪圖相關變數
canvas = None
drawing_layer = None
current_points = []
paths = []
predictions = [] # 儲存預測結果
recognized_letters = [] # 儲存已辨識的字母序列

# 設置清除按鈕
button_width = 100
button_height = 50
button_margin = 20 # 按鈕之間的間距
clear_button_rect = (20, 480 - 60, button_width, button_height) # 調整按鈕位置
undo_button_rect = (20 + button_width + button_margin, 480 - 60, button_width, button_height)

buttons = {
    "清除": {"rect": clear_button_rect, "action": "clear"},
    "撤銷": {"rect": undo_button_rect, "action": "undo"}
}

# 追蹤鼠標點擊事件 (用於按鈕)
mouse_clicked = False
def mouse_callback(event, x, y, flags, param):
    global mouse_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_clicked = True
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_clicked = False

cv2.namedWindow('Hand Tracking')
cv2.setMouseCallback('Hand Tracking', mouse_callback)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1) # 水平翻轉
    image_height, image_width, _ = image.shape

    if canvas is None:
        canvas = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        drawing_layer = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        bounding_boxes_layer = np.zeros((image_height, image_width, 3), dtype=np.uint8) # 新增邊界框圖層

    # 將圖像轉換為 RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # 重置繪圖層和邊界框層
    drawing_layer.fill(0)
    bounding_boxes_layer.fill(0)

    # 繪製手部追蹤網格
    # draw_hand_tracking_grid(image, grid_counts, image_width, image_height) # 如果需要，取消註釋

    current_hand_landmarks = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            current_hand_landmarks = hand_landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=COLOR_BLUE, thickness=2, circle_radius=4), # 藍色點
                mp.solutions.drawing_utils.DrawingSpec(color=COLOR_WHITE, thickness=2, circle_radius=2)) # 白色線

            # 獲取食指尖端座標
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            tip_x = int(index_finger_tip.x * image_width)
            tip_y = int(index_finger_tip.y * image_height)

            # 獲取中指尖端座標
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            middle_x = int(middle_finger_tip.x * image_width)
            middle_y = int(middle_finger_tip.y * image_height)

            # 計算食指尖和中指尖之間的距離
            distance = np.linalg.norm(np.array([tip_x, tip_y]) - np.array([middle_x, middle_y]))

            # 偵測握拳狀態 (用於橡皮擦) - 所有手指尖和掌根距離較近
            is_fist = False
            if hand_landmarks.landmark[mp_hands.HandLandmark.WRIST] and \
               hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP] and \
               hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP] and \
               hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP] and \
               hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP] and \
               hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]:
                
                wrist = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x,
                                  hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y])
                thumb_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x,
                                      hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y])
                index_tip_norm = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                                           hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y])
                middle_tip_norm = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,
                                            hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y])
                ring_tip_norm = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x,
                                          hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y])
                pinky_tip_norm = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x,
                                           hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y])
                
                # 簡單判斷：所有手指尖到掌根的垂直距離是否都較小
                # 這是一個簡化的判斷，可能需要更精確的模型或規則
                if (index_tip_norm[1] > wrist[1] and middle_tip_norm[1] > wrist[1] and
                    ring_tip_norm[1] > wrist[1] and pinky_tip_norm[1] > wrist[1]):
                    is_fist = True

            # 橡皮擦功能
            if is_fist:
                # 橡皮擦中心點可以是手掌中心或某個指關節
                eraser_center_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width)
                eraser_center_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height)
                eraser_radius = 30 # 橡皮擦半徑
                
                # 繪製橡皮擦指示
                cv2.circle(image, (eraser_center_x, eraser_center_y), eraser_radius, COLOR_RED, 2)
                cv2.putText(image, "Eraser", (eraser_center_x + 30, eraser_center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_RED, 1)

                # 清除畫布上的區域
                # 創建一個白色圓形蒙版
                mask = np.zeros_like(canvas)
                cv2.circle(mask, (eraser_center_x, eraser_center_y), eraser_radius, (255, 255, 255), -1)
                
                # 使用蒙版來清除 canvas 和 paths 中的筆劃
                # 先清除畫布上的像素
                canvas = cv2.bitwise_and(canvas, cv2.bitwise_not(mask))

                # 從 paths 中移除被擦除的點
                new_paths = []
                for path in paths:
                    new_path = []
                    for point in path:
                        if np.linalg.norm(np.array(point) - np.array([eraser_center_x, eraser_center_y])) > eraser_radius:
                            new_path.append(point)
                    if new_path:
                        new_paths.append(new_path)
                paths = new_paths
                
                # 重置 current_points
                current_points = []
            else:
                # 判斷是否在繪圖
                if distance > 40: # 食指和中指分開，進行繪圖
                    if is_hand_in_recognition_area(hand_landmarks, recognition_area):
                        current_points.append((tip_x, tip_y))
                    # 重置之前的預測，因為正在書寫
                    predictions = []
                else: # 食指和中指靠近，停止繪圖
                    if current_points:
                        paths.append(list(current_points)) # 將當前筆劃添加到 paths
                        current_points = [] # 清空當前筆劃

    # 繪製已完成的筆劃
    for path in paths:
        for i in range(1, len(path)):
            cv2.line(canvas, path[i-1], path[i], DRAWING_COLOR, 5) # 使用繪圖顏色

    # 繪製當前筆劃 (如果正在繪圖)
    if current_points:
        for i in range(1, len(current_points)):
            cv2.line(drawing_layer, current_points[i-1], current_points[i], DRAWING_COLOR, 5)

    # 繪製偵測區域
    x_min, y_min, x_max, y_max = recognition_area
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), COLOR_YELLOW, 2) # 黃色框
    cv2.putText(image, "Recognition Area", (x_min + 10, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_YELLOW, 2)

    # 當手部不在畫面中或不在識別區域時，觸發辨識
    if not results.multi_hand_landmarks or not is_hand_in_recognition_area(current_hand_landmarks, recognition_area):
        if paths and not predictions: # 避免重複預測
            # 將所有筆劃合併到一個臨時圖像上進行辨識
            temp_drawing_for_prediction = np.zeros_like(canvas)
            for path in paths:
                for i in range(1, len(path)):
                    cv2.line(temp_drawing_for_prediction, path[i-1], path[i], (255, 255, 255), 5) # 用白色筆劃
            
            predictions = detect_shapes_and_predict(temp_drawing_for_prediction, letter_model, class_names)
            # print("Predicted:", predictions) # 偵錯用

    # 繪製邊界框和預測結果
    bounding_boxes_layer = draw_bounding_box(bounding_boxes_layer, predictions)


    # 繪製控制面板
    image = draw_control_panel(image, current_hand_landmarks, buttons)

    # 處理按鈕點擊
    if mouse_clicked:
        for btn_name, btn_info in buttons.items():
            x, y, w, h = btn_info['rect']
            # 檢查鼠標點擊是否在按鈕範圍內
            if x < mouse_x < x + w and y < mouse_y < y + h:
                if btn_info['action'] == "clear":
                    paths = []
                    current_points = []
                    canvas.fill(0)
                    drawing_layer.fill(0)
                    predictions = [] # 清除所有預測
                    recognized_letters = []
                    print("畫布已清除")
                elif btn_info['action'] == "undo":
                    if paths:
                        paths.pop() # 移除最後一個筆劃
                        canvas.fill(0) # 清空畫布
                        for path in paths: # 重新繪製所有剩餘的筆劃
                            for i in range(1, len(path)):
                                cv2.line(canvas, path[i-1], path[i], DRAWING_COLOR, 5)
                        predictions = [] # 重新觸發預測
                        recognized_letters = []
                        print("撤銷操作")
                mouse_clicked = False # 重置點擊狀態以避免連續觸發

    # 當前點上繪製十字準星
    if current_points:
        x, y = current_points[-1]
        crosshair_size = 15
        crosshair_thickness = 2
        cv2.circle(image, (x, y), 5, ACCENT_COLOR, -1)
        cv2.line(image, (x - crosshair_size, y), (x + crosshair_size, y), ACCENT_COLOR, crosshair_thickness)
        cv2.line(image, (x, y - crosshair_size), (x, y + crosshair_size), ACCENT_COLOR, crosshair_thickness)

    # 合并图像：即時攝像頭畫面、繪圖層和邊界框層
    # 將繪圖層和邊界框層疊加到畫布上，然後再疊加到攝像頭畫面
    overlay_drawing = cv2.addWeighted(canvas, 1, drawing_layer, 1, 0)
    overlay_bounding = cv2.addWeighted(overlay_drawing, 1, bounding_boxes_layer, 1, 0)
    
    # 疊加到主圖像上
    # 因為 overlay_bounding 和 image 都是 BGR，可以直接疊加
    combined_image = cv2.addWeighted(image, 1, overlay_bounding, 0.8, 0) # 疊加透明度調整

    # 更新並添加字母窗口
    letter_window = update_letter_window(predictions, combined_image.shape[1])
    final_image = np.vstack((combined_image, letter_window))

    # 顯示最終圖像
    cv2.imshow('Hand Tracking', final_image)

    # 獲取鼠標位置（用於按鈕懸停效果）
    mouse_x, mouse_y = 0, 0
    if cv2.getWindowProperty('Hand Tracking', cv2.WND_PROP_AUTOSIZE) >= 0: # 確保視窗存在
        mouse_x, mouse_y = cv2.getMouseProperty('Hand Tracking', cv2.EVENT_MOUSEMOVE)[0:2] # 獲取X,Y座標

    key = cv2.waitKey(1) & 0xFF

    # 檢查視窗是否被關閉 (X 按鈕)
    if cv2.getWindowProperty('Hand Tracking', cv2.WND_PROP_VISIBLE) < 1:
        break

    if key == 27:  # 按 ESC 鍵
        break

cap.release()
cv2.destroyAllWindows()