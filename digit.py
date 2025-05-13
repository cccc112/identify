import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import mediapipe as mp
import time
from tensorflow.keras.preprocessing.image import img_to_array

# 載入訓練好的模型
digit_model = tf.keras.models.load_model("C:/hand/best_model.h5")
symbol_model = tf.keras.models.load_model("C:/hand/symbol.h5")
#letter_model = tf.keras.models.load_model("C:/hand/model.h5")
'''
def delete_canvas_image(recognized_digits):
    for digits in recognized_digits:
        if digits:
            digits.pop(0)
    print("Earliest recognized digit deleted")
'''
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
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)# 將圖像轉換為灰度圖像
    image = cv2.resize(image, size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0# 將圖像像素值歸一化到0到1之間
    # 保存預處理後的圖像
    '''
    cv2.imwrite("C:/hand/image/Z/Z19.png", image[0] * 255)  # 反歸一化並保存圖像
    cv2.imshow("Preprocessed Image", image[0])  # 展示第一個預處理的圖像
    cv2.waitKey(0)  # 等待按鍵，然後關閉視窗
    '''
    return image

# 預處理圖像
def make_prediction(image, model):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)# 使用模型進行預測
    return np.argmax(prediction), np.max(prediction)# 返回預測的數字和概率
# 創建一個空白圖像來顯示數字
def update_digit_window(grid_digits):
    digit_image = np.zeros((400, 400, 3), dtype=np.uint8)
    text_color = (255, 255, 255)
    font_scale = 0.7
    thickness = 2
    '''
    labels = ['Units', 'Tens', 'Hundreds', 'Thousands', 'Ten Thousands', 'Hundred Thousands', 'Millions', 'Ten Millions']
    
    max_display_digits = 10
# 只顯示最近的數字
    for i, (label, digits) in enumerate(zip(labels, grid_digits)):
        display_digits = digits[-max_display_digits:]# 構建顯示文本
        text = f'{label} ({i+1}): {" ".join(map(str, display_digits))}'# 在圖像上繪製文本
        cv2.putText(digit_image, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
    '''
def draw_grid(canvas, num_rows=4, num_cols=2, color=(200, 200, 200), thickness=1):
    # 獲取畫布的高度和寬度
    height, width = canvas.shape[:2]
    # 計算每行和每列的高度和寬度
    row_height = height // num_rows
    col_width = width // num_cols
    # 繪製垂直線
    for x in range(0, width, col_width):
        cv2.line(canvas, (x, 0), (x, height), color, thickness)
    
    for y in range(0, height, row_height):
        cv2.line(canvas, (0, y), (width, y), color, thickness)
        '''
    for i in range(num_rows):
        for j in range(num_cols):
            grid_index = i * num_cols + j
            text_position = (j * col_width + 10, i * row_height + 30)
            text = f'{grid_index + 1}'
            cv2.putText(canvas, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        '''
def draw_hand_tracking_grid(image, num_rows=4, num_cols=2, color=(200, 200, 200), thickness=1):
    height, width = image.shape[:2]
    row_height = height // num_rows
    col_width = width // num_cols
    for x in range(0, width, col_width):
        cv2.line(image, (x, 0), (x, height), color, thickness)
    for y in range(0, height, row_height):
        cv2.line(image, (0, y), (width, y), color, thickness)

def draw_control_panel(image, width, height):
    # 設定控制面板的高度和顏色
    panel_height = 50
    panel_color = (240, 255, 255)
    button_color = (200, 200, 200)
    border_color = (255, 0, 0)  # Blue border color in BGR
    button_size = (100, 40)
    alpha = 0.7  # Transparency factor for buttons

    # 繪製控制面板背景
    cv2.rectangle(image, (0, 0), (width, panel_height), panel_color, 1)
    
    # 定義按鈕區域
    button_area_clear = (10, 5, 10 + button_size[0], panel_height - 5)
    button_area_save = (140, 5, 140 + button_size[0], panel_height - 5)
    button_area_delete = (270, 5, 270 + button_size[0], panel_height - 5)
    
    # 使用半透明的白色按鈕
    overlay = image.copy()
    '''
    for button_area in [button_area_clear, button_area_save, button_area_delete]:
        cv2.rectangle(overlay, button_area[:2], button_area[2:], button_color, 1)
        cv2.rectangle(overlay, button_area[:2], button_area[2:], border_color, 2)  # 畫外框
    '''
    for button_area in [button_area_clear]:
        cv2.rectangle(overlay, button_area[:2], button_area[2:], button_color, 1)
        cv2.rectangle(overlay, button_area[:2], button_area[2:], border_color, 2)  # 畫外框
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # 添加按鈕文字
    cv2.putText(image, 'Clear-c', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    '''
    cv2.putText(image, 'Save-s', (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(image, 'Delete-d', (280, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    '''
    return button_area_clear, button_area_save, button_area_delete
def get_grid_position(x, y, width, height, num_rows=4, num_cols=2):
    # 計算每個格子的寬度和高度
    grid_width = width // num_cols
    grid_height = height // num_rows
    # 計算點擊位置所在的列和行
    col = x // grid_width
    row = y // grid_height
    return row * num_cols + col

def draw_grid_info(image, grid_counts, width, height, num_rows=4, num_cols=2):
    grid_width = width // num_cols
    grid_height = height // num_rows
    for i in range(num_rows):
        for j in range(num_cols):
            grid_index = i * num_cols + j
            text_position = (j * grid_width + 10, i * grid_height + 30)
#繪製邊界框
def draw_bounding_box(image, points, color=(0, 245, 255), thickness=2):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    if len(x_coords) > 0 and len(y_coords) > 0:
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

def detect_shapes_and_predict(drawing_layer, digit_model, is_digit_mode):
    gray = cv2.cvtColor(drawing_layer, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kernel = np.ones((3,3),np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    # 獲取所有邊界框
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    
    # 合併邊界框的閾值
    merge_threshold = 70  # 可以根據需要調整此值
    
    merged_boxes = []
    for box in bounding_boxes:
        x, y, w, h = box
        # 檢查是否與已合併的框重疊
        merged = False
        for merged_box in merged_boxes:
            mx, my, mw, mh = merged_box
            # 計算邊界框的中心點
            box_center = (x + w // 2, y + h // 2)
            merged_center = (mx + mw // 2, my + mh // 2)
            # 計算距離
            distance = np.sqrt((box_center[0] - merged_center[0]) ** 2 + (box_center[1] - merged_center[1]) ** 2)
            if distance < merge_threshold:
                # 更新合併框的邊界
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
        # 預處理ROI
        preprocessed_roi = preprocess_image(cv2.cvtColor(cv2.resize(roi, (28, 28)), cv2.COLOR_GRAY2BGR))
        if is_digit_mode:
            model = digit_model
            prediction = model.predict(preprocessed_roi)
            predicted_label = np.argmax(prediction)
            predictions.append((box, predicted_label))
    
    # 根據x和y坐標進行排序
    predictions.sort(key=lambda x: (x[0][1], x[0][0]))
    return predictions
def draw_recognized_digits_window(image, recognized_digits, canvas_width, canvas_height, num_digits=8):
       # Define the position and size of the recognized digits window
    window_x = canvas_width + 10
    window_y = 10
    window_width = 200
    window_height = 400

def draw_recognized_text_window(image, recognized_texts, canvas_width, canvas_height, num_entries=8):
    window_x = canvas_width + 10
    window_y = 10
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
    
    # Display the window with recognized texts
    combined_image = np.hstack((image, text_image))
    return combined_image
def hand_tracking():
    alpha = 0.5   # 透明度因子
    is_drawing_enabled = False  # 初始狀態為不繪圖
    button_area_draw = (250, 230, 380, 270)  # 按鈕的區域 (x1, y1, x2, y2)
    erase_start_time = None  # 變量來跟踪橡皮擦觸發時間
    is_erasing = False  # 變量來跟踪是否在橡皮擦模式
    mp_hands = mp.solutions.hands# 初始化MediaPipe手部解決方案
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=2,
                           min_detection_confidence=0.4,
                           min_tracking_confidence=0.3)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0) # 打開攝像頭
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 設置攝像頭分辨率

    points = []  # 用於存儲手部點的列表
    paths = []  # 用於存儲繪圖路徑的列表
    drawing = False  # 繪圖狀態
    clear_canvas = False  # 清除畫布狀態
    hand_in_frame = False  # 手部是否在畫面中的狀態
    recognized_digits = [[] for _ in range(8)]  # 用於存儲識別數字的列表
    grid_counts = [0] * 8  # 用於存儲每個格子的計數

    canvas_width = 640
    canvas_height = 480
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    drawing_layer = np.zeros_like(canvas)
    bounding_boxes_layer = np.zeros_like(canvas)  # 創建邊界框層

    draw_grid(canvas, num_rows=4, num_cols=2, color=(200, 200, 200), thickness=1)# 在畫布上繪製網格
    #準星
    crosshair_size = 20
    crosshair_color = (255, 255, 255)
    crosshair_thickness = 2
    #橡皮擦
    erase_radius = 50
    erase_color = (0, 245, 255)
    #繪圖
    drawing_threshold = 50
    line_thickness = 4

    is_digit_mode = True

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.flip(image, 1)# 翻轉圖像
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)# 將圖像轉換為RGB格式
        image.flags.writeable = False  # 設置圖像為不可寫
        results = hands.process(image)  # 使用MediaPipe處理圖像
        image.flags.writeable = True  # 設置圖像為可寫
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 將圖像轉換回BGR格式

        # 繪製網格和控制面板
        draw_hand_tracking_grid(image, num_rows=4, num_cols=2, color=(200, 200, 200), thickness=1)
        button_area_clear, button_area_save, button_area_delete = draw_control_panel(image, canvas_width, canvas_height)

        if not is_drawing_enabled:
            # 繪製具有透明效果的「開始繪製」 按鈕
            border_color = (255, 0, 0)      # Blue border color in BGR
            border_thickness = 2
            overlay = image.copy()
            cv2.rectangle(overlay, button_area_draw[:2], button_area_draw[2:], (255, 255, 255), 1)
            cv2.rectangle(overlay, button_area_draw[:2], button_area_draw[2:], border_color, border_thickness)
            cv2.putText(overlay, 'Start Draw', (button_area_draw[0] + 10, button_area_draw[1] + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        current_points = []

        if results.multi_hand_landmarks:
            hand_in_frame = True
            if len(results.multi_hand_landmarks) > 1:
                drawing = False
            else:
                drawing = True

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                index_finger_tip = hand_landmarks.landmark[8]
                middle_finger_tip = hand_landmarks.landmark[12]
                x_index, y_index = int(index_finger_tip.x * image.shape[1]), int(index_finger_tip.y * image.shape[0])
                x_middle, y_middle = int(middle_finger_tip.x * image.shape[1]), int(middle_finger_tip.y * image.shape[0])

                x_index = np.clip(x_index, 10, image.shape[1] - 10)
                y_index = np.clip(y_index, 10, image.shape[0] - 10)

                current_points.append((x_index, y_index))

                #檢查是否按下了開始繪圖按鈕
                if not is_drawing_enabled and (button_area_draw[0] <= x_index <= button_area_draw[2] and 
                                               button_area_draw[1] <= y_index <= button_area_draw[3]):
                    is_drawing_enabled = True
                # 检查手指是否进入「Clear」按钮的区域
                if (button_area_clear[0] <= x_index <= button_area_clear[2] and 
                    button_area_clear[1] <= y_index <= button_area_clear[3]):
                    clear_canvas = True
                # 計算食指和中指之间的距離
                distance = np.sqrt((x_index - x_middle) ** 2 + (y_index - y_middle) ** 2)
                # 如果食指和中指靠近，則停止繪圖
                if distance < 40:  # 距離阈值可以调整
                    drawing = False
                    points = []  # 清除當前繪製點，停止繪圖
                else:
                    drawing = True
                # 檢查是否拳頭模式（所有手指折疊）
                fist_state = True
                for id in [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
                           mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP,
                           mp_hands.HandLandmark.PINKY_TIP]:
                    fingertip = hand_landmarks.landmark[id]
                    palm = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    if np.linalg.norm(np.array([fingertip.x, fingertip.y]) - np.array([palm.x, palm.y])) > 0.1:
                        fist_state = False
                        break
                # 如果檢測到拳头，進入橡皮擦模式
                if fist_state:
                    mid_x, mid_y = (x_index + x_middle) // 2, (y_index + y_middle) // 2
                    cv2.circle(image, (mid_x, mid_y), erase_radius, erase_color, 2)

                    if erase_start_time is None:
                        erase_start_time = time.time()
                    is_erasing = True
                if clear_canvas:
                    # Clear the canvas and reset the layers
                    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
                    drawing_layer = np.zeros_like(canvas)
                    bounding_boxes_layer = np.zeros_like(canvas)
                    
                    # Redraw the grid with num_rows=4, num_cols=2
                    draw_grid(canvas, num_rows=4, num_cols=2)
                    
                    clear_canvas = False

        else:
            if hand_in_frame:
                # 偵測形狀並根據模式預測數字或符號
                predictions = detect_shapes_and_predict(drawing_layer, digit_model if is_digit_mode else symbol_model, is_digit_mode)
                bounding_boxes_layer = np.zeros_like(canvas)  # 清除之前的邊界框層
                # 為每個偵測到的形狀繪製邊界框和預測文字
                for (box, prediction) in predictions:
                    x, y, w, h = box
                    cv2.rectangle(bounding_boxes_layer, (x, y), (x + w, y + h), (255, 245, 0), 2)
                    prediction_text = str(prediction)
                    cv2.putText(bounding_boxes_layer, prediction_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if drawing and len(current_points) > 0 and not is_erasing and is_drawing_enabled:
            points.extend(current_points)
            #繪製線條
        if len(points) > 1 and not is_erasing:
            path = []
            for i in range(1, len(points)):
                path.append(points[i - 1])
                cv2.line(drawing_layer, points[i - 1], points[i], (240, 202, 166), line_thickness)
                cv2.line(image, points[i - 1], points[i], (240, 202, 166), line_thickness)
            paths.append(path)
            points = [points[-1]]
        
        # 檢查是否需要執行橡皮擦操作
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
            is_erasing = False  # 完成擦除後重置橡皮擦模式

        if len(paths) == 0:
            bounding_boxes_layer = np.zeros_like(canvas)  # 如果沒有筆劃，隱藏邊界框

        # 合併圖像：首先是畫布，然後是繪圖層，最後是邊界框
        combined_image = np.hstack((image, canvas + drawing_layer + bounding_boxes_layer))
        # 準備繪圖層以進行數字或符號預測
        gray_layer = cv2.cvtColor(drawing_layer, cv2.COLOR_BGR2GRAY)
        resized_layer = cv2.resize(gray_layer, (28, 28))
        normalized_layer = resized_layer / 255.0
        reshaped_layer = normalized_layer.reshape(1, 28, 28, 1)
        # 根據模式預測數字或符號
        prediction = np.argmax(digit_model.predict(reshaped_layer), axis=-1) if is_digit_mode else np.argmax(symbol_model.predict(reshaped_layer), axis=-1)
        prediction_text = f'Predicted {"Digit" if is_digit_mode else "Symbol"}: {prediction[0]}'
        # 更新網格計數和識別的數字
        if len(current_points) > 0:
            x, y = current_points[-1]
            grid_index = get_grid_position(x, y, canvas_width, canvas_height)

            grid_counts[grid_index] += 1

            if len(points) > 1:
                recognized_digits[grid_index].append(prediction[0])
                draw_bounding_box(bounding_boxes_layer, points, color=(0, 255, 0), thickness=2)

            cv2.circle(image, (x, y), 5, (255, 255, 255), -1)
        # 在合併圖像上繪製網格信息
        draw_grid_info(combined_image, grid_counts, canvas_width, canvas_height)
        # 當前點上繪製十字準星
        if len(current_points) > 0:
            x, y = current_points[-1]
            cv2.circle(image, (x, y), 5, crosshair_color, -1)
            cv2.line(image, (x - crosshair_size, y), (x + crosshair_size, y), crosshair_color, crosshair_thickness)
            cv2.line(image, (x, y - crosshair_size), (x, y + crosshair_size), crosshair_color, crosshair_thickness)

         # 繪製識別的數字窗口
        draw_recognized_digits_window(combined_image, recognized_digits, canvas_width, canvas_height, num_digits=8)

        #cv2.putText(combined_image, prediction_text, (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 顯示最終圖像
        cv2.imshow('Hand Tracking', combined_image)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
            '''
        elif key == ord('s'):
            save_canvas_image(canvas + drawing_layer, paths)
            
        elif key == ord('d'):
            delete_canvas_image(recognized_digits)
        '''
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