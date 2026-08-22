import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import mediapipe as mp
import time
from tensorflow.keras.preprocessing.image import img_to_array

# --- 輔助函式：繪製圓角矩形 ---
def draw_rounded_rectangle(img, pt1, pt2, color, thickness, radius=1, filled=False):
    """繪製一個帶有圓角的矩形，可選擇是否填滿"""
    x1, y1 = pt1
    x2, y2 = pt2

    # 確保 pt1 是左上角, pt2 是右下角
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1

    # 圓角半徑不能超過矩形的一半大小
    radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)

    if filled:
        # 繪製四個角落的圓形
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)

        # 繪製中心的大矩形和兩側的小矩形
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    else:
        # 確保厚度是正數，避免錯誤
        if thickness <= 0: return

        # 繪製四個角落的圓弧
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
        
        # 繪製四條邊線
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)

# --- 核心辨識邏輯 ---
def preprocess_image(image, size=(28, 28)):
    """預處理圖像以供模型辨識"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

def detect_shapes_and_predict(drawing_layer, digit_model):
    """偵測繪圖層中的形狀並進行數字辨識"""
    gray = cv2.cvtColor(drawing_layer, cv2.COLOR_BGR2GRAY)
    # 使用較低的閾值來確保能捕捉到較淡的筆觸
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return [], ""

    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    
    # 根據筆劃位置合併相近的邊界框，形成完整的數字
    merge_threshold = 60  
    bounding_boxes.sort(key=lambda box: box[1]) 
    
    merged_boxes = []
    while bounding_boxes:
        base_box = bounding_boxes.pop(0)
        x, y, w, h = base_box
        
        other_boxes = []
        for other_box in bounding_boxes:
            ox, oy, ow, oh = other_box
            # 判斷兩個框是否足夠接近以進行合併
            is_close_x = abs((x + w/2) - (ox + ow/2)) < merge_threshold
            is_close_y = abs((y + h/2) - (oy + oh/2)) < merge_threshold
            
            if is_close_x and is_close_y:
                x = min(x, ox)
                y = min(y, oy)
                w = max(x + w, ox + ow) - x
                h = max(y + h, oy + oh) - y
            else:
                other_boxes.append(other_box)
        
        merged_boxes.append((x, y, w, h))
        bounding_boxes = other_boxes

    predictions = []
    for box in merged_boxes:
        x, y, w, h = box
        roi = gray[y:y+h, x:x+w]
        if roi.size == 0: continue
        
        # 為了給數字留出足夠的邊界，以便模型更好地辨識
        bordered_roi = cv2.copyMakeBorder(roi, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[0,0,0])
        preprocessed_roi = preprocess_image(cv2.cvtColor(bordered_roi, cv2.COLOR_GRAY2BGR))
        
        prediction = digit_model.predict(preprocessed_roi, verbose=0)
        predicted_label = np.argmax(prediction)
        predictions.append({'box': box, 'label': predicted_label})
    
    predictions.sort(key=lambda p: p['box'][0]) # 依據 X 座標排序
    result_text = "".join([str(p['label']) for p in predictions])
    return predictions, result_text


def hand_tracking():
    # --- UI 美化設定 ---
    COLOR_BG = (30, 30, 30)           # 深灰背景
    COLOR_CANVAS = (45, 45, 45)       # 畫布背景
    COLOR_PRIMARY = (128, 0, 0)       # 主題深藍 (用於辨識框)
    COLOR_ACCENT = (255, 255, 150)    # 繪圖淺藍色 (用於繪圖線條)
    COLOR_TEXT = (240, 240, 240)      # 文字白
    COLOR_GRID = (60, 60, 60)         # 網格灰
    COLOR_ERASER = (0, 215, 255)      # 橡皮擦金色
    ALPHA = 0.8 # UI 疊加層的透明度
    
    # --- MediaPipe 初始化 ---
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=1,
                           min_detection_confidence=0.6,
                           min_tracking_confidence=0.6)
    mp_drawing = mp.solutions.drawing_utils
    # --- 自定義手的節點與線條顏色 ---
    # MediaPipe 預設是 BGR，所以這裡的顏色也應該是 BGR 格式
    drawing_spec_connections = mp_drawing.DrawingSpec(color=(255, 191, 0), thickness=2) # 淺藍色線條
    drawing_spec_landmarks = mp_drawing.DrawingSpec(color=(255, 100, 0), thickness=-1, circle_radius=4) # 飽和藍色節點

    # --- 攝影機與畫布設定 ---
    cap = cv2.VideoCapture(0)
    # 嘗試設定攝影機尺寸，但重要的是要獲取實際設定的尺寸
    CAM_WIDTH_REQUEST = 1024 # 請求的寬度
    CAM_HEIGHT_REQUEST = 576 # 請求的高度
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH_REQUEST)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT_REQUEST)

    # 獲取攝影機實際的輸出尺寸
    actual_cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"攝影機實際解析度：{actual_cam_width}x{actual_cam_height}") # 打印實際解析度方便調試

    # 定義視窗各部分的尺寸，確保它們與攝影機實際輸出尺寸匹配
    WINDOW_WIDTH = actual_cam_width // 2 # 畫布寬度為攝影機寬度的一半
    WINDOW_HEIGHT = actual_cam_height    # 畫布高度與攝影機實際高度一致
    WINDOW_NAME = 'Handwriting Recognition' # 定義視窗名稱

    # 使用實際獲取到的尺寸來初始化畫布層
    canvas = np.full((WINDOW_HEIGHT, WINDOW_WIDTH, 3), COLOR_CANVAS, dtype=np.uint8)
    drawing_layer = np.zeros_like(canvas)
    ui_layer = np.zeros_like(canvas) # 用於繪製按鈕、結果面板等 UI 元素

    # 繪製畫布上的網格線
    for x in range(0, WINDOW_WIDTH, 50):
        cv2.line(canvas, (x, 0), (x, WINDOW_HEIGHT), COLOR_GRID, 1)
    for y in range(0, WINDOW_HEIGHT, 50):
        cv2.line(canvas, (0, y), (WINDOW_WIDTH, y), COLOR_GRID, 1)

    # --- 狀態變數 ---
    app_interaction_active = False # 控制應用程式是否處於互動狀態 (顯示提示、允許繪圖/擦除)
    is_drawing_stroke = False # 控制是否正在繪製一條連續的筆劃 (筆尖按下)
    is_erasing = False # 控制是否在橡皮擦模式 (握拳)
    last_hand_presence = False # 追蹤上一幀手部是否在畫面中，用於觸發辨識
    paths = [] # 儲存已繪製的線條路徑，每個元素是一個筆劃 (多個點)
    current_stroke_points = [] # 儲存當前正在繪製的點序列
    predictions = [] # 儲存辨識結果 (數字及其邊界框)
    result_text = "" # 儲存最終辨識的數字文字
    erase_radius = 40 # 橡皮擦的半徑
    line_thickness = 5 # 繪圖線條的粗細
    
    # 載入模型
    try:
        digit_model = tf.keras.models.load_model("C:/hand/best_model.h5")
    except Exception as e:
        print(f"錯誤：無法載入模型，請檢查路徑是否正確。 {e}")
        error_screen = np.full((300, 800, 3), COLOR_CANVAS, dtype=np.uint8)
        cv2.putText(error_screen, "Model Not Found!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(error_screen, "Please check the model path in the code.", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        while True:
            cv2.imshow(WINDOW_NAME, error_screen)
            if cv2.waitKey(1) & 0xFF == 27 or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()
        return

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # --- 影像前處理 ---
        frame = cv2.flip(frame, 1) # 左右翻轉圖像，使其符合鏡像模式
        
        # 從攝影機幀中切割出左半部分作為手部追蹤區域
        image = frame[:, :WINDOW_WIDTH] 
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 轉換為 RGB 格式供 MediaPipe 處理
        results = hands.process(rgb_image) # 處理圖像以偵測手部
        
        ui_overlay = image.copy() # 用於繪製 UI 元素的疊加層，避免直接修改原始圖像
        
        # 定義按鈕位置
        clear_btn_pos = (20, 20, 140, 70)
        undo_btn_pos = (clear_btn_pos[2] + 20, 20, clear_btn_pos[2] + 20 + (clear_btn_pos[2] - clear_btn_pos[0]), 70)
        finger_pos = None # 用於儲存食指尖端的位置

        # --- 手部偵測與手勢邏輯 ---
        hand_in_frame = bool(results.multi_hand_landmarks) # 檢查手部是否在畫面中
        
        # 重置狀態
        is_erasing = False
        
        if hand_in_frame:
            hand_landmarks = results.multi_hand_landmarks[0]
            # --- 使用自定義顏色繪製手部骨架 ---
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=drawing_spec_landmarks, # 應用節點顏色
                connection_drawing_spec=drawing_spec_connections # 應用線條顏色
            )

            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST] # 新增腕部地標

            # 將 MediaPipe 正規化座標轉換為像素座標
            x_index, y_index = int(index_finger_tip.x * WINDOW_WIDTH), int(index_finger_tip.y * WINDOW_HEIGHT)
            x_middle, y_middle = int(middle_finger_tip.x * WINDOW_WIDTH), int(middle_finger_tip.y * WINDOW_HEIGHT)
            finger_pos = (x_index, y_index) # 更新食指位置

            distance_index_middle = np.sqrt((x_index - x_middle)**2 + (y_index - y_middle)**2) # 食指和中指距離

            # --- 握拳手勢檢測 (橡皮擦) ---
            # 判斷是否握拳：主要檢測指尖是否相對其根部關節捲曲
            fist_state = True
            fingers_to_check = [
                mp_hands.HandLandmark.INDEX_FINGER_TIP,
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                mp_hands.HandLandmark.RING_FINGER_TIP,
                mp_hands.HandLandmark.PINKY_TIP
            ]
            mcp_joints = [ # 指根關節
                mp_hands.HandLandmark.INDEX_FINGER_MCP,
                mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
                mp_hands.HandLandmark.RING_FINGER_MCP,
                mp_hands.HandLandmark.PINKY_MCP
            ]
            
            # 檢查除了拇指之外的所有指尖是否都「低於」其指根關節 (表示捲曲)
            for i in range(len(fingers_to_check)):
                tip_y = hand_landmarks.landmark[fingers_to_check[i]].y
                mcp_y = hand_landmarks.landmark[mcp_joints[i]].y
                # 如果指尖的 Y 座標明顯小於其 MCP 關節的 Y 座標 (在畫面翻轉後，y 越大越下方)
                # 或者說，指尖距離手掌根部 (wrist) 的 Y 距離遠大於其 MCP 關節距離 wrist 的 Y 距離
                # 簡單來說，如果指尖沒有足夠地「捲曲」到接近手掌根部，就不算握拳
                if tip_y < mcp_y * 0.95: # 這裡的 0.95 是個經驗值，可能需要調整
                    fist_state = False
                    break
            
            # 輔助判斷：拇指是否也收攏
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP] # 拇指中間關節
            
            # 判斷拇指尖是否靠近拇指中間關節或手掌，而非伸直
            if abs(thumb_tip.x - thumb_ip.x) * WINDOW_WIDTH > 30: # 拇指伸直時 x 距離會較大
                 fist_state = False
            # 更進一步判斷拇指尖是否在其他手指的「內側」
            if thumb_tip.x < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x:
                fist_state = False # 如果拇指伸向手掌外側，不算握拳

            # --- 應用手勢邏輯 ---
            if fist_state: # 握拳：啟用橡皮擦
                is_erasing = True
                is_drawing_stroke = False # 停止繪圖
                
                if current_stroke_points: # 如果有未完成的筆劃，先保存
                    paths.append(list(current_stroke_points))
                    current_stroke_points = [] # 清空當前筆劃點
                    predictions, result_text = detect_shapes_and_predict(drawing_layer, digit_model) # 觸發辨識

                # 橡皮擦視覺效果：在手腕位置繪製橡皮擦圓圈
                erase_center_x = int(wrist.x * WINDOW_WIDTH)
                erase_center_y = int(wrist.y * WINDOW_HEIGHT)
                cv2.circle(image, (erase_center_x, erase_center_y), erase_radius, COLOR_ERASER, 2) # 攝影機畫面上的圓圈
                cv2.circle(drawing_layer, (erase_center_x, erase_center_y), erase_radius, (0,0,0), -1) # 實際擦除繪圖層
                
                # 重新評估 paths (簡化處理：如果橡皮擦觸及到某個筆劃的任何部分，則該筆劃將被移除)
                # 更精確的橡皮擦功能需要更複雜的線段分割和點判斷邏輯。
                # 目前保持較簡單的處理：只要筆劃上的任何一點被擦除，該筆劃即被視為失效。
                updated_paths = []
                for path_segment in paths:
                    keep_segment = True
                    for px, py in path_segment:
                        dist_to_eraser = np.sqrt((px - erase_center_x)**2 + (py - erase_center_y)**2)
                        if dist_to_eraser < erase_radius:
                            keep_segment = False
                            break
                    if keep_segment:
                        updated_paths.append(path_segment)
                paths = updated_paths
                predictions, result_text = detect_shapes_and_predict(drawing_layer, digit_model) # 橡皮擦後重新辨識

            elif distance_index_middle <= 50: # 食指中指合併 (抬筆)
                is_erasing = False # 不是橡皮擦模式
                if is_drawing_stroke: # 如果之前正在繪圖，則此為筆劃結束
                    if current_stroke_points:
                        paths.append(list(current_stroke_points)) # 將完成的筆劃加入 paths
                    current_stroke_points = [] # 清空當前筆劃點
                    predictions, result_text = detect_shapes_and_predict(drawing_layer, digit_model) # 觸發辨識
                is_drawing_stroke = False # 停止繪圖

            else: # 手指張開 (繪圖) 且不是握拳
                is_erasing = False # 不是橡皮擦模式
                is_drawing_stroke = True # 啟用繪圖
                current_stroke_points.append((x_index, y_index)) # 將當前點加入筆劃


        # --- 處理手部離開畫面時的辨識邏輯 ---
        # 當手從畫面中消失時，如果之前有正在繪圖的筆劃，則將其保存並觸發辨識
        if last_hand_presence and not hand_in_frame:
            if current_stroke_points: 
                paths.append(list(current_stroke_points))
                current_stroke_points = []
            if paths: 
                predictions, result_text = detect_shapes_and_predict(drawing_layer, digit_model)
        
        last_hand_presence = hand_in_frame # 更新手部在畫面中的狀態

        # --- 繪製當前正在畫的線條 ---
        if is_drawing_stroke and not is_erasing and len(current_stroke_points) > 1:
            # 只在繪圖模式且非橡皮擦模式下繪製
            cv2.line(drawing_layer, current_stroke_points[-2], current_stroke_points[-1], COLOR_ACCENT, line_thickness)
            # 這裡只繪製最新的線段，筆劃的完成和添加到 paths 由手勢邏輯控制

        # --- UI 按鈕與互動 ---
        # 清除按鈕
        clear_hover = finger_pos and (clear_btn_pos[0] < finger_pos[0] < clear_btn_pos[2] and clear_btn_pos[1] < finger_pos[1] < clear_btn_pos[3])
        clear_color = COLOR_PRIMARY if clear_hover else COLOR_TEXT
        draw_rounded_rectangle(ui_overlay, clear_btn_pos[:2], clear_btn_pos[2:], clear_color, 2, radius=15, filled=clear_hover)
        cv2.putText(ui_overlay, 'Clear (C)', (clear_btn_pos[0]+15, clear_btn_pos[1]+45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, clear_color, 2)
        if clear_hover and not is_erasing and app_interaction_active: # 只有在互動啟用且非擦除模式下觸發
            drawing_layer.fill(0) # 清空繪圖層
            paths = [] # 清空所有路徑
            predictions = [] # 清空辨識結果
            result_text = "" # 清空結果文字
            current_stroke_points = [] # 清空當前筆劃點
            time.sleep(0.2) # 增加延遲避免重複觸發

        # 撤銷按鈕
        undo_hover = finger_pos and (undo_btn_pos[0] < finger_pos[0] < undo_btn_pos[2] and undo_btn_pos[1] < finger_pos[1] < undo_btn_pos[3])
        undo_color = COLOR_PRIMARY if undo_hover else COLOR_TEXT
        draw_rounded_rectangle(ui_overlay, undo_btn_pos[:2], undo_btn_pos[2:], undo_color, 2, radius=15, filled=undo_hover)
        cv2.putText(ui_overlay, 'Undo (Z)', (undo_btn_pos[0]+20, undo_btn_pos[1]+45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, undo_color, 2)
        if undo_hover and not is_erasing and app_interaction_active: # 只有在互動啟用且非擦除模式下觸發
            if paths:
                paths.pop() # 移除最後一個筆劃
                drawing_layer.fill(0) # 清空畫布
                # 重新繪製所有剩餘的路徑
                for path in paths:
                    for i in range(1, len(path)):
                        cv2.line(drawing_layer, path[i-1], path[i], COLOR_ACCENT, line_thickness)
                # 重新辨識以更新結果
                predictions, result_text = detect_shapes_and_predict(drawing_layer, digit_model)
            time.sleep(0.2) # 增加延遲避免重複觸發

        # 如果手不在畫面中且應用程式未啟用互動，顯示提示
        if not app_interaction_active and not hand_in_frame:
            cv2.putText(image, "Bring your hand into the frame to start", (30, WINDOW_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2)
            if hand_in_frame: # 當手進入畫面時，啟用應用程式互動
                app_interaction_active = True
        
        # 將 UI 疊加層與原始圖像合併，實現透明效果
        image = cv2.addWeighted(ui_overlay, ALPHA, image, 1 - ALPHA, 0)
        
        # --- 繪製畫布上的辨識結果 ---
        ui_layer.fill(0) # 清空 UI 疊加層，準備繪製新的 UI 元素
        
        # 繪製辨識結果的邊界框和標籤
        if predictions:
            for p in predictions:
                x, y, w, h = p['box']
                # 繪製圓角矩形邊界框
                draw_rounded_rectangle(ui_layer, (x, y), (x + w, y + h), COLOR_PRIMARY, 2, radius=5)
                # 確保文字不會超出邊界或與邊框重疊
                text_x = x
                text_y = y - 10 if y - 10 > 0 else y + h + 20 # 調整文字位置
                cv2.putText(ui_layer, str(p['label']), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_PRIMARY, 2)

        # 繪製結果顯示面板
        output_panel_y = WINDOW_HEIGHT - 100
        draw_rounded_rectangle(ui_layer, (10, output_panel_y), (WINDOW_WIDTH - 10, WINDOW_HEIGHT - 10), COLOR_BG, thickness=0, radius=15, filled=True)
        cv2.putText(ui_layer, "Result:", (30, output_panel_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_TEXT, 2)
        cv2.putText(ui_layer, result_text, (200, output_panel_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_ACCENT, 3)

        # --- 合併與顯示畫面 ---
        final_canvas = cv2.add(canvas, drawing_layer) # 畫布和繪圖層合併
        final_canvas = cv2.add(final_canvas, ui_layer) # 再與 UI 疊加層合併
        
        # 水平堆疊攝影機畫面和畫布畫面
        combined_image = np.hstack((image, final_canvas))
        cv2.imshow(WINDOW_NAME, combined_image) # 顯示最終畫面

        # --- 鍵盤與視窗控制 ---
        key = cv2.waitKey(1) & 0xFF
        # 退出鍵 (Esc) 或關閉視窗按鈕
        if key == 27 or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break
        elif key == ord('z'): # 撤銷
            if paths:
                paths.pop()
                drawing_layer.fill(0)
                for path in paths:
                    for i in range(1, len(path)):
                        cv2.line(drawing_layer, path[i - 1], path[i], COLOR_ACCENT, line_thickness)
                predictions, result_text = detect_shapes_and_predict(drawing_layer, digit_model) # 撤銷後重新辨識
        elif key == ord('c'): # 清除
            drawing_layer.fill(0)
            paths = []
            predictions = []
            result_text = ""
            current_stroke_points = [] # 清除鍵盤清除時也要清空當前筆劃點

    cap.release() # 釋放攝影機資源
    cv2.destroyAllWindows() # 關閉所有 OpenCV 視窗

if __name__ == "__main__":
    hand_tracking()
