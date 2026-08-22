import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import mediapipe as mp
import time
from tensorflow.keras.preprocessing.image import img_to_array

# 载入训练好的模型
symbol_model = tf.keras.models.load_model("C:/hand/symbol.h5")
# 类别名称映射
class_names = ['+', '-', '*', '/', '=','!', '(', ')', 'sqrt', 'pi', 'sin', 'cos', 'tan', 'log', 'exp','!',',',"[","]",'{','}','alpha','ascii_124','beta','Delta','exists','forall',
               'forward_slash','gamma','geq','gt','in','infty','int','lambda','ldots','leq','lt','mu','neq','phi','pm','prime','rightarrow','sigma','sum','theta','times']  # 根据您的模型添加或删除符号

recognition_area = (100, 100, 540, 380)

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

def preprocess_image(image, size=(64, 64)):
    # 将图像转换为灰度图
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 反转图像颜色（如果需要）
    image = cv2.bitwise_not(image)
    # 调整大小以适应模型输入
    image = cv2.resize(image, size)
    # 转换回3通道BGR格式
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # 将图像转换为数组
    image = img_to_array(image)
    # 增加一个维度
    image = np.expand_dims(image, axis=0)
    # 归一化像素值
    image = image / 255.0
    # 保存预处理图像用于调试
    cv2.imwrite("C:/hand/test.png", image[0] * 255)
    return image

# 预处理图像
def make_prediction(image, model):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    return np.argmax(prediction), np.max(prediction)

def update_digit_window(grid_digits):
    digit_image = np.zeros((400, 400, 3), dtype=np.uint8)
    text_color = (255, 255, 255)
    font_scale = 0.7
    thickness = 2
    labels = ['Units', 'Tens', 'Hundreds', 'Thousands', 'Ten Thousands', 'Hundred Thousands', 'Millions', 'Ten Millions']
    max_display_digits = 10

    for i, (label, digits) in enumerate(zip(labels, grid_digits)):
        display_digits = digits[-max_display_digits:]
        text = f'{label} ({i+1}): {" ".join(map(str, display_digits))}'
        cv2.putText(digit_image, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
    return digit_image

def draw_grid(canvas, num_rows=4, num_cols=2, color=(200, 200, 200), thickness=1):
    height, width = canvas.shape[:2]
    row_height = height // num_rows
    col_width = width // num_cols
    for x in range(0, width, col_width):
        cv2.line(canvas, (x, 0), (x, height), color, thickness)
    
    for y in range(0, height, row_height):
        cv2.line(canvas, (0, y), (width, y), color, thickness)

def draw_hand_tracking_grid(image, num_rows=4, num_cols=2, color=(200, 200, 200), thickness=1):
    height, width = image.shape[:2]
    row_height = height // num_rows
    col_width = width // num_cols
    for x in range(0, width, col_width):
        cv2.line(image, (x, 0), (x, height), color, thickness)
    for y in range(0, height, row_height):
        cv2.line(image, (0, y), (width, y), color, thickness)

def draw_control_panel(image, width, height):
    panel_height = 50
    panel_color = (240, 255, 255)
    button_color = (200, 200, 200)
    border_color = (255, 0, 0)  
    button_size = (100, 40)
    alpha = 0.7  

    cv2.rectangle(image, (0, 0), (width, panel_height), panel_color, 1)
    
    button_area_clear = (10, 5, 10 + button_size[0], panel_height - 5)
    button_area_save = (140, 5, 140 + button_size[0], panel_height - 5)
    button_area_delete = (270, 5, 270 + button_size[0], panel_height - 5)
    
    overlay = image.copy()
    for button_area in [button_area_clear]:
        cv2.rectangle(overlay, button_area[:2], button_area[2:], button_color, 1)
        cv2.rectangle(overlay, button_area[:2], button_area[2:], border_color, 2)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    cv2.putText(image, 'Clear-c', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return button_area_clear, button_area_save, button_area_delete

def get_grid_position(x, y, width, height, num_rows=4, num_cols=2):
    grid_width = width // num_cols
    grid_height = height // num_rows
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

def draw_bounding_box(image, points, color=(0, 0, 255), thickness=2):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    if len(x_coords) > 0 and len(y_coords) > 0:
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

def detect_shapes_and_predict(drawing_layer, symbol_model):
    gray = cv2.cvtColor(drawing_layer, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kernel = np.ones((3,3),np.uint8)
    
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    
    merge_threshold = 70  
    
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
        preprocessed_roi = preprocess_image(cv2.cvtColor(cv2.resize(roi, (64, 64)), cv2.COLOR_GRAY2BGR))
        prediction = symbol_model.predict(preprocessed_roi)
        predicted_label = np.argmax(prediction)
        predictions.append((box, predicted_label))
    
    predictions.sort(key=lambda x: (x[0][1], x[0][0]))
    return predictions

def handle_hand_tracking(hand_landmarks, image):
    index_finger_tip = hand_landmarks.landmark[8]
    middle_finger_tip = hand_landmarks.landmark[12]
    x_index, y_index = int(index_finger_tip.x * image.shape[1]), int(index_finger_tip.y * image.shape[0])
    x_middle, y_middle = int(middle_finger_tip.x * image.shape[1]), int(middle_finger_tip.y * image.shape[0])

    x_index = np.clip(x_index, 10, image.shape[1] - 10)
    y_index = np.clip(y_index, 10, image.shape[0] - 10)

    return x_index, y_index, x_middle, y_middle

def detect_and_predict_image(image, shapes, model):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    return np.argmax(prediction), np.max(prediction)

def is_hand_in_recognition_area(current_points):
    if not current_points:
        return False
    x, y = current_points[-1]
    return (recognition_area[0] <= x <= recognition_area[2] and
            recognition_area[1] <= y <= recognition_area[3])

def update_symbol_window(predictions, width):
    max_symbols_per_row = 20
    symbol_height = 30
    symbol_width = width // max_symbols_per_row
    rows = (len(predictions) + max_symbols_per_row - 1) // max_symbols_per_row
    
    symbol_image = np.zeros((rows * symbol_height, width, 3), dtype=np.uint8)
    text_color = (255, 255, 255)
    font_scale = 0.5
    thickness = 1
    
    for i, (box, prediction) in enumerate(predictions):
        row = i // max_symbols_per_row
        col = i % max_symbols_per_row
        x = col * symbol_width + 5
        y = row * symbol_height + 20
        
        text = f'{class_names[prediction]}'
        cv2.putText(symbol_image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
    
    return symbol_image

def hand_tracking():
    recognized_symbols = []
    predictions = []
    prediction_text = ""
    alpha = 0.5
    is_drawing_enabled = False
    button_area_draw = (250, 230, 380, 270)
    erase_start_time = None
    is_erasing = False
    recognition_area = (100, 100, 540, 380)
    undo_area = (400, 10, 500, 50)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=2,
                           min_detection_confidence=0.4,
                           min_tracking_confidence=0.3)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

    points = []
    paths = []
    drawing = False
    clear_canvas = False
    hand_in_frame = False
    recognized_digits = [[] for _ in range(8)]
    grid_counts = [0] * 8

    canvas_width = 640
    canvas_height = 480
    canvas_color = (255, 255, 255)
    canvas = np.full((canvas_height, canvas_width, 3), canvas_color, dtype=np.uint8)
    drawing_layer = np.zeros_like(canvas)
    bounding_boxes_layer = np.zeros_like(canvas)

    draw_grid(canvas, num_rows=4, num_cols=2, color=(200, 200, 200), thickness=1)
    crosshair_size = 20
    crosshair_color = (255, 255, 255)
    crosshair_thickness = 2
    erase_radius = 50
    erase_color = (0, 245, 255)
    drawing_threshold = 55
    line_thickness = 3

    is_digit_mode = True
    undo_cooldown = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        draw_hand_tracking_grid(image, num_rows=4, num_cols=2, color=(200, 200, 200), thickness=1)
        button_area_clear, button_area_save, button_area_delete = draw_control_panel(image, canvas_width, canvas_height)
        cv2.rectangle(image, undo_area[:2], undo_area[2:], (0, 255, 0), 2)
        cv2.putText(image, 'Undo', (undo_area[0]+10, undo_area[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if not is_drawing_enabled:
            border_color = (255, 0, 0)
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
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                x_index, y_index, x_middle, y_middle = handle_hand_tracking(hand_landmarks, image)
                current_points.append((x_index, y_index))
                
                if not is_drawing_enabled and (button_area_draw[0] <= x_index <= button_area_draw[2] and
                                               button_area_draw[1] <= y_index <= button_area_draw[3]):
                    is_drawing_enabled = True

                if (button_area_clear[0] <= x_index <= button_area_clear[2] and
                    button_area_clear[1] <= y_index <= button_area_clear[3]):
                    clear_canvas = True

                if (undo_area[0] <= x_index <= undo_area[2] and
                    undo_area[1] <= y_index <= undo_area[3] and
                    undo_cooldown == 0):
                    if paths:
                        paths.pop()
                        drawing_layer = np.zeros_like(canvas)
                        for path in paths:
                            for i in range(1, len(path)):
                                cv2.line(drawing_layer, path[i - 1], path[i], (240, 202, 166), line_thickness)
                        undo_cooldown = 30

                distance = np.sqrt((x_index - x_middle) ** 2 + (y_index - y_middle) ** 2)
                if distance < 30:
                    drawing = False
                    points = []
                else:
                    drawing = True

                fist_state = True
                for id in [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
                           mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP,
                           mp_hands.HandLandmark.PINKY_TIP]:
                    fingertip = hand_landmarks.landmark[id]
                    palm = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    if np.linalg.norm(np.array([fingertip.x, fingertip.y]) - np.array([palm.x, palm.y])) > 0.1:
                        fist_state = False
                        break

                if fist_state:
                    mid_x, mid_y = (x_index + x_middle) // 2, (y_index + y_middle) // 2
                    cv2.circle(image, (mid_x, mid_y), erase_radius, erase_color, 2)
                    cv2.circle(drawing_layer, (mid_x, mid_y), erase_radius, (0, 0, 0), -1)
                    new_predictions = []
                    for pred in predictions:
                        box, _ = pred
                        x, y, w, h = box
                        if not ((x <= mid_x <= x+w) and (y <= mid_y <= y+h)):
                            new_predictions.append(pred)
                    predictions = new_predictions
                    if erase_start_time is None:
                        erase_start_time = time.time()
                    is_erasing = True

            if not is_hand_in_recognition_area(current_points):
                predictions = detect_shapes_and_predict(drawing_layer, symbol_model)
                for _, prediction in predictions:
                    recognized_symbols.append(class_names[prediction])
                bounding_boxes_layer = np.zeros_like(canvas)
                for (box, prediction) in predictions:
                    x, y, w, h = box
                    cv2.rectangle(bounding_boxes_layer, (x, y), (x + w, y + h), (0, 100, 255), 2)
                    prediction_text = class_names[prediction]
                    cv2.putText(bounding_boxes_layer, prediction_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        else:
            if hand_in_frame:
                hand_in_frame = False
                if clear_canvas:
                    canvas = np.full((canvas_height, canvas_width, 3), canvas_color, dtype=np.uint8)
                    drawing_layer = np.zeros_like(canvas)
                    bounding_boxes_layer = np.zeros_like(canvas)
                    draw_grid(canvas, num_rows=4, num_cols=2)
                    clear_canvas = False

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
        gray_layer = cv2.cvtColor(drawing_layer, cv2.COLOR_BGR2GRAY)
        resized_layer = cv2.resize(gray_layer, (64, 64))
        normalized_layer = resized_layer / 255.0
        reshaped_layer = normalized_layer.reshape(1, 64, 64, 1)
        
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
        
        combined_image = np.hstack((image, canvas + drawing_layer + bounding_boxes_layer))

        symbol_window = update_symbol_window(predictions, combined_image.shape[1])
        symbol_window_with_padding = np.zeros((symbol_window.shape[0], combined_image.shape[1], 3), dtype=np.uint8)
        symbol_window_with_padding[:, :symbol_window.shape[1]] = symbol_window
        final_image = np.vstack((combined_image, symbol_window_with_padding))

        for i, symbol in enumerate(recognized_symbols):
            cv2.putText(image, symbol, (10, 50 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        symbol_window = update_symbol_window(predictions, image.shape[1])
        combined_image = np.vstack((image, symbol_window))

        cv2.imshow('Hand Tracking', final_image)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC键
            break
        
        if undo_cooldown > 0:
            undo_cooldown -= 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    hand_tracking()