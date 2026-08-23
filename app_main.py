import cv2
import numpy as np
import time
from core.hand_tracker import HandTracker
from core.canvas import CanvasManager
from core.model_manager import ModelManager

def is_point_in_rect(pt, rect):
    x, y = pt
    rx1, ry1, rx2, ry2 = rect
    return rx1 <= x <= rx2 and ry1 <= y <= ry2

def draw_ar_ui(image, mode, recognized_text, hover_point=None):
    """繪製 AR 使用者介面 (按鈕與文字框)"""
    width = image.shape[1]
    height = image.shape[0]
    overlay = image.copy()
    
    # --- 繪製頂部按鈕 ---
    btn_clear = (width - 220, 10, width - 130, 50)
    btn_backspace = (width - 120, 10, width - 10, 50)
    btn_mode = (10, 10, 110, 50)
    
    cv2.rectangle(overlay, btn_clear[:2], btn_clear[2:], (80, 80, 220), -1)
    cv2.rectangle(overlay, btn_backspace[:2], btn_backspace[2:], (80, 150, 220), -1)
    cv2.rectangle(overlay, btn_mode[:2], btn_mode[2:], (80, 220, 80), -1)
    
    cv2.putText(overlay, 'Clear', (btn_clear[0]+15, btn_clear[1]+28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(overlay, '< Back', (btn_backspace[0]+15, btn_backspace[1]+28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(overlay, f"{mode.upper()}", (btn_mode[0]+15, btn_mode[1]+28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # --- 繪製底部文字輸入框 ---
    text_bar_height = 80
    text_bar_rect = (0, height - text_bar_height, width, height)
    cv2.rectangle(overlay, text_bar_rect[:2], text_bar_rect[2:], (30, 30, 30), -1)
    
    # 顯示文字
    display_text = f"Result: {recognized_text}"
    cv2.putText(overlay, display_text, (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    # 合併半透明
    cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
    
    # 按鈕互動反饋 (如果懸停，畫出白色高亮框)
    hovered_btn = None
    if hover_point:
        for btn, name in [(btn_clear, 'clear'), (btn_backspace, 'backspace'), (btn_mode, 'mode')]:
            if is_point_in_rect(hover_point, btn):
                cv2.rectangle(image, btn[:2], btn[2:], (255, 255, 255), 3)
                hovered_btn = name
                break
                
    return btn_clear, btn_backspace, btn_mode, hovered_btn

def main():
    print("[系統] 正在啟動空中手勢繪圖系統 (AR 手寫板模式)...")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    tracker = HandTracker()
    canvas_mgr = CanvasManager(width=640, height=480)
    model_mgr = ModelManager()
    
    current_mode = "digit"
    recognized_text = ""
    
    # 狀態變數
    last_draw_time = time.time()
    button_cooldown = 0
    btn_hover_start_time = 0
    last_hovered_btn = None
    
    prev_time = time.time()
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
            
        frame = cv2.flip(frame, 1)
        
        # 取得追蹤結果 (關閉光線優化)
        results, processed_frame = tracker.process_frame(frame, optimize_lighting=False)
        processed_frame = tracker.draw_landmarks(processed_frame, results)
        
        is_drawing = False
        hover_point = None
        
        if results.hand_landmarks:
            hand_landmarks = results.hand_landmarks[0]
            
            index_tip = hand_landmarks[8]
            middle_tip = hand_landmarks[12]
            
            x_idx = int(np.clip(index_tip.x * 640, 10, 630))
            y_idx = int(np.clip(index_tip.y * 480, 10, 470))
            x_mid = int(middle_tip.x * 640)
            y_mid = int(middle_tip.y * 480)
            
            hover_point = (x_idx, y_idx)
            
            # 手勢判定：兩指距離大於 40 -> 畫筆，否則 -> 懸浮移動
            dist_fingers = np.sqrt((x_idx - x_mid)**2 + (y_idx - y_mid)**2)
            
            if dist_fingers > 40:
                is_drawing = True
                canvas_mgr.add_point((x_idx, y_idx))
                last_draw_time = time.time()
                # 畫筆準心
                cv2.circle(processed_frame, (x_idx, y_idx), 8, (0, 255, 0), -1)
                last_hovered_btn = None # 畫圖時不觸發按鈕
            else:
                canvas_mgr.end_stroke()
                # 懸浮準心
                cv2.circle(processed_frame, (x_idx, y_idx), 8, (150, 150, 150), -1)
                cv2.drawMarker(processed_frame, (x_idx, y_idx), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
        else:
            canvas_mgr.end_stroke()
            last_hovered_btn = None
            
        # 繪製介面與取得懸停狀態
        btns = draw_ar_ui(processed_frame, current_mode, recognized_text, hover_point)
        hovered_btn = btns[3]
        
        # 處理按鈕懸停邏輯 (懸停 1 秒觸發)
        current_time = time.time()
        if hovered_btn and not is_drawing:
            if hovered_btn == last_hovered_btn:
                if current_time - btn_hover_start_time > 1.0 and current_time - button_cooldown > 1.0:
                    if hovered_btn == 'clear':
                        recognized_text = ""
                        canvas_mgr.clear()
                    elif hovered_btn == 'backspace':
                        recognized_text = recognized_text[:-1]
                        canvas_mgr.clear()
                    elif hovered_btn == 'mode':
                        current_mode = "letter" if current_mode == "digit" else "digit"
                    button_cooldown = current_time
                    btn_hover_start_time = current_time # 重置計時
            else:
                last_hovered_btn = hovered_btn
                btn_hover_start_time = current_time
        else:
            last_hovered_btn = None
            
        # --- 自動分字邏輯 (Auto-Segmentation) ---
        # 如果畫布上有東西，且超過 1.2 秒沒有下筆
        if not is_drawing and canvas_mgr.has_content() and (current_time - last_draw_time > 1.2):
            # 進行預測
            predictions = model_mgr.predict_canvas_content(canvas_mgr.drawing_layer, mode=current_mode)
            
            # 將預測結果串接
            for box, label in predictions:
                recognized_text += str(label)
                
            # 預測完自動清空畫布，準備寫下一個字
            canvas_mgr.clear()
        
        # 將筆跡疊加到畫面上 (AR 效果)
        # 把黑底的 drawing_layer 疊加 (只保留非黑色的線條)
        mask = cv2.cvtColor(canvas_mgr.drawing_layer, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        processed_frame[mask > 0] = canvas_mgr.drawing_layer[mask > 0]
        
        # 即時筆跡也要畫上去
        canvas_mgr.draw_current_stroke(processed_frame)
        
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(processed_frame, f"FPS: {int(fps)}", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow('AR Handwriting Keyboard', processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
