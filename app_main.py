import cv2
import numpy as np
import time
from core.hand_tracker import HandTracker
from core.canvas import CanvasManager
from core.model_manager import ModelManager

def create_glass_overlay(image, rect, corner_radius=15, alpha=0.4, color=(30, 30, 30), border_color=(100, 100, 100)):
    """建立帶有圓角和邊框的毛玻璃(半透明)特效遮罩"""
    x1, y1, x2, y2 = rect
    
    # 取出 ROI
    roi = image[y1:y2, x1:x2]
    
    # 建立一個全黑背景，畫圓角矩形當作 Mask
    mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
    
    # OpenCV 的矩形畫法，自己刻一個簡易圓角 (利用多邊形與圓形組合，或簡單直接畫)
    # 簡易圓角矩形 (直接用 cv2 filled polygon 或多個圓與矩形)
    cv2.rectangle(mask, (corner_radius, 0), (x2-x1-corner_radius, y2-y1), 255, -1)
    cv2.rectangle(mask, (0, corner_radius), (x2-x1, y2-y1-corner_radius), 255, -1)
    cv2.circle(mask, (corner_radius, corner_radius), corner_radius, 255, -1)
    cv2.circle(mask, (x2-x1-corner_radius, corner_radius), corner_radius, 255, -1)
    cv2.circle(mask, (corner_radius, y2-y1-corner_radius), corner_radius, 255, -1)
    cv2.circle(mask, (x2-x1-corner_radius, y2-y1-corner_radius), corner_radius, 255, -1)
    
    # 毛玻璃模糊
    blurred_roi = cv2.GaussianBlur(roi, (21, 21), 0)
    
    # 套上顏色
    colored_roi = cv2.addWeighted(blurred_roi, 1-alpha, np.full_like(blurred_roi, color), alpha, 0)
    
    # 邊框
    border_mask = cv2.Canny(mask, 100, 200)
    border_mask = cv2.dilate(border_mask, np.ones((3,3), np.uint8), iterations=1)
    
    # 合成回原圖
    np.copyto(roi, colored_roi, where=(mask==255)[:, :, None])
    np.copyto(roi, np.full_like(roi, border_color), where=(border_mask>0)[:, :, None])
    
    image[y1:y2, x1:x2] = roi


def is_point_in_rect(pt, rect):
    x, y = pt
    rx1, ry1, rx2, ry2 = rect
    return rx1 <= x <= rx2 and ry1 <= y <= ry2

def draw_ar_ui(image, mode, recognized_text, hover_point=None, last_hovered_btn=None, hover_progress=0.0):
    """繪製美化版 AR 使用者介面"""
    width = image.shape[1]
    height = image.shape[0]
    
    # --- 繪製頂部按鈕 ---
    # 定義三個按鈕區塊
    btn_clear = (width - 240, 15, width - 130, 65)
    btn_backspace = (width - 110, 15, width - 15, 65)
    btn_mode = (15, 15, 140, 65)
    
    hovered_btn = None
    
    # 繪製 Mode 切換按鈕
    create_glass_overlay(image, btn_mode, color=(40, 40, 80))
    cv2.putText(image, f"Mode: {mode.upper()}", (btn_mode[0]+12, btn_mode[1]+33), cv2.FONT_HERSHEY_DUPLEX, 0.55, (200, 255, 200), 1, cv2.LINE_AA)
    
    # 繪製 Backspace 按鈕
    create_glass_overlay(image, btn_backspace, color=(80, 50, 40))
    cv2.putText(image, "<- Back", (btn_backspace[0]+15, btn_backspace[1]+33), cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 200, 255), 1, cv2.LINE_AA)
    
    # 繪製 Clear 按鈕
    create_glass_overlay(image, btn_clear, color=(80, 30, 30))
    cv2.putText(image, "Clear All", (btn_clear[0]+15, btn_clear[1]+33), cv2.FONT_HERSHEY_DUPLEX, 0.6, (150, 150, 255), 1, cv2.LINE_AA)
    
    # 處理 Hover 狀態 (畫出讀條或發光外框)
    if hover_point:
        for btn, name in [(btn_clear, 'clear'), (btn_backspace, 'backspace'), (btn_mode, 'mode')]:
            if is_point_in_rect(hover_point, btn):
                hovered_btn = name
                # 如果是目前正在 hover 的按鈕，繪製進度條外框
                if name == last_hovered_btn and hover_progress > 0:
                    thickness = 3
                    # 簡單的光暈發光效果
                    cv2.rectangle(image, btn[:2], btn[2:], (255, 255, 255), thickness, cv2.LINE_AA)
                    
                    # 畫一條小進度條在按鈕底部
                    p_w = int((btn[2]-btn[0]) * hover_progress)
                    cv2.line(image, (btn[0], btn[3]-2), (btn[0]+p_w, btn[3]-2), (0, 255, 0), 4, cv2.LINE_AA)
                break
    
    # --- 繪製底部文字輸入框 ---
    text_bar_height = 80
    text_bar_rect = (15, height - text_bar_height - 15, width - 15, height - 15)
    create_glass_overlay(image, text_bar_rect, alpha=0.3, color=(10, 10, 10))
    
    # 顯示文字 (置中或靠左美化)
    display_text = f"Output: {recognized_text}"
    cv2.putText(image, display_text, (text_bar_rect[0] + 20, text_bar_rect[1] + 50), 
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (50, 255, 100), 2, cv2.LINE_AA)
                
    return btn_clear, btn_backspace, btn_mode, hovered_btn

def main():
    print("[系統] 正在啟動空中手勢繪圖系統 (Aesthetics AR 升級版)...")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    tracker = HandTracker()
    canvas_mgr = CanvasManager(width=640, height=480, line_thickness=12)
    model_mgr = ModelManager()
    
    modes = ["digit", "letter", "symbol"]
    current_mode_idx = 0
    
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
        
        # 取得追蹤結果
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
            
            # 手勢判定：兩指距離大於 45 -> 畫筆，否則 -> 懸浮移動
            dist_fingers = np.sqrt((x_idx - x_mid)**2 + (y_idx - y_mid)**2)
            
            if dist_fingers > 45:
                is_drawing = True
                canvas_mgr.add_point((x_idx, y_idx))
                last_draw_time = time.time()
                # 畫筆準心 (實心圓)
                cv2.circle(processed_frame, (x_idx, y_idx), 8, (100, 255, 100), -1, cv2.LINE_AA)
                last_hovered_btn = None
            else:
                canvas_mgr.end_stroke()
                # 懸浮準心 (十字線，代表可選擇)
                cv2.circle(processed_frame, (x_idx, y_idx), 6, (200, 200, 200), -1, cv2.LINE_AA)
                cv2.circle(processed_frame, (x_idx, y_idx), 14, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            canvas_mgr.end_stroke()
            last_hovered_btn = None
            
        # 計算 Hover 進度
        current_time = time.time()
        hover_progress = 0.0
        if last_hovered_btn and not is_drawing:
            hover_progress = min(1.0, (current_time - btn_hover_start_time) / 1.0)
            
        # 繪製介面
        current_mode = modes[current_mode_idx]
        btns = draw_ar_ui(processed_frame, current_mode, recognized_text, hover_point, last_hovered_btn, hover_progress)
        hovered_btn = btns[3]
        
        # 處理按鈕懸停邏輯 (懸停 1.0 秒觸發)
        if hovered_btn and not is_drawing:
            if hovered_btn == last_hovered_btn:
                if current_time - btn_hover_start_time > 1.0 and current_time - button_cooldown > 0.8:
                    if hovered_btn == 'clear':
                        recognized_text = ""
                        canvas_mgr.clear()
                    elif hovered_btn == 'backspace':
                        recognized_text = recognized_text[:-1]
                        canvas_mgr.clear()
                    elif hovered_btn == 'mode':
                        current_mode_idx = (current_mode_idx + 1) % len(modes)
                        # 切換模式時給點反饋，清空畫布防誤判
                        canvas_mgr.clear()
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
        mask = cv2.cvtColor(canvas_mgr.drawing_layer, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        # 建立抗鋸齒融合
        brush_bgr = canvas_mgr.drawing_layer
        
        # 這裡採用簡單遮罩疊加，因為前面已經開了 LINE_AA，邊緣有半透明像素
        # 把非黑色的部分疊加上去
        roi_mask = mask > 0
        processed_frame[roi_mask] = brush_bgr[roi_mask]
        
        # 即時筆跡也要畫上去
        canvas_mgr.draw_current_stroke(processed_frame)
        
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(processed_frame, f"FPS: {int(fps)}", (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('AR Handwriting Keyboard', processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
