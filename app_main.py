import cv2
import numpy as np
import time
from core.hand_tracker import HandTracker
from core.canvas import CanvasManager
from core.model_manager import ModelManager

def draw_control_panel(image, mode):
    """繪製半透明控制面板 (App Navbar 風格)"""
    panel_height = 50
    width = image.shape[1]
    
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (width, panel_height), (30, 30, 30), -1)
    
    btn_clear = (10, 5, 110, 45)
    btn_mode = (130, 5, 270, 45)
    
    cv2.rectangle(overlay, btn_clear[:2], btn_clear[2:], (80, 80, 220), -1)
    cv2.rectangle(overlay, btn_mode[:2], btn_mode[2:], (80, 220, 80), -1)
    
    cv2.putText(overlay, 'Clear', (25, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(overlay, f"Mode: {mode.upper()}", (140, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
    return btn_clear, btn_mode

def is_point_in_rect(pt, rect):
    x, y = pt
    rx1, ry1, rx2, ry2 = rect
    return rx1 <= x <= rx2 and ry1 <= y <= ry2

def main():
    print("[系統] 正在啟動空中手勢繪圖系統 (UI 升級版)...")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    tracker = HandTracker()
    canvas_mgr = CanvasManager(width=640, height=480)
    model_mgr = ModelManager()
    
    current_mode = "digit"
    button_cooldown = 0
    prev_time = time.time()
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
            
        frame = cv2.flip(frame, 1)
        
        # 1. 關閉光線優化，避免雜訊干擾 MediaPipe
        results, processed_frame = tracker.process_frame(frame, optimize_lighting=False)
        processed_frame = tracker.draw_landmarks(processed_frame, results)
        btn_clear, btn_mode = draw_control_panel(processed_frame, current_mode)
        
        # 準備右側畫布與歷史面板 (640 + 250 = 890 寬度)
        final_canvas = canvas_mgr.get_combined_canvas()
        history_panel = np.zeros((480, 250, 3), dtype=np.uint8)
        right_side = np.hstack((final_canvas, history_panel))
        
        is_drawing = False
        if results.hand_landmarks:
            hand_landmarks = results.hand_landmarks[0]
            
            index_tip = hand_landmarks[8]
            middle_tip = hand_landmarks[12]
            
            x_idx = int(np.clip(index_tip.x * 640, 10, 630))
            y_idx = int(np.clip(index_tip.y * 480, 10, 470))
            x_mid = int(middle_tip.x * 640)
            y_mid = int(middle_tip.y * 480)
            
            current_time = time.time()
            if current_time - button_cooldown > 1.0:
                if is_point_in_rect((x_idx, y_idx), btn_clear):
                    canvas_mgr.clear()
                    button_cooldown = current_time
                elif is_point_in_rect((x_idx, y_idx), btn_mode):
                    current_mode = "letter" if current_mode == "digit" else "digit"
                    button_cooldown = current_time
            
            # 檢查是否為握拳 (橡皮擦)
            fist_state = True
            palm = hand_landmarks[0]
            fingertips = [4, 8, 12, 16, 20]
            for tip_idx in fingertips:
                pt = hand_landmarks[tip_idx]
                dist = np.sqrt((pt.x - palm.x)**2 + (pt.y - palm.y)**2)
                if dist > 0.15: 
                    fist_state = False
                    break
                    
            if fist_state:
                canvas_mgr.apply_eraser(x_idx, y_idx)
                # 左側攝影機畫大黃圈
                cv2.circle(processed_frame, (x_idx, y_idx), canvas_mgr.erase_radius, (0, 245, 255), 2)
                # 右側畫布也畫大黃圈 (Virtual Pointer)
                cv2.circle(right_side, (x_idx, y_idx), canvas_mgr.erase_radius, (0, 245, 255), 2)
            else:
                # 兩指距離大於 40 才判定為繪圖
                dist_fingers = np.sqrt((x_idx - x_mid)**2 + (y_idx - y_mid)**2)
                if dist_fingers > 40:  
                    is_drawing = True
                    canvas_mgr.add_point((x_idx, y_idx))
                    
                    # 虛擬準心：左側白十字，右側綠色準心
                    cv2.drawMarker(processed_frame, (x_idx, y_idx), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
                    cv2.circle(right_side, (x_idx, y_idx), 8, (0, 255, 0), -1)
                else:
                    canvas_mgr.end_stroke()
                    # 兩指靠攏暫停：右側灰色準心
                    cv2.circle(right_side, (x_idx, y_idx), 8, (150, 150, 150), -1)
                    
            # 預測觸發條件：剛結束一筆劃，或者提筆狀態下
            if not is_drawing and len(canvas_mgr.paths) > 0:
                predictions = model_mgr.predict_canvas_content(canvas_mgr.drawing_layer, mode=current_mode)
                
                # 每次重新繪製 Bounding Boxes
                canvas_mgr.bounding_boxes_layer = np.zeros_like(canvas_mgr.canvas)
                for box, label in predictions:
                    x, y, w, h = box
                    cv2.rectangle(canvas_mgr.bounding_boxes_layer, (x, y), (x+w, y+h), (255, 245, 0), 2)
                    cv2.putText(canvas_mgr.bounding_boxes_layer, str(label), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    
                    # 更新歷史紀錄 (只在提筆瞬間做一次，避免重複塞入，這邊簡化處理)
                    grid_idx = canvas_mgr.get_grid_position(x + w//2, y + h//2)
                    # 為了避免每幀都加，只有當有剛結束的筆畫時 (此處邏輯簡化，實際上應該綁定筆畫事件)
        else:
            canvas_mgr.end_stroke()
            
        canvas_mgr.draw_current_stroke(processed_frame)
        
        # 重新取得更新後的 Bounding Box layer
        final_canvas = canvas_mgr.get_combined_canvas()
        right_side = np.hstack((final_canvas, history_panel))
        
        # 繪製歷史紀錄到右側的 panel 區域
        canvas_mgr.draw_history_window(right_side)
        
        # 組合總畫面 (640 + 640 + 250 = 1530)
        combined_image = np.hstack((processed_frame, right_side))
        
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(combined_image, f"FPS: {int(fps)}", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow('Hand Tracking Application', combined_image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('z'):
            canvas_mgr.undo_stroke()
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
