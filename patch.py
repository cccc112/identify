import re

with open('C:/hand/app_main.py', 'r', encoding='utf-8') as f:
    code = f.read()

start_str = "        hovered_btn, gaze_hovered_btn = draw_ui("
end_str = "                    btn_cooldown = curr_time"

start_idx = code.find(start_str)
end_idx = code.find(end_str, start_idx)

if start_idx != -1 and end_idx != -1:
    new_block = """        hovered_btn, gaze_hovered_btn = draw_ui(
            disp, mode_name, palette_idx, thickness_idx, active_tool, gesture_name,
            recognized_text, hover_point, last_hovered_btn,
            hover_progress, ar_ans=ar_ans_display, gaze_pt=gaze_pt,
            track_mode_str=TRACK_MODES[track_mode_idx], active_menu=active_menu
        )
        
        # ── Blink 觸發 UI ─────────────────────────────────
        if is_blinking and gaze_hovered_btn and (curr_time - btn_cooldown > 1.0):
            hovered_btn = gaze_hovered_btn # 模擬成按鈕被按下
            hover_start_time = 0.0 # bypass hover time check
            last_hovered_btn = hovered_btn

        # ── Hover 按鈕觸發 ────────────────────────────────
        if hovered_btn and not is_drawing:
            if hovered_btn == last_hovered_btn:
                req_time = 0.6 if hovered_btn.startswith('menu_') else 1.0
                if (curr_time - hover_start_time > req_time and
                        curr_time - btn_cooldown > 0.8):
                    
                    if hovered_btn == 'clear':
                        recognized_text = ""
                        canvas.clear()
                        _recognized_path_cnt = 0
                        active_menu = None
                    elif hovered_btn == 'back':
                        recognized_text = recognized_text[:-1]
                        active_menu = None
                    elif hovered_btn in ('mode', 'size', 'track'):
                        active_menu = hovered_btn if active_menu != hovered_btn else None
                    elif hovered_btn.startswith('menu_mode_'):
                        mode_idx = int(hovered_btn.split('_')[-1])
                        canvas.clear()
                        _recognized_path_cnt = 0
                        active_menu = None
                    elif hovered_btn.startswith('menu_size_'):
                        thickness_idx = int(hovered_btn.split('_')[-1])
                        canvas.line_thickness = THICKNESSES[thickness_idx]
                        active_menu = None
                    elif hovered_btn.startswith('menu_track_'):
                        track_mode_idx = int(hovered_btn.split('_')[-1])
                        active_menu = None
                        if TRACK_MODES[track_mode_idx] == 'HEAD' and mode_name != 'type':
                            try:
                                mode_idx = MODES.index('type')
                            except: pass
                    elif hovered_btn == 'ink':
                        color_picker_active = not color_picker_active
                        color_hover_start = curr_time
                        active_menu = None
                    elif hovered_btn == 'tool':
                        active_tool = 'bucket' if active_tool == 'brush' else 'brush'
                        active_menu = None
                    elif hovered_btn == 'save':
                        if mode_name == 'art':
                            from datetime import datetime
                            import os
                            save_dir = "C:/hand/saved_art"
                            os.makedirs(save_dir, exist_ok=True)
                            fname = f"{save_dir}/art_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                            save_img = np.zeros_like(canvas.drawing_layer)
                            wm = cv2.inRange(canvas.drawing_layer, (200,200,200), (255,255,255))
                            save_img[wm > 0] = ink_bgr
                            cv2.imwrite(fname, save_img)
                            last_recog_boxes = [(-1, -1, -1, -1, f"Saved!")]
                            recog_box_expire = curr_time + 2.5
                        elif mode_name == 'train':
                            import subprocess
                            subprocess.Popen(["python", "C:/hand/retrain.py"])
                            last_recog_boxes = [(-1, -1, -1, -1, f"Retraining started!")]
                            recog_box_expire = curr_time + 3.0
                        active_menu = None
                    elif hovered_btn == 'next':
                        if mode_name == 'art':
                            coloring.load_random_image()
                            canvas.clear()
                            _recognized_path_cnt = 0
                        active_menu = None
                    
                    btn_cooldown = curr_time"""
    
    new_code = code[:start_idx] + new_block + code[end_idx + len(end_str):]
    with open('C:/hand/app_main.py', 'w', encoding='utf-8') as f:
        f.write(new_code)
    print("Success")
else:
    print("Not found")
