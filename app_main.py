import cv2
import numpy as np
import time
import math
from core.hand_tracker import HandTracker
from core.canvas import CanvasManager
from core.model_manager import ModelManager
from core.magic_effects import ParticleSystem, MagicMandala
from core.gesture_solver import detect_gesture, get_palm_center, safe_math_eval


# ──────────────────────────────
#  Glassmorphism UI 輔助函式
# ──────────────────────────────

def create_glass_overlay(image, rect, corner_radius=15, alpha=0.45, 
                         color=(30, 30, 30), border_color=(130, 130, 130)):
    x1, y1, x2, y2 = rect
    if x2 <= x1 or y2 <= y1:
        return
    
    roi = image[y1:y2, x1:x2].copy()
    h, w = roi.shape[:2]
    
    # 圓角 Mask
    mask = np.zeros((h, w), dtype=np.uint8)
    r = min(corner_radius, h // 2, w // 2)
    cv2.rectangle(mask, (r, 0), (w - r, h), 255, -1)
    cv2.rectangle(mask, (0, r), (w, h - r), 255, -1)
    cv2.circle(mask, (r, r), r, 255, -1)
    cv2.circle(mask, (w - r, r), r, 255, -1)
    cv2.circle(mask, (r, h - r), r, 255, -1)
    cv2.circle(mask, (w - r, h - r), r, 255, -1)
    
    # 毛玻璃模糊 + 著色
    blurred = cv2.GaussianBlur(roi, (21, 21), 0)
    colored = cv2.addWeighted(blurred, 1.0 - alpha, np.full_like(blurred, color[::-1] if len(color) == 3 else color), alpha, 0)
    
    np.copyto(roi, colored, where=(mask == 255)[:, :, None])
    
    # 邊框（只在 mask 邊緣）
    edge = cv2.Canny(mask, 100, 200)
    edge = cv2.dilate(edge, np.ones((2, 2), np.uint8), iterations=1)
    roi[edge > 0] = border_color
    
    image[y1:y2, x1:x2] = roi


def is_in_rect(pt, rect):
    return rect[0] <= pt[0] <= rect[2] and rect[1] <= pt[1] <= rect[3]


def draw_progress_arc(image, center, radius, progress, color=(0, 255, 0)):
    """在按鈕旁繪製圓弧進度指示（更有魔法感）"""
    if progress <= 0:
        return
    angle = int(360 * progress)
    cv2.ellipse(image, center, (radius, radius), -90, 0, angle, color, 3, cv2.LINE_AA)


# ──────────────────────────────
#  主 UI 繪製
# ──────────────────────────────

PALETTE_COLORS = [
    ('Draw',  (255, 200,  50)),  # 橙金（預設）
    ('Ice',   ( 80, 220, 255)),  # 冰藍
    ('Magic', (200,  80, 255)),  # 紫魔
    ('Blood', ( 50,  50, 255)),  # 赤紅
]

def draw_ar_ui(image, mode, color_idx, recognized_text,
               hover_point=None, last_hovered_btn=None, hover_progress=0.0):
    W, H = image.shape[1], image.shape[0]

    btn_mode     = (15,  15, 170, 65)
    btn_color    = (180, 15, 270, 65)
    btn_back     = (W - 230, 15, W - 120, 65)
    btn_clear    = (W - 110, 15, W -  15, 65)

    # --- 毛玻璃按鈕 ---
    create_glass_overlay(image, btn_mode,  color=(40, 30, 60))
    create_glass_overlay(image, btn_color, color=(20, 50, 40))
    create_glass_overlay(image, btn_back,  color=(60, 40, 30))
    create_glass_overlay(image, btn_clear, color=(70, 20, 20))

    label_color, _ = PALETTE_COLORS[color_idx]
    cv2.putText(image, f"Mode: {mode.upper()}", (btn_mode[0]+10, btn_mode[1]+34),
                cv2.FONT_HERSHEY_DUPLEX, 0.52, (200, 255, 200), 1, cv2.LINE_AA)
    cv2.putText(image, f"Ink: {label_color}", (btn_color[0]+10, btn_color[1]+34),
                cv2.FONT_HERSHEY_DUPLEX, 0.52, (220, 255, 180), 1, cv2.LINE_AA)
    cv2.putText(image, "< Back", (btn_back[0]+15, btn_back[1]+34),
                cv2.FONT_HERSHEY_DUPLEX, 0.62, (200, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "Clear", (btn_clear[0]+20, btn_clear[1]+34),
                cv2.FONT_HERSHEY_DUPLEX, 0.62, (150, 150, 255), 1, cv2.LINE_AA)

    # --- 底部輸出框 ---
    bar = (15, H - 90, W - 15, H - 15)
    create_glass_overlay(image, bar, alpha=0.35, color=(10, 10, 10))
    display = f"Output: {recognized_text[-60:]}"  # 最多顯示 60 個字元
    cv2.putText(image, display, (bar[0]+20, bar[1]+52),
                cv2.FONT_HERSHEY_DUPLEX, 0.95, (60, 255, 100), 2, cv2.LINE_AA)

    # --- Hover 高亮 + 進度弧 ---
    hovered_btn = None
    if hover_point:
        for btn, name in [(btn_mode, 'mode'), (btn_color, 'color'),
                          (btn_back, 'back'), (btn_clear, 'clear')]:
            if is_in_rect(hover_point, btn):
                hovered_btn = name
                if name == last_hovered_btn:
                    cv2.rectangle(image, btn[:2], btn[2:], (255, 255, 255), 2, cv2.LINE_AA)
                    cx = (btn[0] + btn[2]) // 2
                    cy = btn[3] + 18
                    draw_progress_arc(image, (cx, cy), 12, hover_progress)
                break

    return hovered_btn


def draw_ar_answer(image, answer_str, anchor_x, anchor_y):
    """在算式旁邊顯示浮空的 AR 解答，帶有光暈效果"""
    text = f"= {answer_str}"
    # 發光底層
    for offset, col in [(4, (0, 80, 0)), (2, (0, 180, 30))]:
        cv2.putText(image, text, (anchor_x + offset, anchor_y),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, col, 3 + offset, cv2.LINE_AA)
    # 亮字
    cv2.putText(image, text, (anchor_x, anchor_y),
                cv2.FONT_HERSHEY_DUPLEX, 1.4, (100, 255, 120), 2, cv2.LINE_AA)


# ──────────────────────────────
#  主程式
# ──────────────────────────────

def main():
    print("[系統] 正在啟動空中魔法系統 (Doctor Strange Edition)...")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    tracker   = HandTracker()
    canvas    = CanvasManager(width=640, height=480, line_thickness=12)
    model_mgr = ModelManager()
    particles = ParticleSystem()
    mandala   = MagicMandala()

    MODES = ["digit", "letter", "symbol"]
    mode_idx   = 0
    color_idx  = 0

    recognized_text = ""
    ar_answer   = None          # (answer_str, expire_time)
    last_draw_time = time.time()

    btn_cooldown      = 0.0
    last_hovered_btn  = None
    hover_start_time  = 0.0

    # 搖滾手勢連擊偵測
    rock_gesture_time  = 0.0
    rock_triggered     = False

    prev_draw_pt = None
    prev_time    = time.time()

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        results, disp = tracker.process_frame(frame, optimize_lighting=False)
        # 不顯示骨架，保持畫面乾淨（選擇性）
        # disp = tracker.draw_landmarks(disp, results)

        curr_time  = time.time()
        mode_name  = MODES[mode_idx]
        ink_color  = PALETTE_COLORS[color_idx][1]  # BGR

        is_drawing    = False
        hover_point   = None
        gesture       = 'unknown'

        if results.hand_landmarks:
            lm = results.hand_landmarks[0]

            # 食指 & 中指座標
            x_idx = int(np.clip(lm[8].x * 640, 5, 635))
            y_idx = int(np.clip(lm[8].y * 480, 5, 475))
            x_mid = int(lm[12].x * 640)
            y_mid = int(lm[12].y * 480)
            hover_point = (x_idx, y_idx)

            gesture = detect_gesture(lm, 640, 480)

            # ── 手勢分支 ──────────────────────────────────────

            if gesture == 'palm_open':
                # 掌心法陣
                palm_cx, palm_cy = get_palm_center(lm, 640, 480)
                # 根據手掌深度 (z) 調整法陣大小
                avg_z = sum(lm[i].z for i in [0, 5, 9, 13, 17]) / 5
                radius = int(np.clip(80 - avg_z * 400, 50, 130))
                mandala.draw(disp, palm_cx, palm_cy, radius=radius, color=ink_color)
                # 指尖噴射粒子
                for tip_id in [4, 8, 12, 16, 20]:
                    tx = int(lm[tip_id].x * 640)
                    ty = int(lm[tip_id].y * 480)
                    particles.spawn(tx, ty, color=ink_color, count=2)
                canvas.end_stroke()

            elif gesture == 'rock':
                # 搖滾手勢 → 炸裂清空 (觸發一次)
                if not rock_triggered and curr_time - rock_gesture_time < 0.1:
                    pass
                elif not rock_triggered:
                    rock_gesture_time = curr_time
                    rock_triggered = True
                    # 爆炸粒子噴射
                    for _ in range(60):
                        import random
                        particles.spawn(
                            random.randint(100, 540),
                            random.randint(80, 400),
                            color=ink_color, count=1
                        )
                    canvas.clear()
                    recognized_text = ""
                canvas.end_stroke()

            elif gesture == 'draw':
                rock_triggered = False
                is_drawing = True
                canvas.add_point((x_idx, y_idx))
                last_draw_time = curr_time
                # 指尖粒子（少量，避免性能下降）
                if prev_draw_pt:
                    dist = math.sqrt((x_idx - prev_draw_pt[0])**2 + (y_idx - prev_draw_pt[1])**2)
                    if dist > 12:
                        particles.spawn(x_idx, y_idx, color=ink_color, count=3)
                prev_draw_pt = (x_idx, y_idx)
                # 畫筆準心
                cv2.circle(disp, (x_idx, y_idx), 10, ink_color, -1, cv2.LINE_AA)
                cv2.circle(disp, (x_idx, y_idx), 14, (255, 255, 255), 1, cv2.LINE_AA)

            else:  # hover / unknown
                rock_triggered = False
                canvas.end_stroke()
                prev_draw_pt = None
                # 懸浮準心
                cv2.circle(disp, (x_idx, y_idx), 8, (200, 200, 200), -1, cv2.LINE_AA)
                cv2.circle(disp, (x_idx, y_idx), 18, (255, 255, 255), 2, cv2.LINE_AA)

        else:
            canvas.end_stroke()
            prev_draw_pt  = None
            last_hovered_btn = None
            rock_triggered = False

        # ── 粒子更新 ──────────────────────────────────────────
        particles.update_and_draw(disp)

        # ── Hover 按鈕計算 ───────────────────────────────────
        hover_progress = 0.0
        if last_hovered_btn and not is_drawing:
            hover_progress = min(1.0, (curr_time - hover_start_time) / 1.0)

        btns = draw_ar_ui(disp, mode_name, color_idx, recognized_text,
                          hover_point, last_hovered_btn, hover_progress)
        hovered_btn = btns

        if hovered_btn and not is_drawing:
            if hovered_btn == last_hovered_btn:
                if curr_time - hover_start_time > 1.0 and curr_time - btn_cooldown > 0.8:
                    if hovered_btn == 'clear':
                        recognized_text = ""
                        canvas.clear()
                    elif hovered_btn == 'back':
                        recognized_text = recognized_text[:-1]
                    elif hovered_btn == 'mode':
                        mode_idx = (mode_idx + 1) % len(MODES)
                        canvas.clear()
                    elif hovered_btn == 'color':
                        color_idx = (color_idx + 1) % len(PALETTE_COLORS)
                    btn_cooldown = curr_time
                    hover_start_time = curr_time
            else:
                last_hovered_btn = hovered_btn
                hover_start_time  = curr_time
        else:
            last_hovered_btn = None

        # ── 自動辨識 (Auto-Segmentation 1.2s) ────────────────
        if not is_drawing and canvas.has_content() and (curr_time - last_draw_time > 1.2):
            preds = model_mgr.predict_canvas_content(canvas.drawing_layer, mode=mode_name)
            new_chars = "".join(str(label) for _, label in preds)
            recognized_text += new_chars

            # ── AR 數學解題器 ───────────────────────────────
            # 如果最後一個字元是 '=' 或輸入尾端已包含 '='，嘗試求解
            if '=' in recognized_text:
                # 取最後一段等號之前的算式
                expr = recognized_text.rsplit('=', 1)[0]
                answer, ok = safe_math_eval(expr)
                if ok:
                    ar_answer = (answer, curr_time + 4.0)  # 顯示 4 秒

            canvas.clear()

        # ── AR 解答浮空顯示 ──────────────────────────────────
        if ar_answer:
            ans_str, expire = ar_answer
            if curr_time < expire:
                draw_ar_answer(disp, ans_str, 80, 240)
            else:
                ar_answer = None

        # ── 筆跡 AR 疊加 ─────────────────────────────────────
        draw_layer = canvas.drawing_layer
        # 用色彩替換：把白色筆跡染成 ink_color
        white_mask = cv2.inRange(draw_layer, (200, 200, 200), (255, 255, 255))
        colored_layer = np.zeros_like(draw_layer)
        colored_layer[white_mask > 0] = ink_color
        
        mask_bool = white_mask > 0
        disp[mask_bool] = colored_layer[mask_bool]
        canvas.draw_current_stroke(disp)

        # ── FPS ──────────────────────────────────────────────
        now = time.time()
        fps = 1.0 / max(now - prev_time, 0.001)
        prev_time = now
        cv2.putText(disp, f"FPS {int(fps)}", (15, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Doctor Strange AR', disp)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC 離開
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
