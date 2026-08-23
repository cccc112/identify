"""
Doctor Strange AR 手寫板 — 穩定版
修復：
  - EMA 濾波消除手抖
  - 容錯緩衝 (4幀) 防止追蹤短暫遺失導致筆畫斷裂
  - 最小像素門檻 (MIN_PIXELS) 過濾誤判噪點
  - 信心值門檻 (55%) 過濾低品質辨識
  - 更細的筆觸 (thickness=7)
  - 重新設計的毛玻璃 UI
"""

import cv2
import numpy as np
import time
import math
import random

from core.hand_tracker import HandTracker
from core.canvas import CanvasManager
from core.model_manager import ModelManager
from core.magic_effects import ParticleSystem, MagicMandala
from core.gesture_solver import detect_gesture, get_palm_center, safe_math_eval


# ─────────────────────────────────────────────────────────────────
#  常數設定
# ─────────────────────────────────────────────────────────────────
W, H = 640, 480
MODES        = ["digit", "letter", "symbol"]
PALETTES     = [
    ("Gold",   (50, 200, 255)),   # 橙金
    ("Cyan",   (255, 220,  80)),  # 冰藍
    ("Violet", (255,  80, 220)),  # 紫魔
    ("Red",    (60,  60,  255)),  # 赤紅
]
DRAW_DIST_THRESHOLD = 40   # 食指與中指距離 > 此值 → 畫圖
MIN_PIXELS          = 300  # 畫布上至少要有這麼多白色像素才觸發辨識
SEG_TIMEOUT         = 1.5  # 停筆超過此秒數才觸發自動辨識


# ─────────────────────────────────────────────────────────────────
#  UI 工具函式
# ─────────────────────────────────────────────────────────────────

def glass_rect(img, rect, alpha=0.50, bg=(25, 25, 35), radius=12, border=(100, 100, 130)):
    """半透明圓角毛玻璃矩形"""
    x1, y1, x2, y2 = rect
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return

    sub  = img[y1:y2, x1:x2].copy()
    h, w = sub.shape[:2]
    r    = min(radius, h // 2, w // 2)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (r, 0), (w - r, h), 255, -1)
    cv2.rectangle(mask, (0, r), (w, h - r), 255, -1)
    for cx, cy in [(r, r), (w-r, r), (r, h-r), (w-r, h-r)]:
        cv2.circle(mask, (cx, cy), r, 255, -1)

    blurred = cv2.GaussianBlur(sub, (15, 15), 0)
    color_layer = np.full_like(sub, bg[::-1] if len(bg) == 3 else bg)
    blended = cv2.addWeighted(blurred, 1 - alpha, color_layer, alpha, 0)

    np.copyto(sub, blended, where=(mask == 255)[:, :, None])

    # 邊框
    edge_mask = cv2.Canny(mask, 100, 200)
    edge_mask = cv2.dilate(edge_mask, np.ones((2, 2), np.uint8))
    sub[edge_mask > 0] = border

    img[y1:y2, x1:x2] = sub


def label(img, text, x, y, scale=0.55, color=(220, 220, 220), thick=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX,
                scale, color, thick, cv2.LINE_AA)


def hover_arc(img, cx, cy, r, progress, color=(60, 255, 120)):
    if progress <= 0:
        return
    cv2.ellipse(img, (cx, cy), (r, r), -90, 0,
                int(360 * progress), color, 3, cv2.LINE_AA)


def is_in(pt, rect):
    return rect[0] <= pt[0] <= rect[2] and rect[1] <= pt[1] <= rect[3]


# ─────────────────────────────────────────────────────────────────
#  主 UI 繪製
# ─────────────────────────────────────────────────────────────────

def draw_ui(img, mode, palette_idx, gesture_name,
            text, hover_pt, last_btn, hover_prog, ar_ans=None):
    """
    繪製整個 AR UI。
    回傳: hovered_btn_name | None
    """
    # ── 頂部按鈕列 ──────────────────────────────────────
    BTN_H = 50
    PAD   = 12

    btn_mode  = (PAD,        PAD, 175,           PAD + BTN_H)
    btn_ink   = (185,        PAD, 310,           PAD + BTN_H)
    btn_back  = (W - 215,   PAD, W - 110,       PAD + BTN_H)
    btn_clear = (W - 100,   PAD, W - PAD,       PAD + BTN_H)

    for rect, bg in [(btn_mode,  (40, 25, 70)),
                     (btn_ink,   (20, 55, 40)),
                     (btn_back,  (65, 35, 25)),
                     (btn_clear, (70, 15, 15))]:
        glass_rect(img, rect, bg=bg)

    ink_name = PALETTES[palette_idx][0]
    label(img, f"Mode: {mode.upper()}", btn_mode[0]+10,  btn_mode[1]+32,  color=(160, 255, 180))
    label(img, f"Ink: {ink_name}",      btn_ink[0]+10,   btn_ink[1]+32,   color=(160, 255, 220))
    label(img, "< Undo",               btn_back[0]+14,  btn_back[1]+32,  color=(190, 190, 255))
    label(img, "Clear",                btn_clear[0]+16, btn_clear[1]+32, color=(200, 140, 255))

    # ── 手勢狀態提示 (左下角) ────────────────────────────
    gesture_colors = {
        'draw':      (60,  255, 100),
        'hover':     (200, 200,  60),
        'palm_open': (60,  160, 255),
        'rock':      (60,   60, 255),
    }
    gc = gesture_colors.get(gesture_name, (160, 160, 160))
    gesture_icons = {
        'draw': '✏ DRAW', 'hover': '🖱 HOVER',
        'palm_open': '✋ MAGIC', 'rock': '🤘 CAST',
    }
    gi = gesture_icons.get(gesture_name, '· ·  ·')
    glass_rect(img, (PAD, H - 110, 170, H - PAD - 90), alpha=0.4, bg=(20, 20, 20))
    label(img, gi, PAD + 10, H - 110 + 32, scale=0.52, color=gc)

    # ── 底部輸出文字框 ───────────────────────────────────
    bar = (PAD, H - 85, W - PAD, H - PAD)
    glass_rect(img, bar, alpha=0.42, bg=(10, 10, 15))
    display = recognized_text_display(text)
    label(img, display, bar[0] + 16, bar[1] + 52,
          scale=1.0, color=(60, 255, 100), thick=2)

    # ── AR 數學解答 ──────────────────────────────────────
    if ar_ans:
        ans_text = f"= {ar_ans}"
        glass_rect(img, (160, 185, 480, 265), alpha=0.55, bg=(10, 40, 10))
        # 外發光
        for off, col in [(3, (0, 80, 0)), (1, (0, 200, 50))]:
            cv2.putText(img, ans_text, (170 + off, 244),
                        cv2.FONT_HERSHEY_DUPLEX, 1.35, col, 3 + off, cv2.LINE_AA)
        cv2.putText(img, ans_text, (170, 244),
                    cv2.FONT_HERSHEY_DUPLEX, 1.35, (80, 255, 120), 2, cv2.LINE_AA)

    # ── Hover 高亮與進度弧 ───────────────────────────────
    hovered = None
    if hover_pt:
        for rect, name in [(btn_mode, 'mode'), (btn_ink, 'ink'),
                           (btn_back, 'back'), (btn_clear, 'clear')]:
            if is_in(hover_pt, rect):
                hovered = name
                if name == last_btn and hover_prog > 0:
                    cv2.rectangle(img, rect[:2], rect[2:], (255, 255, 255), 2, cv2.LINE_AA)
                    cx = (rect[0] + rect[2]) // 2
                    hover_arc(img, cx, rect[3] + 16, 11, hover_prog)
                break

    return hovered


def recognized_text_display(text, max_chars=55):
    """只顯示最近 max_chars 個字元，避免 overflow"""
    return ("Output: " + text)[-max_chars:]


# ─────────────────────────────────────────────────────────────────
#  主程式
# ─────────────────────────────────────────────────────────────────

def main():
    print("[系統] Doctor Strange AR — 穩定版啟動中...")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    tracker   = HandTracker()
    canvas    = CanvasManager(width=W, height=H, line_thickness=7)
    model_mgr = ModelManager()
    particles = ParticleSystem()
    mandala   = MagicMandala()

    mode_idx    = 0
    palette_idx = 0

    recognized_text = ""
    ar_answer       = None   # (str, expire_time)
    last_draw_time  = time.time()

    btn_cooldown     = 0.0
    last_hovered_btn = None
    hover_start_time = 0.0

    rock_triggered = False
    prev_draw_pt   = None
    prev_time      = time.time()
    gesture_name   = 'unknown'

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        results, disp = tracker.process_frame(frame, optimize_lighting=False)

        curr_time   = time.time()
        mode_name   = MODES[mode_idx]
        ink_bgr     = PALETTES[palette_idx][1]    # (B, G, R)

        is_drawing  = False
        hover_point = None

        # ── 手部追蹤 ─────────────────────────────────────
        if results.hand_landmarks:
            lm = results.hand_landmarks[0]

            x_idx = int(np.clip(lm[8].x * W, 5, W - 5))
            y_idx = int(np.clip(lm[8].y * H, 5, H - 5))
            x_mid = int(lm[12].x * W)
            y_mid = int(lm[12].y * H)
            hover_point  = (x_idx, y_idx)
            gesture_name = detect_gesture(lm, W, H)

            dist_fingers = math.sqrt((x_idx - x_mid) ** 2 + (y_idx - y_mid) ** 2)

            # ── 手勢分支 ──────────────────────────────────

            if gesture_name == 'palm_open':
                palm_cx, palm_cy = get_palm_center(lm, W, H)
                avg_z  = sum(lm[i].z for i in [0, 5, 9, 13, 17]) / 5
                radius = int(np.clip(85 - avg_z * 380, 50, 130))
                mandala.draw(disp, palm_cx, palm_cy, radius=radius, color=ink_bgr)
                for tip_id in [4, 8, 12, 16, 20]:
                    tx = int(lm[tip_id].x * W)
                    ty = int(lm[tip_id].y * H)
                    particles.spawn(tx, ty, color=ink_bgr, count=2)
                canvas.notify_tracking_lost()  # 張手時不應畫圖
                rock_triggered = False

            elif gesture_name == 'rock' and not rock_triggered:
                rock_triggered = True
                for _ in range(80):
                    particles.spawn(
                        random.randint(80, 560), random.randint(60, 420),
                        color=ink_bgr, count=1)
                canvas.clear()
                recognized_text = ""
                canvas.notify_tracking_lost()

            elif dist_fingers > DRAW_DIST_THRESHOLD:
                # 畫圖模式
                is_drawing     = True
                rock_triggered = False
                canvas.add_point(x_idx, y_idx)
                last_draw_time = curr_time

                if prev_draw_pt:
                    d = math.sqrt((x_idx - prev_draw_pt[0]) ** 2 + (y_idx - prev_draw_pt[1]) ** 2)
                    if d > 10:
                        particles.spawn(x_idx, y_idx, color=ink_bgr, count=2)
                prev_draw_pt = (x_idx, y_idx)

                # 準心：實心圓 + 白色外環
                cv2.circle(disp, (x_idx, y_idx), 9,  ink_bgr,       -1, cv2.LINE_AA)
                cv2.circle(disp, (x_idx, y_idx), 14, (255, 255, 255), 1, cv2.LINE_AA)

            else:
                # 懸浮模式
                rock_triggered = False
                canvas.notify_tracking_lost()
                prev_draw_pt = None
                # 準心：空心雙環
                cv2.circle(disp, (x_idx, y_idx), 7,  (180, 180, 180), -1, cv2.LINE_AA)
                cv2.circle(disp, (x_idx, y_idx), 16, (255, 255, 255),  1, cv2.LINE_AA)
                cv2.circle(disp, (x_idx, y_idx), 20, (200, 200, 200),  1, cv2.LINE_AA)

        else:
            # 完全沒偵測到手
            canvas.notify_tracking_lost()
            prev_draw_pt   = None
            last_hovered_btn = None
            rock_triggered = False
            gesture_name   = 'unknown'

        # ── 粒子更新 ──────────────────────────────────────
        particles.update_and_draw(disp)

        # ── Hover 進度計算 ────────────────────────────────
        hover_progress = 0.0
        if last_hovered_btn and not is_drawing:
            hover_progress = min(1.0, (curr_time - hover_start_time) / 1.0)

        # ── 繪製 UI ───────────────────────────────────────
        ar_ans_display = None
        if ar_answer and curr_time < ar_answer[1]:
            ar_ans_display = ar_answer[0]
        elif ar_answer and curr_time >= ar_answer[1]:
            ar_answer = None

        hovered_btn = draw_ui(
            disp, mode_name, palette_idx, gesture_name,
            recognized_text, hover_point, last_hovered_btn,
            hover_progress, ar_ans=ar_ans_display
        )

        # ── Hover 按鈕觸發 ────────────────────────────────
        if hovered_btn and not is_drawing:
            if hovered_btn == last_hovered_btn:
                if (curr_time - hover_start_time > 1.0 and
                        curr_time - btn_cooldown > 0.8):
                    if hovered_btn == 'clear':
                        recognized_text = ""
                        canvas.clear()
                    elif hovered_btn == 'back':
                        recognized_text = recognized_text[:-1]
                    elif hovered_btn == 'mode':
                        mode_idx = (mode_idx + 1) % len(MODES)
                        canvas.clear()
                    elif hovered_btn == 'ink':
                        palette_idx = (palette_idx + 1) % len(PALETTES)
                    btn_cooldown     = curr_time
                    hover_start_time = curr_time
            else:
                last_hovered_btn = hovered_btn
                hover_start_time = curr_time
        else:
            last_hovered_btn = None

        # ── 自動分字辨識 ──────────────────────────────────
        pixel_count = canvas.get_pixel_count()
        time_elapsed = curr_time - last_draw_time

        if (not is_drawing
                and canvas.has_content()
                and pixel_count >= MIN_PIXELS
                and time_elapsed > SEG_TIMEOUT):

            preds = model_mgr.predict_canvas_content(
                canvas.drawing_layer, mode=mode_name)

            new_chars = ""
            for box, char, conf in preds:
                new_chars += char

            if new_chars:
                recognized_text += new_chars

                # AR 數學解題
                if '=' in recognized_text:
                    expr = recognized_text.rsplit('=', 1)[0]
                    ans, ok = safe_math_eval(expr)
                    if ok:
                        ar_answer = (ans, curr_time + 4.0)

            canvas.clear()

        # ── 筆跡 AR 疊加 (著色) ───────────────────────────
        draw_lyr = canvas.drawing_layer
        white_mask = cv2.inRange(draw_lyr, (200, 200, 200), (255, 255, 255))
        colored_lyr = np.zeros_like(draw_lyr)
        colored_lyr[white_mask > 0] = ink_bgr
        disp[white_mask > 0] = colored_lyr[white_mask > 0]

        # 即時筆跡（當幀尚未 commit 的點）
        canvas.draw_current_stroke(disp, ink_color=ink_bgr)

        # ── 輕量骨架（可選：不顯示以保持畫面清爽）─────────
        # disp = tracker.draw_landmarks(disp, results)

        # ── FPS 顯示 ──────────────────────────────────────
        now  = time.time()
        fps  = 1.0 / max(now - prev_time, 0.001)
        prev_time = now
        label(disp, f"FPS {int(fps)}", 16, 120, scale=0.55, color=(0, 220, 220))

        cv2.imshow('Doctor Strange AR', disp)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
