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
import os

from core.hand_tracker import HandTracker
from core.canvas import CanvasManager
from core.model_manager import ModelManager
from core.magic_effects import ParticleSystem, MagicMandala, NeuralBloom, AnimalSpawner
from core.gesture_solver import GestureStateMachine, get_palm_center, safe_math_eval
from core.coloring_manager import ColoringManager
from core.color_picker import ColorPicker
from core.face_tracker import FaceTracker
from core.gaze_solver import GazeSolver
from core.keyboard import GazeKeyboard


# ─────────────────────────────────────────────────────────────────
#  常數設定
# ─────────────────────────────────────────────────────────────────
W, H = 640, 480
MODES        = ["digit", "letter", "symbol", "magic", "art", "type", "train"]
MIN_BLOOM_RADIUS = 50
PALETTES     = [
    ("Gold",    ( 50, 200, 255)),  # 橙金
    ("Cyan",    (255, 220,  80)),  # 冰藍
    ("Violet",  (255,  80, 220)),  # 紫魔
    ("Red",     ( 60,  60, 255)),  # 赤紅
    ("Green",   ( 50, 220,  80)),  # 翠綠
    ("Pink",    (180,  80, 255)),  # 粉紅
    ("White",   (240, 240, 240)),  # 白
    ("Sky",     (255, 190,  80)),  # 天藍
    ("Orange",  ( 30, 140, 255)),  # 橘色
    ("Yellow",  ( 30, 240, 255)),  # 黃色
    ("Lime",    ( 30, 255, 180)),  # 萊姆綠
    ("Navy",    (180,  50,  30)),  # 海軍藍
    ("Brown",   ( 30,  80, 150)),  # 棕色
    ("Black",   ( 20,  20,  20)),  # 黑色
    ("Gray",    (130, 130, 130)),  # 灰色
]
THICKNESSES  = [4, 7, 12, 20]    # 筆刷粗細選項
DRAW_DIST_THRESHOLD = 40
MIN_PIXELS          = 50
SEG_TIMEOUT         = 2.5


# ─────────────────────────────────────────────────────────────────
#  上游座標平滑器（所有邏輯都用平滑後座標）
# ─────────────────────────────────────────────────────────────────

class PointerSmoother:
    """
    在 MediaPipe 原始座標讀出後立刻套 EMA 低通濾波。
    - alpha 越小 → 越平滑但反應越慢
    - alpha=0.30 在書寫流暢度和抗抖動之間取得平衡
    """
    def __init__(self, alpha=0.28):
        self._a = alpha
        self._x = self._y = None

    def update(self, raw_x, raw_y):
        if self._x is None:
            self._x, self._y = float(raw_x), float(raw_y)
        else:
            self._x = self._a * raw_x + (1 - self._a) * self._x
            self._y = self._a * raw_y + (1 - self._a) * self._y
        return int(self._x), int(self._y)

    def reset(self):
        self._x = self._y = None


# ─────────────────────────────────────────────────────────────────
#  Material You 色彩 Token (BGR)
# ─────────────────────────────────────────────────────────────────
M3 = {
    'bg':            (14,  13,  16),   # 近黑背景
    'surface':       (38,  34,  42),   # 卡片底色
    'surface_hi':    (70,  64,  80),   # 懸浮高亮
    'primary':       (255, 177, 130),  # 鮮藍 #82B1FF (BGR)
    'primary_dim':   (160, 100,  70),
    'secondary':     (220, 215, 225),  # 淺灰白
    'tertiary':      ( 64, 171, 255),  # 鮮橙 #FFAB40 (BGR)
    'on_surface':    (245, 245, 250),  # 幾乎純白文字
    'on_surface_lo': (170, 165, 180),  # 次要文字
    'error':         ( 80,  80, 255),  # 鮮紅 #FF5050 (BGR)
    'success':       ( 80, 230, 100),  # 鮮綠 #64E650 (BGR)
    # 模式專屬
    'magic_col':     (255,  80, 200),  # 鮮紫 #C850FF (BGR)
    'art_col':       ( 40, 200, 255),  # 鮮橙黃 (BGR)
}

# 各模式的 tonal surface 色（也配合提高飽和度）
MODE_COLORS = {
    'digit':  ((55, 45, 70),   M3['primary']),
    'letter': ((40, 65, 45),   M3['success']),
    'symbol': ((55, 50, 38),   M3['tertiary']),
    'magic':  ((65, 35, 75),   M3['magic_col']),
    'art':    ((35, 60, 65),   M3['art_col']),
}

# ─────────────────────────────────────────────────────────────────
#  UI 工具函式（Material You 版）
# ─────────────────────────────────────────────────────────────────

def pill(img, rect, bg=None, alpha=0.60, border=None, glow=False):
    """Pill 形狀（完全圓角）的毛玻璃卡片，radius = height/2"""
    x1, y1, x2, y2 = rect
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return

    if bg is None:
        bg = M3['surface']

    sub = img[y1:y2, x1:x2].copy()
    h, w = sub.shape[:2]
    r    = h // 2   # pill: 半圓角

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (r, 0), (w - r, h), 255, -1)
    for cx2, cy2 in [(r, r), (w - r, r), (r, h - r), (w - r, h - r)]:
        cv2.circle(mask, (cx2, cy2), r, 255, -1)

    # 毛玻璃底
    blurred      = cv2.GaussianBlur(sub, (11, 11), 0)
    color_layer  = np.full_like(sub, bg)
    blended      = cv2.addWeighted(blurred, 1 - alpha, color_layer, alpha, 0)
    np.copyto(sub, blended, where=(mask > 0)[:, :, None])

    # 邊框（細 1px，用 border 色或白半透）
    if border is None:
        border = M3['surface_hi']
    edge = cv2.Canny(mask, 100, 200)
    edge = cv2.dilate(edge, np.ones((1, 1), np.uint8))
    sub[edge > 0] = border

    # 可選：底部外發光
    if glow and border:
        img[min(y2, img.shape[0]-1), x1:x2] = border  # subtle line glow

    img[y1:y2, x1:x2] = sub


def glass_rect(img, rect, alpha=0.50, bg=(25, 25, 35), radius=12, border=(80, 80, 100)):
    """保留相容性：對較大面板用較大 radius"""
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
    for cx2, cy2 in [(r, r), (w-r, r), (r, h-r), (w-r, h-r)]:
        cv2.circle(mask, (cx2, cy2), r, 255, -1)
    blurred     = cv2.GaussianBlur(sub, (15, 15), 0)
    color_layer = np.full_like(sub, bg)
    blended     = cv2.addWeighted(blurred, 1 - alpha, color_layer, alpha, 0)
    np.copyto(sub, blended, where=(mask == 255)[:, :, None])
    edge = cv2.Canny(mask, 100, 200)
    edge = cv2.dilate(edge, np.ones((2, 2), np.uint8))
    sub[edge > 0] = border
    img[y1:y2, x1:x2] = sub


def label(img, text, x, y, scale=0.52, color=None, thick=1):
    if color is None:
        color = M3['on_surface']
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
#  主 UI 繪製（Material You / Pixel 風格）
# ─────────────────────────────────────────────────────────────────

def draw_ui(img, mode, palette_idx, thickness_idx, active_tool, gesture_name,
            text, hover_pt, last_btn, hover_prog, ar_ans=None, gaze_pt=None, use_eye_only=False):
    """
    Material You 風格 UI。
    回傳: (hand_hovered_btn, gaze_hovered_btn)
    """
    PAD   = 10
    BTN_H = 44   # pill 高度

    # 第二排永遠固定一個 Tracker Toggle
    btn_track = (PAD, PAD + BTN_H + 10, 110, PAD + BTN_H * 2 + 10)

    # ── 頂部 pill 按鈕列 ─────────────────────────────────
    if mode == 'art':
        btn_mode  = (PAD, PAD, 110, PAD + BTN_H)
        btn_ink   = (115, PAD, 205, PAD + BTN_H)
        btn_size  = (210, PAD, 285, PAD + BTN_H)
        btn_tool  = (290, PAD, 385, PAD + BTN_H)
        btn_save  = (390, PAD, 465, PAD + BTN_H)
        btn_next  = (470, PAD, 545, PAD + BTN_H)
        btn_clear = (550, PAD, W - PAD, PAD + BTN_H)
        btn_back  = (-10, -10, -10, -10)
    else:
        btn_mode  = (PAD, PAD, 155, PAD + BTN_H)
        btn_ink   = (162, PAD, 280, PAD + BTN_H)
        btn_size  = (287, PAD, 375, PAD + BTN_H)
        btn_tool  = (-10, -10, -10, -10)
        btn_save  = (-10, -10, -10, -10)
        btn_next  = (-10, -10, -10, -10)
        btn_back  = (W - 195, PAD, W - 105, PAD + BTN_H)
        btn_clear = (W - 98, PAD, W - PAD, PAD + BTN_H)
        if mode == 'train':
            btn_save = (382, PAD, 490, PAD + BTN_H) # We will call it RETRAIN but use btn_save variable for hitboxes

    mode_bg, mode_col = MODE_COLORS.get(mode, (M3['surface'], M3['primary']))
    ink_bgr_color     = PALETTES[palette_idx][1]  # 取得目前墨水色

    # 各按鈕 tonal surface
    pill(img, btn_mode,  bg=mode_bg,       alpha=0.70, border=mode_col)
    pill(img, btn_ink,   bg=M3['surface'], alpha=0.65, border=ink_bgr_color)
    pill(img, btn_size,  bg=M3['surface'], alpha=0.65, border=M3['secondary'])
    pill(img, btn_clear, bg=(45, 25, 25),  alpha=0.65, border=M3['error'])
    
    if mode == 'art':
        pill(img, btn_tool, bg=M3['surface'], alpha=0.65, border=M3['secondary'])
        pill(img, btn_save, bg=M3['surface'], alpha=0.65, border=M3['success'])
        pill(img, btn_next, bg=M3['surface'], alpha=0.65, border=M3['secondary'])
    else:
        pill(img, btn_back, bg=(35, 35, 55),  alpha=0.65, border=M3['secondary'])

    # 按鈕文字
    ink_name = PALETTES[palette_idx][0]
    thick_val = THICKNESSES[thickness_idx]
    
    mode_label_str = mode.upper()[:3] if mode == 'art' else mode.upper()

    label(img, f' {mode_label_str}', btn_mode[0]+4, btn_mode[1]+28, color=mode_col, scale=0.52)
    label(img, f' {ink_name[:4]}',   btn_ink[0]+2,  btn_ink[1]+28, color=ink_bgr_color, scale=0.52)
    label(img, f' {thick_val}px',    btn_size[0]+2, btn_size[1]+28, color=M3['secondary'], scale=0.48)
    label(img, ' Clear',             btn_clear[0]+2, btn_clear[1]+28, color=M3['error'], scale=0.52)

    if mode == 'art':
        tool_label = 'BRUSH' if active_tool == 'brush' else 'BUCKET'
        label(img, f' {tool_label}', btn_tool[0]+4, btn_tool[1]+28, color=M3['secondary'], scale=0.52)
        label(img, ' Save', btn_save[0]+4, btn_save[1]+28, color=M3['success'], scale=0.52)
        label(img, ' Next', btn_next[0]+4, btn_next[1]+28, color=M3['secondary'], scale=0.52)
    else:
        label(img, ' Undo', btn_back[0]+4, btn_back[1]+28, color=M3['secondary'], scale=0.52)
        if mode == 'train':
            pill(img, btn_save, bg=(40, 25, 40), alpha=0.65, border=(200, 100, 200))
            label(img, ' Retrain', btn_save[0]+4, btn_save[1]+28, color=(255, 150, 255), scale=0.52)

    # 繪製全域第二排 Tracker Toggle
    track_label = ' EYE' if use_eye_only else ' HEAD'
    pill(img, btn_track, bg=(20, 40, 50), alpha=0.8, border=(0, 200, 200) if use_eye_only else (100, 100, 100))
    label(img, track_label, btn_track[0]+4, btn_track[1]+28, color=(0, 255, 255) if use_eye_only else M3['secondary'], scale=0.52)

    # 小圓點色塊（對應墨水色）
    dot_cx = btn_ink[0] + 16 if mode != 'art' else btn_ink[0] + 12
    dot_cy = (btn_ink[1] + btn_ink[3]) // 2
    cv2.circle(img, (dot_cx, dot_cy), 5, ink_bgr_color, -1, cv2.LINE_AA)
    cv2.circle(img, (dot_cx, dot_cy), 5, M3['surface_hi'], 1, cv2.LINE_AA)

    # ── 手勢狀態 badge（右下角 pill chip）────────────────
    gesture_info = {
        'draw':      (M3['success'],   'DRAW ✏'),
        'hover':     (M3['secondary'], 'HOVER'),
        'palm_open': (M3['magic_col'], 'MAGIC ✋'),
        'rock':      (M3['error'],     'CAST  '),
        'fist':      (M3['tertiary'],  'SUBMIT ✊'),
        'thumb_up':  (M3['primary'],   'UNDO  '),
    }
    gcol, gtext = gesture_info.get(gesture_name, (M3['on_surface_lo'], '  ·  '))
    chip_w = 130
    chip_rect = (W - chip_w - PAD, H - 105, W - PAD, H - 105 + BTN_H)
    pill(img, chip_rect, bg=M3['surface'], alpha=0.75, border=gcol)
    label(img, gtext, chip_rect[0] + 14, chip_rect[1] + 28, color=gcol, scale=0.50)

    # ── 底部輸出文字框（大圓角，更高）─────────────────────
    bar_y1, bar_y2 = H - 78, H - PAD
    bar = (PAD, bar_y1, W - PAD, bar_y2)
    glass_rect(img, bar, alpha=0.55, bg=M3['bg'], radius=22,
               border=M3['surface_hi'])
    # 左側色條指示目前模式
    indicator_col = mode_col
    cv2.rectangle(img,
                  (PAD + 4, bar_y1 + 8),
                  (PAD + 8, bar_y2 - 8),
                  indicator_col, -1, cv2.LINE_AA)
    # 輸出文字
    display = recognized_text_display(text)
    label(img, display, PAD + 22, bar_y1 + 48,
          scale=1.05, color=M3['on_surface'], thick=2)

    # ── AR 數學解答浮卡 ──────────────────────────────────
    if ar_ans:
        ans_text = f'= {ar_ans}'
        card = (150, 175, W - 150, 265)
        glass_rect(img, card, alpha=0.65, bg=(20, 40, 20), radius=18,
                   border=M3['success'])
        # 外發光字
        for off, col in [(3, (0, 60, 0)), (1, (0, 160, 40))]:
            cv2.putText(img, ans_text, (167 + off, 238),
                        cv2.FONT_HERSHEY_DUPLEX, 1.3, col, 3 + off, cv2.LINE_AA)
        cv2.putText(img, ans_text, (167, 238),
                    cv2.FONT_HERSHEY_DUPLEX, 1.3, M3['success'], 2, cv2.LINE_AA)

    # ── Hover 高亮與進度弧 ───────────────────────────────
    hovered = None
    gaze_hovered = None
    
    btn_list = [(btn_mode, 'mode'), (btn_ink, 'ink'), (btn_size, 'size'),
                (btn_tool, 'tool'), (btn_save, 'save'), (btn_next, 'next'),
                (btn_back, 'back'), (btn_clear, 'clear'), (btn_track, 'track')]
                
    if hover_pt:
        for rect, name in btn_list:
            if is_in(hover_pt, rect):
                hovered = name
                if name == last_btn and hover_prog > 0:
                    # 亮邊框高亮
                    cv2.rectangle(img, rect[:2], rect[2:],
                                  M3['on_surface'], 2, cv2.LINE_AA)
                    cx = (rect[0] + rect[2]) // 2
                    hover_arc(img, cx, rect[3] + 14, 10, hover_prog,
                              color=M3['primary'])
                break

    if gaze_pt:
        for rect, name in btn_list:
            if is_in(gaze_pt, rect):
                gaze_hovered = name
                # 視線游標如果在按鈕上，加一個光暈
                cv2.rectangle(img, rect[:2], rect[2:], (255, 200, 100), 1, cv2.LINE_AA)
                break

    return hovered, gaze_hovered



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

    tracker   = HandTracker(
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.45
    )
    face_tracker = FaceTracker()
    gaze_solver  = GazeSolver()
    
    canvas    = CanvasManager(width=W, height=H, line_thickness=7)
    model_mgr = ModelManager()
    particles = ParticleSystem()
    mandala   = MagicMandala()
    coloring  = ColoringManager(canvas_w=W, canvas_h=H)
    keyboard  = GazeKeyboard(W, H)
    color_picker = ColorPicker(W//2, H//2, radius=130)
    color_picker_active = False
    preview_color = None
    color_hover_pt = None
    color_hover_start = 0.0

    mode_idx       = 0
    palette_idx    = 0
    thickness_idx  = 1   # 預設 index=1 → 7px
    active_tool    = 'brush' # 'brush' 或 'bucket'

    gesture_sm   = GestureStateMachine()
    neural_bloom = NeuralBloom()
    animals      = AnimalSpawner()
    pointer      = PointerSmoother(alpha=0.28)
    recognized_text       = ""
    ar_answer             = None
    last_draw_time        = time.time()
    _recognized_path_cnt  = 0   # 上次辨識後 canvas.paths 的數量
    #   → 只有新筆畫加入後才重新觸發辨識

    btn_cooldown     = 0.0
    last_hovered_btn = None
    hover_start_time = 0.0

    rock_triggered    = False
    fist_triggered    = False
    _fist_submit      = False
    last_recog_boxes  = []
    recog_box_expire  = 0.0
    prev_draw_pt      = None
    prev_time         = time.time()
    gesture_name      = 'hover'
    gesture_progress  = 0.0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        results, disp = tracker.process_frame(frame, optimize_lighting=False)
        
        # ── 臉部/眼球追蹤 ──────────────────────────────────
        face_results = face_tracker.process_frame(frame)
        gaze_pt = None
        is_blinking = False
        ear_val = 1.0
        if face_results.multi_face_landmarks:
            gx, gy, is_blinking, ear_val = gaze_solver.update(face_results.multi_face_landmarks[0], W, H)
            
            # 只在 TYPE 模式啟用並顯示視線游標，避免干擾手部畫畫
            if MODES[mode_idx] == 'type':
                gaze_pt = (gx, gy)

        curr_time   = time.time()
        mode_name   = MODES[mode_idx]
        ink_bgr     = PALETTES[palette_idx][1]    # (B, G, R)

        is_drawing  = False
        hover_point = None

        # ── 手部追蹤 ─────────────────────────────────────
        if results.hand_landmarks:
            lm = results.hand_landmarks[0]

            # ── 上游座標平滑：先對 raw landmark 做 EMA，後續所有操作都用平滑後的座標 ──
            raw_x = int(np.clip(lm[8].x * W, 5, W - 5))
            raw_y = int(np.clip(lm[8].y * H, 5, H - 5))
            x_idx, y_idx = pointer.update(raw_x, raw_y)

            # 中指只用於手勢判斷，對中指也稍微平滑
            x_mid = int(lm[12].x * W)
            y_mid = int(lm[12].y * H)
            hover_point = (x_idx, y_idx)

            # ── 防抖狀態機更新 ────────────────────────────
            gesture_name, gesture_progress = gesture_sm.update(lm, W, H)

            dist_fingers = math.sqrt((x_idx - x_mid) ** 2 + (y_idx - y_mid) ** 2)

            # ── 手勢分支 ──────────────────────────────────

            if gesture_name == 'palm_open' and mode_name == 'magic':
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
                # 搖滾 → 粒子爆炸 + 清空畫布+文字+填色
                for _ in range(80):
                    particles.spawn(
                        random.randint(80, 560), random.randint(60, 420),
                        color=ink_bgr, count=1)
                canvas.clear()
                if mode_name == 'art':
                    coloring.clear_fill()
                recognized_text = ""
                _recognized_path_cnt = 0
                canvas.notify_tracking_lost()

            elif gesture_name == 'thumb_up' and not fist_triggered:
                # ── 豎大拇指 → 撤銷最後一筆 / 或取消調色盤 ───────
                fist_triggered = True   # 用同一個 flag 防重複觸發
                
                if color_picker_active:
                    color_picker_active = False
                else:
                    canvas.end_stroke()
                    canvas.undo_last_stroke()
                    _recognized_path_cnt = min(_recognized_path_cnt, len(canvas.paths))

            elif gesture_name == 'fist' and not fist_triggered:
                fist_triggered = True
                canvas.end_stroke()
                canvas.notify_tracking_lost()
                if color_picker_active:
                    _fist_submit = False
                elif mode_name == 'art':
                    # ── Art 模式：握拳 → 儲存畫布為 PNG ──────
                    _fist_submit = False
                    from datetime import datetime
                    import os
                    save_dir = "C:/hand/saved_art"
                    os.makedirs(save_dir, exist_ok=True)
                    fname = f"{save_dir}/art_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    save_img = np.zeros_like(canvas.drawing_layer)
                    wm = cv2.inRange(canvas.drawing_layer, (200,200,200), (255,255,255))
                    save_img[wm > 0] = ink_bgr
                    cv2.imwrite(fname, save_img)
                    last_recog_boxes = [(-1, -1, -1, -1, f"Saved! {os.path.basename(fname)}")]
                    recog_box_expire = curr_time + 2.5
                elif not canvas.has_content() and recognized_text:
                    # ── 空畫布握拳 → 快速清除輸出文字 ─────────
                    recognized_text = ""
                    _fist_submit = False
                else:
                    # ── 一般模式：握拳 → 立即辨識 ─────────────
                    _fist_submit = True

            elif gesture_name == 'draw':
                # ── 確認進入畫圖模式 ──────────────────────
                is_drawing     = True
                rock_triggered = False
                last_draw_time = curr_time

                if mode_name == 'art' and active_tool == 'bucket':
                    if not getattr(canvas, '_bucket_triggered', False):
                        coloring.flood_fill(x_idx, y_idx, ink_bgr)
                        canvas._bucket_triggered = True
                        # 小粒子特效
                        for _ in range(15):
                            particles.spawn(x_idx, y_idx, color=ink_bgr, count=1)
                else:
                    # ── 跳躍過濾：單幀位移超過 60px → 視為雜訊（其他手指干擾）
                    _spike = False
                    if prev_draw_pt:
                        _dx = x_idx - prev_draw_pt[0]
                        _dy = y_idx - prev_draw_pt[1]
                        if math.sqrt(_dx*_dx + _dy*_dy) > 60:
                            _spike = True
    
                    if not _spike:
                        canvas.add_point(x_idx, y_idx)
                    if prev_draw_pt:
                        d = math.sqrt((x_idx - prev_draw_pt[0])**2 + (y_idx - prev_draw_pt[1])**2)
                        if d > 10:
                            particles.spawn(x_idx, y_idx, color=ink_bgr, count=2)
                            if mode_name == 'magic' and random.random() < 0.15:
                                animals.spawn(x_idx, y_idx)
                    prev_draw_pt = (x_idx, y_idx)

                # 準心：實心圓 + 白色外環
                cv2.circle(disp, (x_idx, y_idx), 9,  ink_bgr,        -1, cv2.LINE_AA)
                cv2.circle(disp, (x_idx, y_idx), 14, (255, 255, 255),  1, cv2.LINE_AA)

            else:
                # ── hover / 確認中 ────────────────────────
                rock_triggered = False
                canvas._bucket_triggered = False

                # 每次從畫圖切換回 hover，結束筆畫並偵測圓形
                was_drawing = canvas.points  # 如果還有未提交的點
                canvas.notify_tracking_lost()
                prev_draw_pt = None

                # 只有在 Magic 模式下才偵測圓形 → 觸發 Neural Bloom
                # 同時要求半徑 > MIN_BLOOM_RADIUS，避免與手寫 O/0 衝突
                if mode_name == 'magic' and not neural_bloom.active:
                    circle = canvas.detect_circle_in_last_path()
                    if circle:
                        bcx, bcy, brad = circle
                        if brad >= MIN_BLOOM_RADIUS:
                            neural_bloom.trigger(bcx, bcy, brad, color=ink_bgr)
                            # 清掉圓形筆跡，不進入辨識
                            canvas.paths.pop()
                            cv2.rectangle(canvas.drawing_layer,
                                          (bcx - brad - 20, bcy - brad - 20),
                                          (bcx + brad + 20, bcy + brad + 20),
                                          (0, 0, 0), -1)

                # 如果正在「充能」進入畫圖模式，顯示橘色弧形進度
                pending_draw = (gesture_sm._candidate == 'draw' and gesture_sm.state != 'draw')
                if pending_draw and gesture_progress > 0:
                    # 橘色充能環
                    angle = int(360 * gesture_progress)
                    cv2.circle(disp, (x_idx, y_idx), 7, (60, 180, 255), -1, cv2.LINE_AA)
                    cv2.ellipse(disp, (x_idx, y_idx), (20, 20), -90, 0, angle,
                                (60, 180, 255), 3, cv2.LINE_AA)
                    cv2.circle(disp, (x_idx, y_idx), 24, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    # 一般懸浮準心：空心雙環
                    cv2.circle(disp, (x_idx, y_idx), 7,  (180, 180, 180), -1, cv2.LINE_AA)
                    cv2.circle(disp, (x_idx, y_idx), 16, (255, 255, 255),  1, cv2.LINE_AA)
                    cv2.circle(disp, (x_idx, y_idx), 20, (200, 200, 200),  1, cv2.LINE_AA)

        else:
            gesture_sm.reset()
            pointer.reset()
            gesture_name     = 'hover'
            gesture_progress = 0.0
            fist_triggered    = False
            canvas.notify_tracking_lost()
            prev_draw_pt     = None
            last_hovered_btn = None
            rock_triggered   = False

        # ── 粒子更新 ──────────────────────────────────────
        particles.update_and_draw(disp)
        neural_bloom.update_and_draw(disp)
        animals.update_and_draw(disp)

        # ── 填色底圖 AR 疊加 ─────────────────────────────
        if mode_name == 'art' and coloring.has_image:
            coloring.blend_onto(disp)

        # ── 自動辨識倒數進度條 ────────────────────────────
        pixel_count  = canvas.get_pixel_count()
        time_elapsed = curr_time - last_draw_time
        # 只有「有新筆畫加入」且「非 magic/art」才顯示倒數
        _new_strokes = len(canvas.paths) > _recognized_path_cnt
        has_enough   = (_new_strokes and canvas.has_content()
                        and pixel_count >= MIN_PIXELS
                        and not is_drawing
                        and mode_name not in ('magic', 'art'))

        if has_enough:
            seg_progress = min(1.0, time_elapsed / SEG_TIMEOUT)
            bar_w = int((W - 30) * seg_progress)
            bar_y = H - 93
            cv2.line(disp, (15, bar_y), (15 + bar_w, bar_y), M3['success'], 3, cv2.LINE_AA)
            hint = f"Fist=submit  Thumb=undo  |  Auto in {max(0, SEG_TIMEOUT - time_elapsed):.1f}s"
            label(disp, hint, 20, bar_y - 6, scale=0.38, color=M3['success'])
        elif mode_name == 'art' and canvas.has_content():
            bar_y = H - 93
            img_label = f"[{coloring.current_name}]  " if coloring.has_image else ""
            hint = f"{img_label}Strokes:{canvas.stroke_count}  Fist=save  Thumb=undo  Rock=next img"
            label(disp, hint, 20, bar_y - 6, scale=0.38, color=M3['art_col'])

        # ── 辨識函式 (閉包) ───────────────────────────────
        def _do_recognize():
            nonlocal recognized_text, ar_answer, last_recog_boxes
            nonlocal recog_box_expire, last_draw_time, _recognized_path_cnt
            # 只辨識「尚未辨識」的新筆畫
            new_paths = canvas.paths[_recognized_path_cnt:]
            if not new_paths:
                return
            preds = model_mgr.predict_from_paths(
                new_paths,
                canvas_w=W, canvas_h=H,
                line_thickness=canvas.line_thickness,
                mode=mode_name)
            new_chars = ""
            last_recog_boxes = []
            for box, char, conf in preds:
                new_chars += char
                last_recog_boxes.append((box[0], box[1], box[2], box[3], char))
            if new_chars:
                recognized_text  += new_chars
                recog_box_expire  = curr_time + 2.0
                _recognized_path_cnt = len(canvas.paths)  # 更新已辨識筆畫數
                if '=' in recognized_text:
                    expr = recognized_text.rsplit('=', 1)[0]
                    ans, ok = safe_math_eval(expr)
                    if ok:
                        ar_answer = (ans, curr_time + 4.0)
            # !! 不再自動清空畫布 !! 等使用者主動搖滾或按 Clear

        def _do_train_save():
            nonlocal _recognized_path_cnt, last_draw_time, recognized_text
            import os
            new_paths = canvas.paths[_recognized_path_cnt:]
            if not new_paths: return
            
            groups = model_mgr._group_strokes(new_paths, gap_threshold=80)
            if not groups: return
            
            for grp in groups:
                (gx, gy, gw, gh), path_indices = grp
                if gw * gh < 100: continue
                mini = np.zeros((H, W), dtype=np.uint8)
                for idx in path_indices:
                    path = new_paths[idx]
                    if len(path) > 1:
                        for i in range(1, len(path)):
                            cv2.line(mini, path[i-1], path[i], 255, canvas.line_thickness, cv2.LINE_AA)
                            cv2.circle(mini, path[i], canvas.line_thickness//2, 255, -1, cv2.LINE_AA)
                    elif len(path) == 1:
                        cv2.circle(mini, path[0], canvas.line_thickness//2+1, 255, -1, cv2.LINE_AA)
                pad = 20
                x1, y1 = max(0, gx-pad), max(0, gy-pad)
                x2, y2 = min(W, gx+gw+pad), min(H, gy+gh+pad)
                roi = mini[y1:y2, x1:x2]
                if roi.size == 0: continue
                
                win_name = "Train Mode: Press key for this char (ESC to skip)"
                cv2.imshow(win_name, roi)
                key = cv2.waitKey(0)
                cv2.destroyWindow(win_name)
                
                if key not in (27, -1):
                    char_str = chr(key & 0xFF)
                    # Window cannot have some characters in folder names like '*', '?', '<', '>'
                    folder_name = char_str
                    if char_str == '*': folder_name = 'times'
                    elif char_str == '/': folder_name = 'div'
                    elif char_str == '<': folder_name = 'lt'
                    elif char_str == '>': folder_name = 'gt'
                    elif char_str == '?': folder_name = 'question'
                    elif char_str == '|': folder_name = 'pipe'
                    
                    save_dir = f"C:/hand/custom_dataset/{folder_name}"
                    os.makedirs(save_dir, exist_ok=True)
                    fname = f"{save_dir}/{int(time.time()*1000)}.png"
                    cv2.imwrite(fname, roi)
                    print(f"Saved {fname} for character '{char_str}'")
                    recognized_text += f" [Saved {char_str}] "
            
            _recognized_path_cnt = len(canvas.paths)
            canvas.clear()
            _recognized_path_cnt = 0

        # Magic/Art 模式不做辨識
        if mode_name == 'train':
            if _fist_submit:
                _fist_submit = False
                if canvas.has_content() and canvas.get_pixel_count() >= MIN_PIXELS:
                    _do_train_save()
        else:
            if has_enough and time_elapsed > SEG_TIMEOUT:
                _do_recognize()
            elif _fist_submit:
                _fist_submit = False
                if canvas.has_content() and canvas.get_pixel_count() >= MIN_PIXELS:
                    _do_recognize()

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

        hovered_btn, gaze_hovered_btn = draw_ui(
            disp, mode_name, palette_idx, thickness_idx, active_tool, gesture_name,
            recognized_text, hover_point, last_hovered_btn,
            hover_progress, ar_ans=ar_ans_display, gaze_pt=gaze_pt,
            use_eye_only=gaze_solver.use_eye_only
        )
        
        # ── Blink 觸發 UI ─────────────────────────────────
        if is_blinking and gaze_hovered_btn and (curr_time - btn_cooldown > 1.0):
            hovered_btn = gaze_hovered_btn # 模擬成按鈕被按下
            hover_start_time = 0.0 # bypass hover time check
            last_hovered_btn = hovered_btn

        # ── Hover 按鈕觸發 ────────────────────────────────
        if hovered_btn and not is_drawing:
            if hovered_btn == last_hovered_btn:
                if (curr_time - hover_start_time > 1.0 and
                        curr_time - btn_cooldown > 0.8):
                    if hovered_btn == 'clear':
                        recognized_text = ""
                        canvas.clear()
                        _recognized_path_cnt = 0
                    elif hovered_btn == 'back':
                        recognized_text = recognized_text[:-1]
                    elif hovered_btn == 'mode':
                        mode_idx = (mode_idx + 1) % len(MODES)
                        canvas.clear()
                        _recognized_path_cnt = 0
                    elif hovered_btn == 'ink':
                        color_picker_active = not color_picker_active
                        color_hover_start = curr_time
                    elif hovered_btn == 'size':
                        thickness_idx = (thickness_idx + 1) % len(THICKNESSES)
                        canvas.line_thickness = THICKNESSES[thickness_idx]
                    elif hovered_btn == 'tool':
                        active_tool = 'bucket' if active_tool == 'brush' else 'brush'
                    elif hovered_btn == 'track':
                        gaze_solver.use_eye_only = not gaze_solver.use_eye_only
                        if gaze_solver.use_eye_only and mode_name != 'type':
                            try:
                                mode_idx = MODES.index('type')
                            except ValueError:
                                pass
                            canvas.clear()
                    elif hovered_btn == 'save':
                        if mode_name == 'train':
                            import subprocess
                            subprocess.Popen(["cmd", "/c", "python C:\\hand\\retrain.py & echo. & pause"], creationflags=subprocess.CREATE_NEW_CONSOLE)
                            last_recog_boxes = [(-1, -1, -1, -1, f"Retraining...")]
                            recog_box_expire = curr_time + 4.0
                        else:
                            timestamp = int(time.time() * 1000)
                            save_path = os.path.join("C:\\hand", f"artwork_{timestamp}.jpg")
                            cv2.imwrite(save_path, disp)
                            last_recog_boxes = [(-1, -1, -1, -1, f"Saved!")]
                            recog_box_expire = curr_time + 2.0
                    elif hovered_btn == 'next':
                        name = coloring.next_image()
                        canvas.clear()
                        coloring.clear_fill()
                        _recognized_path_cnt = 0
                        last_recog_boxes = [(-1, -1, -1, -1, f"Image: {name}")]
                        recog_box_expire = curr_time + 2.0
                    btn_cooldown     = curr_time
                    hover_start_time = curr_time
            else:
                last_hovered_btn = hovered_btn
                hover_start_time = curr_time
        else:
            last_hovered_btn = None

        if color_picker_active:
            color_picker.draw(disp)
            # 提示文字
            cv2.putText(disp, "Hover 0.8s or Blink to select, THUMB to cancel", (60, H - 40), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 255, 200), 1, cv2.LINE_AA)
            # 允許視線游標預覽與選擇
            active_pt = hover_point
            if gaze_pt and color_picker.get_color(gaze_pt[0], gaze_pt[1]):
                active_pt = gaze_pt
                
            if active_pt:
                col = color_picker.get_color(active_pt[0], active_pt[1])
                if col is not None:
                    preview_color = col
                    if color_hover_pt:
                        dx = active_pt[0] - color_hover_pt[0]
                        dy = active_pt[1] - color_hover_pt[1]
                        if math.sqrt(dx*dx + dy*dy) < 8:
                            if curr_time - color_hover_start > 0.8 or (is_blinking and active_pt == gaze_pt):
                                # 自動選取 或 眨眼選取！
                                PALETTES[palette_idx] = ("Custom", preview_color)
                                color_picker_active = False
                                _fist_submit = False
                                color_hover_start = curr_time # reset
                        else:
                            color_hover_pt = active_pt
                            color_hover_start = curr_time
                    else:
                        color_hover_pt = active_pt
                        color_hover_start = curr_time
                else:
                    color_hover_pt = None
                    
            if preview_color:
                # 畫預覽色塊
                cv2.circle(disp, (color_picker.cx, color_picker.cy), 30, preview_color, -1, cv2.LINE_AA)
                cv2.circle(disp, (color_picker.cx, color_picker.cy), 30, (255, 255, 255), 2, cv2.LINE_AA)
                
                # 畫進度條
                if color_hover_pt and (curr_time - color_hover_start) > 0.1:
                    prog = min(1.0, (curr_time - color_hover_start) / 0.8)
                    angle = int(360 * prog)
                    cv2.ellipse(disp, (color_picker.cx, color_picker.cy), (38, 38),
                                0, 0, angle, (200, 255, 200), 3, cv2.LINE_AA)

        # ── 繪製 TYPE 模式的視線鍵盤 ────────────────────────
        if mode_name == 'type' and not color_picker_active:
            typed_key = keyboard.update_and_draw(disp, gaze_pt, is_blinking, curr_time, recognized_text)
            if typed_key:
                # 粒子特效
                if gaze_pt:
                    for _ in range(30):
                        particles.spawn(gaze_pt[0], gaze_pt[1], color=(200, 255, 100), count=1)
                        
                if typed_key == 'SPACE':
                    recognized_text += " "
                elif typed_key == 'BKSP':
                    recognized_text = recognized_text[:-1]
                elif typed_key == 'CLEAR':
                    recognized_text = ""
                elif typed_key == 'EXIT':
                    mode_idx = 0
                elif typed_key.startswith('SUG_'):
                    word = typed_key[4:]
                    # 把原本正在打的未完成拼字替換掉 (簡單實作：拔掉最後一個單字並接上建議)
                    parts = recognized_text.split(' ')
                    if parts:
                        parts.pop()
                    parts.append(word + " ")
                    recognized_text = " ".join(parts).lstrip()
                else:
                    recognized_text += typed_key



        # ── 辨識結果邊界框顯示 ───────────────────────────
        if curr_time < recog_box_expire:
            fade = min(1.0, (recog_box_expire - curr_time) / 0.5)  # 最後 0.5s 淡出
            for bx, by, bw, bh, bchar in last_recog_boxes:
                pad = 18
                x1, y1 = max(0, bx - pad), max(0, by - pad)
                x2, y2 = min(W, bx + bw + pad), min(H, by + bh + pad)
                col = tuple(int(c * fade) for c in (60, 255, 120))
                cv2.rectangle(disp, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
                # 角落裝飾線
                sz = 12
                for cx2, cy2, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                    cv2.line(disp,(cx2,cy2),(cx2+dx*sz,cy2),col,3,cv2.LINE_AA)
                    cv2.line(disp,(cx2,cy2),(cx2,cy2+dy*sz),col,3,cv2.LINE_AA)
                # 字元標籤
                label(disp, bchar, x1 + 4, y1 - 6, scale=0.7,
                      color=tuple(int(c * fade) for c in (100, 255, 150)), thick=2)

        # ── 筆跡 AR 疊加 (著色) ───────────────────────────
        draw_lyr = canvas.drawing_layer
        white_mask = cv2.inRange(draw_lyr, (200, 200, 200), (255, 255, 255))
        colored_lyr = np.zeros_like(draw_lyr)
        colored_lyr[white_mask > 0] = ink_bgr
        disp[white_mask > 0] = colored_lyr[white_mask > 0]

        # 即時筆跡（當幀尚未 commit 的點）
        canvas.draw_current_stroke(disp, ink_color=ink_bgr)

        # ── 軳架顯示（幫助使用者看到 MediaPipe 實際追蹤到哪裏）──
        disp = tracker.draw_landmarks(disp, results)

        # ── FPS 顯示 ──────────────────────────────────────
        now  = time.time()
        fps  = 1.0 / max(now - prev_time, 0.001)
        prev_time = now
        label(disp, f"FPS {int(fps)}", 16, 120, scale=0.55, color=(0, 220, 220))

        # ── 繪製視線游標 (最上層) ─────────────────────────
        if gaze_pt and MODES[mode_idx] == 'type':
            cv2.circle(disp, gaze_pt, 12, (255, 150, 0), 2, cv2.LINE_AA)
            cv2.circle(disp, gaze_pt, 4, (0, 255, 255), -1, cv2.LINE_AA)

        cv2.imshow('Doctor Strange AR', disp)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    face_tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
