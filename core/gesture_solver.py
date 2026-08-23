import math
import numpy as np


# 簡單的安全 eval：只允許數學相關 token
_ALLOWED_NAMES = {k: v for k, v in math.__dict__.items() if not k.startswith('_')}
_ALLOWED_NAMES.update({'abs': abs, 'round': round})

def safe_math_eval(expr: str):
    """
    嘗試安全地對數學運算式求值。
    支援：+ - * / ** sqrt sin cos tan log exp pi
    回傳 (result_str, ok) 兩個值。
    """
    # 符號替換：把手寫辨識出的特殊符號對應成 Python 能算的語法
    subs = {
        'sqrt': 'sqrt', 'pi': 'pi', 'sin': 'sin', 'cos': 'cos',
        'tan': 'tan', 'log': 'log', 'exp': 'exp',
        'times': '*', '×': '*', '÷': '/',
        'pm': '+', 'neq': '!=', 'leq': '<=', 'geq': '>=',
        'infty': 'inf'
    }
    
    cleaned = expr.strip().rstrip('=').strip()
    for k, v in subs.items():
        cleaned = cleaned.replace(k, v)
    
    # 移除非法字元（只保留數學符號）
    import re
    if not re.match(r'^[0-9+\-*/^().a-zA-Z_ ]+$', cleaned):
        return None, False
    
    # 替換 ** 風格的次方 (symbol 模型輸出的 ^ 對應 ** )
    cleaned = cleaned.replace('^', '**')
    
    try:
        result = eval(cleaned, {"__builtins__": {}}, _ALLOWED_NAMES)
        if isinstance(result, float):
            if result == int(result):
                return str(int(result)), True
            return f"{result:.4g}", True
        return str(result), True
    except Exception:
        return None, False


def detect_gesture(hand_landmarks, frame_w, frame_h):
    """
    從 MediaPipe hand_landmarks 中偵測目前的手勢類型。
    回傳字串：'draw', 'hover', 'palm_open', 'rock', 'unknown'
    """
    if hand_landmarks is None:
        return 'unknown'
    
    lm = hand_landmarks
    
    def tip(idx): return (lm[idx].x, lm[idx].y)
    def pip(idx): return (lm[idx].x, lm[idx].y)
    
    # 判斷每根手指是否伸展 (Tip y < PIP y 代表伸直，y 軸由上至下增大)
    # 大拇指用 x 軸判斷
    def finger_up(tip_idx, pip_idx, is_thumb=False):
        t = lm[tip_idx]
        p = lm[pip_idx]
        if is_thumb:
            return abs(t.x - lm[0].x) > abs(p.x - lm[0].x)
        return t.y < p.y

    index_up  = finger_up(8,  6)
    middle_up = finger_up(12, 10)
    ring_up   = finger_up(16, 14)
    pinky_up  = finger_up(20, 18)
    thumb_up  = finger_up(4,  3, is_thumb=True)
    
    all_up = index_up and middle_up and ring_up and pinky_up
    
    # 計算食指與中指距離
    x_idx, y_idx = int(lm[8].x * frame_w), int(lm[8].y * frame_h)
    x_mid, y_mid = int(lm[12].x * frame_w), int(lm[12].y * frame_h)
    dist = math.sqrt((x_idx - x_mid)**2 + (y_idx - y_mid)**2)
    
    # ---- 手勢判定邏輯 ----
    
    # 張開手掌 (五指全開)：魔法陣
    if all_up and thumb_up:
        return 'palm_open'
    
    # 搖滾手勢 (食指 + 小指，中指與無名指收)：施法大絕
    if index_up and pinky_up and not middle_up and not ring_up:
        return 'rock'
    
    # 食指伸出、中指也伸出但距離較遠：畫圖模式
    if index_up and dist > 45:
        return 'draw'
    
    # 雙指合攏或中指單獨伸出：懸浮/選單模式
    if index_up or middle_up:
        return 'hover'
    
    return 'unknown'


def get_palm_center(hand_landmarks, frame_w, frame_h):
    """取得手掌中心座標"""
    # 取 0 (腕), 5, 9, 13, 17 的平均
    palm_ids = [0, 5, 9, 13, 17]
    xs = [hand_landmarks[i].x * frame_w for i in palm_ids]
    ys = [hand_landmarks[i].y * frame_h for i in palm_ids]
    return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))
