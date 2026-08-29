import cv2
import time
import math

LAYOUTS = {
    'EN': [
        ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
        ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
        ['Z', 'X', 'C', 'V', 'B', 'N', 'M'],
        ['LANG', 'SPACE', 'BKSP', 'CLEAR', 'EXIT']
    ],
    'NUM / SYM': [
        ['7', '8', '9', '+', '-'],
        ['4', '5', '6', '*', '/'],
        ['1', '2', '3', '(', ')'],
        ['LANG', '0', '.', '=', 'BKSP', 'EXIT']
    ],
    'ZHUYIN (注音)': [
        ['ㄅ','ㄉ','ˇ','ˋ','ㄓ','ˊ','˙','ㄚ','ㄞ','ㄢ'],
        ['ㄆ','ㄊ','ㄍ','ㄐ','ㄔ','ㄗ','ㄧ','ㄛ','ㄟ','ㄣ'],
        ['ㄇ','ㄋ','ㄎ','ㄑ','ㄕ','ㄘ','ㄨ','ㄜ','ㄠ','ㄤ'],
        ['ㄈ','ㄌ','ㄏ','ㄒ','ㄖ','ㄙ','ㄩ','ㄝ','ㄡ','ㄥ'],
        ['LANG', 'SPACE', 'BKSP', 'CLEAR', 'EXIT']
    ]
}

class GazeKeyboard:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.layout_names = list(LAYOUTS.keys())
        self.layout_idx = 0
        self.keys = [] # list of (rect, label)
        
        self.hovered_key = None
        self.hover_start = 0.0
        self._build_layout()
        
    def _build_layout(self):
        self.keys = []
        layout = LAYOUTS[self.layout_names[self.layout_idx]]
        
        key_h = 45
        gap = 8
        start_y = self.h - (len(layout) * (key_h + gap)) - 20
        
        for r_idx, row in enumerate(layout):
            # 以每列 10 個鍵為基準計算寬度，確保按鍵不會太大
            max_keys = max(len(r) for r in layout)
            key_w = (self.w - 20 - (max_keys - 1) * gap) // max_keys
            
            # 若某些按鍵需要比較寬 (例如 SPACE)
            actual_row_w = 0
            for k in row:
                if k in ('SPACE', 'LANG', 'BKSP', 'CLEAR', 'EXIT'):
                    actual_row_w += int(key_w * 1.5) + gap
                else:
                    actual_row_w += key_w + gap
            actual_row_w -= gap
            
            start_x = (self.w - actual_row_w) // 2
            cx = start_x
            
            for key in row:
                kw = int(key_w * 1.5) if key in ('SPACE', 'LANG', 'BKSP', 'CLEAR', 'EXIT') else key_w
                x1 = cx
                x2 = cx + kw
                y1 = start_y + r_idx * (key_h + gap)
                y2 = y1 + key_h
                self.keys.append(((x1, y1, x2, y2), key))
                cx += kw + gap

    def is_in(self, pt, rect):
        return rect[0] <= pt[0] <= rect[2] and rect[1] <= pt[1] <= rect[3]

    def update_and_draw(self, img, gaze_pt, is_blinking, curr_time):
        """
        繪製鍵盤，並根據視線游標與眨眼更新狀態。
        回傳被觸發的按鍵字元 (若有)。
        """
        triggered_key = None
        current_hover = None
        
        # 畫鍵盤底板
        first_y = self.keys[0][0][1] - 15
        bg_rect = (5, first_y, self.w - 5, self.h - 5)
        overlay = img.copy()
        cv2.rectangle(overlay, bg_rect[:2], bg_rect[2:], (20, 20, 30), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # 尋找 Hover 的按鍵
        if gaze_pt:
            for rect, label in self.keys:
                if self.is_in(gaze_pt, rect):
                    current_hover = label
                    break
                    
        if current_hover != self.hovered_key:
            self.hovered_key = current_hover
            self.hover_start = curr_time
            
        # 繪製所有按鍵
        for rect, label in self.keys:
            x1, y1, x2, y2 = rect
            is_hover = (label == self.hovered_key)
            
            bg_col = (80, 150, 255) if is_hover else (50, 60, 70)
            text_col = (255, 255, 255)
            
            if label in ('LANG', 'SPACE', 'BKSP', 'CLEAR'):
                bg_col = (100, 180, 100) if is_hover else (40, 80, 50)
            elif label == 'EXIT':
                bg_col = (60, 60, 200) if is_hover else (40, 40, 120)
                
            cv2.rectangle(img, (x1, y1), (x2, y2), bg_col, -1, cv2.LINE_AA)
            cv2.rectangle(img, (x1, y1), (x2, y2), (100, 100, 120), 1, cv2.LINE_AA)
            
            # 文字置中
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6 if len(label) > 1 else 0.7
            (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
            tx = x1 + (x2 - x1 - tw) // 2
            ty = y1 + (y2 - y1 + th) // 2
            cv2.putText(img, label, (tx, ty), font, scale, text_col, 1, cv2.LINE_AA)
            
            # 畫進度條
            if is_hover and current_hover:
                prog = min(1.0, (curr_time - self.hover_start) / 0.8) # 0.8s hover
                if prog > 0.1:
                    cv2.rectangle(img, (x1, y2 - 4), (x1 + int((x2 - x1) * prog), y2), (0, 255, 255), -1)
                
                if prog >= 1.0 or is_blinking:
                    triggered_key = label
                    self.hover_start = curr_time + 0.5 # cooldown
                    
        # 處理特殊功能鍵
        if triggered_key == 'LANG':
            self.layout_idx = (self.layout_idx + 1) % len(self.layout_names)
            self._build_layout()
            return None # 攔截不輸出
            
        return triggered_key
