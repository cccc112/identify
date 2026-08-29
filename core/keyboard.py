import cv2
import time
import math

LAYOUTS = {
    'EN': [
        ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
        ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
        ['SHIFT', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', 'BKSP'],
        ['EN/中', ',', 'SPACE', '.', 'ENTER', 'EXIT']
    ],
    'NUM / SYM': [
        ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
        ['@', '#', '$', '%', '&', '-', '+', '(', ')'],
        ['*', '"', "'", ':', ';', '!', '?', 'BKSP'],
        ['EN/中', ',', 'SPACE', '.', 'ENTER', 'EXIT']
    ],
    'ZHUYIN (注音)': [
        ['ㄅ','ㄉ','ˇ','ˋ','ㄓ','ˊ','˙','ㄚ','ㄞ','ㄢ'],
        ['ㄆ','ㄊ','ㄍ','ㄐ','ㄔ','ㄗ','ㄧ','ㄛ','ㄟ','ㄣ'],
        ['ㄇ','ㄋ','ㄎ','ㄑ','ㄕ','ㄘ','ㄨ','ㄜ','ㄠ','ㄤ', 'BKSP'],
        ['EN/中', 'ㄈ', 'SPACE', 'ㄥ', 'ENTER', 'EXIT'] # Simplified for spacing, or standard:
    ]
}

# Fix standard Zhuyin bottom rows (Mobile puts BKSP on row 3 usually, but Zhuyin has 10 keys in row 3).
LAYOUTS['ZHUYIN (注音)'] = [
    ['ㄅ','ㄉ','ˇ','ˋ','ㄓ','ˊ','˙','ㄚ','ㄞ','ㄢ'],
    ['ㄆ','ㄊ','ㄍ','ㄐ','ㄔ','ㄗ','ㄧ','ㄛ','ㄟ','ㄣ'],
    ['ㄇ','ㄋ','ㄎ','ㄑ','ㄕ','ㄘ','ㄨ','ㄜ','ㄠ','ㄤ'],
    ['ㄈ','ㄌ','ㄏ','ㄒ','ㄖ','ㄙ','ㄩ','ㄝ','ㄡ','ㄥ', 'BKSP'],
    ['EN/中', ',', 'SPACE', '.', 'ENTER', 'EXIT']
]

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
                if k in ('SPACE', 'EN/中', 'BKSP', 'CLEAR', 'EXIT'):
                    actual_row_w += int(key_w * 1.5) + gap
                else:
                    actual_row_w += key_w + gap
            actual_row_w -= gap
            
            start_x = (self.w - actual_row_w) // 2
            cx = start_x
            
            for key in row:
                kw = int(key_w * 1.5) if key in ('SPACE', 'EN/中', 'BKSP', 'CLEAR', 'EXIT') else key_w
                x1 = cx
                x2 = cx + kw
                y1 = start_y + r_idx * (key_h + gap)
                y2 = y1 + key_h
                self.keys.append(((x1, y1, x2, y2), key))
                cx += kw + gap

    def is_in(self, pt, rect):
        return rect[0] <= pt[0] <= rect[2] and rect[1] <= pt[1] <= rect[3]

    def _get_recommendations(self, current_word):
        if not current_word:
            return ["你好", "我", "謝謝"]
            
        # 簡單的注音與英文詞庫
        dict_map = {
            'ㄋ': ["你", "你好", "哪裡", "那個", "能"],
            'ㄋㄧ': ["你", "你好", "你們", "年輕"],
            'ㄏ': ["好", "很", "還", "會", "和"],
            'ㄏㄠ': ["好", "好像", "好笑"],
            'ㄨ': ["我", "我們", "為什麼", "問題"],
            'ㄨㄛ': ["我", "我們"],
            'ㄕ': ["是", "什麼", "時間"],
            'ㄕㄉ': ["是的", "誰的"],
            'ㄉ': ["的", "對", "大", "到"],
            'ㄉㄨㄟ': ["對", "對不起", "對吧"],
            'ㄅ': ["不", "把", "被", "比"],
            'ㄅㄨ': ["不", "不是", "不用", "不會"],
            'ㄒ': ["謝謝", "想", "喜歡", "小"],
            'ㄒㄧㄝ': ["謝謝", "些", "寫"],
            'ㄗ': ["在", "做", "怎麼", "走"],
            'ㄗㄞ': ["在", "再見", "再來"],
            'ㄐ': ["就", "家", "今天", "見"],
            'ㄇ': ["嗎", "沒", "買", "賣"],
            'ㄇㄟ': ["沒有", "沒事", "妹妹"],
            'ㄧ': ["一", "有", "要", "也"],
            'ㄧㄡ': ["有", "右邊", "優秀"]
        }
        
        # 嘗試直接匹配
        if current_word in dict_map:
            return dict_map[current_word][:3]
            
        # 嘗試英文前綴 (增加常見基礎單字)
        en_words = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
            "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
            "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
            "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
            "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
            "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
            "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
            "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
            "hello", "morning", "night", "thanks", "sorry", "please", "yes", "where", "why"
        ]
        matches = [w for w in en_words if w.startswith(current_word.lower())]
        if matches:
            return matches[:3]
            
        # 兜底：直接返回使用者目前打的注音字串作為一個「選項」
        return [current_word]

    def update_and_draw(self, img, gaze_pt, is_blinking, curr_time, recognized_text=""):
        """
        繪製鍵盤，並根據視線游標與眨眼更新狀態。
        回傳被觸發的按鍵字元 (若有)。
        """
        triggered_key = None
        current_hover = None
        
        # 畫鍵盤底板
        first_y = self.keys[0][0][1] - 40 # 增加空間給推薦列
        bg_rect = (5, first_y, self.w - 5, self.h - 5)
        overlay = img.copy()
        cv2.rectangle(overlay, bg_rect[:2], bg_rect[2:], (20, 20, 30), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # 推薦字
        # 擷取字串最後的連續注音或英文字母作為「正在拼的字」
        import re
        match = re.search(r'([a-zA-Zㄅ-ㄩㄚ-ㄦ˙ˊˇˋ]+)$', recognized_text)
        current_word = match.group(1) if match else ""
        suggestions = self._get_recommendations(current_word)
        
        s_rects = []
        if suggestions:
            sw = (self.w - 20 - (len(suggestions)-1)*8) // len(suggestions)
            cx = 10
            sy = first_y + 5
            for s in suggestions:
                s_rect = (cx, sy, cx+sw, sy+30)
                s_rects.append((s_rect, s))
                cx += sw + 8

        # 尋找 Hover 的按鍵或推薦字
        if gaze_pt:
            for rect, label in s_rects:
                if self.is_in(gaze_pt, rect):
                    current_hover = f"SUG_{label}"
                    break
            if not current_hover:
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
            
            if label in ('EN/中', 'SPACE', 'BKSP', 'CLEAR'):
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

        # 繪製推薦字
        for rect, label in s_rects:
            x1, y1, x2, y2 = rect
            is_hover = (self.hovered_key == f"SUG_{label}")
            bg_col = (100, 180, 255) if is_hover else (30, 40, 50)
            cv2.rectangle(img, (x1, y1), (x2, y2), bg_col, -1, cv2.LINE_AA)
            cv2.rectangle(img, (x1, y1), (x2, y2), (150, 150, 180), 1, cv2.LINE_AA)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            # 若為中文，建議換能支援中文的字體 (此處先用簡化的 putText 或仰賴系統)
            # 在 OpenCV 原生 putText 不支援中文，但我們還是先丟進去
            (tw, th), _ = cv2.getTextSize(label, font, 0.6, 1)
            tx = x1 + (x2 - x1 - tw) // 2
            ty = y1 + (y2 - y1 + th) // 2
            cv2.putText(img, label, (tx, ty), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            if is_hover:
                prog = min(1.0, (curr_time - self.hover_start) / 0.8)
                if prog > 0.1:
                    cv2.rectangle(img, (x1, y2 - 4), (x1 + int((x2 - x1) * prog), y2), (0, 255, 255), -1)
                if prog >= 1.0 or is_blinking:
                    triggered_key = f"SUG_{label}"
                    self.hover_start = curr_time + 0.5

        # 處理特殊功能鍵
        if triggered_key == 'EN/中':
            self.layout_idx = (self.layout_idx + 1) % len(self.layout_names)
            self._build_layout()
            return None # 攔截不輸出
            
        return triggered_key
