import cv2
import time
import math

LAYOUTS = {
    'EN': [
        ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
        ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
        ['SHIFT', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', 'BKSP'],
        ['EN/中', '?123', ',', 'SPACE', '.', 'ENTER', 'EXIT']
    ],
    'NUM / SYM': [
        ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
        ['@', '#', '$', '%', '&', '-', '+', '=', '(', ')'],
        ['*', '"', "'", ':', ';', '!', '?', 'BKSP'],
        ['EN/中', '?123', ',', 'SPACE', '.', 'ENTER', 'EXIT']
    ],
    'ZHUYIN (注音)': [
        ['ㄅ','ㄉ','ˇ','ˋ','ㄓ','ˊ','˙','ㄚ','ㄞ','ㄢ','ㄦ'],
        ['ㄆ','ㄊ','ㄍ','ㄐ','ㄔ','ㄗ','ㄧ','ㄛ','ㄟ','ㄣ','?123'],
        ['ㄇ','ㄋ','ㄎ','ㄑ','ㄕ','ㄘ','ㄨ','ㄜ','ㄠ','ㄤ','BKSP'],
        ['ㄈ','ㄌ','ㄏ','ㄒ','ㄖ','ㄙ','ㄩ','ㄝ','ㄡ','ㄥ','ENTER'],
        ['EN/中', ',', 'SPACE', '.', 'EXIT']
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
                if k in ('SPACE', 'EN/中', '?123', 'BKSP', 'CLEAR', 'EXIT'):
                    actual_row_w += int(key_w * 1.5) + gap
                else:
                    actual_row_w += key_w + gap
            actual_row_w -= gap
            
            start_x = (self.w - actual_row_w) // 2
            cx = start_x
            
            for key in row:
                kw = int(key_w * 1.5) if key in ('SPACE', 'EN/中', '?123', 'BKSP', 'CLEAR', 'EXIT') else key_w
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
            'ㄋㄧㄣ': ["您", "您好"],
            'ㄏ': ["好", "很", "還", "會", "和"],
            'ㄏㄠ': ["好", "好像", "好笑"],
            'ㄏㄨㄟ': ["會", "回去", "回來"],
            'ㄨ': ["我", "我們", "為什麼", "問題"],
            'ㄨㄛ': ["我", "我們"],
            'ㄨㄟ': ["為", "喂", "為什麼"],
            'ㄕ': ["是", "什麼", "時間"],
            'ㄕㄉ': ["是的", "誰的"],
            'ㄕㄜ': ["什麼", "設計"],
            'ㄉ': ["的", "對", "大", "到"],
            'ㄉㄨㄟ': ["對", "對不起", "對吧"],
            'ㄉㄠ': ["到", "到底", "道"],
            'ㄅ': ["不", "把", "被", "比"],
            'ㄅㄨ': ["不", "不是", "不用", "不會"],
            'ㄒ': ["謝謝", "想", "喜歡", "小"],
            'ㄒㄧㄝ': ["謝謝", "些", "寫"],
            'ㄒㄧㄤ': ["想", "相信", "向"],
            'ㄒㄧㄠ': ["小", "笑", "效果"],
            'ㄒㄧㄢ': ["現在", "先生", "先"],
            'ㄗ': ["在", "做", "怎麼", "走"],
            'ㄗㄞ': ["在", "再見", "再來"],
            'ㄗㄨㄛ': ["做", "左邊", "作"],
            'ㄐ': ["就", "家", "今天", "見"],
            'ㄐㄧㄣ': ["今天", "進來", "近"],
            'ㄐㄧㄢ': ["見", "簡單", "件"],
            'ㄇ': ["嗎", "沒", "買", "賣", "明"],
            'ㄇㄟ': ["沒有", "沒事", "妹妹"],
            'ㄇㄧㄥ': ["明天", "明白", "名字"],
            'ㄧ': ["一", "有", "要", "也", "以"],
            'ㄧㄡ': ["有", "右邊", "優秀"],
            'ㄧㄠ': ["要", "要求"],
            'ㄧㄥ': ["應該", "英文", "贏"],
            'ㄓ': ["這", "知道", "中", "只"],
            'ㄓㄜ': ["這", "這個", "這裡"],
            'ㄓㄉ': ["知道", "直到"],
            'ㄓㄨㄥ': ["中文", "中間", "中國"],
            'ㄊ': ["他", "她", "它", "太", "天"],
            'ㄊㄚ': ["他", "她", "它們"],
            'ㄊㄞ': ["太", "台灣", "台北"],
            'ㄊㄧㄢ': ["天", "天氣", "天天"],
            'ㄎ': ["可", "看", "開", "快"],
            'ㄎㄜ': ["可以", "可能", "可愛"],
            'ㄎㄢ': ["看", "看到", "看見"],
            'ㄌ': ["了", "來", "裡", "老"],
            'ㄌㄞ': ["來", "來到"],
            'ㄌㄧ': ["裡面", "力量", "理"],
            'ㄍ': ["個", "高", "過", "給"],
            'ㄍㄜ': ["個", "哥哥", "各"],
            'ㄍㄨㄛ': ["過", "國家", "過去"],
            'ㄑ': ["去", "請", "前", "起"],
            'ㄑㄩ': ["去", "去年", "區域"],
            'ㄑㄧㄥ': ["請", "清楚", "情況"],
        }
        
        # 先嘗試完全匹配
        if current_word in dict_map:
            return dict_map[current_word][:3]
            
        # 再嘗試前綴匹配注音
        z_matches = []
        for k, v in dict_map.items():
            if k.startswith(current_word) or current_word.startswith(k):
                z_matches.extend(v)
        if z_matches:
            # 去重複並維持順序
            seen = set()
            res = []
            for w in z_matches:
                if w not in seen:
                    seen.add(w)
                    res.append(w)
            if len(res) > 0:
                return res[:3]
            
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
            
            if label in ('EN/中', '?123', 'SPACE', 'BKSP', 'CLEAR'):
                bg_col = (100, 180, 100) if is_hover else (40, 80, 50)
            elif label == 'EXIT':
                bg_col = (60, 60, 200) if is_hover else (40, 40, 120)
                
            cv2.rectangle(img, (x1, y1), (x2, y2), bg_col, -1, cv2.LINE_AA)
            cv2.rectangle(img, (x1, y1), (x2, y2), (100, 100, 120), 1, cv2.LINE_AA)
            
            # 暫存文字以批次繪製 (避免過多 PIL/numpy 轉換)
            if not hasattr(self, 'texts_to_draw'):
                self.texts_to_draw = []
            
            # 使用統一大小的字體
            tx = x1 + (x2 - x1) // 2
            ty = y1 + (y2 - y1) // 2
            self.texts_to_draw.append((label, (tx, ty), text_col, True)) # True = center align
            
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
            
            tx = x1 + (x2 - x1) // 2
            ty = y1 + (y2 - y1) // 2
            self.texts_to_draw.append((label, (tx, ty), (255, 255, 255), True))
            
            if is_hover:
                prog = min(1.0, (curr_time - self.hover_start) / 0.8)
                if prog > 0.1:
                    cv2.rectangle(img, (x1, y2 - 4), (x1 + int((x2 - x1) * prog), y2), (0, 255, 255), -1)
                if prog >= 1.0 or is_blinking:
                    triggered_key = f"SUG_{label}"
                    self.hover_start = curr_time + 0.5

        # 批次使用 PIL 畫中文
        if self.texts_to_draw:
            from PIL import ImageFont, ImageDraw, Image
            import numpy as np
            try:
                font = ImageFont.truetype("msjh.ttc", 20)
            except:
                font = ImageFont.load_default()
            
            img_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(img_pil)
            for text, pos, color, center in self.texts_to_draw:
                # anchor='mm' 讓文字置中於 pos
                if hasattr(font, 'getbbox'):
                    draw.text(pos, text, font=font, fill=color, anchor="mm")
                else:
                    # fallback
                    draw.text((pos[0]-10, pos[1]-10), text, font=font, fill=color)
            img[:] = np.array(img_pil)
            self.texts_to_draw.clear()

        # 處理特殊功能鍵
        if triggered_key == 'EN/中':
            # 只在 EN 和 ZHUYIN 之間切換
            try:
                self.layout_idx = self.layout_names.index('ZHUYIN (注音)') if self.layout_names[self.layout_idx] != 'ZHUYIN (注音)' else self.layout_names.index('EN')
            except ValueError:
                self.layout_idx = 0
            self._build_layout()
            return None
            
        if triggered_key == '?123':
            try:
                num_idx = self.layout_names.index('NUM / SYM')
                en_idx = self.layout_names.index('EN')
                self.layout_idx = en_idx if self.layout_idx == num_idx else num_idx
            except ValueError:
                pass
            self._build_layout()
            return None

        return triggered_key
