import cv2
import numpy as np
import os
import glob

COLORING_DIR = r'C:/hand/coloring'

def _to_coloring_style(img_bgr, canvas_w, canvas_h):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    white_ratio = np.sum(gray > 240) / gray.size
    if white_ratio > 0.5:
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        outline = binary
    else:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(blurred, 40, 120)
        kernel  = np.ones((3, 3), np.uint8)
        edges   = cv2.dilate(edges, kernel, iterations=1)
        outline = cv2.bitwise_not(edges)
    h, w = outline.shape[:2]
    scale = min(canvas_w / w, canvas_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(outline, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)
    x0 = (canvas_w - nw) // 2
    y0 = (canvas_h - nh) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

class ColoringManager:
    def __init__(self, canvas_w=640, canvas_h=480):
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self._images  = []
        self._idx     = -1
        self.current  = None
        self._load_all()

    def load_file(self, path):
        img = cv2.imread(path)
        if img is None:
            return False
        name    = os.path.basename(path)
        outline = _to_coloring_style(img, self.canvas_w, self.canvas_h)
        self._images.append((name, outline))
        self._idx   = len(self._images) - 1
        self.current = outline
        return True

    def next_image(self):
        n = len(self._images)
        if n == 0:
            self._idx = -1
        else:
            self._idx = (self._idx + 1) % (n + 1)
            if self._idx == n:
                self._idx = -1
        self.current = self._images[self._idx][1] if self._idx >= 0 else None
        return self.current_name

    @property
    def current_name(self):
        if self._idx < 0:
            return 'No image'
        return self._images[self._idx][0]

    @property
    def has_image(self):
        return self.current is not None

    def blend_onto(self, frame_bgr, alpha=0.95):
        if self.current is None:
            return
        # 將背景調淡（白化/霧化），降低攝影機畫面干擾
        white_bg = np.full_like(frame_bgr, 255)
        bg = cv2.addWeighted(frame_bgr, 0.25, white_bg, 0.75, 0)
        
        # 疊加黑色線稿
        mask = self.current < 128
        bg[mask] = [30, 20, 30]  # 深灰色線條
        
        # 蓋回原本的畫面中
        np.copyto(frame_bgr, bg)

    def get_outline_bgr(self):
        if self.current is None:
            return None
        return cv2.cvtColor(self.current, cv2.COLOR_GRAY2BGR)

    def image_count(self):
        return len(self._images)

    def _load_all(self):
        patterns = [os.path.join(COLORING_DIR, f'*.{ext}')
                    for ext in ('jpg', 'jpeg', 'png')]
        files = []
        for p in patterns:
            files.extend(sorted(glob.glob(p)))
        for f in files:
            img = cv2.imread(f)
            if img is None:
                continue
            outline = _to_coloring_style(img, self.canvas_w, self.canvas_h)
            self._images.append((os.path.basename(f), outline))
            print(f'[Coloring] Loaded: {os.path.basename(f)}')
        if self._images:
            print(f'[Coloring] {len(self._images)} pages ready. Rock in Art mode to cycle.')
            self._idx = 0
            self.current = self._images[0][1]
