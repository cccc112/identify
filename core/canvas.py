import cv2
import numpy as np
import math


class CanvasManager:
    def __init__(self, width=640, height=480, line_thickness=7):
        self.width = width
        self.height = height
        self.line_thickness = line_thickness

        # 白色筆跡 (黑底)，方便 CNN 辨識
        self.drawing_layer = np.zeros((height, width, 3), dtype=np.uint8)
        self.points   = []   # 目前正在進行中的筆畫點
        self.paths    = []   # 已完成的所有筆畫

        # EMA 濾波器：平滑手部抖動
        self._ema_x = None
        self._ema_y = None
        self._ema_alpha = 0.40  # 越小越滑順但越有延遲，0.4 是體感最佳平衡

        # 容錯 buffer：短暫漏偵測時不要切斷筆畫
        self._miss_frames  = 0
        self._max_miss     = 4   # 最多容忍連續 4 幀遺失

        # 最小採樣間距：避免同一位置重複採樣造成密集噪點
        self._min_dist = 5

    # ── 公開 API ──────────────────────────────────────────────

    def smooth_point(self, raw_x, raw_y):
        """EMA 低通濾波，消除手部抖動"""
        if self._ema_x is None:
            self._ema_x, self._ema_y = float(raw_x), float(raw_y)
        else:
            a = self._ema_alpha
            self._ema_x = a * raw_x + (1 - a) * self._ema_x
            self._ema_y = a * raw_y + (1 - a) * self._ema_y
        return int(self._ema_x), int(self._ema_y)

    def add_point(self, raw_x, raw_y):
        """加入新座標（自動套用 EMA 濾波與最小間距過濾）"""
        self._miss_frames = 0
        sx, sy = self.smooth_point(raw_x, raw_y)

        # 如果與上一個點太近，跳過（避免鋸齒密點）
        if self.points:
            lx, ly = self.points[-1]
            if math.sqrt((sx - lx) ** 2 + (sy - ly) ** 2) < self._min_dist:
                return

        self.points.append((sx, sy))

    def notify_tracking_lost(self):
        """通知系統這一幀追蹤失敗；容錯 N 幀後才真正結束筆畫"""
        self._miss_frames += 1
        if self._miss_frames >= self._max_miss:
            self.end_stroke()

    def end_stroke(self):
        """強制結束目前筆畫，寫入 drawing_layer"""
        self._miss_frames = 0
        self._ema_x = self._ema_y = None  # 重置濾波器

        if len(self.points) > 1:
            for i in range(1, len(self.points)):
                cv2.line(self.drawing_layer,
                         self.points[i - 1], self.points[i],
                         (255, 255, 255), self.line_thickness, cv2.LINE_AA)
                cv2.circle(self.drawing_layer, self.points[i],
                           self.line_thickness // 2, (255, 255, 255), -1, cv2.LINE_AA)
            self.paths.append(self.points.copy())
        elif len(self.points) == 1:
            cv2.circle(self.drawing_layer, self.points[0],
                       self.line_thickness // 2 + 1, (255, 255, 255), -1, cv2.LINE_AA)
            self.paths.append(self.points.copy())

        self.points = []

    def clear(self):
        self.drawing_layer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.points = []
        self.paths  = []
        self._ema_x = self._ema_y = None
        self._miss_frames = 0

    def draw_current_stroke(self, image, ink_color=(255, 200, 50)):
        """即時把正在畫的筆跡疊加到 display frame（用 ink_color 著色）"""
        pts = self.points
        if len(pts) > 1:
            for i in range(1, len(pts)):
                cv2.line(image, pts[i - 1], pts[i],
                         ink_color, self.line_thickness, cv2.LINE_AA)
                cv2.circle(image, pts[i],
                           self.line_thickness // 2, ink_color, -1, cv2.LINE_AA)
        elif len(pts) == 1:
            cv2.circle(image, pts[0],
                       self.line_thickness // 2, ink_color, -1, cv2.LINE_AA)

    def has_content(self):
        return len(self.paths) > 0

    def get_pixel_count(self):
        """回傳畫布上有效筆跡的像素數量，用來過濾噪點"""
        gray = cv2.cvtColor(self.drawing_layer, cv2.COLOR_BGR2GRAY)
        return cv2.countNonZero(gray)

    def detect_circle_in_last_path(self, min_points=25,
                                   circularity_threshold=0.72,
                                   closure_ratio=0.28):
        """
        判斷最後一筆畫是否形成圓形。
        - min_points        : 路徑最少要有幾個採樣點
        - circularity_threshold : 各點到圓心距離的 (1 - std/mean)，越接近 1 越圓
        - closure_ratio     : 起點終點距離 / 總路徑長度 的最大允許比例
        回傳 (cx, cy, radius) 或 None
        """
        if not self.paths:
            return None

        path = self.paths[-1]
        if len(path) < min_points:
            return None

        pts = np.array(path, dtype=np.float32)

        # 1. 閉合度：起點與終點必須夠接近
        start, end = pts[0], pts[-1]
        dist_close = float(np.linalg.norm(end - start))
        # 計算總路徑長度
        diffs = np.diff(pts, axis=0)
        path_len = float(np.linalg.norm(diffs, axis=1).sum())
        if path_len < 40:          # 太短的線不算圓
            return None
        if dist_close > path_len * closure_ratio:
            return None

        # 2. 圓度：各點到重心距離的 std/mean 要小
        cx, cy = pts.mean(axis=0)
        dists   = np.linalg.norm(pts - np.array([cx, cy]), axis=1)
        mean_r  = float(dists.mean())
        std_r   = float(dists.std())

        if mean_r < 18:            # 半徑太小不算
            return None

        circularity = 1.0 - (std_r / mean_r)
        if circularity < circularity_threshold:
            return None

        return int(cx), int(cy), int(mean_r)

