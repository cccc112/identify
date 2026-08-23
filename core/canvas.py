import cv2
import numpy as np

class CanvasManager:
    def __init__(self, width=640, height=480, line_thickness=8):
        self.width = width
        self.height = height
        self.line_thickness = line_thickness
        
        # 只保留最純粹的繪圖層，背景必須為純黑，以便於 CNN 辨識
        self.drawing_layer = np.zeros((height, width, 3), dtype=np.uint8)
        self.points = []
        self.paths = []
        
    def clear(self):
        """清空畫布與所有軌跡"""
        self.drawing_layer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.points = []
        self.paths = []

    def add_point(self, pt):
        """加入新座標"""
        self.points.append(pt)

    def end_stroke(self):
        """結束目前筆畫"""
        if len(self.points) > 1:
            for i in range(1, len(self.points)):
                cv2.line(self.drawing_layer, self.points[i - 1], self.points[i], (255, 255, 255), self.line_thickness)
            self.paths.append(self.points.copy())
        self.points = []

    def draw_current_stroke(self, image):
        """在原畫面上即時繪製正在畫的線條 (讓使用者能看到筆跡 AR 疊加)"""
        if len(self.points) > 1:
            for i in range(1, len(self.points)):
                # 繪製半透明或不同顏色的即時筆跡，這裡選擇亮黃色
                cv2.line(image, self.points[i - 1], self.points[i], (0, 255, 255), self.line_thickness)

    def has_content(self):
        """檢查畫布上是否有筆跡"""
        # 如果有存下來的路徑，或者是正在畫圖的狀態，都算有內容
        return len(self.paths) > 0 or len(self.points) > 0
