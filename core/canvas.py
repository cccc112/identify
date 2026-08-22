import cv2
import numpy as np

class CanvasManager:
    def __init__(self, width=640, height=480, line_thickness=4):
        self.width = width
        self.height = height
        self.line_thickness = line_thickness
        
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.drawing_layer = np.zeros_like(self.canvas)
        self.bounding_boxes_layer = np.zeros_like(self.canvas)
        
        self.points = []
        self.paths = []
        
        # 橡皮擦相關
        self.erase_radius = 50
        
        # 網格歷史紀錄
        self.num_rows = 4
        self.num_cols = 2
        self.grid_counts = [0] * (self.num_rows * self.num_cols)
        self.recognized_history = [[] for _ in range(self.num_rows * self.num_cols)]
        
        self.draw_grid(self.canvas)
        
    def clear(self):
        """清空畫布與所有軌跡"""
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.drawing_layer = np.zeros_like(self.canvas)
        self.bounding_boxes_layer = np.zeros_like(self.canvas)
        self.points = []
        self.paths = []
        self.grid_counts = [0] * (self.num_rows * self.num_cols)
        self.recognized_history = [[] for _ in range(self.num_rows * self.num_cols)]
        self.draw_grid(self.canvas)

    def draw_grid(self, target, color=(200, 200, 200), thickness=1):
        """繪製分割網格"""
        row_height = self.height // self.num_rows
        col_width = self.width // self.num_cols
        for x in range(0, self.width, col_width):
            cv2.line(target, (x, 0), (x, self.height), color, thickness)
        for y in range(0, self.height, row_height):
            cv2.line(target, (0, y), (self.width, y), color, thickness)

    def get_grid_position(self, x, y):
        grid_width = self.width // self.num_cols
        grid_height = self.height // self.num_rows
        col = min(x // grid_width, self.num_cols - 1)
        row = min(y // grid_height, self.num_rows - 1)
        return row * self.num_cols + col

    def add_point(self, pt):
        """加入新座標 (移除 EMA，改為直接繪製最跟手)"""
        self.points.append(pt)

    def end_stroke(self):
        """結束目前筆畫"""
        if len(self.points) > 1:
            for i in range(1, len(self.points)):
                cv2.line(self.drawing_layer, self.points[i - 1], self.points[i], (240, 202, 166), self.line_thickness)
            self.paths.append(self.points.copy())
        self.points = []

    def undo_stroke(self):
        """復原上一步"""
        if self.paths:
            self.paths.pop()
            self.drawing_layer = np.zeros_like(self.canvas)
            for path in self.paths:
                for i in range(1, len(path)):
                    cv2.line(self.drawing_layer, path[i - 1], path[i], (240, 202, 166), self.line_thickness)

    def apply_eraser(self, x, y):
        """圓形橡皮擦消除軌跡"""
        erase_mask = np.zeros_like(self.drawing_layer)
        cv2.circle(erase_mask, (x, y), self.erase_radius, (255, 255, 255), -1)
        self.drawing_layer = cv2.bitwise_and(self.drawing_layer, cv2.bitwise_not(erase_mask))
        
        new_paths = []
        for path in self.paths:
            new_path = [pt for pt in path if erase_mask[pt[1], pt[0]].sum() == 0]
            if new_path:
                new_paths.append(new_path)
        self.paths = new_paths
        self.points = []

    def draw_current_stroke(self, image):
        """在原畫面上繪製正在畫的線條"""
        if len(self.points) > 1:
            for i in range(1, len(self.points)):
                cv2.line(image, self.points[i - 1], self.points[i], (240, 202, 166), self.line_thickness)

    def add_prediction_to_history(self, grid_index, label):
        """將預測結果加入該網格的歷史紀錄"""
        self.recognized_history[grid_index].append(label)

    def draw_history_window(self, target_image):
        """繪製歷史辨識紀錄"""
        labels = ['Units', 'Tens', 'Hundreds', 'Thousands', 'Ten Thousands', 'Hundred Thousands', 'Millions', 'Ten Millions']
        for i in range(min(8, len(self.recognized_history))):
            digits = self.recognized_history[i][-10:] # 只顯示最近 10 個
            text = f"{labels[i]}: {' '.join(map(str, digits))}"
            cv2.putText(target_image, text, (self.width + 10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def get_combined_canvas(self):
        """取得右側整合後的畫布影像"""
        return self.canvas + self.drawing_layer + self.bounding_boxes_layer
