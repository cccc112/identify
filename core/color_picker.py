import cv2
import numpy as np
import math

class ColorPicker:
    def __init__(self, cx, cy, radius=120):
        self.cx = cx
        self.cy = cy
        self.radius = radius
        self.wheel_img = None
        self.mask = None
        self._generate_wheel()
        
    def _generate_wheel(self):
        size = self.radius * 2
        img_hsv = np.zeros((size, size, 3), dtype=np.uint8)
        
        y, x = np.ogrid[-self.radius:self.radius, -self.radius:self.radius]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x) * 180 / np.pi
        
        hue = (theta + 180) / 2
        sat = np.clip(r / self.radius * 255, 0, 255)
        
        mask = r <= self.radius
        
        img_hsv[..., 0] = hue
        img_hsv[..., 1] = sat
        img_hsv[..., 2] = 255
        
        self.wheel_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        self.wheel_img[~mask] = 0
        self.mask = mask

    def draw(self, frame, alpha=0.95):
        y1, y2 = self.cy - self.radius, self.cy + self.radius
        x1, x2 = self.cx - self.radius, self.cx + self.radius
        
        if y1 < 0 or y2 > frame.shape[0] or x1 < 0 or x2 > frame.shape[1]:
            return
            
        roi = frame[y1:y2, x1:x2]
        colored_roi = np.where(self.mask[..., None], 
                               cv2.addWeighted(self.wheel_img, alpha, roi, 1 - alpha, 0), 
                               roi)
        frame[y1:y2, x1:x2] = colored_roi
        
        # Draw nice border
        cv2.circle(frame, (self.cx, self.cy), self.radius, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(frame, (self.cx, self.cy), self.radius+2, (50, 50, 50), 1, cv2.LINE_AA)

    def get_color(self, px, py):
        dx = px - self.cx
        dy = py - self.cy
        r = math.sqrt(dx**2 + dy**2)
        
        if r > self.radius:
            return None
            
        theta = math.atan2(dy, dx) * 180 / math.pi
        hue = (theta + 180) / 2
        sat = np.clip(r / self.radius * 255, 0, 255)
        
        hsv_color = np.uint8([[[hue, sat, 255]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
        return tuple(int(x) for x in bgr_color[0][0])
