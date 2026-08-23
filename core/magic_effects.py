import cv2
import numpy as np
import math
import time
import random


class ParticleSystem:
    """奇異博士風格的火花粒子系統"""
    
    def __init__(self):
        self.particles = []
    
    def spawn(self, x, y, color=(255, 160, 30), count=6):
        """在指定位置噴出火花"""
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 7)
            life = random.uniform(0.3, 0.8)
            size = random.randint(2, 5)
            
            # 顏色輕微隨機偏移，更有真實火花感
            r = min(255, color[2] + random.randint(-30, 30))
            g = min(255, color[1] + random.randint(-30, 30))
            b = min(255, color[0] + random.randint(-30, 60))
            
            self.particles.append({
                'x': float(x), 'y': float(y),
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed - random.uniform(0, 3),  # 向上漂
                'life': life, 'max_life': life,
                'size': size,
                'color': (b, g, r)
            })
    
    def update_and_draw(self, frame):
        """更新粒子位置並繪製，自動回收消亡的粒子"""
        alive = []
        for p in self.particles:
            p['life'] -= 0.04
            if p['life'] <= 0:
                continue
            
            # 更新位置 (加上一點重力)
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.25  # 重力
            p['vx'] *= 0.96  # 空氣阻力
            
            # 根據剩餘壽命計算透明度與大小
            ratio = p['life'] / p['max_life']
            current_size = max(1, int(p['size'] * ratio))
            
            # 繪製發光效果 (外大內小，疊加)
            cx, cy = int(p['x']), int(p['y'])
            if 0 < cx < frame.shape[1] and 0 < cy < frame.shape[0]:
                # 外光暈
                glow_color = tuple(int(c * ratio) for c in p['color'])
                cv2.circle(frame, (cx, cy), current_size + 2, glow_color, -1, cv2.LINE_AA)
                # 亮核心
                cv2.circle(frame, (cx, cy), current_size, (255, 255, 255), -1, cv2.LINE_AA)
            
            alive.append(p)
        self.particles = alive


class MagicMandala:
    """掌心法陣：奇異博士風格的旋轉幾何魔法陣"""
    
    def __init__(self):
        self.angle = 0.0          # 外圈旋轉角度
        self.angle_inner = 0.0   # 內圈（反向旋轉）
        self.pulse = 0.0          # 光暈脈衝計時
        
    def update(self, dt=0.03):
        self.angle += 2.5         # 外圈每幀轉 2.5 度
        self.angle_inner -= 1.8  # 內圈反方向
        self.pulse += dt * 2
        if self.angle > 360:
            self.angle -= 360
    
    def draw(self, frame, cx, cy, radius=90, color=(30, 120, 255)):
        """在掌心 (cx, cy) 位置繪製發光旋轉魔法陣"""
        
        # --- 光暈底層 (Bloom Glow) ---
        pulse_r = int(radius * (1.0 + 0.08 * math.sin(self.pulse)))
        glow_overlay = frame.copy()
        for r_offset, alpha in [(30, 0.03), (18, 0.05), (8, 0.10)]:
            cv2.circle(glow_overlay, (cx, cy), pulse_r + r_offset, color, -1, cv2.LINE_AA)
        cv2.addWeighted(glow_overlay, 0.25, frame, 0.75, 0, frame)
        
        # --- 外圈：旋轉的六角幾何 ---
        n_outer = 8  # 外圈幾何邊數
        for i in range(n_outer):
            theta1 = math.radians(self.angle + (360 / n_outer) * i)
            theta2 = math.radians(self.angle + (360 / n_outer) * (i + 1))
            p1 = (int(cx + radius * math.cos(theta1)), int(cy + radius * math.sin(theta1)))
            p2 = (int(cx + radius * math.cos(theta2)), int(cy + radius * math.sin(theta2)))
            # 向圓心連線，形成星芒
            cv2.line(frame, (cx, cy), p1, color, 1, cv2.LINE_AA)
            cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)
        
        # 外圈圓環
        cv2.circle(frame, (cx, cy), pulse_r, color, 2, cv2.LINE_AA)
        
        # --- 中圈：反向旋轉的三角形 ---
        mid_r = int(radius * 0.58)
        n_mid = 3
        for i in range(n_mid):
            theta = math.radians(self.angle_inner + (360 / n_mid) * i)
            px = int(cx + mid_r * math.cos(theta))
            py = int(cy + mid_r * math.sin(theta))
            theta_next = math.radians(self.angle_inner + (360 / n_mid) * (i + 1))
            px2 = int(cx + mid_r * math.cos(theta_next))
            py2 = int(cy + mid_r * math.sin(theta_next))
            cv2.line(frame, (px, py), (px2, py2), (255, 200, 80), 2, cv2.LINE_AA)
        
        # 中圈圓環
        cv2.circle(frame, (cx, cy), mid_r, (255, 200, 80), 1, cv2.LINE_AA)
        
        # --- 內核：小旋轉方塊 ---
        inner_r = int(radius * 0.22)
        for i in range(4):
            theta = math.radians(self.angle * 2 + 45 * i)
            px = int(cx + inner_r * math.cos(theta))
            py = int(cy + inner_r * math.sin(theta))
            theta_next = math.radians(self.angle * 2 + 45 * (i + 1))
            px2 = int(cx + inner_r * math.cos(theta_next))
            py2 = int(cy + inner_r * math.sin(theta_next))
            cv2.line(frame, (px, py), (px2, py2), (255, 255, 200), 2, cv2.LINE_AA)
        
        # 中心亮點
        cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 9, color, 1, cv2.LINE_AA)
        
        self.update()
