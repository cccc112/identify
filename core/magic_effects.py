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


class NeuralBloom:
    """
    意念視覺化：畫圓後從圓心爆出的神經突觸樹狀網路。
    完全原創的非線性動畫效果。
    """

    TOTAL_DURATION = 4.0   # 整個效果持續秒數
    FADE_OUT_TIME  = 0.9   # 最後淡出秒數

    def __init__(self):
        self.active   = False
        self._branches = []
        self._t0       = 0.0
        self._color    = (50, 200, 255)

    # ── 觸發 ──────────────────────────────────────────────────────

    def trigger(self, cx, cy, radius, color=(50, 200, 255)):
        """在 (cx, cy) 觸發神經網路爆發，radius 決定主幹長度"""
        self._color    = color
        self._t0       = time.time()
        self._branches = []
        self.active    = True

        rng = random.Random(int(self._t0 * 1000) % 99991)

        # 主幹（從圓心向外放射）
        n_main = rng.randint(7, 11)
        for i in range(n_main):
            base_angle = (2 * math.pi / n_main) * i + rng.uniform(-0.18, 0.18)
            main_len   = radius * rng.uniform(0.85, 1.55)
            # 每條主幹錯開一點點開始時間，有層疊爆發感
            t_start    = i * 0.038
            self._gen_branch(cx, cy, base_angle, main_len, depth=0,
                             t_start=t_start, rng=rng)

    def _gen_branch(self, x1, y1, angle, length, depth, t_start, rng, max_depth=4):
        """遞歸生成所有分支（預先計算，動畫時只做插值）"""
        if depth > max_depth or length < 11:
            return

        x2 = x1 + math.cos(angle) * length
        y2 = y1 + math.sin(angle) * length

        # 深度越深生長越快（更靠近末梢反應更快）
        grow_dur = max(0.055, 0.20 - depth * 0.025)

        self._branches.append({
            'x1': x1, 'y1': y1,
            'x2': x2, 'y2': y2,
            'depth': depth,
            't_start': t_start,
            'grow_dur': grow_dur,
        })

        # 子分支數量：越深越少
        n_children = rng.randint(2, 3) if depth < 2 else rng.randint(1, 2)
        child_t = t_start + grow_dur + rng.uniform(0.0, 0.025)
        for _ in range(n_children):
            spread   = rng.uniform(0.35, 1.05)
            side     = 1 if rng.random() > 0.45 else -1
            c_angle  = angle + side * spread
            c_length = length * rng.uniform(0.42, 0.65)
            self._gen_branch(x2, y2, c_angle, c_length,
                             depth + 1, child_t, rng, max_depth)

    # ── 更新與繪製 ─────────────────────────────────────────────────

    def update_and_draw(self, frame):
        if not self.active:
            return

        elapsed = time.time() - self._t0
        if elapsed >= self.TOTAL_DURATION:
            self.active = False
            return

        # 全域透明度（淡入 0.08s，保持，淡出 FADE_OUT_TIME）
        if elapsed < 0.08:
            global_alpha = elapsed / 0.08
        elif elapsed > self.TOTAL_DURATION - self.FADE_OUT_TIME:
            global_alpha = (self.TOTAL_DURATION - elapsed) / self.FADE_OUT_TIME
        else:
            global_alpha = 1.0
        global_alpha = max(0.0, min(1.0, global_alpha))

        bc, bg, br = self._color  # BGR

        for b in self._branches:
            t = elapsed - b['t_start']
            if t <= 0:
                continue

            progress = min(1.0, t / b['grow_dur'])
            depth    = b['depth']

            x1, y1 = int(b['x1']), int(b['y1'])
            # 根據 progress 插值當前末端
            ex = int(b['x1'] + (b['x2'] - b['x1']) * progress)
            ey = int(b['y1'] + (b['y2'] - b['y1']) * progress)

            # 越深越暗 + 全域 alpha
            depth_factor = max(0.25, 1.0 - depth * 0.18) * global_alpha
            col = (
                int(bc * depth_factor),
                int(bg * depth_factor),
                int(br * depth_factor),
            )
            glow = (
                int(bc * depth_factor * 0.3),
                int(bg * depth_factor * 0.3),
                int(br * depth_factor * 0.3),
            )

            thick = max(1, 3 - depth)

            # 外發光（粗線）
            if thick >= 2:
                cv2.line(frame, (x1, y1), (ex, ey), glow, thick + 5, cv2.LINE_AA)

            # 主線
            cv2.line(frame, (x1, y1), (ex, ey), col, thick, cv2.LINE_AA)

            # 末端節點（完全生長後才亮起）
            if progress >= 1.0:
                nr = max(2, 5 - depth)
                # 光暈
                cv2.circle(frame, (ex, ey), nr + 4, glow, -1, cv2.LINE_AA)
                # 主色節點
                cv2.circle(frame, (ex, ey), nr, col, -1, cv2.LINE_AA)
                # 高亮白芯（只在淺層節點）
                if depth <= 1:
                    white = tuple(int(255 * global_alpha) for _ in range(3))
                    cv2.circle(frame, (ex, ey), max(1, nr - 1), white, -1, cv2.LINE_AA)

