"""
Autonomous Car Navigation using T3D (Twin Delayed DDPG)
Continuous control with steering and speed actions
"""

import sys
import os
import math
import numpy as np
import random

# --- PYTORCH ---
import torch

# --- PYQT ---
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QGraphicsScene, 
                             QGraphicsView, QGraphicsItem, QFrame, QFileDialog,
                             QTextEdit, QGridLayout)
from PyQt6.QtGui import (QImage, QPixmap, QColor, QPen, QBrush, QPainter, 
                         QPolygonF, QFont, QPainterPath)
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF

# Import T3D
from t3d import T3D, ReplayBuffer

# ==========================================
# 1. CONFIGURATION & THEME
# ==========================================
# Nordic Theme
C_BG_DARK   = QColor("#2E3440") 
C_PANEL     = QColor("#3B4252")
C_INFO_BG   = QColor("#4C566A") 
C_ACCENT    = QColor("#88C0D0") 
C_TEXT      = QColor("#ECEFF4") 
C_SUCCESS   = QColor("#A3BE8C") 
C_FAILURE   = QColor("#BF616A") 
C_SENSOR_ON = QColor("#A3BE8C")
C_SENSOR_OFF= QColor("#BF616A")

# Physics Parameters
CAR_WIDTH = 14     
CAR_HEIGHT = 8   
SENSOR_DIST = 20  # Match DQN solution
SENSOR_ANGLE = 45
TARGET_RADIUS = 20

# T3D Hyperparameters
BATCH_SIZE = 100
GAMMA = 0.99
TAU = 0.005
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
POLICY_FREQ = 2
LR = 3e-4
EXPL_NOISE = 0.1
START_TIMESTEPS = 3000  # Reduced for faster learning

# Action bounds
MAX_STEERING = 8.0   # degrees per step (increased for better maneuverability)
MAX_SPEED = 2.5      # pixels/step
MIN_SPEED = 1.5      # minimum speed

# Target Colors
TARGET_COLORS = [
    QColor(0, 255, 255),      # Cyan
    QColor(255, 100, 255),    # Magenta
    QColor(0, 255, 100),      # Green
    QColor(255, 150, 0),      # Orange
    QColor(100, 150, 255),    # Blue
    QColor(255, 50, 150),     # Pink
    QColor(150, 255, 50),     # Lime
    QColor(255, 255, 0),      # Yellow
]

# ==========================================
# 2. CAR BRAIN (T3D Agent)
# ==========================================
class CarBrain:
    def __init__(self, map_image: QImage):
        self.map = map_image
        self.w, self.h = map_image.width(), map_image.height()
        
        # State and action dimensions
        self.state_dim = 9  # 7 sensors + angle_to_target + distance_to_target
        self.action_dim = 2  # steering angle, speed
        
        # Action bounds (for normalization)
        self.max_action = np.array([MAX_STEERING, MAX_SPEED])
        
        # T3D Agent
        self.agent = T3D(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            max_action=self.max_action,
            discount=GAMMA,
            tau=TAU,
            policy_noise=POLICY_NOISE,
            noise_clip=NOISE_CLIP,
            policy_freq=POLICY_FREQ,
            lr=LR
        )
        
        # Replay Buffer
        self.replay_buffer = ReplayBuffer(max_size=int(1e6))
        
        # Training state
        self.total_timesteps = 0
        self.episode_num = 0
        self.episode_timesteps = 0
        self.episode_reward = 0
        
        # Car state
        self.start_pos = QPointF(100, 100) 
        self.car_pos = QPointF(100, 100)   
        self.car_angle = 0
        self.car_speed = 2.0  # Current speed
        
        # Target management
        self.targets = []
        self.current_target_idx = 0
        self.targets_reached = 0
        self.target_pos = QPointF(200, 200)
        
        self.alive = True
        self.score = 0
        self.sensor_coords = [] 
        self.prev_dist = None
        
        # Episode tracking
        self.episode_scores = []

    def set_start_pos(self, point):
        self.start_pos = point
        self.car_pos = point

    def reset(self):
        self.alive = True
        self.score = 0
        self.car_pos = QPointF(self.start_pos.x(), self.start_pos.y())
        self.car_angle = random.randint(0, 360)
        self.car_speed = 2.0
        self.current_target_idx = 0
        self.targets_reached = 0
        if len(self.targets) > 0:
            self.target_pos = self.targets[0]
        state, dist = self.get_state()
        self.prev_dist = dist
        
        # Episode tracking
        if self.episode_num > 0:
            self.episode_scores.append(self.episode_reward)
        self.episode_reward = 0
        self.episode_timesteps = 0
        self.episode_num += 1
        
        return state
    
    def add_target(self, point):
        self.targets.append(QPointF(point.x(), point.y()))
        if len(self.targets) == 1:
            self.target_pos = self.targets[0]
            self.current_target_idx = 0
    
    def switch_to_next_target(self):
        if self.current_target_idx < len(self.targets) - 1:
            self.current_target_idx += 1
            self.target_pos = self.targets[self.current_target_idx]
            self.targets_reached += 1
            return True
        return False

    def get_state(self):
        sensor_vals = []
        self.sensor_coords = []
        angles = [-45, -30, -15, 0, 15, 30, 45]
        
        for a in angles:
            rad = math.radians(self.car_angle + a)
            sx = self.car_pos.x() + math.cos(rad) * SENSOR_DIST
            sy = self.car_pos.y() + math.sin(rad) * SENSOR_DIST
            self.sensor_coords.append(QPointF(sx, sy))
            
            val = 0.0
            if 0 <= sx < self.w and 0 <= sy < self.h:
                c = QColor(self.map.pixel(int(sx), int(sy)))
                brightness = (c.red() + c.green() + c.blue()) / 3.0
                val = brightness / 255.0
            sensor_vals.append(val)
            
        dx = self.target_pos.x() - self.car_pos.x()
        dy = self.target_pos.y() - self.car_pos.y()
        dist = math.sqrt(dx*dx + dy*dy)
        
        rad_to_target = math.atan2(dy, dx)
        angle_to_target = math.degrees(rad_to_target)
        
        angle_diff = (angle_to_target - self.car_angle) % 360
        if angle_diff > 180: angle_diff -= 360
        
        norm_dist = min(dist / 800.0, 1.0)
        norm_angle = angle_diff / 180.0
        
        state = sensor_vals + [norm_angle, norm_dist]
        return np.array(state, dtype=np.float32), dist

    def step(self, action):
        """
        Execute action in environment
        action: [steering_angle, speed] from actor network
        """
        # Denormalize actions
        steering_delta = action[0]  # Steering change per step
        # Map speed from [-MAX_SPEED, MAX_SPEED] to [MIN_SPEED, MAX_SPEED]
        speed = MIN_SPEED + (action[1] + MAX_SPEED) * (MAX_SPEED - MIN_SPEED) / (2 * MAX_SPEED)
        speed = np.clip(speed, MIN_SPEED, MAX_SPEED)
        
        # No speed-steering coupling - let the agent learn this
        
        # Update car state
        self.car_angle += steering_delta
        self.car_speed = speed
        
        rad = math.radians(self.car_angle)
        new_x = self.car_pos.x() + math.cos(rad) * speed
        new_y = self.car_pos.y() + math.sin(rad) * speed
        self.car_pos = QPointF(new_x, new_y)
        
        next_state, dist = self.get_state()
        
        # Simplified 2-component reward
        reward = -0.1
        done = False
        
        car_center_val = self.check_pixel(self.car_pos.x(), self.car_pos.y())
        
        # Component 1: Crash detection
        if car_center_val < 0.4:
            reward = -100
            done = True
            self.alive = False
        # Component 2: Target reached
        elif dist < TARGET_RADIUS: 
            reward = 100
            has_next = self.switch_to_next_target()
            if has_next:
                done = False
                _, new_dist = self.get_state()
                self.prev_dist = new_dist
            else:
                done = True
        else:
            # CORRECTED REWARD - sensor values are brightness (1.0 = road, 0.0 = obstacle)
            # The DQN formula was inverted! We want HIGH sensor values (road), not low
            sensors = next_state[:7]
            
            # 1. Reward staying on road - use average of all sensors
            avg_brightness = np.mean(sensors)
            reward += avg_brightness * 15
            
            # 2. Strong reward for approaching target
            if self.prev_dist is not None:
                dist_improvement = self.prev_dist - dist
                reward += dist_improvement * 20
            self.prev_dist = dist
            
            # 3. Reward facing toward target (critical for continuous steering)
            angle_to_target = next_state[7]  # normalized [-1, 1]
            reward += (1.0 - abs(angle_to_target)) * 5
            
        self.score += reward
        self.episode_reward += reward
        return next_state, reward, done

    def check_pixel(self, x, y):
        if 0 <= x < self.w and 0 <= y < self.h:
            c = QColor(self.map.pixel(int(x), int(y)))
            return ((c.red() + c.green() + c.blue()) / 3.0) / 255.0
        return 0.0

    def select_action(self, state):
        """Select action with exploration noise"""
        if self.total_timesteps < START_TIMESTEPS:
            # Random exploration
            action = np.random.uniform(-self.max_action, self.max_action, size=self.action_dim)
        else:
            # Use policy with Gaussian noise
            action = self.agent.select_action(state)
            noise = np.random.normal(0, EXPL_NOISE * self.max_action, size=self.action_dim)
            action = np.clip(action + noise, -self.max_action, self.max_action)
        
        return action

    def train_step(self):
        """Train the agent if enough samples"""
        if len(self.replay_buffer.storage) > BATCH_SIZE:
            return self.agent.train(self.replay_buffer, BATCH_SIZE)
        return None, None

# ==========================================
# 3. CUSTOM WIDGETS (VISUALS)
# ==========================================
class RewardChart(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(150)
        self.setStyleSheet(f"background-color: {C_PANEL.name()}; border-radius: 5px;")
        self.scores = []
        self.max_points = 50

    def update_chart(self, new_score):
        self.scores.append(new_score)
        if len(self.scores) > self.max_points:
            self.scores.pop(0)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        painter.fillRect(0, 0, w, h, C_PANEL)
        
        if len(self.scores) < 2:
            return

        min_val = min(self.scores)
        max_val = max(self.scores)
        if max_val == min_val: max_val += 1
        
        points = []
        step_x = w / (self.max_points - 1)
        
        for i, score in enumerate(self.scores):
            x = i * step_x
            ratio = (score - min_val) / (max_val - min_val)
            y = h - (ratio * (h * 0.8) + (h * 0.1))
            points.append(QPointF(x, y))

        path = QPainterPath()
        path.moveTo(points[0])
        for p in points[1:]:
            path.lineTo(p)
            
        pen = QPen(C_ACCENT, 2)
        painter.setPen(pen)
        painter.drawPath(path)
        
        if len(self.scores) >= 2:
            avg_points = []
            window_size = 10
            
            for i in range(len(self.scores)):
                start_idx = max(0, i - window_size + 1)
                avg_score = sum(self.scores[start_idx:i+1]) / (i - start_idx + 1)
                
                x = i * step_x
                ratio = (avg_score - min_val) / (max_val - min_val)
                y = h - (ratio * (h * 0.8) + (h * 0.1))
                avg_points.append(QPointF(x, y))
            
            if len(avg_points) > 1:
                avg_path = QPainterPath()
                avg_path.moveTo(avg_points[0])
                for p in avg_points[1:]:
                    avg_path.lineTo(p)
                
                avg_pen = QPen(QColor(255, 215, 0), 3)
                painter.setPen(avg_pen)
                painter.drawPath(avg_path)
        
        if min_val < 0 and max_val > 0:
            zero_ratio = (0 - min_val) / (max_val - min_val)
            y_zero = h - (zero_ratio * (h * 0.8) + (h * 0.1))
            painter.setPen(QPen(QColor(255, 255, 255, 50), 1, Qt.PenStyle.DashLine))
            painter.drawLine(0, int(y_zero), w, int(y_zero))
        
        legend_x = 10
        legend_y = 15
        
        painter.setPen(QPen(C_ACCENT, 2))
        painter.drawLine(legend_x, legend_y, legend_x + 20, legend_y)
        painter.setPen(QPen(QColor(200, 200, 200)))
        painter.setFont(QFont("Segoe UI", 9))
        painter.drawText(legend_x + 25, legend_y + 4, "Raw")
        
        painter.setPen(QPen(QColor(255, 215, 0), 3))
        painter.drawLine(legend_x + 60, legend_y, legend_x + 80, legend_y)
        painter.setPen(QPen(QColor(200, 200, 200)))
        painter.drawText(legend_x + 85, legend_y + 4, "Avg (10)")

class SensorItem(QGraphicsItem):
    def __init__(self):
        super().__init__()
        self.setZValue(90)
        self.pulse = 0
        self.pulse_speed = 0.3
        self.is_detecting = True
        
    def set_detecting(self, detecting):
        self.is_detecting = detecting
        self.update()
    
    def boundingRect(self):
        return QRectF(-4, -4, 8, 8)
    
    def paint(self, painter, option, widget):
        self.pulse += self.pulse_speed
        if self.pulse > 1.0:
            self.pulse = 0
        
        if self.is_detecting:
            color = C_SENSOR_ON
            outer_alpha = int(150 * (1 - self.pulse))
        else:
            color = C_SENSOR_OFF
            outer_alpha = int(200 * (1 - self.pulse))
        
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        outer_size = 3 + (2 * self.pulse)
        outer_color = QColor(color)
        outer_color.setAlpha(outer_alpha)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(outer_color))
        painter.drawEllipse(QPointF(0, 0), outer_size, outer_size)
        
        painter.setBrush(QBrush(color))
        painter.drawEllipse(QPointF(0, 0), 2, 2)

class CarItem(QGraphicsItem):
    def __init__(self):
        super().__init__()
        self.setZValue(100)
        self.brush = QBrush(C_ACCENT)
        self.pen = QPen(Qt.GlobalColor.white, 1)

    def boundingRect(self):
        return QRectF(-CAR_WIDTH/2, -CAR_HEIGHT/2, CAR_WIDTH, CAR_HEIGHT)

    def paint(self, painter, option, widget):
        painter.setBrush(self.brush)
        painter.setPen(self.pen)
        painter.drawRoundedRect(self.boundingRect(), 2, 2)
        painter.setBrush(Qt.GlobalColor.white)
        painter.drawRect(int(CAR_WIDTH/2)-2, -3, 2, 6)

class TargetItem(QGraphicsItem):
    def __init__(self, color=None, is_active=True, number=1):
        super().__init__()
        self.setZValue(50)
        self.pulse = 0
        self.growing = True
        self.color = color if color else QColor(0, 255, 255)
        self.is_active = is_active
        self.number = number

    def set_active(self, active):
        self.is_active = active
        self.update()
    
    def set_color(self, color):
        self.color = color
        self.update()

    def boundingRect(self):
        return QRectF(-20, -20, 40, 40)

    def paint(self, painter, option, widget):
        if self.is_active:
            if self.growing:
                self.pulse += 0.5
                if self.pulse > 10: self.growing = False
            else:
                self.pulse -= 0.5
                if self.pulse < 0: self.growing = True
            
            r = 10 + self.pulse
            painter.setPen(Qt.PenStyle.NoPen)
            outer_color = QColor(self.color)
            outer_color.setAlpha(100)
            painter.setBrush(QBrush(outer_color)) 
            painter.drawEllipse(QPointF(0,0), r, r)
            painter.setBrush(QBrush(self.color)) 
            painter.setPen(QPen(Qt.GlobalColor.white, 2))
            painter.drawEllipse(QPointF(0,0), 8, 8)
        else:
            dimmed_color = QColor(self.color)
            dimmed_color.setAlpha(120)
            painter.setPen(QPen(Qt.GlobalColor.white, 1))
            painter.setBrush(QBrush(dimmed_color))
            painter.drawEllipse(QPointF(0,0), 6, 6)
        
        painter.setPen(QPen(Qt.GlobalColor.white))
        painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        painter.drawText(QRectF(-10, -10, 20, 20), Qt.AlignmentFlag.AlignCenter, str(self.number))

# ==========================================
# 4. MAIN APPLICATION
# ==========================================
class T3DNavApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("T3D Car Navigation - Continuous Control")
        self.resize(1300, 850)
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {C_BG_DARK.name()}; }}
            QLabel {{ color: {C_TEXT.name()}; font-family: Segoe UI; font-size: 13px; }}
            QPushButton {{ background-color: {C_PANEL.name()}; color: white; border: 1px solid {C_INFO_BG.name()}; padding: 8px; border-radius: 4px; }}
            QPushButton:hover {{ background-color: {C_INFO_BG.name()}; }}
            QPushButton:checked {{ background-color: {C_ACCENT.name()}; color: black; }}
            QTextEdit {{ background-color: {C_PANEL.name()}; color: #D8DEE9; border: none; font-family: Consolas; font-size: 11px; }}
            QFrame {{ border: none; }}
        """)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # LEFT PANEL
        panel = QFrame()
        panel.setFixedWidth(280)
        panel.setStyleSheet(f"background-color: {C_BG_DARK.name()};")
        vbox = QVBoxLayout(panel)
        vbox.setSpacing(10)
        
        lbl_title = QLabel("T3D CONTROLS")
        lbl_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 5px;")
        vbox.addWidget(lbl_title)
        
        self.lbl_status = QLabel("1. Click Map → CAR\n2. Click Map → TARGET(S)\n   (Multiple clicks for sequence)")
        self.lbl_status.setStyleSheet(f"background-color: {C_INFO_BG.name()}; padding: 10px; border-radius: 5px; color: #E5E9F0;")
        vbox.addWidget(self.lbl_status)

        self.btn_run = QPushButton("▶ START (Space)")
        self.btn_run.setCheckable(True)
        self.btn_run.setEnabled(False) 
        self.btn_run.clicked.connect(self.toggle_training)
        vbox.addWidget(self.btn_run)
        
        self.btn_reset = QPushButton("↺ RESET ALL")
        self.btn_reset.clicked.connect(self.full_reset)
        vbox.addWidget(self.btn_reset)
        
        self.btn_load = QPushButton("📂 LOAD MAP")
        self.btn_load.clicked.connect(self.load_map_dialog)
        vbox.addWidget(self.btn_load)
        
        self.btn_save = QPushButton("💾 SAVE MODEL")
        self.btn_save.clicked.connect(self.save_model)
        vbox.addWidget(self.btn_save)
        
        self.btn_load_model = QPushButton("📥 LOAD MODEL")
        self.btn_load_model.clicked.connect(self.load_model)
        vbox.addWidget(self.btn_load_model)

        vbox.addSpacing(15)
        vbox.addWidget(QLabel("REWARD HISTORY"))
        self.chart = RewardChart()
        vbox.addWidget(self.chart)

        stats_frame = QFrame()
        stats_frame.setStyleSheet(f"background-color: {C_PANEL.name()}; border-radius: 5px;")
        sf_layout = QGridLayout(stats_frame)
        sf_layout.setContentsMargins(10, 10, 10, 10)
        
        self.val_timesteps = QLabel("0")
        self.val_timesteps.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf_layout.addWidget(QLabel("Timesteps:"), 0, 0)
        sf_layout.addWidget(self.val_timesteps, 0, 1)
        
        self.val_episode = QLabel("0")
        self.val_episode.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf_layout.addWidget(QLabel("Episode:"), 1, 0)
        sf_layout.addWidget(self.val_episode, 1, 1)
        
        self.val_rew = QLabel("0")
        self.val_rew.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf_layout.addWidget(QLabel("Ep Reward:"), 2, 0)
        sf_layout.addWidget(self.val_rew, 2, 1)
        
        vbox.addWidget(stats_frame)

        vbox.addWidget(QLabel("LOGS"))
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        vbox.addWidget(self.log_console)

        main_layout.addWidget(panel)

        # RIGHT PANEL
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setStyleSheet(f"border: 2px solid {C_PANEL.name()}; background-color: {C_BG_DARK.name()}")
        self.view.mousePressEvent = self.on_scene_click
        main_layout.addWidget(self.view)

        # Logic
        self.setup_map("city_map.png") 
        self.setup_state = 0 
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self.game_loop)
        
        self.car_item = CarItem()
        self.target_items = []
        self.sensor_items = []
        for _ in range(7):
            si = SensorItem()
            self.scene.addItem(si)
            self.sensor_items.append(si)

    def log(self, msg):
        self.log_console.append(msg)
        sb = self.log_console.verticalScrollBar()
        sb.setValue(sb.maximum())

    def setup_map(self, path):
        if not os.path.exists(path):
            self.create_dummy_map(path)
        self.map_img = QImage(path).convertToFormat(QImage.Format.Format_RGB32)
        self.scene.clear()
        self.scene.addPixmap(QPixmap.fromImage(self.map_img))
        if hasattr(self, "sensor_items"):
            for s in self.sensor_items:
                if s.scene() != self.scene:
                    self.scene.addItem(s)
        self.brain = CarBrain(self.map_img)
        self.log(f"Map Loaded. T3D Agent Initialized.")
        self.log(f"Action Space: Continuous [steering: ±{MAX_STEERING}°, speed: 0-{MAX_SPEED}px/s]")
        self.log(f"Warm-up: {START_TIMESTEPS} random steps before policy")

    def create_dummy_map(self, path):
        img = QImage(1000, 800, QImage.Format.Format_RGB32)
        img.fill(C_BG_DARK)
        p = QPainter(img)
        p.setBrush(Qt.GlobalColor.white)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(100, 100, 800, 600)
        p.setBrush(C_BG_DARK)
        p.drawEllipse(250, 250, 500, 300)
        p.end()
        img.save(path)

    def load_map_dialog(self):
        f, _ = QFileDialog.getOpenFileName(self, "Load Map", "", "Images (*.png *.jpg)")
        if f: 
            self.full_reset()
            self.setup_map(f)

    def save_model(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save Model", "", "Model Files (*.pth)")
        if filename:
            base = filename.replace("_actor.pth", "").replace("_critic.pth", "")
            self.brain.agent.save(base)
            self.log(f"<font color='#A3BE8C'>✓ Model saved: {base}_actor.pth, {base}_critic.pth</font>")
    
    def load_model(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Model", "", "Model Files (*.pth)")
        if filename:
            base = filename.replace("_actor.pth", "").replace("_critic.pth", "")
            self.brain.agent.load(base)
            self.log(f"<font color='#A3BE8C'>✓ Model loaded: {base}</font>")

    def on_scene_click(self, event):
        pt = self.view.mapToScene(event.pos())
        if self.setup_state == 0:
            self.brain.set_start_pos(pt) 
            self.scene.addItem(self.car_item)
            self.car_item.setPos(pt)
            self.setup_state = 1
            self.lbl_status.setText("Click Map → TARGET(S)\nRight-click when done")
        elif self.setup_state == 1:
            if event.button() == Qt.MouseButton.LeftButton:
                self.brain.add_target(pt)
                target_idx = len(self.brain.targets) - 1
                color = TARGET_COLORS[target_idx % len(TARGET_COLORS)]
                is_active = (target_idx == 0)
                num_targets = len(self.brain.targets)
                
                target_item = TargetItem(color, is_active, num_targets)
                target_item.setPos(pt)
                self.scene.addItem(target_item)
                self.target_items.append(target_item)
                
                self.lbl_status.setText(f"Targets: {num_targets}\nRight-click to finish setup")
                self.log(f"Target #{num_targets} added at ({pt.x():.0f}, {pt.y():.0f})")
            
            elif event.button() == Qt.MouseButton.RightButton:
                if len(self.brain.targets) > 0:
                    self.setup_state = 2
                    self.brain.reset()  # Initialize episode state
                    self.lbl_status.setText(f"READY. {len(self.brain.targets)} target(s). Press SPACE.")
                    self.lbl_status.setStyleSheet(f"background-color: {C_SUCCESS.name()}; color: #2E3440; font-weight: bold; padding: 10px; border-radius: 5px;")
                    self.btn_run.setEnabled(True)
                    self.update_visuals()

    def full_reset(self):
        self.sim_timer.stop()
        self.btn_run.setChecked(False)
        self.btn_run.setEnabled(False)
        self.setup_state = 0
        self.scene.removeItem(self.car_item)
        for target_item in self.target_items:
            self.scene.removeItem(target_item)
        self.target_items = []
        self.brain.targets = []
        self.brain.current_target_idx = 0
        self.brain.targets_reached = 0
        
        for s in self.sensor_items: 
            if s.scene() == self.scene: self.scene.removeItem(s)
        for s in self.sensor_items:
            if s.scene() != self.scene:
                self.scene.addItem(s)
        self.lbl_status.setText("1. Click Map → CAR\n2. Click Map → TARGET(S)")
        self.lbl_status.setStyleSheet(f"background-color: {C_INFO_BG.name()}; color: white; padding: 10px; border-radius: 5px;")
        self.log("--- RESET ---")
        self.chart.scores = []
        self.chart.update()

    def toggle_training(self):
        if self.btn_run.isChecked():
            self.sim_timer.start(16)
            self.btn_run.setText("⏸ PAUSE")
        else:
            self.sim_timer.stop()
            self.btn_run.setText("▶ RESUME")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space and self.setup_state == 2:
            self.btn_run.click()

    def game_loop(self):
        if self.setup_state != 2: return

        state, _ = self.brain.get_state()
        
        prev_target_idx = self.brain.current_target_idx
        
        # Select action
        action = self.brain.select_action(state)
        
        # Execute action
        next_state, reward, done = self.brain.step(action)
        
        # Store transition
        self.brain.replay_buffer.add((state, next_state, action, reward, float(done)))
        
        # Train agent
        critic_loss, actor_loss = self.brain.train_step()
        
        # Update counters
        self.brain.total_timesteps += 1
        self.brain.episode_timesteps += 1
        
        # Check for target switch
        if self.brain.current_target_idx != prev_target_idx:
            target_num = self.brain.current_target_idx + 1
            total = len(self.brain.targets)
            self.log(f"<font color='#88C0D0'>🎯 Target {prev_target_idx + 1} reached! Moving to target {target_num}/{total}</font>")
            for i, item in enumerate(self.target_items):
                item.set_active(i == self.brain.current_target_idx)
        
        if done:
            # Log episode
            status = "SUCCESS" if self.brain.alive else "CRASH"
            color = "#A3BE8C" if self.brain.alive else "#BF616A"
            
            phase = "EXPLORATION" if self.brain.total_timesteps < START_TIMESTEPS else "TRAINING"
            
            self.log(f"<font color='{color}'>Episode {self.brain.episode_num} [{phase}]: {status} | "
                    f"Reward: {self.brain.episode_reward:.1f} | Steps: {self.brain.episode_timesteps} | "
                    f"Total: {self.brain.total_timesteps}</font>")
            
            self.chart.update_chart(self.brain.episode_reward)
            
            # Reset environment
            state = self.brain.reset()
            
            # Reset target visuals
            for i, item in enumerate(self.target_items):
                item.set_active(i == 0)

        self.update_visuals()
        self.val_timesteps.setText(f"{self.brain.total_timesteps}")
        self.val_episode.setText(f"{self.brain.episode_num}")
        self.val_rew.setText(f"{self.brain.episode_reward:.1f}")

    def update_visuals(self):
        self.car_item.setPos(self.brain.car_pos)
        self.car_item.setRotation(self.brain.car_angle)
        
        for i, target_item in enumerate(self.target_items):
            is_active = (i == self.brain.current_target_idx)
            target_item.set_active(is_active)
        
        state, _ = self.brain.get_state()
        for i, (coord, sensor_val) in enumerate(zip(self.brain.sensor_coords, state[:7])):
            self.sensor_items[i].setPos(coord)
            self.sensor_items[i].set_detecting(sensor_val > 0.4)

# ==========================================
# 5. MAIN
# ==========================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = T3DNavApp()
    window.show()
    sys.exit(app.exec())
