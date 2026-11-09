import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class TrafficIntersectionEnv(gym.Env):
  metadata = {"render_modes": ["human"], "render_fps": 30}

  def __init__(self, render_mode=None):
    super(TrafficIntersectionEnv, self).__init__()

    self.action_space = spaces.Discrete(2)
    self.observation_space = spaces.Box(
        low=0, high=100, shape=(6,), dtype=np.float32
    )

    self.render_mode = render_mode
    self.current_phase = 0
    self.time_in_phase = 0
    self.min_green_time = 8

    self.queues = {"N": 0, "S": 0, "E": 0, "W": 0}
    self.vehicles = []

    self.window = None
    self.clock = None
    self.window_size = 800

  def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    self.queues = {"N": 0, "S": 0, "E": 0, "W": 0}
    self.vehicles = []
    self.current_phase = 0
    self.time_in_phase = 0

    if self.render_mode == "human":
      self._init_render()

    return self._get_obs(), {}

  def _get_obs(self):
    return np.array(
        [
            self.queues["N"],
            self.queues["S"],
            self.queues["E"],
            self.queues["W"],
            self.current_phase,
            self.time_in_phase,
        ],
        dtype=np.float32,
    )

  def step(self, action):
    if action == 1 and self.time_in_phase >= self.min_green_time:
      self.current_phase = 1 - self.current_phase
      self.time_in_phase = 0
    else:
      self.time_in_phase += 1

    center = self.window_size // 2
    road_width = 140

    for direction in ["N", "S", "E", "W"]:
      if np.random.rand() < 0.25:
        self.queues[direction] = min(35, self.queues[direction] + 1)
        
        car_color = (
            np.random.randint(120, 255),
            np.random.randint(120, 255),
            np.random.randint(150, 255),
        )

        if direction == "N":
          self.vehicles.append({"x": center - 35, "y": 20, "dir": "N", "color": car_color, "speed": 3.5})
        elif direction == "S":
          self.vehicles.append({"x": center + 15, "y": self.window_size - 20, "dir": "S", "color": car_color, "speed": 3.5})
        elif direction == "E":
          self.vehicles.append({"x": self.window_size - 20, "y": center + 15, "dir": "E", "color": car_color, "speed": 3.5})
        elif direction == "W":
          self.vehicles.append({"x": 20, "y": center - 35, "dir": "W", "color": car_color, "speed": 3.5})

    discharge_rate = 1
    if self.current_phase == 0:
      self.queues["N"] = max(0, self.queues["N"] - discharge_rate)
      self.queues["S"] = max(0, self.queues["S"] - discharge_rate)
    else:
      self.queues["E"] = max(0, self.queues["E"] - discharge_rate)
      self.queues["W"] = max(0, self.queues["W"] - discharge_rate)

    stop_n = center - road_width // 2 - 25
    stop_s = center + road_width // 2 + 5
    stop_w = center - road_width // 2 - 25
    stop_e = center + road_width // 2 + 5

    new_vehicles = []
    for v in self.vehicles:
      d = v["dir"]
      is_green = (self.current_phase == 0 and d in ["N", "S"]) or \
                 (self.current_phase == 1 and d in ["E", "W"])

      passed_line = False
      if d == "N" and v["y"] > stop_n + 25: passed_line = True
      if d == "S" and v["y"] < stop_s - 25: passed_line = True
      if d == "W" and v["x"] > stop_w + 25: passed_line = True
      if d == "E" and v["x"] < stop_e - 25: passed_line = True

      blocked = False
      for other in new_vehicles:
        if other["dir"] == d:
          if d == "N" and 0 < other["y"] - v["y"] < 28: blocked = True
          if d == "S" and 0 < v["y"] - other["y"] < 28: blocked = True
          if d == "E" and 0 < v["x"] - other["x"] < 28: blocked = True
          if d == "W" and 0 < other["x"] - v["x"] < 28: blocked = True

      if not is_green and not passed_line:
        if d == "N" and v["y"] >= stop_n: v["y"] = stop_n; blocked = True
        elif d == "S" and v["y"] <= stop_s: v["y"] = stop_s; blocked = True
        elif d == "W" and v["x"] >= stop_w: v["x"] = stop_w; blocked = True
        elif d == "E" and v["x"] <= stop_e: v["x"] = stop_e; blocked = True

      if not blocked:
        if d == "N": v["y"] += v["speed"]
        elif d == "S": v["y"] -= v["speed"]
        elif d == "E": v["x"] -= v["speed"]
        elif d == "W": v["x"] += v["speed"]

      if -40 <= v["x"] <= self.window_size + 40 and -40 <= v["y"] <= self.window_size + 40:
        new_vehicles.append(v)
    
    self.vehicles = new_vehicles

    total_waiting = sum(self.queues.values())
    reward = -float(total_waiting)

    if self.render_mode == "human":
      self._render_frame()

    return self._get_obs(), reward, False, False, {}

  def _init_render(self):
    pygame.init()
    pygame.display.init()
    self.window = pygame.display.set_mode((self.window_size, self.window_size))
    pygame.display.set_caption("AI Autonomous Traffic Controller - Pro Studio")
    self.clock = pygame.time.Clock()

  def render_frame_with_state(self, is_paused=False, running=True):
    if self.window is None:
      return

    canvas = pygame.Surface((self.window_size, self.window_size))
    canvas.fill((18, 20, 24))

    road_width = 140
    center = self.window_size // 2

    # Roads
    pygame.draw.rect(canvas, (35, 39, 42), (center - road_width // 2, 0, road_width, self.window_size))
    pygame.draw.rect(canvas, (35, 39, 42), (0, center - road_width // 2, self.window_size, road_width))
    pygame.draw.rect(canvas, (28, 31, 34), (center - road_width // 2, center - road_width // 2, road_width, road_width))

    # Lane Markings
    for y in range(0, self.window_size, 35):
      if y < center - road_width // 2 or y > center + road_width // 2:
        pygame.draw.line(canvas, (120, 125, 130), (center, y), (center, y + 20), 2)
    for x in range(0, self.window_size, 35):
      if x < center - road_width // 2 or x > center + road_width // 2:
        pygame.draw.line(canvas, (120, 125, 130), (x, center), (x + 20, center), 2)

    # Stop Bars
    bar_color = (241, 196, 15)
    pygame.draw.line(canvas, bar_color, (center - road_width // 2, center - road_width // 2), (center, center - road_width // 2), 5)
    pygame.draw.line(canvas, bar_color, (center, center + road_width // 2), (center + road_width // 2, center + road_width // 2), 5)
    pygame.draw.line(canvas, bar_color, (center + road_width // 2, center - road_width // 2), (center + road_width // 2, center), 5)
    pygame.draw.line(canvas, bar_color, (center - road_width // 2, center), (center - road_width // 2, center + road_width // 2), 5)

    # Traffic Signals
    ns_color = (46, 204, 113) if self.current_phase == 0 else (231, 76, 60)
    ew_color = (46, 204, 113) if self.current_phase == 1 else (231, 76, 60)

    for pos in [
        (center - 30, center - road_width // 2 - 20),
        (center + 30, center + road_width // 2 + 20),
        (center + road_width // 2 + 20, center - 30),
        (center - road_width // 2 - 20, center + 30),
    ]:
      pygame.draw.circle(canvas, (10, 10, 10), pos, 12)

    pygame.draw.circle(canvas, ns_color, (center - 30, center - road_width // 2 - 20), 7)
    pygame.draw.circle(canvas, ns_color, (center + 30, center + road_width // 2 + 20), 7)
    pygame.draw.circle(canvas, ew_color, (center + road_width // 2 + 20, center - 30), 7)
    pygame.draw.circle(canvas, ew_color, (center - road_width // 2 - 20, center + 30), 7)

    # Render Cars
    for v in self.vehicles:
      if v["dir"] in ["N", "S"]:
        car_rect = pygame.Rect(v["x"] - 7, v["y"] - 12, 14, 24)
      else:
        car_rect = pygame.Rect(v["x"] - 12, v["y"] - 7, 24, 14)
      
      pygame.draw.rect(canvas, v["color"], car_rect, border_radius=4)
      pygame.draw.rect(canvas, (200, 220, 240), car_rect.inflate(-6, -6), border_radius=2)

    # --- HUD Dashboard & Interactive Control Buttons ---
    hud = pygame.Surface((280, 150), pygame.SRCALPHA)
    hud.fill((15, 18, 22, 230))
    canvas.blit(hud, (20, 20))

    font_title = pygame.font.SysFont("Segoe UI", 15, bold=True)
    font_body = pygame.font.SysFont("Segoe UI", 13)

    canvas.blit(font_title.render("AI TRAFFIC MANAGEMENT", True, (255, 255, 255)), (32, 28))
    canvas.blit(font_body.render(f"Phase: {'N-S Green' if self.current_phase == 0 else 'E-W Green'} | Timer: {self.time_in_phase}s", True, (52, 152, 219)), (32, 50))
    canvas.blit(font_body.render(f"Total Waiting: {sum(self.queues.values())} vehicles", True, (231, 76, 60)), (32, 72))
    
    status_text = "Status: PAUSED" if is_paused else "Status: RUNNING"
    canvas.blit(font_body.render(status_text, True, (241, 196, 15) if is_paused else (46, 204, 113)), (32, 94))

    # Draw Buttons
    font_btn = pygame.font.SysFont("Segoe UI", 11, bold=True)
    
    # Pause Button Rect
    pygame.draw.rect(canvas, (52, 73, 94), (32, 120, 75, 26), border_radius=4)
    canvas.blit(font_btn.render("PAUSE", True, (255, 255, 255)), (50, 126))

    # Resume Button Rect
    pygame.draw.rect(canvas, (39, 174, 96), (113, 120, 75, 26), border_radius=4)
    canvas.blit(font_btn.render("RESUME", True, (255, 255, 255)), (128, 126))

    # Stop Button Rect
    pygame.draw.rect(canvas, (192, 57, 43), (194, 120, 78, 26), border_radius=4)
    canvas.blit(font_btn.render("STOP/QUIT", True, (255, 255, 255)), (202, 126))

    # Queue Counter Callouts
    canvas.blit(font_body.render(f"N Queue: {self.queues['N']}", True, (255, 255, 255)), (center - 60, 15))
    canvas.blit(font_body.render(f"S Queue: {self.queues['S']}", True, (255, 255, 255)), (center - 60, self.window_size - 25))
    canvas.blit(font_body.render(f"E Queue: {self.queues['E']}", True, (255, 255, 255)), (self.window_size - 120, center - 40))
    canvas.blit(font_body.render(f"W Queue: {self.queues['W']}", True, (255, 255, 255)), (20, center - 40))

    self.window.blit(canvas, (0, 0))
    pygame.display.update()
    self.clock.tick(15)

  def _render_frame(self):
    self.render_frame_with_state(is_paused=False, running=True)

  def close(self):
    if self.window is not None:
      pygame.display.quit()
      pygame.quit()