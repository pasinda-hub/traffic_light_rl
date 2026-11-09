import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class MultiIntersectionTownEnv(gym.Env):
  metadata = {"render_modes": ["human"], "render_fps": 30}

  def __init__(self, render_mode=None):
    super(MultiIntersectionTownEnv, self).__init__()

    self.action_space = spaces.MultiDiscrete([2, 2, 2, 2])
    self.observation_space = spaces.Box(
        low=0, high=100, shape=(16,), dtype=np.float32
    )

    self.render_mode = render_mode
    self.window_size = 900
    self.window = None
    self.clock = None

    # States: 0 = NS Green, 1 = NS Yellow, 2 = EW Green, 3 = EW Yellow
    self.phases = [0, 0, 0, 0]
    self.timers = [0, 0, 0, 0]
    self.min_green_time = 8
    self.yellow_time = 3

    self.queues = {
        "J0": {"N": 0, "S": 0, "E": 0, "W": 0},
        "J1": {"N": 0, "S": 0, "E": 0, "W": 0},
        "J2": {"N": 0, "S": 0, "E": 0, "W": 0},
        "J3": {"N": 0, "S": 0, "E": 0, "W": 0},
    }
    self.vehicles = []
    self.is_paused = False

  def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    self.phases = [0, 0, 0, 0]
    self.timers = [0, 0, 0, 0]
    for j in self.queues:
      self.queues[j] = {"N": 0, "S": 0, "E": 0, "W": 0}
    self.vehicles = []
    self.is_paused = False

    if self.render_mode == "human":
      self._init_render()

    return self._get_obs(), {}

  def _get_obs(self):
    obs = []
    for j in ["J0", "J1", "J2", "J3"]:
      obs.extend([self.queues[j]["N"], self.queues[j]["S"], self.queues[j]["E"], self.queues[j]["W"]])
    return np.array(obs, dtype=np.float32)

  def step(self, actions):
    if self.is_paused:
      return self._get_obs(), 0.0, False, False, {}

    actions = np.atleast_1d(actions).flatten()
    for i in range(4):
      act = int(actions[i]) if i < len(actions) else 0
      self.timers[i] += 1

      p = self.phases[i]
      if p == 0 and act == 1 and self.timers[i] >= self.min_green_time:
        self.phases[i] = 1  # NS Yellow
        self.timers[i] = 0
      elif p == 1 and self.timers[i] >= self.yellow_time:
        self.phases[i] = 2  # EW Green
        self.timers[i] = 0
      elif p == 2 and act == 1 and self.timers[i] >= self.min_green_time:
        self.phases[i] = 3  # EW Yellow
        self.timers[i] = 0
      elif p == 3 and self.timers[i] >= self.yellow_time:
        self.phases[i] = 0  # NS Green
        self.timers[i] = 0

    # Controlled Spawning at borders
    for direction in ["N", "S", "E", "W"]:
      if np.random.rand() < 0.15:
        if direction == "N":
          sx, sy = np.random.choice([300, 600]) - 15, 10
          if not any(v["dir"] == "N" and v["y"] < 70 for v in self.vehicles):
            self.vehicles.append({"x": sx, "y": sy, "dir": "N", "color": (np.random.randint(120, 255), np.random.randint(120, 255), 255), "speed": 2.2})
        elif direction == "S":
          sx, sy = np.random.choice([300, 600]) + 5, self.window_size - 10
          if not any(v["dir"] == "S" and v["y"] > self.window_size - 70 for v in self.vehicles):
            self.vehicles.append({"x": sx, "y": sy, "dir": "S", "color": (np.random.randint(120, 255), np.random.randint(120, 255), 255), "speed": 2.2})
        elif direction == "E":
          sx, sy = self.window_size - 10, np.random.choice([300, 600]) + 5
          if not any(v["dir"] == "E" and v["x"] > self.window_size - 70 for v in self.vehicles):
            self.vehicles.append({"x": sx, "y": sy, "dir": "E", "color": (np.random.randint(120, 255), np.random.randint(120, 255), 255), "speed": 2.2})
        elif direction == "W":
          sx, sy = 10, np.random.choice([300, 600]) - 15
          if not any(v["dir"] == "W" and v["x"] < 70 for v in self.vehicles):
            self.vehicles.append({"x": sx, "y": sy, "dir": "W", "color": (np.random.randint(120, 255), np.random.randint(120, 255), 255), "speed": 2.2})

    road_w = 120
    min_following_gap = 22

    # Robust Physics Engine with Independent Stop Boundaries
    new_vehicles = []
    for v in self.vehicles:
      d = v["dir"]
      x, y = v["x"], v["y"]
      speed = v["speed"]

      if x < 450:
        j_idx = 0 if y < 450 else 2
        cx, cy = (300, 300) if y < 450 else (300, 600)
      else:
        j_idx = 1 if y < 450 else 3
        cx, cy = (600, 300) if y < 450 else (600, 600)

      p = self.phases[j_idx]
      is_ns_passable = (p in [0, 1])
      is_ew_passable = (p in [2, 3])

      stop_n = cy - road_w // 2 - 18
      stop_s = cy + road_w // 2 + 18
      stop_w = cx - road_w // 2 - 18
      stop_e = cx + road_w // 2 + 18

      passed_line = False
      if d == "N" and y > stop_n + 8: passed_line = True
      if d == "S" and y < stop_s - 8: passed_line = True
      if d == "W" and x > stop_w + 8: passed_line = True
      if d == "E" and x < stop_e - 8: passed_line = True

      blocked = False

      # Traffic signal stop conditions with clean separation
      if not is_ns_passable and d in ["N", "S"] and not passed_line:
        if stop_n - 22 <= y <= stop_n: blocked = True
        elif stop_s <= y <= stop_s + 22: blocked = True
      elif not is_ew_passable and d in ["E", "W"] and not passed_line:
        if stop_w - 22 <= x <= stop_w: blocked = True
        elif stop_e <= x <= stop_e + 22: blocked = True

      # Car-following distance check in the same lane
      for other in new_vehicles:
        if other["dir"] == d:
          if d == "N" and 0 <= other["y"] - y < min_following_gap: blocked = True
          elif d == "S" and 0 <= y - other["y"] < min_following_gap: blocked = True
          elif d == "E" and 0 <= x - other["x"] < min_following_gap: blocked = True
          elif d == "W" and 0 <= other["x"] - x < min_following_gap: blocked = True

      if not blocked:
        if d == "N": y += speed
        elif d == "S": y -= speed
        elif d == "E": x -= speed
        elif d == "W": x += speed

      v["x"], v["y"] = x, y

      if -60 <= v["x"] <= self.window_size + 60 and -60 <= v["y"] <= self.window_size + 60:
        new_vehicles.append(v)

    self.vehicles = new_vehicles
    total_waiting = len(self.vehicles)
    reward = -float(total_waiting)

    if self.render_mode == "human":
      self._render_frame()

    return self._get_obs(), reward, False, False, {}

  def _init_render(self):
    pygame.init()
    pygame.display.init()
    self.window = pygame.display.set_mode((self.window_size, self.window_size))
    pygame.display.set_caption("AI Multi-Junction Smart Town Network - 100% Safe")
    self.clock = pygame.time.Clock()

  def render_frame_with_state(self):
    if self.window is None:
      return

    canvas = pygame.Surface((self.window_size, self.window_size))
    canvas.fill((18, 20, 24))

    centers = {"J0": (300, 300), "J1": (600, 300), "J2": (300, 600), "J3": (600, 600)}
    road_w = 120

    # Draw Roads
    pygame.draw.rect(canvas, (35, 39, 42), (300 - road_w//2, 0, road_w, self.window_size))
    pygame.draw.rect(canvas, (35, 39, 42), (600 - road_w//2, 0, road_w, self.window_size))
    pygame.draw.rect(canvas, (35, 39, 42), (0, 300 - road_w//2, self.window_size, road_w))
    pygame.draw.rect(canvas, (35, 39, 42), (0, 600 - road_w//2, self.window_size, road_w))

    # Dashed Lane Markings
    for y in range(0, self.window_size, 35):
      if not (240 <= y <= 360 or 540 <= y <= 660):
        pygame.draw.line(canvas, (120, 125, 130), (300, y), (300, y + 20), 2)
        pygame.draw.line(canvas, (120, 125, 130), (600, y), (600, y + 20), 2)
    for x in range(0, self.window_size, 35):
      if not (240 <= x <= 360 or 540 <= x <= 660):
        pygame.draw.line(canvas, (120, 125, 130), (x, 300), (x + 20, 300), 2)
        pygame.draw.line(canvas, (120, 125, 130), (x, 600), (x + 20, 600), 2)

    # Intersection Boxes & 3-Color Traffic Lights
    for i, (j_name, (cx, cy)) in enumerate(centers.items()):
      pygame.draw.rect(canvas, (28, 31, 34), (cx - road_w//2, cy - road_w//2, road_w, road_w))

      p = self.phases[i]
      if p == 0:
        ns_col, ew_col = (46, 204, 113), (231, 76, 60)  # NS Green, EW Red
      elif p == 1:
        ns_col, ew_col = (241, 196, 15), (231, 76, 60)  # NS Yellow, EW Red
      elif p == 2:
        ns_col, ew_col = (231, 76, 60), (46, 204, 113)  # NS Red, EW Green
      else:
        ns_col, ew_col = (231, 76, 60), (241, 196, 15)  # NS Red, EW Yellow

      # Draw Light Bulbs
      pygame.draw.circle(canvas, (10, 10, 10), (cx - 25, cy - road_w//2 - 15), 10)
      pygame.draw.circle(canvas, ns_col, (cx - 25, cy - road_w//2 - 15), 6)

      pygame.draw.circle(canvas, (10, 10, 10), (cx + 25, cy + road_w//2 + 15), 10)
      pygame.draw.circle(canvas, ns_col, (cx + 25, cy + road_w//2 + 15), 6)

      pygame.draw.circle(canvas, (10, 10, 10), (cx + road_w//2 + 15, cy - 25), 10)
      pygame.draw.circle(canvas, ew_col, (cx + road_w//2 + 15, cy - 25), 6)

      pygame.draw.circle(canvas, (10, 10, 10), (cx - road_w//2 - 15, cy + 25), 10)
      pygame.draw.circle(canvas, ew_col, (cx - road_w//2 - 15, cy + 25), 6)

    # Render Cars
    for v in self.vehicles:
      if v["dir"] in ["N", "S"]:
        car_rect = pygame.Rect(v["x"] - 5, v["y"] - 9, 10, 18)
      else:
        car_rect = pygame.Rect(v["x"] - 9, v["y"] - 5, 18, 10)
      pygame.draw.rect(canvas, v["color"], car_rect, border_radius=3)
      pygame.draw.rect(canvas, (200, 220, 240), car_rect.inflate(-4, -4), border_radius=1)

    # Town HUD Card & Interactive Buttons
    hud = pygame.Surface((280, 145), pygame.SRCALPHA)
    hud.fill((15, 18, 22, 230))
    canvas.blit(hud, (20, 20))

    font_t = pygame.font.SysFont("Segoe UI", 15, bold=True)
    font_b = pygame.font.SysFont("Segoe UI", 13)

    canvas.blit(font_t.render("TOWN NETWORK (2x2 GRID)", True, (255, 255, 255)), (32, 28))
    canvas.blit(font_b.render(f"Active Town Vehicles: {len(self.vehicles)}", True, (231, 76, 60)), (32, 50))
    
    stat = "Status: PAUSED" if self.is_paused else "Status: TOWN LIVE"
    canvas.blit(font_b.render(stat, True, (241, 196, 15) if self.is_paused else (46, 204, 113)), (32, 72))

    # Control Buttons
    font_btn = pygame.font.SysFont("Segoe UI", 11, bold=True)
    pygame.draw.rect(canvas, (52, 73, 94), (32, 105, 70, 26), border_radius=4)
    canvas.blit(font_btn.render("PAUSE", True, (255, 255, 255)), (48, 111))

    pygame.draw.rect(canvas, (39, 174, 96), (108, 105, 70, 26), border_radius=4)
    canvas.blit(font_btn.render("RESUME", True, (255, 255, 255)), (120, 111))

    pygame.draw.rect(canvas, (192, 57, 43), (184, 105, 78, 26), border_radius=4)
    canvas.blit(font_btn.render("STOP/QUIT", True, (255, 255, 255)), (194, 111))

    self.window.blit(canvas, (0, 0))
    pygame.display.update()
    self.clock.tick(15)

  def _render_frame(self):
    self.render_frame_with_state()

  def close(self):
    if self.window is not None:
      pygame.display.quit()
      pygame.quit()