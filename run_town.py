import matplotlib.pyplot as plt
import pygame
from multi_env import MultiIntersectionTownEnv
from stable_baselines3 import PPO

env = MultiIntersectionTownEnv(render_mode="human")
try:
  model = PPO.load("ppo_town_controller")
  print("Loaded trained PPO multi-junction model successfully.")
except:
  model = None
  print("Trained model not found. Running random baseline policy.")

obs, info = env.reset()

steps = []
congestion_history = []
running = True

print("Running multi-junction town simulation. Click PAUSE, RESUME, or STOP/QUIT on the HUD.")

while running:
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      print("Window closed by user.")
      running = False
    elif event.type == pygame.MOUSEBUTTONDOWN:
      mx, my = pygame.mouse.get_pos()
      print(f"Mouse Clicked at X: {mx}, Y: {my}")

      # Pause Button: X [30 to 105], Y [100 to 140]
      if 30 <= mx <= 105 and 100 <= my <= 140:
        env.is_paused = True
        print(">>> Town Simulation Paused Successfully")
      # Resume Button: X [105 to 180], Y [100 to 140]
      elif 105 <= mx <= 180 and 100 <= my <= 140:
        env.is_paused = False
        print(">>> Town Simulation Resumed Successfully")
      # Stop/Quit Button: X [180 to 270], Y [100 to 140]
      elif 180 <= mx <= 270 and 100 <= my <= 140:
        print(">>> Town Simulation Stopped by User")
        running = False

  if not running:
    break

  if env.is_paused:
    env._render_frame()
    pygame.time.wait(100)
    continue

  if model is not None:
    actions, _states = model.predict(obs, deterministic=True)
  else:
    actions = env.action_space.sample()

  obs, reward, terminated, truncated, info = env.step(actions)

  steps.append(len(steps))
  congestion_history.append(len(env.vehicles))

env.close()

# --- Generate Analytics Graph after Manual Stop ---
if len(congestion_history) > 0:
  print("Generating Multi-Junction Performance Analytics Chart...")
  plt.figure(figsize=(10, 5))
  plt.plot(
      steps,
      congestion_history,
      color="crimson",
      linewidth=2,
      label="Total Town Traffic Congestion",
  )
  plt.title("AI Multi-Junction Town Performance Over Time", fontsize=14, fontweight="bold")
  plt.xlabel("Simulation Timesteps", fontsize=12)
  plt.ylabel("Total Active Vehicles in Town", fontsize=12)
  plt.grid(True, linestyle="--", alpha=0.6)
  plt.legend(fontsize=12)
  plt.tight_layout()
  plt.show()