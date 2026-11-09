import matplotlib.pyplot as plt
import pygame
from stable_baselines3 import DQN
from traffic_env import TrafficIntersectionEnv

env = TrafficIntersectionEnv(render_mode="human")
model = DQN.load("dqn_traffic_controller")

obs, info = env.reset()

steps = []
queue_history = []
is_paused = False
running = True

print("Running live simulation. Click PAUSE, RESUME, or STOP/QUIT on the HUD.")

while running:
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      print("Window closed by user.")
      running = False
    elif event.type == pygame.MOUSEBUTTONDOWN:
      mx, my = pygame.mouse.get_pos()
      print(f"Mouse Clicked at X: {mx}, Y: {my}")

      # Corrected button hitboxes based on your terminal logs (Y: 115 to 155)
      # Pause Button: X [30 to 110], Y [115 to 150]
      if 30 <= mx <= 110 and 115 <= my <= 150:
        is_paused = True
        print(">>> Simulation Paused Successfully")
      # Resume Button: X [110 to 190], Y [115 to 150]
      elif 110 <= mx <= 190 and 115 <= my <= 150:
        is_paused = False
        print(">>> Simulation Resumed Successfully")
      # Stop/Quit Button: X [190 to 280], Y [115 to 150]
      elif 190 <= mx <= 280 and 115 <= my <= 150:
        print(">>> Simulation Stopped/Quit by User")
        running = False

  if not running:
    break

  if is_paused:
    env.render_frame_with_state(is_paused=True, running=running)
    pygame.time.wait(100)
    continue

  action, _states = model.predict(obs, deterministic=True)
  obs, reward, terminated, truncated, info = env.step(action)

  steps.append(len(steps))
  total_q = obs[0] + obs[1] + obs[2] + obs[3]
  queue_history.append(total_q)

env.close()

# --- Plot Performance Chart ---
if len(queue_history) > 0:
  print("Generating Performance Analytics Chart...")
  plt.figure(figsize=(10, 5))
  plt.plot(
      steps,
      queue_history,
      color="crimson",
      linewidth=2,
      label="Total Intersection Queue (Congestion)",
  )
  plt.title("AI Traffic Controller Performance Over Time", fontsize=14, fontweight="bold")
  plt.xlabel("Simulation Timesteps", fontsize=12)
  plt.ylabel("Total Vehicles Waiting in Queues", fontsize=12)
  plt.grid(True, linestyle="--", alpha=0.6)
  plt.legend(fontsize=12)
  plt.tight_layout()
  plt.show()