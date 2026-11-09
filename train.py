from stable_baselines3 import DQN
from traffic_env import TrafficIntersectionEnv

env = TrafficIntersectionEnv()

model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-3,
    buffer_size=10000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,
    verbose=1,
)

print("Starting AI Traffic Controller Training...")
model.learn(total_timesteps=20000)
model.save("dqn_traffic_controller")
print("Training complete! Model saved as 'dqn_traffic_controller.zip'.")