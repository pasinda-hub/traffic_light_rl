from multi_env import MultiIntersectionTownEnv
from stable_baselines3 import PPO

# PPO handles MultiDiscrete action spaces smoothly for multi-junction networks
env = MultiIntersectionTownEnv()

model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    verbose=1,
)

print("Starting Multi-Junction Smart Town AI Training...")
model.learn(total_timesteps=20000)
model.save("ppo_town_controller")
print("Training complete! Model saved as 'ppo_town_controller.zip'.")