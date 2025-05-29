from stable_baselines3 import PPO
from env.turtlebot3_env import TurtleBot3Env

# Create environment instance
env = TurtleBot3Env()




seed = 42
env.seed(seed)
# np.random.seed(seed)
# random.seed(seed)
# torch.manual_seed(seed)


# Create PPO model with optimized parameters
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,  # Keep minimal logging
    learning_rate=3e-4,  # Slightly higher learning rate
    n_steps=1024,  # Reduced steps to match batch size
    batch_size=1024,  # Must be equal to or smaller than n_steps
    n_epochs=10,  # More epochs per update
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,  # Encourage exploration
    policy_kwargs=dict(
        net_arch=[dict(pi=[64, 64], vf=[64, 64])],  # Simpler network architecture
    ),
    tensorboard_log="logs/ppo_turtlebot3"
)

# Train the model with minimal logging
model.learn(
    total_timesteps=2000000,
    log_interval=1000,  # Log less frequently
    tb_log_name="run_1"
)

# Save the trained model
model.save("ppo_turtlebot3")

print("✅ Training complete and model saved.")

env.close()
