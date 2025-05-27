from stable_baselines3 import PPO
from env.turtlebot3_env import TurtleBot3Env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement


# Create environment instance
env = TurtleBot3Env()

# Optional: check environment for compliance
# check_env(env, warn=True)
# eval_callback = EvalCallback(
#     env,
#     best_model_save_path="./logs/best_model",
#     log_path="./logs/",
#     eval_freq=10000,             # כל כמה צעדים לבדוק
#     n_eval_episodes=20,           # כמה פרקים להריץ בכל הערכה
#     deterministic=True,
#     render=False
# )


# # Create PPO model with MLP policy
# model = PPO(
#     "MlpPolicy",
#     env,
#     verbose=1,
#     learning_rate=2.5e-4,
#     gamma=0.99,
#     gae_lambda=0.95,
#     n_steps=2048,
#     batch_size=256,
#     ent_coef=0.01,
#     clip_range=0.2,
#     policy_kwargs=dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])]),
# )


# Train the model
# model.learn(total_timesteps=50000, log_interval=1, callback=eval_callback)




# Train using PPO with default hyperparameters
model = PPO("MlpPolicy", env, verbose=0)

# Start learning
model.learn(total_timesteps=50000)

# 


# Save the trained model
model.save("ppo_turtlebot3")

print("✅ Training complete and model saved.")
