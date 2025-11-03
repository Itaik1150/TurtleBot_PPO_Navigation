from stable_baselines3 import PPO
from env.turtlebot3_env import TurtleBot3Env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement


# Create environment instance
env = TurtleBot3Env()

# Optional: check environment for compliance
check_env(env, warn=True)
eval_callback = EvalCallback(
    env,
    best_model_save_path="./logs/best_model",
    log_path="./logs/",
    eval_freq=10000,             # כל כמה צעדים לבדוק
    n_eval_episodes=5,           # כמה פרקים להריץ בכל הערכה
    deterministic=True,
    render=False
)



model = PPO.load("ppo_turtlebot3")
model.set_env(env)
model.learn(total_timesteps=10000, reset_num_timesteps=False)

# Save the trained model
model.save("ppo_turtlebot3_rest")

print("✅ Training complete and model saved.")
