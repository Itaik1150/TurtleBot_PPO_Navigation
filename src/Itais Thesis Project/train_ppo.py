import os, random, numpy as np, torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from env.turtlebot3_env import TurtleBot3Env

# ---- Seeding ----
seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

# ---- Env ----
env = TurtleBot3Env()
env = Monitor(env)                    # logs episode reward/len
_ = env.reset(seed=seed)              # proper seeding

# (Optional) a tiny eval env for periodic evaluation
eval_env = TurtleBot3Env()
eval_env = Monitor(eval_env)
_ = eval_env.reset(seed=seed + 1)

# ---- Callbacks ----
ckpt_cb = CheckpointCallback(save_freq=50_000, save_path="checkpoints", name_prefix="ppo_tb3")
eval_cb = EvalCallback(eval_env, best_model_save_path="best_model",
                       log_path="eval_logs", eval_freq=20_000, deterministic=True, render=False)

# ---- Model ----
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=1024,            # rollout length
    batch_size=1024,         # <= n_steps * n_envs
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    device="cpu",
    tensorboard_log="logs/ppo_turtlebot3",
    policy_kwargs=dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])]),
)

model.learn(
    total_timesteps=2_000_000,
    log_interval=1,          # see logs every rollout
    tb_log_name="run_1",
    callback=[ckpt_cb, eval_cb],
    progress_bar=True
)

model.save("ppo_turtlebot3")
env.close(); eval_env.close()
print("✅ Training complete and model saved.")
