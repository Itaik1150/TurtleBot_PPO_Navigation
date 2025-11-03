from stable_baselines3 import PPO
import rospy
from env.turtlebot3_env import TurtleBot3Env
from math import sqrt


model = PPO.load("ppo_turtlebot3")

env = TurtleBot3Env()

positions = []
total_time = 0
step_time = 0.05  # שניות

obs = env.reset()
done = False
positions.append(env.current_position[:])

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    print(f"action: {action[0]:.3f}, position: {env.current_position}, reward: {reward:.2f}")

    positions.append(env.current_position[:])
    total_time += step_time
    rospy.sleep(step_time)




total_distance = 0.0
for i in range(1, len(positions)):
    x1, y1 = positions[i-1]
    x2, y2 = positions[i]
    dist = sqrt((x2 - x1)**2 + (y2 - y1)**2)
    total_distance += dist


print(f"Total distance: {total_distance:.3f} m")
print(f"Total time: {total_time:.3f} s")

average_speed = total_distance / total_time
print(f"🟢 Average linear speed: {average_speed:.3f} m/s")
