import gym
from gym import spaces
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
import tf.transformations as tf_trans
from nav_msgs.msg import Odometry
import random



class TurtleBot3Env(gym.Env):
    def __init__(self):
      
        super(TurtleBot3Env, self).__init__()

        # Initialize ROS node
        rospy.init_node('turtlebot3_gym_env', anonymous=True)

        # Publisher and subscriber
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)


        # Action space: [linear_vel, angular_vel]
        self.action_space = spaces.Box(low=np.array([-0.1, -1.0]), high=np.array([0.35, 1.0]), dtype=np.float32)

        # Observation space
        self.observation_space = spaces.Box(
                low=np.array([0.0]*24 + [-np.pi, 0.0]),
                high=np.array([3.5]*24 + [np.pi, 200.0]),
                dtype=np.float32)

        self.laser_data = np.full((24,), 3.5, dtype=np.float32)

        # Position and goal tracking
        self.start_position = [0,0]     # Set this according to your environment
        self.goal_position = [3, -1]      # Set this according to your environment
        self.start_yaw = 0 
        
        self.current_position = self.start_position.copy()
              # Robot's current orientation
        self.current_yaw = self.start_yaw



        #start from 0 velocity and initial position
         # Teleport to start positin
        success = self.teleport_robot(model_name="turtlebot3",
                                x=self.start_position[0],
                                y=self.start_position[1],
                                yaw=self.current_yaw)
            
        # Publish zero velocity to stop any motion
        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)

        self.seed(42)
        self.step_count = 0
        self.max_steps = 1000
        self.stuck_steps = 0

        self.initial_distance_to_goal = self.get_distance_to_goal()

        self.min_dist = 0.15 #distance which is considered a crash
        self.last_time = rospy.Time.now()
        self.prev_position = None
        self.previous_distance = None
        self.prev_min_dist = None
        self.prev_reward = 0.0
        self.gamma = 0.99  # discount factor for advantage calculation
        self.previous_distance = None
        self.previous_goal_potential = 0.0



    def lidar_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges = np.clip(ranges, 0.0, 3.5)
        step = len(ranges) // 24
        self.laser_data = ranges[::step][:24]

    def odom_callback(self, msg):
        self.current_position[0] = msg.pose.pose.position.x
        self.current_position[1] = msg.pose.pose.position.y

        
        orientation_q = msg.pose.pose.orientation
        _, _, yaw = tf_trans.euler_from_quaternion([
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w
        ])
        self.current_yaw = yaw


    def change_start_point(self):
        x = random.uniform(0, 2.5)
        y = 0
        self.start_position = [x, y]


# evironment settings 

    def compute_direction_to_goal(self):
        dx = self.goal_position[0] - self.current_position[0]
        dy = self.goal_position[1] - self.current_position[1]
        angle_to_goal = np.arctan2(dy, dx)
        direction = angle_to_goal - self.current_yaw
        # Normalize to [-π, π]
        direction = np.arctan2(np.sin(direction), np.cos(direction))
        return direction

    def yaw_to_quaternion(self, yaw):
        q = tf_trans.quaternion_from_euler(0, 0, yaw)
        return q  # [x, y, z, w]

    def teleport_robot(self, model_name="turtlebot3", x=0.0, y=0.0, yaw=0.0):
        rospy.wait_for_service('/gazebo/set_model_state')
        
        q = self.yaw_to_quaternion(yaw)

        state_msg = ModelState()
        state_msg.model_name = model_name
        state_msg.pose.position.x = x
        state_msg.pose.position.y = y
        state_msg.pose.position.z = 0.0
        state_msg.pose.orientation.x = q[0]
        state_msg.pose.orientation.y = q[1]
        state_msg.pose.orientation.z = q[2]
        state_msg.pose.orientation.w = q[3]
        state_msg.twist.linear.x = 0.0
        state_msg.twist.angular.z = 0.0
        state_msg.reference_frame = "world"

        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(state_msg)
            # if resp.success:
            #     print(f"[INFO] Teleport successful: {model_name} to ({x}, {y}, yaw={yaw})")
            # else:
            #     print(f"[ERROR] Failed to teleport: {resp.status_message}")
        except rospy.ServiceException as e:
            print(f"[ERROR] Service call failed: {e}")

    def euclidean_distance(self, pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def get_distance_to_goal(self):
        return self.euclidean_distance(self.current_position, self.goal_position)




    # def reward(self, distance_to_wall, goal_reached, crushed_a_wall, reached_max_step, stuck_too_long):

        distance_to_goal = self.get_distance_to_goal()

        # === 1. פוטנציאל למטרה (shaping)
        goal_potential = 1.0 - (distance_to_goal / (self.initial_distance_to_goal + 1e-5))
        if self.previous_distance is None:
            self.previous_distance = distance_to_goal
            self.previous_goal_potential = goal_potential

        delta_goal = goal_potential - self.previous_goal_potential
        goal_reward = 5.0 * delta_goal  # מתאים לטווח [-5, +5]

        # === 2. ענישה על קרבה לקיר (רק אם מתחת ל־0.3 מטר)
        if distance_to_wall < 0.3:
            wall_penalty_ratio = 1.0 - ((distance_to_wall - 0.15) / (0.3 - 0.15))  # normalized: 0 at 0.3, 1 at 0.15
            wall_penalty = 3.0 * np.clip(wall_penalty_ratio, 0.0, 1.0)
        else:
            wall_penalty = 0.0

        # === 3. עונש על חוסר תזוזה
        progress = abs(distance_to_goal - self.previous_distance)
        no_progress_penalty = 0.2 if progress < 0.01 else 0.0

        # === 4. עונש קל על כל צעד
        step_penalty = 0.01

        # === 5. תגמולים סופיים
        terminal_reward = 0.0
        if goal_reached:
            terminal_reward += 10.0
        if crushed_a_wall:
            terminal_reward -= 10.0
        if stuck_too_long:
            terminal_reward -= 5.0

        # === 6. סכימה
        reward = goal_reward - wall_penalty - step_penalty - no_progress_penalty + terminal_reward

        # === 7. עדכון מצב קודם
        self.previous_distance = distance_to_goal
        self.previous_goal_potential = goal_potential

        # === 8. הדפסה (לפי הצורך)
        if self.step_count % 200 == 0:
            print(f"▶️ REWARD DEBUG")
            print(f"   ▪ Δgoal: {delta_goal:.3f} → Goal reward: {goal_reward:.2f}")
            print(f"   ▪ Wall distance: {distance_to_wall:.2f} → Penalty: {wall_penalty:.2f}")
            print(f"   ▪ Progress: {progress:.3f} → No progress penalty: {no_progress_penalty:.2f}")
            print(f"   ▪ Total reward: {reward:.2f}")

        return reward


    def reward(self, distance_to_wall, goal_reached, crushed_a_wall, reached_max_step, stuck_too_long):
        distance_to_goal = self.get_distance_to_goal()

        # 1. Goal reward (linear potential based on current distance)
        progress_ratio = 1.0 - (distance_to_goal / (self.initial_distance_to_goal + 1e-5))
        goal_progress_reward = 5.0 * progress_ratio**2
        goal_progress_reward = np.clip(goal_progress_reward, 0.0, 5.0)  # keep in [0, 5]

        # 2. Wall penalty (linear, only starts below 0.3m)
        if distance_to_wall < 0.3:
            wall_penalty = 2.0 * (0.3 - distance_to_wall) / 0.15  # 0.3→0, 0.15→2.0
        else:
            wall_penalty = 0.0

        # 3. Step penalty
        step_penalty = 0.05

        # 4. Terminal reward
        terminal_reward = 0.0
        if goal_reached:
            terminal_reward = 10.0
        elif crushed_a_wall or stuck_too_long or reached_max_step:
            terminal_reward = -10.0

        survival_reward = 0.01


        # 5. Final reward
        reward = goal_progress_reward - wall_penalty - step_penalty + terminal_reward + survival_reward

        # 6. Debug print
        if self.step_count % 100 == 0:
            print("▶️ REWARD DEBUG")
            print(f"   ▪ Distance to goal: {distance_to_goal:.2f} → Goal reward: {goal_progress_reward:.2f}")
            print(f"   ▪ Distance to wall: {distance_to_wall:.2f} → Wall penalty: {wall_penalty:.2f}")
            print(f"   ▪ Step penalty: {step_penalty:.2f}")
            print(f"   ▪ Terminal reward: {terminal_reward:.2f}")
            print(f"   ✅ Final reward: {reward:.2f}")

        return reward


    def reset(self):
        
        # Reset step counter
        self.prev_position = None

        self.step_count = 0

        # Reset internal state
        self.current_position = self.start_position.copy()
        self.current_yaw = 0.0

        # Reset time tracking
        self.last_time = rospy.Time.now()

        # Publish zero velocity to stop any motion
        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)

        
        # Teleport
        self.change_start_point()
        success = self.teleport_robot(model_name="turtlebot3",
                                x=self.start_position[0],
                                y=self.start_position[1],
                                yaw=self.start_yaw)



        rospy.sleep(1.0)

        print("[RESET] Environment reset. Robot returned to start position.")


        # Wait for first valid laser scan before returning
        while self.laser_data is None or np.isnan(self.laser_data).any():
            rospy.sleep(0.1)
       
       
        obs = np.concatenate([
        self.laser_data,
        [self.compute_direction_to_goal()],
        [self.get_distance_to_goal()]
        ]).astype(np.float32)

        self.prev_distance_to_goal = None
        self.prev_min_dist = None
        self.prev_reward = 0.0
        self.previous_distance = None
        self.previous_state_value = 0.0

        return obs

    def step(self, action):
        self.step_count += 1

        # Check laser data is valid
        if self.laser_data is None or len(self.laser_data) != 24 or np.isnan(self.laser_data).any():
            print(f"[STEP {self.step_count}] WARNING: Invalid laser data. Returning safe fallback.")
            fallback_obs = np.concatenate([np.full((24,), 3.5), [0.0], [10.0]])
            return fallback_obs, 0.0, False, {}

        # 1. Calculate min distance to obstacle
        distance_to_wall = np.min(self.laser_data)

        # 2. Save previous distance to goal
        prev_distance_to_goal = self.get_distance_to_goal()

        # 3. Clip actions to action space bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        vel = Twist()
        vel.linear.x = action[0]
        vel.angular.z = action[1]
        
        self.cmd_vel_pub.publish(vel)
 

        # 4. Wait a short time to simulate motion
        rospy.sleep(0.02)

        # 6. Calculate new distance to goal
        current_distance_to_goal = self.get_distance_to_goal()

        # 7. Check termination
        goal_reached = self.current_position[1] <= -1
        crushed_a_wall = distance_to_wall < self.min_dist
        reached_max_step = self.step_count >= self.max_steps

        # Track if stuck for many steps
        progress = abs(prev_distance_to_goal - current_distance_to_goal)
        if progress < 0.001 and self.step_count > 100:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0

        stuck_too_long = self.stuck_steps >= 200

        # Combined done condition
        done = bool(crushed_a_wall or goal_reached or reached_max_step or stuck_too_long)

        if goal_reached:
            print("✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅" \
            "✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅")
        
        # 8. Compute reward
        reward = self.reward(distance_to_wall, goal_reached,crushed_a_wall, reached_max_step, stuck_too_long,)

        # print(f"done: {done}")
        # print(f"goal_reached: {goal_reached}")

        if done:
            stop_msg = Twist()
            self.cmd_vel_pub.publish(stop_msg)

        # 9. Debug print
        # print(f"[STEP {self.step_count}] action: {action} | min_dist: {min_dist:.2f} | dist_to_goal: {current_distance_to_goal:.2f} | done: {done}")

        obs = np.concatenate([
        self.laser_data,
        [self.compute_direction_to_goal()],
        [self.get_distance_to_goal()]
        ]).astype(np.float32)


        if self.step_count % 200 == 10:
            print(f"current distance from goal: {current_distance_to_goal}")
            # print(f"obs: {obs}")
        return obs, reward, done, {}


    def render(self, mode='human'):
        pass

    def close(self):
        rospy.signal_shutdown("Closing TurtleBot3Env")




# check_env = TurtleBot3Env()
# check_env.teleport_robot(model_name="turtlebot3",
#                                 x=,
#                                 y=-1,
#                                 )

# env = TurtleBot3Env()
# obs = env.reset()
# action = np.array([0.2, 0.0])
# obs, reward, done, _ = env.step(action)
# print("obs shape:", obs.shape)
# print("reward:", reward, "done:", done)
# env.close()