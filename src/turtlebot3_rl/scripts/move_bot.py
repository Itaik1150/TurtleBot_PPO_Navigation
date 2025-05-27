#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist

def move():
    print("הקוד רץ")
    rospy.init_node('simple_move_node', anonymous=True)
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    # מחכה עד שה-publisher מתחבר
    while pub.get_num_connections() == 0 and not rospy.is_shutdown():
        rospy.loginfo("Waiting for subscriber to connect to /cmd_vel...")
        rospy.sleep(0.1)

    # יוצרים הודעת תנועה
    vel = Twist()
    vel.linear.x = 0.2  # תנועה קדימה
    vel.angular.z = 0.0

    rate = rospy.Rate(10)  # 10Hz
    rospy.loginfo("Publishing velocity commands...")

    # שולח פקודות תנועה למשך שנייה
    for _ in range(10):
        pub.publish(vel)
        rate.sleep()

    # עוצר את הרובוט
    vel.linear.x = 0.0
    pub.publish(vel)
    rospy.loginfo("Robot stopped.")

if __name__ == '__main__':
    try:
        move()
    except rospy.ROSInterruptException:
        pass
