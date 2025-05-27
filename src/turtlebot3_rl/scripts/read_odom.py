#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry

def odom_callback(data):
    position = data.pose.pose.position
    orientation = data.pose.pose.orientation
    linear_vel = data.twist.twist.linear
    angular_vel = data.twist.twist.angular

    rospy.loginfo(f"Position -> x: {position.x:.2f}, y: {position.y:.2f}")
    rospy.loginfo(f"Linear Velocity -> x: {linear_vel.x:.2f}")
    rospy.loginfo(f"Angular Velocity -> z: {angular_vel.z:.2f}")

def listener():
    rospy.init_node('odom_listener', anonymous=True)
    rospy.Subscriber('/odom', Odometry, odom_callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
