#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan

def scan_callback(data):
    # מדפיס כמה קריאות לדוגמה
    rospy.loginfo(f"מספר קריאות: {len(data.ranges)}")
    rospy.loginfo(f"מרחק קדימה (אמצע): {data.ranges[len(data.ranges)//2]:.2f} מטר")

def listener():
    rospy.init_node('lidar_listener', anonymous=True)
    rospy.Subscriber('/scan', LaserScan, scan_callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
