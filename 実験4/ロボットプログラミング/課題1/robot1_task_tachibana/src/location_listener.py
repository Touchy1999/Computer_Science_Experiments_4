#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry

def callback(data):

  rospy.loginfo("(x, y) = (%s, %s)", data.pose.pose.position.x, data.pose.pose.position.y)
 
def location_listener():
  rospy.init_node('location_listener')

  
  rospy.Subscriber("odom", Odometry, callback)

  rospy.spin()

if __name__ == '__main__':
    location_listener()
