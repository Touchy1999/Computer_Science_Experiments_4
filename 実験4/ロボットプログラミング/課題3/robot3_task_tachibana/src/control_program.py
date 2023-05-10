#!/usr/bin/env python  
import rospy
import math
import tf2_ros
import geometry_msgs.msg
import nav_msgs.msg
import sensor_msgs.msg
import tf
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_srvs.srv import *

cmd_vel = Twist()
destination_x = 0
destination_y = 0
distance = 0
angle = 0

def callback1(data): ##(data is LaserScan type)
  global destination_x
  global destination_y
  global distance
  global angle
  minimum_of_ranges = min(data.ranges)
  index_of_minimum_of_ranges = data.ranges.index(minimum_of_ranges)
  angle = data.angle_min + index_of_minimum_of_ranges * data.angle_increment
  destination_x = minimum_of_ranges * math.cos(angle)
  destination_y = minimum_of_ranges * math.sin(angle)
  distance = minimum_of_ranges
  
def callback2(data): ##(data is Odometry type)
  global cmd_vel
  global destination_x
  global destination_y
  global distance
  K = 1
  v_const = 0.25
  w_max = 0.5
  v_factor = 0.25
  if distance < 0.3:
    cmd_vel.linear.x = 0
    cmd_vel.angular.z = 0
  else: 
    cmd_vel.linear.x = min(v_const, distance*v_factor)
    cmd_vel.angular.z = min(K*angle, w_max)

def goto_pillar(self):
  pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
  sub1 = rospy.Subscriber("/scan", LaserScan, callback1)
  sub2 = rospy.Subscriber("/odom", Odometry, callback2)
  tfBuffer = tf2_ros.Buffer()
  listener = tf2_ros.TransformListener(tfBuffer)
  rate = rospy.Rate(20)
  while not rospy.is_shutdown():
      try:
        trans_from_odom_into_base_link = tfBuffer.lookup_transform("base_link", "odom", rospy.Time(0))
        trans_from_base_scan_into_odom = tfBuffer.lookup_transform("odom", "base_scan", rospy.Time(0))
      except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        rate.sleep()
        continue
      pub.publish(cmd_vel)
      rate.sleep()

def server():
  s = rospy.Service('goto_closest', Trigger, goto_pillar)
  rospy.spin()

if __name__ == '__main__':
  rospy.init_node('control_program')
  server()
  
