#!/usr/bin/env python  
import rospy

import math
import tf2_ros
import geometry_msgs.msg
import nav_msgs.msg
import tf
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

cmd_vel = Twist()
destination_x = 0
destination_y = 0

def callback1(data): ##(data is PoseStamped type)
  global destination_x
  global destination_y
  destination_x = data.pose.position.x
  destination_y = data.pose.position.y

def callback2(data): ##(data is Odometry type)
  global cmd_vel
  pose = data.pose.pose
  dx = destination_x - pose.position.x
  dy = destination_y - pose.position.y
  d = (dx*dx+dy*dy)**0.5
  K = 1
  muki = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
  theta = tf.transformations.euler_from_quaternion(muki)
  alpha = math.atan2(dy,dx) - theta[2]
  v_const = 0.25
  w_max = 0.5
  v_factor = 0.25
  cmd_vel.linear.x = min(v_const, d*v_factor)
  cmd_vel.angular.z = min(K*alpha, w_max)

if __name__ == '__main__':
  rospy.init_node('control_program')
  pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
  sub1 = rospy.Subscriber("move_base_simple/goal", PoseStamped, callback1)
  sub2 = rospy.Subscriber("odom", Odometry, callback2)
  tfBuffer = tf2_ros.Buffer()
  listener = tf2_ros.TransformListener(tfBuffer)
  rate = rospy.Rate(20)
  while not rospy.is_shutdown():
      try:
        trans = tfBuffer.lookup_transform("base_link", "odom", rospy.Time(0))
        ## print(cmd_vel)
        pub.publish(cmd_vel)
      except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        rate.sleep()
        continue
      rate.sleep()