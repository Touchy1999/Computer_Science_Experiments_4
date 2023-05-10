#!/usr/bin/env python
import rospy
import cv2
import math
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped, Twist, PointStamped, Point

def callbackImage(msg):
  global bridge
  twist = Twist()
    
  try:
    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
  except CvBridgeError as e:
    rospy.logerr(e)

  low = (0, 0, 100)
  high = (50, 50, 255)
  masked_image = cv2.inRange(cv_image, low, high)
  mu = cv2.moments(masked_image, False)
  try:
    x_inImage = mu["m10"]/mu["m00"]
    detection = True
  except ZeroDivisionError:
    detection = False
  
  if detection:
    angle = math.atan2(160.5 - x_inImage, 265.23)
    if angle >= 0:
      twist.angular.z = min(0.5, 4 * angle)
    else:
      twist.angular.z = max(-0.5, 4 * angle)
    if angle > (math.pi / 4) or angle < (- math.pi / 4):
      twist.linear.x = 0
    else:
      twist.linear.x = 0.25
  else:
    twist.angular.z = 0.5
    twist.linear.x = 0

  # plot image
  #cv2.imshow("Original image", cv_image)
  #cv2.imshow("Processed image", masked_image)
  #cv2.waitKey(3)

  pub.publish(twist)


if __name__ == '__main__':
	rospy.init_node('controll')

	# subscribe to original image
  	sub = rospy.Subscriber("camera/image", Image, callbackImage)
	# publisher for cmd_vel
	pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
  	# CvBridge for converting Ros <-> OpenCV
  	bridge = CvBridge()

	try:
    		rospy.spin()
  	except KeyboardInterrupt:
		print("Shutting down")
  	cv2.destroyAllWindows()
