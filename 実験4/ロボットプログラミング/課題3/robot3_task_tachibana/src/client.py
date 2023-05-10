#!/usr/bin/env python2

import sys
import rospy
import std_srvs.srv import *

def client():
    rospy.wait_for_service('goto_closest')
    try:
        service = rospy.ServiceProxy('goto_closest', Trigger)
        service()
    except rospy.ServiceException, e:
        rospy.logerr("Error Message : %s", %e)

if __name__ == "__main__":
    client()
