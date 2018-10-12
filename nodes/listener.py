#!/usr/bin/env python
# -*- coding: utf-8 -*-
import roslib; roslib.load_manifest('beginner_tutorials')
import rospy
from std_msgs.msg import String
def callback(data):
    rospy.loginfo("Обнаружено лицо: %s",data.data)

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("face", String, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
