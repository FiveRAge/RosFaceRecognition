#!/usr/bin/env python
# -*- coding: utf-8 -*-
from skimage import io
from imutils.video import VideoStream
from imutils import face_utils
from scipy.spatial import distance
import numpy as np
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import face_base
import rospy
from std_msgs.msg import String


print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/home/eva/ros_tutorials/face_recognize/nodes/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('/home/eva/ros_tutorials/face_recognize/nodes/dlib_face_recognition_resnet_model_v1.dat')    
print("[INFO] linit ros 'face recognize'...")
rospy.init_node('facerecognize')
pub = rospy.Publisher('face', String)
def get_face_descriptor(imge):    # from image
    img = io.imread(imge)
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        shape = predictor(img, d)
    face_descriptor = facerec.compute_face_descriptor(img, shape)
    return face_descriptor

def index_min(array, n):
    array_new=[]
    for i in range(len(array)):
        array_new.append(array[i][n-1])
    minimym = min(array_new)
    index=array_new.index(minimym)
    return minimym, index

a = face_base.a

for i in range(len(a)):
    a[i][2]=get_face_descriptor(a[i][1])




print("[INFO] camera sensor warming up...")
vs = VideoStream(1).start()
time.sleep(2.0)



while True:
    frame = vs.read()
    
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    rects = detector(gray, 0)

    # loop over the face detections
    if len(rects)>0:
        for rect in rects:
            shape_cam = predictor(gray, rect)
            shape2 = face_utils.shape_to_np(shape_cam)
            for (x, y) in shape2:
                cv2.circle(frame, (x, y), 0, (0, 255, 255), -1)
    
        cv2.imshow("Frame", frame)
        face_descriptor_cam= facerec.compute_face_descriptor(frame, shape_cam)
        for i in range(len(a)):
            a[i][3]=distance.euclidean(a[i][2], face_descriptor_cam)
        minimym, index =index_min(a,4)
        if (minimym < 0.4):
            print(a[index][0])
            pub.publish(String(a[index][0]))
            
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
        # do a bit of cleanup
        cv2.destroyAllWindows()
        vs.stop()
