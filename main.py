import cv2
import cloudvision
import motiondetection
import json

config = json.load('config.json')
detection = motiondetection.MotionDetection()
vision = cloudvision.CloudVision()

def onMotionTrigger(frame):
    labels = vision.getLabels(frame)
    descriptions = []
    for label in labels:
        descriptions.append(label.description)

    

# add handler to trigger event
detection.onTrigger += onMotionTrigger

detection.start(cv2.VideoCapture(1))
