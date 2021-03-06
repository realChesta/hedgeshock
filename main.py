import cv2
import cloudvision
import motiondetection
import triggerhandler

from configloader import config

detection = motiondetection.MotionDetection(config['show_debug'])
vision = cloudvision.CloudVision()
handler = triggerhandler.TriggerHandler()


def onDetectionTrigger(frame):
    handler.handle(frame)


def onMotionTrigger(frame):
    print('motion trigger!')

    labels = vision.getLabels(cv2.imencode('.jpg', frame)[1].tostring())
    descriptions = []
    for label in labels:
        descriptions.append(label.description)
        print(label.description)

    detecTrigger = False

    # detect if the detected possible subjects are whitelisted
    detecTrigger = config['mode'] == 'whitelist' and set(
        descriptions).isdisjoint(config['whitelist'])

    # detect if any possible subjects are blacklisted
    detecTrigger = config['mode'] == 'blacklist' and not set(
        descriptions).isdisjoint(config['blacklist'])

    if detecTrigger:
        onDetectionTrigger(frame)


# add handler to trigger event
detection.onTrigger += onMotionTrigger

detection.start(cv2.VideoCapture(config['camera_index']))
