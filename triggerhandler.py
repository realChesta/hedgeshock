import time
import cv2

from configloader import config

class TriggerHandler:
    from pygame import mixer

    def __init__(self):
        self.mixer.init()
        self.mixer.music.load(config['scare_media'])

    def handle(self, frame):
        print('detection!')
        cv2.imshow("Detection", frame)
        self.playScare()
        cv2.waitKey(0)

    def playScare(self):
        self.mixer.music.rewind()
        self.mixer.music.play()
        time.sleep(config['media_timeout'])
        self.mixer.music.stop()