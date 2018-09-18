import cv2
import pygame

class TriggerHandler:

    def handle(self, frame):
        print('detection!')
        cv2.imshow("Detection", frame)
        cv2.waitKey(0)
        