import time
import cv2
import imutils
import eventhook

from configloader import config


class MotionDetection:
    minArea = config['min_contour_area']  # minimum contour area (500)
    # maximum frames before firstFrame reset (60)
    maxIdle = config['max_idle_frames']
    # minimum needed consecutive motion-containing frames (60)
    minContMotion = config['min_continuous_motion']
    onTrigger = eventhook.EventHook()  # event that fires when motion is confirmed

    def __init__(self, showDebug=False):
        self.showDebug = showDebug

    def start(self, camera):
        firstFrame = None
        prevFrame = None
        # amount of contours
        contAmount = 0

        # amount of idle frames that were the same
        # used to determine if that frame should become the new firstFrame
        idleCount = 0

        # amount of consecutive motion frames
        consecFrames = 0
        triggered = False

        # grab one frame and wait 5s while the camera focuses
        # (grabbed, frame) = camera.read()
        # time.sleep(5)

        while True:
            (grabbed, frame) = camera.read()
            if frame is None:
                break

            # convert grabbed frame to gray and blur
            frame = imutils.resize(frame, width=500)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if firstFrame is None:
                firstFrame = gray
            if prevFrame is None:
                prevFrame = gray

            frameDelta = cv2.absdiff(firstFrame, gray)
            thresh = self.calcThresh(frameDelta)
            cnts = self.detectContours(thresh)

            # we have motion, possible new firstFrame?
            if len(cnts) > 0:
                consecFrames += 1
                prevDelta = cv2.absdiff(prevFrame, gray)
                prevThresh = self.calcThresh(prevDelta)
                # we have no changes from the previous frame
                if cv2.countNonZero(prevThresh) == 0:
                    idleCount += 1
                else:
                    idleCount = 0
            else:
                consecFrames = 0
                triggered = False

            # we have now seen the same image for too long, reset firstImage
            if idleCount > self.maxIdle:
                firstFrame = prevFrame
                idleCount = 0

            if consecFrames > self.minContMotion and not triggered:
                triggered = True
                self.onTrigger.fire(frame.copy())

            prevFrame = gray

            # loop over the contours
            for c in cnts:
                # compute the bounding box for the contour, draw it on the frame
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if self.showDebug:
                cv2.putText(frame, "contours: " + str(contAmount),
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, "idle: " + str(idleCount), (140, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, "consecutive: " + str(consecFrames),
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                cv2.imshow("Frame", frame)
                cv2.imshow("Delta", frameDelta)
                cv2.imshow("Threshold", thresh)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        camera.release()
        cv2.destroyAllWindows()

    def calcThresh(self, frame):
        thresh = cv2.threshold(frame, 25, 255, cv2.THRESH_BINARY)[1]
        return cv2.dilate(thresh, None, iterations=2)

    def detectContours(self, thresh):
        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        validCnts = []

        # loop over the contours
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < self.minArea:
                continue

            validCnts.append(c)

        return validCnts

# ------------


# MotionDetection().start(cv2.VideoCapture(1))
