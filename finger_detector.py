import cv2 as cv
import numpy as np

# Parameters
cap_region_x_begin = 0.5  # start point/total width
cap_region_y_end = 0.8  # start point/total width
threshold = 80  # BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0


class FingerDetector:

    def __init__(self):
        self.keepRunning = True
        self.isBgCaptured = False
        self.bgModel = None
        try:
            self.camera = cv.VideoCapture(0)
            self.main()
        except cv.error as e:
            print("Error: " + e)

    def capture_bg(self):
        self.bgModel = cv.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        self.isBgCaptured = True
        print("Background Captured")

    def remove_bg(self, frame):
        fg_mask = self.bgModel.apply(frame, learningRate=learningRate)
        kernel = np.ones((3, 3), np.uint8)
        fg_mask = cv.erode(fg_mask, kernel, iterations=1)
        res = cv.bitwise_and(frame, frame, mask=fg_mask)
        return res

    def end_detection(self):
        self.camera.release()
        cv.destroyAllWindows()
        print("Ending finger detection")
        self.keepRunning = False

    def get_frame(self):
        _, frame = self.camera.read()
        return cv.flip(frame, 1)

    def draw_rectangle(self, frame):
        cv.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                     (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)

    def main(self):
        while self.keepRunning:
            frame = self.get_frame()
            self.draw_rectangle(frame)
            cv.imshow('original', frame)

            if self.isBgCaptured:
                img = self.remove_bg(frame)
                img = img[0:int(cap_region_y_end * frame.shape[0]),
                          int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]
                cv.imshow('mask', img)

            k = cv.waitKey(10)
            if k == 27:  # ESC to exit
                self.end_detection()
            elif k == ord('b'):
                self.capture_bg()


if __name__ == '__main__':
    finger_detector = FingerDetector()
