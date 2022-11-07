import math
import cv2 as cv
import numpy as np

# Parameters
cap_region_x_begin = 0.5  # start point/total width
cap_region_y_end = 0.8  # start point/total width
threshold = 80  # BINARY threshold
blurValue = 41  # Gaussian Blur parameter
bgSubThreshold = 50
learningRate = 0


def draw_rectangle(frame):
    cv.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                 (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)


def remove_bg(bg_model, frame):
    fg_mask = bg_model.apply(frame, learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fg_mask = cv.erode(fg_mask, kernel, iterations=1)
    res = cv.bitwise_and(frame, frame, mask=fg_mask)
    return res


def to_binary(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (blurValue, blurValue), 0)
    cv.imshow('blur', blur)
    _, thresh = cv.threshold(blur, threshold, 255, cv.THRESH_BINARY)
    cv.imshow('ori', thresh)
    return thresh


def find_contour(thresh, img):
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    length = len(contours)
    max_area = -1
    ci = 0
    if length > 0:
        for i in range(length):
            temp = contours[i]
            area = cv.contourArea(temp)
            if area > max_area:
                max_area = area
                ci = i
        res = contours[ci]
        hull = cv.convexHull(res)
        drawing = np.zeros(img.shape, np.uint8)
        cv.drawContours(drawing, [res], 0, (0, 255, 0), 2)
        cv.drawContours(drawing, [hull], 0, (0, 0, 255), 3)
        return res, drawing


def calculate_fingers(res, drawing):
    hull = cv.convexHull(res, returnPoints=False)
    finger_count = 0
    if len(hull) > 3:
        defects = cv.convexityDefects(res, hull)
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i][0]
            start = tuple(res[s][0])
            end = tuple(res[e][0])
            far = tuple(res[f][0])
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
            if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                finger_count += 1
                cv.circle(drawing, far, 8, [211, 84, 0], -1)
    return finger_count


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

    def end_detection(self):
        self.camera.release()
        cv.destroyAllWindows()
        print("Ending finger detection")
        self.keepRunning = False

    def get_frame(self):
        _, frame = self.camera.read()
        return cv.flip(frame, 1)

    def main(self):
        while self.keepRunning:
            frame = self.get_frame()
            draw_rectangle(frame)
            cv.imshow('original', frame)

            if self.isBgCaptured:
                img = remove_bg(self.bgModel, frame)
                img = img[0:int(cap_region_y_end * frame.shape[0]),
                          int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]
                cv.imshow('mask', img)
                thresh = to_binary(img)
                try:
                    res, drawing = find_contour(thresh, img)
                except TypeError:
                    continue
                finger_num = calculate_fingers(res, drawing)
                print(finger_num)
                cv.imshow('output', drawing)

            k = cv.waitKey(10)
            if k == 27:  # ESC to exit
                self.end_detection()
            elif k == ord('b'):
                self.capture_bg()


if __name__ == '__main__':
    finger_detector = FingerDetector()
