import math
import cv2 as cv
import numpy as np
import pyautogui
import time

REGION_X_START = 0.5
REGION_Y_END = 0.8
BINARY_THRESHOLD = 80
BLUR = 41
BG_THRESHOLD = 50


def remove_bg(bg_model, frame):
    fg_mask = bg_model.apply(frame, learningRate=0)
    kernel = np.ones((3, 3), np.uint8)
    fg_mask = cv.erode(fg_mask, kernel, iterations=1)
    result = cv.bitwise_and(frame, frame, mask=fg_mask)
    return result


def to_binary(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_blur = cv.GaussianBlur(img_gray, (BLUR, BLUR), 0)
    cv.imshow('Blurred background', img_blur)
    _, img_binary = cv.threshold(img_blur, BINARY_THRESHOLD, 255, cv.THRESH_BINARY)
    return img_binary


def find_contour(img, img_no_bg):
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
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
        drawing = np.zeros(img_no_bg.shape, np.uint8)
        cv.drawContours(drawing, [res], 0, (0, 255, 0), 2)
        cv.drawContours(drawing, [hull], 0, (0, 0, 255), 3)
        return res, drawing


def calculate_fingers(hand, img):
    hull = cv.convexHull(hand, returnPoints=False)
    finger_count = 0
    defects = cv.convexityDefects(hand, hull)
    if len(hull) > 3 and defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i][0]
            start = tuple(hand[s][0])
            end = tuple(hand[e][0])
            far = tuple(hand[f][0])
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
            if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                finger_count += 1
                cv.circle(img, far, 8, [211, 84, 0], -1)
    return finger_count


def gest_out(self, finger_count):
    if 0 <= finger_count <= 5:
        self.fingersHistory.append(finger_count)
        if len(self.fingersHistory) > 5:
            self.fingersHistory.pop(0)
        # check the average finger count over the last 5 frames
        self.last_finger_count = int(sum(self.fingersHistory) / len(self.fingersHistory))
        if finger_count != self.last_finger_count:
            self.last_finger_count = finger_count
        if finger_count == 2:
            pyautogui.press('volumeup')
            print("Volume Up")
        elif finger_count == 3:
            pyautogui.press('volumedown')
            print("Volume down")
        elif finger_count == 1:
            pyautogui.press('playpause')
            print("Play / Pause")
    self.last_action_time = time.time()


class FingerDetector:

    def __init__(self):
        self.keepRunning = True
        self.isBgCaptured = False
        self.bgModel = None
        self.fingersHistory = []
        try:
            self.camera = cv.VideoCapture(1)
        except cv.error as e:
            print("Error: " + e)
        self.main()

    def end_detection(self):
        self.camera.release()
        cv.destroyAllWindows()
        print("Ending finger detection")
        self.keepRunning = False

    def get_frame(self):
        _, frame = self.camera.read()
        return cv.flip(frame, 1)

    def capture_bg(self):
        self.bgModel = cv.createBackgroundSubtractorMOG2(0, BG_THRESHOLD)
        self.isBgCaptured = True
        print("Background Captured")

    def main(self):
        while self.keepRunning:
            frame = self.get_frame()
            cv.rectangle(frame, (int(REGION_X_START * frame.shape[1]), 0),
                         (frame.shape[1], int(REGION_Y_END * frame.shape[0])), (255, 0, 0), 2)
            cv.imshow('Original', frame)
            if self.isBgCaptured:
                img_no_bg = remove_bg(self.bgModel, frame)
                img_with_mask = img_no_bg[0:int(REGION_Y_END * frame.shape[0]),
                                          int(REGION_X_START * frame.shape[1]):frame.shape[1]]
                cv.imshow('Image no background', img_with_mask)
                img_binary = to_binary(img_with_mask)
                cv.imshow('Binary image', img_binary)
                try:
                    hand, img_final = find_contour(img_binary, img_with_mask)
                except TypeError:
                    continue
                finger_num = calculate_fingers(hand, img_final)
                print(finger_num)
                cv.imshow('Result', img_final)
                if len(self.fingersHistory) > 10:
                    self.fingersHistory.pop(0)
                self.fingersHistory.append(finger_num)
                current_fingers = int(sum(self.fingersHistory) / len(self.fingersHistory))
                print(current_fingers)
                gest_out(self, current_fingers)

            k = cv.waitKey(10)
            if k == 27:  # ESC to exit
                self.end_detection()
            elif k == ord('b'):
                self.capture_bg()


if __name__ == '__main__':
    finger_detector = FingerDetector()
