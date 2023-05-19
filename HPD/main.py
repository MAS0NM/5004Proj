'''main program that sends data to unity'''

from cvzone.HandTrackingModule import HandDetector
import cv2
import socket
import numpy as np
from LengthBasedCalibrator_inf import lengthBased_inf
from HandPoseBasedCalibrator_inf import handPoseBased_inf
# from MiDaS_depth_inf import MiDaS_based_inf
from DepthCalibrator import depthBased_inf
import joblib
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
success, img = cap.read()
h, w, _ = img.shape
detector = HandDetector(detectionCon=0.8, maxHands=1)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052)
# cali = Calibrator()
counter = 0
# inf = inference()

calibrator = 'length_based'
# calibrator that is based on the finger length and the distance
cali_inf = depthBased_inf(mode=calibrator)
# calibrator based on the handpose
cali_pos = depthBased_inf(mode='handpose_based')
clf = joblib.load('svm_clf.pkl')
while True:
    # Get image frame
    success, img = cap.read()
    # Find the hand and its landmarks
    hands, img = detector.findHands(img)  # with draw
    data = []

    if hands:
        # raw data, 21 x 3
        lmList = hands[0]['lmList']
        # handpose based calibrator
        lmList = cali_pos.inf(lmList)
        # will be extended to 21 x 4 with an additional distance on the rear
        lmList = cali_inf.inf(lmList)
        for idx, lm in enumerate(lmList):
            data.extend([lm[0], h - lm[1], lm[2]])
        label = clf.predict(np.array([data]))
        sock.sendto(str.encode(str(data)+label[0]), serverAddressPort)

    # Display
    
    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    cv2.imshow("Image", img)
    cv2.waitKey(1)