'''main program that sends data to unity'''

from cvzone.HandTrackingModule import HandDetector
import cv2
import socket
import math
import numpy as np
from LengthBasedCalibrator_inf import lengthBased_inf
from HandPoseBasedCalibrator_inf import handPoseBased_inf
from MiDaS_depth_inf import MiDaS_based_inf
from DepthCalibrator import depthBased_inf
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

calibrator = 'depth_based'
# calibrator = 'length_based'
cali_inf = depthBased_inf(mode=calibrator)
while True:
    # Get image frame
    success, img = cap.read()
    # Find the hand and its landmarks
    hands, img = detector.findHands(img)  # with draw
    data = []

    if hands:
        # Hand 1
        hand = hands[0]
        lmList = hand["lmList"]  # List of 21 Landmark points
        # depth enhance
        lmList = hands[0]['lmList']
        
        # fixed length based calibration
        # cali.len_cali_setup(lmList)
        # if counter < 60:
        #     cali.len_cali_setup(lmList)
        #     counter += 1
        #     print(counter)
        # else:
        #     lmList = cali.len_cali(lmList)
            
        # transformer based calibration
        # lmList = inf.cali_predict(lmList)
        print(max(lmList))
        lmList = cali_inf.inf(lmList, img)
        for idx, lm in enumerate(lmList):
            data.extend([lm[0], h - lm[1], lm[2]])
        sock.sendto(str.encode(str(data)), serverAddressPort)

    # Display
    
    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    cv2.imshow("Image", img)
    cv2.waitKey(1)