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
from DynamicModel import LSTMModel
import torch
from DynamicHPD_infer import get_keyframes


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

# load dynamic HPD model
input_size, hidden_size, output_size = 63, 64, 3
dmodel = LSTMModel(input_size, hidden_size, output_size)
dmodel.load_state_dict(torch.load('lstm_model.pt'))
dmodel.eval()

# set the dynamic HPD data
sequence_length = 10
# 存储符合要求的帧
sequence = []
key_hand_points = []
data1 = []

label_dict = {0: 'grasp', 1: 'none', 2: 'throw'}
# last_state = -1
# last_label = -1
# this_count = 0
dynamic_label = 'none'
last_label = 'none'

while True:
    # Get image frame
    success, img = cap.read()
    # Find the hand and its landmarks
    hands, img = detector.findHands(img)  # with draw
    data = []
    # Display

    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

    if hands:
        sequence.append(img)
        # raw data, 21 x 3
        lmList = hands[0]['lmList']
        # handpose based calibrator
        lmList = cali_pos.inf(lmList)
        # will be extended to 21 x 4 with an additional distance on the rear
        lmList, dis_cm = cali_inf.inf(lmList)
        for lm in lmList:
            data.extend([lm[0], h - lm[1], lm[2]])
        data1.append(data)
        label = clf.predict(np.array([data]))

        # 如果序列长度超过要求，保持 sequence_length 个 frame
        if len(sequence) > sequence_length:
            sequence = sequence[-sequence_length:]
            data1 = data1[-sequence_length:]

        # 检查序列是否已经达到要求的长度
        if len(sequence) >= sequence_length:
            # 如果序列长度达到要求，就将序列中的帧作为关键帧进行处理
            key_frames = get_keyframes(sequence)
            input_data = np.array([data1[i] for i in key_frames])
            # 准备输入数据
            # input_data = np.random.rand(5, 63)  # 5*63维的输入数据
            input_data = torch.from_numpy(input_data).unsqueeze(0).float()  # 调整数据形状 (1, 5, 63)

            # 进行推理
            with torch.no_grad():
                outputs = dmodel(input_data)

            # 获取预测结果
            _, predicted = torch.max(outputs.data, 1)
            predicted_label = predicted.item()

            # 打印预测结果
            # print("Predicted Label:", label_dict[predicted_label])
            # if label_dict[predicted_label] == 'none':
            #     last_state = last_state
            # if predicted_label == last_state:
            #     last_state = last_state
            # elif predicted_label == last_label:
            #     this_count += 1
            #     if this_count >= 5:
            #         last_state = predicted_label
            #         this_count = 0
            #         dynamic_label = label_dict[last_state]
            # elif predicted_label != last_label:
            #     this_count = 0
            #     last_label = predicted_label
            dynamic_label = label_dict[predicted_label]

        # print(label, dynamic_label)
        final_label = label[0]
        if final_label in ['up', 'down', 'left', 'right']:
            final_label = final_label
        elif final_label in ['point', 'five']:
            if dynamic_label == 'grasp':
                final_label = 'grasp'
            elif dynamic_label == 'throw':
                final_label = 'throw'
            else:
                final_label = final_label
        # print(final_label)
        if last_label == 'grasp' and final_label == 'throw':
            final_label = 'throw'
        elif last_label != 'grasp' and final_label == 'throw':
            final_label = 'five'
        print(final_label)
        last_label = final_label
        data = []
        for lm in lmList:
            data.extend([lm[0], h - lm[1], lm[2], dis_cm])
        # print(data)
        sock.sendto(str.encode(str(data)+final_label), serverAddressPort)

