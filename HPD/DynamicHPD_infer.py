from cvzone.HandTrackingModule import HandDetector
import cv2
import socket
import math
import numpy as np

import os
import shutil
from pathlib import Path
from cluster_dp import cluster_dp
from skimage.io import imread
from skimage.measure import shannon_entropy
import numpy as np
from scipy.signal import find_peaks
import warnings
import time
import matplotlib.pyplot as plt
import pylab
# import imageio
# import skimage.io
import os
import re
import cv2
from tqdm import tqdm
from DynamicModel import LSTMModel
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def get_keyframes(video, key_frames_num=5):
    frame_num = len(video)
    x = np.zeros(frame_num - 2)
    for k in range(1, frame_num-1):
        ii = video[k]
        image_entropy = shannon_entropy(ii)
        x[k - 1] = image_entropy
    # plt.plot(x, '.')
    # plt.show()
    max, max_label = find_peaks(x)
    min, min_label = find_peaks(-x)
    xlabel_ = np.concatenate((max, min)) #最大最小值索引
    xlabel_new = np.concatenate(([0, frame_num - 3], xlabel_)) #加上头尾值
    xlabel_new = np.sort(xlabel_new)
    ylabel_new = x[xlabel_new]
    point_data = np.column_stack((xlabel_new, ylabel_new))
    # key_frames_labels = np.ones()
    point_data = point_data[:4]
    pointNUM = len(point_data[:, 0])
    if pointNUM == 2:
        dif = np.diff(point_data[:, 0])[0]
        key_frames_label = [int(x) for x in point_data[:, 0]] + [round(dif / 4), round(2 * dif / 4),
                                                            round(3 * dif / 4)]
        key_frames_label = sorted(key_frames_label)
        assert len(key_frames_label) == 5, 'There is a error, key_frames_labels ~ = 5'
    elif pointNUM == 3:
        dif = np.diff(point_data[:, 0])
        dif32 = dif[1]
        dif21 = dif[0]
        if dif32 >= dif21:
            key_frames_label = [int(x) for x in point_data[:, 0]] + [round(dif32 / 3 + point_data[1, 0]),
                                                                round(2 * dif32 / 3 + point_data[1, 0])]
        else:
            key_frames_label = [int(x) for x in point_data[:, 0]] + [round(dif21 / 3 + point_data[0, 0]),
                                                                round(2 * dif21 / 3 + point_data[0, 0])]
        key_frames_label = sorted(key_frames_label)
        assert len(key_frames_label) == 5, 'There is a error, key_frames_labels ~ = 5'
    elif pointNUM == 4:
        if point_data[1, 0] + 1 != point_data[2, 0]:
            key_frames_label = [int(x) for x in point_data[:, 0]] + [round((point_data[2, 0] + point_data[1, 0]) / 2)]
        else:
            if point_data[2, 0] < (point_data[3, 0] + point_data[0, 0]) / 2:
                key_frames_label = [int(x) for x in point_data[:, 0]] + [round((point_data[2, 0] + point_data[3, 0]) / 2)]
            # elif point_data[1, 0] >= (point_data[3, 0] + point_data[0, 0]) / 2:
            else:
                key_frames_label = [int(x) for x in point_data[:, 0]] + [round((point_data[0, 0] + point_data[1, 0]) / 2)]
        key_frames_label = sorted(key_frames_label)
        assert len(key_frames_label) == 5, 'There is a error, key_frames_labels ~ = 5'
    elif pointNUM == key_frames_num:
        key_frames_label = [int(x) for x in point_data[:, 0]]
    else:
        # cluster
        key_frames_label = cluster_dp(point_data)
    # print("keyframes index: ", key_frames_label)
    return key_frames_label


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    success, img = cap.read()
    h, w, _ = img.shape
    detector = HandDetector(detectionCon=0.8, maxHands=1)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    serverAddressPort = ("127.0.0.1", 5052)
    counter = 0
    cnt = 0

    sequence_length = 10
    # 存储符合要求的帧
    sequence = []
    cnt2 = 1
    key_hand_points = []
    data1 = []

    label_dict = {0: 'grab', 1: 'none', 2: 'open'}


    input_size = 63
    hidden_size = 64
    output_size = 3
    loaded_model = LSTMModel(input_size, hidden_size, output_size)
    loaded_model.load_state_dict(torch.load('lstm_model.pt'))
    loaded_model.eval()

    last_state = -1
    last_label = -1
    this_count = 0

    while True:
        # Get image frame
        success, img = cap.read()
        # Find the hand and its landmarks
        hands, img = detector.findHands(img)  # with draw
        # hands = detector.findHands(img, draw=False)  # without draw
        # Display
        img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
        if hands:
            # detected hands, then add img to frame sequence
            sequence.append(img)
            hand = hands[0]
            lmList = hand["lmList"]  # List of 21 Landmark points
            lmList
            this_data = []
            for lm in lmList:
                this_data.extend([lm[0], h - lm[1], lm[2]])
            data1.append(this_data)

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
                    outputs = loaded_model(input_data)

                # 获取预测结果
                _, predicted = torch.max(outputs.data, 1)
                predicted_label = predicted.item()

                # 打印预测结果
                # print("Predicted Label:", label_dict[predicted_label])
                if label_dict[predicted_label] == 'none':
                    continue
                if predicted_label == last_state:
                    continue
                elif predicted_label == last_label:
                    this_count += 1
                    if this_count >= 5:
                        last_state = predicted_label
                        this_count = 0
                        print(label_dict[last_state])
                elif predicted_label != last_label:
                    this_count = 0
                    last_label = predicted_label

        else:
            sequence = sequence

