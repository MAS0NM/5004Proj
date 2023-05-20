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



# allpath = []
# def getallfile(path):
#     file_path = os.listdir(path)
#     # 遍历该文件夹下的所有目录或者文件
#     for file in file_path:
#         fp = os.path.join(path, file)
#         # 如果是文件夹，递归调用函数
#         if os.path.isdir(fp):
#             getallfile(fp)
#         # 如果不是文件夹，保存文件路径及文件名
#         elif os.path.isfile(fp):
#             allpath.append(fp)
#
#
# path = 'Datasets/Northwestern_Hand_Gesture'
# getallfile(path)
# for path in allpath:
#     if re.search(r'_bin200_', path):
#         print(path)
#         print(os.system("rm " + path))
#
# videos = []
# labels = []
# cnt = 0
# for path in tqdm(allpath):
#     frames = []
#     cap = cv2.VideoCapture(path)
#     while(cap.isOpened()):
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frames.append(frame)
#         # cv2.imshow('image', frame)
#     videos.append(np.array(frames))
#     labels.append(path.split('_')[-2])
#     # videos = np.array(frames)
#     cnt += 1
#     if cnt > 30:
#         break
#
# # videos = np.array(videos)
# # 每个视频帧数不同
#
# cap.release()
# cv2.destroyAllWindows()
#
# warnings.filterwarnings('ignore')
# tic = time.time()


# videos: []list of video,
# video: frames, height, width, 3(rgb)
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
    print("keyframes index: ", key_frames_label)
    return key_frames_label




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

# 获取当前目录中的所有文件夹和文件
items = os.listdir()
# 遍历所有文件夹和文件
for item in items:
    # 检查是否为文件夹并且以'frames_'为前缀
    if os.path.isdir(item) and item.startswith('frames_'):
        # 构建文件夹的完整路径
        folder_path = os.path.join(os.getcwd(), item)
        # 删除文件夹及其内容
        shutil.rmtree(folder_path)

data = []  # 用于保存手势数据
labels = []  # 用于保存标签

is_recording = False  # 是否正在录制
label = None  # 录制的标签
key_images = []


while True:
    # Get image frame
    success, img = cap.read()
    # Find the hand and its landmarks
    hands, img = detector.findHands(img)  # with draw
    # hands = detector.findHands(img, draw=False)  # without draw
    # Display
    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    # 弹窗要求输入标签
    if key == ord('l'):
        label = input("Enter label: ")

    # 点击 'r' 键开始录制
    elif key == ord('r'):
        if label:
            is_recording = True
            print("Recording started...")
        else:
            print("Please enter a label first!")

    # 点击 's' 键停止录制
    elif key == ord('s'):
        if is_recording:
            is_recording = False
            print("Recording stopped.")
            # 保存手势数据和标签
            with open("data_output.txt", "a") as f:
                for frame_data in data:
                # frame_data = data[len(data)//2]
                    np.savetxt(f, frame_data, fmt='%f')
            with open("label.txt", "a") as f:
                for i in range(len(data)):
                    f.write(label + '\n')
            # 保存原始帧和关键帧
            # 创建一个新文件夹
            # for keyframe_sequence in key_images:
            #     new_dir = "key_frames_" + label + "_" + str(cnt2)
            #     cnt2 += 1
            #     if not os.path.exists(new_dir):
            #         os.makedirs(new_dir)
            #     # 使用 OpenCV 存储图片
            #     # keyframe_sequence = key_images[len(key_images)//2]
            #     for i, img in enumerate(keyframe_sequence):
            #         file_name = os.path.join(new_dir, f"image_{i}.png")
            #         cv2.imwrite(file_name, img)
            # 清空手势数据和标签
            data = []
            labels = []
            sequence = []
            key_images = []

    if is_recording:
        # 保存当前帧的手势数据
        if hands:
            # detected hands, then add img to frame sequence
            sequence.append(img)
            hand = hands[0]
            lmList = hand["lmList"]  # List of 21 Landmark points
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
                frame_data = []
                for i in key_frames:
                    rows = range(len(data1[i]))
                    row_str = [str(data1[i][j]) for j in rows]
                    frame_data.append(' '.join(row_str))
                data.append([data1[i] for i in key_frames])
                labels.append(label)
                key_images.append(sequence[i] for i in key_frames)

        else:
            sequence = sequence

