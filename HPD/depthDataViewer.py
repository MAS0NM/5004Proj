import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dataset_XY = []
dataset_Dp = []
dataset= []
fileName = 'XY'
filePath = 'HandDepth'
with open(f'./{filePath}/{fileName}.txt', 'r') as f:
    line = f.readline()
    while line:
        line = line.strip().split(',')
        tmp = []
        for i in range(21):
            x, y = int(line[2*i]), int(line[2*i+1])
            tmp.append([x, y])
        dataset_XY.append(tmp)
        line = f.readline()
        
fileName = 'Depth'
filePath = 'HandDepth'

with open(f'./{filePath}/{fileName}.txt', 'r') as f:
    line = f.readline()
    while line:
        line = line.strip().split(',')
        tmp = []
        for i in range(21):
            z = int(line[i])
            tmp.append(z)
        dataset_Dp.append(tmp)
        line = f.readline()
for i in range(len(dataset_XY)):
    print(dataset_XY[i] + dataset_Dp[i])
    dataset.append([dataset_XY[i][j] + [dataset_Dp[i][j]] for j in range(len(dataset_XY[0]))])
    
dataset = np.array(dataset)
print(dataset.shape)
for data in dataset:
    data = data.astype(np.float32)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # ax.scatter3D(data[:, 0], -data[:, 2], data[:, 1]-720)
    ax.scatter3D(data[:, 0]/100, -data[:, 2]/100, data[:, 1]/100)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    x_line_0 = data[0:5, 0]
    x_line_1 = np.append(data[0, 0], data[5:9, 0])
    x_line_2 = np.append(data[5, 0], data[9:13, 0])
    x_line_3 = np.append(data[9, 0], data[13:17, 0])
    x_line_4 = np.append(data[13, 0], data[17:21, 0])
    x_line_5 = np.array([data[0, 0], data[17, 0]])
    tmpls = [x_line_0, x_line_1, x_line_2, x_line_3, x_line_4, x_line_5]
    for idx, _ in enumerate(tmpls):
        tmpls[idx] /= 100
    
    y_line_0 = data[0:5, 2]
    y_line_1 = np.append(data[0, 2], data[5:9, 2])
    y_line_2 = np.append(data[5, 2], data[9:13, 2])
    y_line_3 = np.append(data[9, 2], data[13:17, 2])
    y_line_4 = np.append(data[13, 2], data[17:21, 2])
    y_line_5 = np.array([data[0, 2], data[17, 2]])
    tmpls = [y_line_0, y_line_1, y_line_2, y_line_3, y_line_4, y_line_5]
    for idx, _ in enumerate(tmpls):
        tmpls[idx] *= -1
        tmpls[idx] /= 100
    
    z_line_0 = data[0:5, 1]
    z_line_1 = np.append(data[0, 1], data[5:9, 1])
    z_line_2 = np.append(data[5, 1], data[9:13, 1])
    z_line_3 = np.append(data[9, 1], data[13:17, 1])
    z_line_4 = np.append(data[13, 1], data[17:21, 1])
    z_line_5 = np.array([data[0, 1], data[17, 1]])
    tmpls = [z_line_0, z_line_1, z_line_2, z_line_3, z_line_4, z_line_5]
    for idx, _ in enumerate(tmpls):
        tmpls[idx] /= 100
    
    ax.plot(x_line_0, y_line_0, z_line_0, c = 'r')
    ax.plot(x_line_1, y_line_1, z_line_1, c = 'r')
    ax.plot(x_line_2, y_line_2, z_line_2, c = 'r')
    ax.plot(x_line_3, y_line_3, z_line_3, c = 'r')
    ax.plot(x_line_4, y_line_4, z_line_4, c = 'r')
    ax.plot(x_line_5, y_line_5, z_line_5, c = 'r')
    plt.show()