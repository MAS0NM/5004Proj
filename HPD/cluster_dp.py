import numpy as np
from sklearn.cluster import DBSCAN, KMeans
import pandas as pd


def cluster_dp(points):
    kmeans = KMeans(n_clusters=5)
    # X = numpy.array(points)
    kmeans.fit(points)
    # print(kmeans.cluster_centers_)
    # kmeans的聚类中心可能不在原始点中，选取每个类别图像熵最大的作为关键帧
    # 将坐标和标签合并到DataFrame中
    df = pd.DataFrame({'x': points[:, 0], 'y': points[:, 1], 'label': kmeans.labels_})

    # 对DataFrame按照标签进行分组，并使用apply函数对每个分组进行处理
    representatives = df.groupby('label').apply(
        lambda g: g['x'].iloc[np.argmax(g['y'])] if len(g) > 1 else g['x'].iloc[0])

    # 将处理后的代表点的横坐标组成的list返回
    representatives = representatives.tolist()
    representatives = list(map(int, representatives))
    return representatives
# print(X[:,0], X[:,1])
# import matplotlib.pyplot as plt
# # plt.figure()
# plt.plot(X[:,0], X[:,1],'.')
# plt.show()
