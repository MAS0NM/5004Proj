''' handpose based method to do depth calibration '''

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
from models import TransformerModel

class handPoseBased_inf:
    
    def __init__(self):
        self.model = TransformerModel()
        self.model.load_state_dict(torch.load('transformer_model.pth'))
        self.lmList_Origin = []
        
    def dataLoader(self, lmList):
        
        self.lmList_Origin = lmList.copy()
        ox, oy, oz = lmList[0][0], lmList[0][1], lmList[0][2]
        tmp = [0, 0, 0]
        X, Y = [0], [0]
        for lm in lmList[1:]:
            tmp.extend([lm[0]-ox, lm[1]-oy, lm[2]-oz])
            X.append(lm[0]-ox)
            Y.append(lm[1]-oy)
            
        return X, Y
    
    def cali_predict(self, lmList):
        
        self.dataLoader(lmList)
        new_X1, new_X2 = self.dataLoader(lmList)
        scaler = MinMaxScaler(feature_range=(0, 1))
        new_X1 = np.array(new_X1).reshape(1,-1)
        new_X2 = np.array(new_X2).reshape(1,-1)
        new_X1 = scaler.fit_transform(new_X1)
        new_X2 = scaler.fit_transform(new_X2)

        # 将数据转换为PyTorch张量
        new_X1 = torch.from_numpy(new_X1).float()
        new_X2 = torch.from_numpy(new_X2).float()

        # 进行推理
        with torch.no_grad():
            prediction = self.model(new_X1, new_X2)

        # 反归一化处理
        prediction = prediction.numpy()
        prediction = scaler.inverse_transform(prediction)
        prediction = [int(i) for i in list(prediction[0])]
        # return prediction
        
        lmList_res = self.lmList_Origin.copy()
        
        for i in range(len(prediction)):
            lmList_res[i][2] += prediction[i]
        
                
        return lmList_res

        print('Prediction:', prediction)
