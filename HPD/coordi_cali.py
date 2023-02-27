import numpy as np
import math

class Calibrator:
    def __init__(self):
        self.coff = self.dis_cali_setup()
        self.hand_standard = [0 for _ in range(23)]
    def dis_cali_setup(self):
        
        x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
        y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    
        coff = np.polyfit(x, y, 2)
        return coff
    
    def cal_dis(self, lmList):
        x1, y1, z1 = lmList[5]
        x2, y2, z1 = lmList[17]
        distance = int(math.sqrt((y2-y1)**2 + (x2 - x1) ** 2))
        A, B, C = self.coff
        distanceCM = A*distance**2 + B*distance + C
        print(distance, distanceCM)
        return distance, distanceCM
    
    def len_cali_setup(self, lmList):
        # 0-1 1-2 2-3 3-4 0-5 5-6 ...
        idx = 0
        for l in [list(range(0,5)), [0]+list(range(5,9)),\
            [0]+list(range(9,13)), [0]+list(range(13,17)),\
            [0]+list(range(17,21)),\
            [5,9,13,17]]:
            for i in range(len(l)-1):
                x1, y1, z1 = lmList[l[i+1]]
                x2, y2, z2 = lmList[l[i]]
                length = int(math.sqrt((y2-y1)**2 + (x2-x1)**2))
                self.hand_standard[idx] = (length + self.hand_standard[idx]) // 2
                idx += 1
        # print(tmp)
        # [67, 60, 52, 41, 67, 60, 52, 41, 67, 60, 52, 41, 67, 60, 52, 41, 67, 60, 52, 41, 67, 60, 52]
        # print(self.hand_standard, len(self.hand_standard))
    
    def len_cali(self, lmList):
        idx = 0
        tmp = []
        for j, l in enumerate([list(range(0,5)), [0]+list(range(5,9)),\
            [0]+list(range(9,13)), [0]+list(range(13,17)),\
            [0]+list(range(17,21))]):
            for i in range(len(l)-1):
                x1, y1, z1 = lmList[l[i]]
                x2, y2, z2 = lmList[l[i+1]]
                tmp.append((l[i], l[i+1], idx))
                cor_length = self.hand_standard[idx]
                # print(rate)
                if cor_length**2 > (x2-x1)**2 + (y2-y1)**2:
                    z2 = -int(math.sqrt(cor_length**2 - (x2-x1)**2 - (y2-y1)**2)) + z1
                lmList[i] = [x2, y2, z2]
                idx += 1
        return lmList
    
    def lr_cali(self, lmList):
        pass