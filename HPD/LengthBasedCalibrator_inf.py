''' depth calibrator based on length '''

import numpy as np
import math

class lengthBased_inf:
    def __init__(self):
        self.coff = self.dis_cali_setup()
        self.hand_standard = [0 for _ in range(23)]
        self.hand_standard = [67, 60, 52, 41, 67, 60, 52, 41, 67, 60, 52, 41, 67, 60, 52, 41, 67, 60, 52, 41, 67, 60, 52]
    def dis_cali_setup(self):
        
        x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
        y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    
        coff = np.polyfit(x, y, 2)
        return coff
    
    def cal_dis(self, lmList):
        x0, y0, z0 = lmList[0]
        x1, y1, z1 = lmList[5]
        x2, y2, z1 = lmList[17]
        
        len0 = int(math.sqrt((y2-y1)**2 + (x2 - x1) ** 2))
        len1 = int(math.sqrt((y0-y1)**2 + (x0 - x1) ** 2))
        len2 = int(math.sqrt((y2-y0)**2 + (x2 - x0) ** 2))
                   
        distance, max_idx = max((val, idx) for idx, val in enumerate([len0, len1, len2]))
        std_distance = [sum(self.hand_standard[:-4]), self.hand_standard[4], self.hand_standard[16]][max_idx]
        A, B, C = self.coff
        distanceCM = A*distance**2 + B*distance + C
        scale = distance / std_distance
        # print(distance, distanceCM)
        return distance, distanceCM, scale
    
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
        depth = []
        for l in [list(range(0,5)), [0]+list(range(5,9)),\
            [0]+list(range(9,13)), [0]+list(range(13,17)),\
            [0]+list(range(17,21))]:
            for i in range(len(l)-1):
                x1, y1, z1 = lmList[l[i]]
                x2, y2, z2 = lmList[l[i+1]]
                dis, dis_cm, scale = self.cal_dis(lmList)
                cor_length = self.hand_standard[idx]
                cur_length = scale*cor_length
                # print(rate)
                sqr = cur_length**2 - (x2-x1)**2 - (y2-y1)**2
                sign = 1 if z2 - z1 >= 0 else -1
                if sqr > 0:
                    z2 = sign * int(math.sqrt(sqr)) + z1
                depth.append(z2)
                idx += 1
        for idx, lm in enumerate(lmList):
            # if idx == 0:
            #     lmList[idx][0] += dis
            # elif idx >= 1:
            #     lmList[idx][2] += depth[idx-1] + dis
            lmList[idx].append(dis_cm)
        # print(lmList)
        return lmList