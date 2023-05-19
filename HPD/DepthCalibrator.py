''' the depth calibrator that integrate all the proposed methods for inference '''

from LengthBasedCalibrator_inf import lengthBased_inf
from HandPoseBasedCalibrator_inf import handPoseBased_inf
from MiDaS_depth_inf import MiDaS_based_inf

class depthBased_inf:
    def __init__(self, mode='length_based'):
        self.mode = mode
        self.infer = self.get_calibrator(mode)
        self.counter = 0
    
    def get_calibrator(self, calibrator):
        if calibrator == 'length_based':
            cali = lengthBased_inf()
        elif calibrator == 'handpose_based':
            cali = handPoseBased_inf()
        elif calibrator == 'depth_based':
            cali = MiDaS_based_inf()
            
        return cali
    
    def inf(self, lmList, img=None):
        # print(lmList)
        if self.mode == 'length_based':
            if self.counter < 60:
                self.infer.len_cali_setup(lmList)
                self.counter += 1
                print(self.counter)
                return lmList
            else:
                lmList = self.infer.len_cali(lmList)
                return lmList
            
        elif self.mode == 'handpose_based':
            lmList = self.infer.cali_predict(lmList)
            return lmList
        
        elif self.mode == 'depth_based':
            # print(lmList)
            lmList = self.infer.depth_estimate(lmList, img)
            return lmList