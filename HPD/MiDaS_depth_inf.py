'''just here for reference'''

import cv2
import torch
import matplotlib.pyplot as plt

class MiDaS_based_inf:
    def __init__(self):
        
        self.midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
        self.midas.to('cuda')
        self.midas.eval()

        self.transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
        self.transform = self.transforms.small_transform

    def get_depth(self, img):
        with torch.no_grad():
            imgbatch = self.transform(img).to('cuda')
            
            prediction = self.midas(imgbatch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size = img.shape[:2],
                mode = 'bicubic',
                align_corners=False,
                
            ).squeeze()
            
            output = prediction.cpu().numpy()
        return output
    
    def depth_estimate(self, lmList, img):
        depth = self.get_depth(img)
        for idx, lm in enumerate(lmList):
            x, y = lmList[idx][0], lmList[idx][1]
            print(x, y)
            lmList[idx][2] += depth[y][x] // 10
        # print(lmList)
        return lmList
        
        
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cuda')
midas.eval()
# Input transformation pipeline
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to('cuda')
    
    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size = img.shape[:2],
            mode = 'bicubic',
            align_corners=False,
            
        ).squeeze()
        
        output = prediction.cpu().numpy()
    cv2.imshow('CV2Frame', frame)
    plt.imshow(output)
    
    plt.pause(0.00001)
    # cv2.imshow('CV2Frame', frame)
    # print(output)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()

plt.show()