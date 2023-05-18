''' record the xy and y coordinates data for 3d reconstruction'''

from cvzone.HandTrackingModule import HandDetector
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import os
import numpy as np
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Pose Labeler")
                
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)
        self.detector = HandDetector(detectionCon=0.8, maxHands=1)
        self.data = []
        
        self.button1 = tk.Button(self.root, text="Record X,Y",\
                                command=self.Record_Coord_XY)
        self.button1.pack()
        
        self.button2 = tk.Button(self.root, text="Record Y",\
                                command=self.Record_Coord_Depth)
        self.button2.pack()
        
        self.label = tk.Label(self.root)
        self.label.pack()
        self.update_img()
        
           
    def Record_Coord_XY(self):
        label = 'XY'
        pos = self.data[-1] if self.data else [0,0]*21
        print(label, '\n', pos)
        pos = [pos[i] for i in range(len(pos)) if i%3 != 2]
        folder_path = './HandDepth'
        os.makedirs(folder_path, exist_ok=True)
        with open(f"./{folder_path}/{label}.txt", 'a') as f:
            f.write(str(pos)[1:-1] + '\n')
            
    def Record_Coord_Depth(self):
        label = 'Depth'
        pos = self.data[-1] if self.data else [0]*21
        print(label, '\n', pos)
        
        pos = [pos[i] for i in range(len(pos)) if i%3 == 2]
        folder_path = './HandDepth'
        os.makedirs(folder_path, exist_ok=True)
        with open(f"./{folder_path}/{label}.txt", 'a') as f:
            f.write(str(pos)[1:-1] + '\n')
        
    def update_img(self):
        _, img = self.cap.read()
        hands, img = self.detector.findHands(img)
        h, w, _ = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if len(self.data) > 10:
            self.data.pop(0)
            
        if hands:
            hand = hands[0]
            lmList = hand["lmList"]
            ox, oy, oz = lmList[0][0], lmList[0][1], lmList[0][2]
            lmList[0] = [0, 0, 0]
            tmp = [0, 0, 0]
            for lm in lmList[1:]:
                tmp.extend([lm[0]-ox, lm[1]-oy, lm[2]-oz])
                self.data.append(tmp)
        
        photo = ImageTk.PhotoImage(Image.fromarray(img).resize((w//2, h//2)))
        self.label.config(image = photo)
        self.label.image = photo
        self.root.after(30, self.update_img)
        
root = tk.Tk()
app = App(root)
root.mainloop()