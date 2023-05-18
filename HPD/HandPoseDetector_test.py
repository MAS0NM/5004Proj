''' hand pose detection demo '''

from cvzone.HandTrackingModule import HandDetector
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import os
import numpy as np
import joblib

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Pose Recognizer")
                
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)
        self.detector = HandDetector(detectionCon=0.8, maxHands=1)
        
        self.res_default = '> w <'
        self.res = tk.Label(self.root, text=self.res_default, font=('Arial', 20))
        self.res.pack()
        
        self.label = tk.Label(self.root)
        self.label.pack()
        
        self.update_img()
        
        self.clf = joblib.load('svm_clf.pkl')
        
    def update_img(self):
        _, img = self.cap.read()
        hands, img = self.detector.findHands(img)
        h, w, _ = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        if hands:
            hand = hands[0]
            lmList = hand["lmList"]
            sample = []
            for lm in lmList:
                sample.extend([lm[0], h - lm[1], lm[2]])
            
            sample = np.array([sample])
            prediction = self.clf.predict(sample)
            # print(prediction)
            self.res.config(text=prediction[0])    
        else:
            self.res.config(text=self.res_default)
        
        photo = ImageTk.PhotoImage(Image.fromarray(img).resize((w//2, h//2)))
        self.label.config(image = photo)
        self.label.image = photo
        self.root.after(30, self.update_img)
        
root = tk.Tk()
app = App(root)
root.mainloop()