import cv2 
from cvzone.HandTrackingModule import HandDetector
import numpy as np 
import math
import time 

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
offset = 20 
imgSize = 300 
counter = 0

folder = 'D:\\Experiment-ML4\\HandSigns\\Oo'

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255

        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
        
        # Check if imgCrop is not empty before attempting to resize
        if imgCrop.shape[0] != 0 and imgCrop.shape[1] != 0:
            imgCropShape = imgCrop.shape
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize-wCal)/2)
                imgWhite[:, wGap: wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResize

            cv2.imshow('Image Crop', imgCrop)
            cv2.imshow('Image White', imgWhite)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    
    # Exit the loop if the ESC key is pressed
    if key == 27:  # 27 is the ASCII code for the ESC key
        break
    
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
