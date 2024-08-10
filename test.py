import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Initialize video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Initialize the hand detector and classifier
detector = HandDetector(maxHands=2)
try:
    classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
except Exception as e:
    print(f"Error loading model or labels: {e}")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Constants
offset = 20
imgSize = 300
labels = ["Salamat", "Kumusta", "Okay", "Mahal Kita"]

while True:
    success, img = cap.read()
    
    # Check if the frame was successfully captured
    if not success:
        print("Failed to capture image from camera. Exiting...")
        break

    imgOutput = img.copy()
    
    # Detect hands in the image
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Prepare a white background for the processed hand image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the image around the detected hand
        try:
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            imgCropShape = imgCrop.shape
        except Exception as e:
            print(f"Error cropping the image: {e}")
            continue

        # Calculate the aspect ratio of the hand and resize accordingly
        aspectRatio = h / w

        try:
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Get the prediction from the classifier
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            
            # Draw the bounding box and label on the original image
            cv2.rectangle(imgOutput, (x - offset, y - offset - 70), 
                          (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 30), 
                        cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), 
                          (x + w + offset, y + h + offset), (0, 255, 0), 4)

            # Display the cropped and resized hand image
            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)

        except Exception as e:
            print(f"Error processing hand image: {e}")
            continue

    # Display the output image with bounding box and label
    cv2.imshow('Image', imgOutput)
    
    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
