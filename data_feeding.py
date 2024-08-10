import cv2 
from glob import glob 
import numpy as np 
import random 
from sklearn.utils import shuffle 
import pickle 
import os 

def pickle_img_labels():
    imgs_label = []
    images = glob("HandSigns/*/*.jpg")
    images.sort()
    for image in images:
        print(image)
        label = image[image.find(os.sep)+1: image.rfind(os.sep)]
        img = cv2.imread(image, 0)
        imgs_label.append((np.array(img, dtype = np.uint8), int(label)))
    return imgs_label

img_labels = pickle_img_labels()
img_labels = shuffle(shuffle(shuffle(shuffle(img_labels))))
imgs, labels  = zip(*img_labels)
print("Length of images_labels", len(img_labels))

train_images = imgs[:int(5/6*len(imgs))]
print("Length of train_images", len(train_images))
with open("train_images", "wb") as f: 
    pickle.dump(train_images, f)
del train_images

train_labels = labels[:int(5/6 * len(labels))]
print("Length of train_labels", len(train_labels))
with open("train_labels", "wb") as f: 
    pickle.dump(train_labels, f)
del train_labels

test_images = imgs[int(5/6 * len(imgs)):int(11/12*len(imgs))]
print("Length of test_images", len(test_images))
with open("test_images", "wb") as f:
    pickle.dump(test_images, f)
del test_images

test_labels = labels[int(5/6 * len(labels)):int(11/12 * len(labels))]
print("Length of test_labels", len(test_labels))
with open("test_labels", "wb") as f:
    pickle.dump(test_labels, f)
del test_labels

val_images = imgs[int(5/6 * len(imgs)):int (11/12 * len(imgs))]
print("Length of val_images", len(val_images))
with open("val_images", "wb") as f: 
    pickle.dump(val_images, f)
del val_images

val_labels = labels[int(5/6 * len(labels)) :int (11/12 * len(labels))]
print("Length of val_labels", len(val_labels))
with open("val_labels", "wb") as f: 
    pickle.dump(val_labels, f)
del val_labels 