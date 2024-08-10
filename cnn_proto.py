import numpy as np
import pickle
import cv2
import os
from glob import glob
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras import backend as K

K.set_image_data_format('channels_last')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_image_size(sample_image_path):
    try:
        img = cv2.imread(sample_image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Failed to load the image. Check if the file path is correct.")
        return img.shape
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def get_num_of_classes(gestures_path):
    return len(glob(os.path.join(gestures_path, '*')))

def cnn_model(num_of_classes, imgX, imgY):
    print("Number of classes in your model:", num_of_classes)
    model = Sequential()
    model.add(Conv2D(16, (2, 2), input_shape=(imgX, imgY, 1), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPool2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_of_classes, activation='softmax'))
    sgd = optimizers.SGD(learning_rate=1e-2)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])
    filepath = 'trained_model.h5'
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint1]
    return model, callbacks_list

def create_datasets(train_images_path, train_labels_path, val_images_path, val_labels_path, gestures_path):
    # Directories
    gesture_dirs = glob(os.path.join(gestures_path, '*'))
    num_classes = len(gesture_dirs)

    # Prepare lists to hold data
    train_images = []
    train_labels = []
    val_images = []
    val_labels = []

    for label, gesture_dir in enumerate(gesture_dirs):
        image_files = glob(os.path.join(gesture_dir, '*.jpg'))  # or *.png based on your image format
        for image_file in image_files:
            img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                resized_img = cv2.resize(img, (imgX, imgY))  # Resize image to match model input
                train_images.append(resized_img)
                train_labels.append(label)
                # For simplicity, we'll use the same images for validation in this example
                val_images.append(resized_img)
                val_labels.append(label)

    # Convert lists to numpy arrays
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    val_images = np.array(val_images)
    val_labels = np.array(val_labels)

    # Save data to files
    with open(train_images_path, "wb") as f:
        pickle.dump(train_images, f)
    with open(train_labels_path, "wb") as f:
        pickle.dump(train_labels, f)
    with open(val_images_path, "wb") as f:
        pickle.dump(val_images, f)
    with open(val_labels_path, "wb") as f:
        pickle.dump(val_labels, f)

def load_data(file_path):
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, np.ndarray) and data.size == 0:
                raise ValueError(f"The file {file_path} is empty.")
            return np.array(data)
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def train():
    gestures_path = 'D:/Experiment-ML4/gestures'
    train_images_path = "D:/Experiment-ML4/train_images"
    train_labels_path = "D:/Experiment-ML4/train_labels"
    val_images_path = "D:/Experiment-ML4/val_images"
    val_labels_path = "D:/Experiment-ML4/val_labels"
    
    # Create datasets
    create_datasets(train_images_path, train_labels_path, val_images_path, val_labels_path, gestures_path)

    imgX, imgY = get_image_size('D:/Experiment-ML4/HandSigns/A/Image_1709644697.138582.jpg')
    if imgX is None or imgY is None:
        print("Error: Unable to determine image size.")
        return

    num_of_classes = get_num_of_classes(gestures_path)
    
    train_images = load_data(train_images_path)
    train_labels = load_data(train_labels_path)
    val_images = load_data(val_images_path)
    val_labels = load_data(val_labels_path)
    
    if None in [train_images, train_labels, val_images, val_labels]:
        print("Error loading one or more data files.")
        return
    
    # Reshape images to match input shape requirements
    train_images = np.reshape(train_images, (train_images.shape[0], imgX, imgY, 1))
    val_images = np.reshape(val_images, (val_images.shape[0], imgX, imgY, 1))
    
    # Convert labels to categorical format
    train_labels = to_categorical(train_labels, num_classes=num_of_classes)
    val_labels = to_categorical(val_labels, num_classes=num_of_classes)

    model, callbacks_list = cnn_model(num_of_classes, imgX, imgY)

    print("Model output shape:", model.output_shape)
    print("Train labels shape:", train_labels.shape)
    print("Validation labels shape:", val_labels.shape)
    
    model.summary()
    model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=15, batch_size=500, callbacks=callbacks_list)
    scores = model.evaluate(val_images, val_labels, verbose=0)
    print("CNN Error: %.2f%%" % (100 - scores[1] * 100))
    model.save('trained_model.h5')

train()
K.clear_session()
