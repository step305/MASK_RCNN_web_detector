from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from backend.lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os

import matplotlib

if __name__ == '__main__)':
    matplotlib.use("Agg")

    EPOCHS = 200
    INIT_LR = 1e-4
    BS = 32
    IMAGE_SIZE = (64, 64)

    print("[INFO] loading images...")
    data = []
    labels = []

    imagePaths = sorted(list(paths.list_images('dataset\\classification\\images')))
    random.seed(42)
    random.shuffle(imagePaths)

    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        image = cv2.resize(image, IMAGE_SIZE)
        image = img_to_array(image)
        data.append(image)

        label = imagePath.split(os.path.sep)[-2]
        if label == "Class_1":
            label = 0
        elif label == "Class_2":
            label = 1
        elif label == "Class_3":
            label = 2
        elif label == "Class_4":
            label = 3
        elif label == "Class_5":
            label = 4
        else:
            label = 5

        labels.append(label)

    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    (trainX, testX, trainY, testY) = train_test_split(data,
                                                      labels, test_size=0.25, random_state=42)

    trainY = to_categorical(trainY, num_classes=5)
    testY = to_categorical(testY, num_classes=5)

    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode="nearest")

    print("[INFO] compiling model...")

    model = LeNet.build(width=IMAGE_SIZE[0], height=IMAGE_SIZE[1], depth=3, classes=5)
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    print("[INFO] training network...")
    H = model.fit(x=aug.flow(trainX, trainY, batch_size=BS),
                  validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
                  epochs=EPOCHS, verbose=1)

    print("[INFO] serializing network...")
    model.save('classification_model.h5', save_format="h5")

    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig('plot.jpg')
