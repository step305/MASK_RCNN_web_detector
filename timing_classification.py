from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import time

IMAGE_SIZE = (64, 64)

if __name__ == '__main__':
    model = load_model('classification_model.h5')
    labels = ['Type_1', 'Type_2', 'Type_3', 'Type_4', 'Type_5']
    test_images_path = os.path.abspath(os.path.join(os.path.curdir, 'dataset\\classification\\images\\Class_5'))
    tp_metric = 0
    fp_metric = 0
    fn_metric = 0
    images_count = 0

    images_paths = [os.path.join(test_images_path, f) for f in os.listdir(test_images_path)
                    if (os.path.isfile(os.path.join(test_images_path, f)) and '.jpg' in f.lower())]

    images = []
    t0 = time.time()
    for image_path in images_paths:
        image = cv2.imread(image_path)
        image = cv2.resize(image, IMAGE_SIZE)
        image = img_to_array(image)
        image_arr = np.expand_dims(image, axis=0)
        images.append(image_arr)
    t1 = time.time()
    print('Prepare stage: {:0.3f}ms per image'.format((t1 - t0) / len(images) * 1000.0))

    probs = model.predict(images[0])[0]
    t0 = time.time()
    for image in images:
        probs = model.predict(image)[0]
        proba = max(probs)
    t1 = time.time()

    print('Classification stage: {0:3f}ms per image'.format((t1 - t0) / len(images) * 1000.0))
