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
    times = []
    t0 = time.time()
    for image_path in images_paths:
        image = cv2.imread(image_path)
        t00 = time.time()
        image = cv2.resize(image, IMAGE_SIZE)
        image = img_to_array(image)
        image_arr = np.expand_dims(image, axis=0)
        images.append(image_arr)
        t11 = time.time()
        times.append(t11 - t00)
    t1 = time.time()
    print('Max timing: {:0.3f}ms, mean timing: {:0.3f}ms, std timing: {:0.3f}ms'.format(max(times)*1000,
                                                                                        np.mean(times)*1000,
                                                                                        np.std(times)*1000))
    print('Total images: {}'.format(len(images)))

    probs = model.predict(images[0])[0]
    times = []
    t0 = time.time()
    for image in images:
        t00 = time.time()
        probs = model.predict(image)[0]
        proba = max(probs)
        t11 = time.time()
        times.append(t11 - t00)
    t1 = time.time()

    print('Max timing: {:0.3f}ms, mean timing: {:0.3f}ms, std timing: {:0.3f}ms'.format(max(times)*1000,
                                                                                        np.mean(times)*1000,
                                                                                        np.std(times)*1000))
    print('Total images: {}'.format(len(images)))
