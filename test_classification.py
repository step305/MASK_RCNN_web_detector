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
    test_images_paths = [os.path.abspath(os.path.join(os.path.curdir,
                                                      'dataset\\classification\\images\\Class_{:d}'.format(i)))
                         for i in range(1, 6)]
    tp_metric = 0
    fp_metric = 0
    fn_metric = 0
    images_count = 0

    t0 = time.time()

    for test_images_path, true_label in zip(test_images_paths, labels):
        images = [os.path.join(test_images_path, f) for f in os.listdir(test_images_path)
                  if (os.path.isfile(os.path.join(test_images_path, f)) and '.jpg' in f.lower())]

        print('Processing images in {:}'.format(test_images_path))
        cnt = 0
        N = len(images)
        for image_path in images:
            if cnt % 40:
                print('Processed {:d} images of {:d}'.format(cnt, N), end='\r')
            cnt += 1
            image = cv2.imread(image_path)
            image = cv2.resize(image, IMAGE_SIZE)
            image = img_to_array(image)
            image_arr = np.expand_dims(image, axis=0)
            probs = model.predict(image_arr)[0]
            proba = max(probs)
            if proba < 0.5:
                fn_metric += 1
            else:
                label = labels[np.argmax(probs)]
                if label == true_label:
                    tp_metric += 1
                else:
                    fp_metric += 1
            images_count += 1
    te = time.time()

    print('{:d} images classified in {:.1f}sec with rate {:.1f} images per second'.format(images_count, te - t0,
                                                                                          (images_count / (te - t0))))

    print('True positives = {:d}\nFalse negatives = {:d}\nFalse positives = {:d}'.format(tp_metric,
                                                                                         fn_metric,
                                                                                         fp_metric))
    print('Accuracy = {:.3f}%'.format(tp_metric / images_count * 100.0))
    print('Precision = {:.3f}%'.format(tp_metric / (tp_metric + fp_metric) * 100.0))
    print('Recall = {:.3f}%'.format(tp_metric / (tp_metric + fn_metric) * 100.0))
