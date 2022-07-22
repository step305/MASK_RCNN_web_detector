from backend import DefectDetector
import cv2
import time
import os
import numpy as np

defect_detector = DefectDetector.DefectDetector('mask_rcnn_models\\mask_rcnn_model.resnet101.h5')


if __name__ == '__main__':
    im_p = 'dataset\\test'
    images = []
    images_paths = [os.path.join(im_p, f) for f in os.listdir(im_p)
                    if (os.path.isfile(os.path.join(im_p, f)) and '.jpg' in f.lower())]
    for img in images_paths:
        images.append(cv2.imread(img))

    img_re, result = defect_detector.detect(images[0])

    times = []
    t0 = time.time()
    for img in images:
        t00 = time.time()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (600, 600))
        t11 = time.time()
        times.append(t11 - t00)
    print('Max timing: {:0.3f}ms, mean timing: {:0.3f}ms, std timing: {:0.3f}ms'.format(max(times) * 1000,
                                                                                        np.mean(times) * 1000,
                                                                                        np.std(times) * 1000))
    print('Total images: {}'.format(len(images)))

    for img in images:
        t00 = time.time()
        results = defect_detector.segment_image.model.detect([img])
        t11 = time.time()
        times.append(t11 - t00)

    print('Max timing: {:0.3f}ms, mean timing: {:0.3f}ms, std timing: {:0.3f}ms'.format(max(times) * 1000,
                                                                                        np.mean(times) * 1000,
                                                                                        np.std(times) * 1000))
    print('Total images: {}'.format(len(images)))
