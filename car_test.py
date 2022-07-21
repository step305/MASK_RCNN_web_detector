from backend import DefectDetector
import cv2
import time
import os

defect_detector = DefectDetector.DefectDetector('mask_rcnn_models\\mask_rcnn_model.resnet101.h5')

if __name__ == '__main__':
    test_images_path = 'test_car'
    images = [os.path.join(test_images_path, f) for f in os.listdir(test_images_path)
              if (os.path.isfile(os.path.join(test_images_path, f)) and '.jpg' in f.lower())]

    for image_path in images:
        img = cv2.imread(image_path)
        img_re, result = defect_detector.detect(cv2.resize(img, (600, 600)))
        t1 = time.time()
        print(result)

        # timing = []
        # for _ in range(100):
        #    t0 = time.time()
        #    img_re, result = defect_detector.detect(cv2.resize(img, (600, 600)))
        #    t1 = time.time()
        #    timing.append(t1 - t0)

        # print(timing)
        cv2.imwrite(image_path + '_detected.jpg', img_re)

