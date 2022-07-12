from backend import DefectDetector
import cv2
import time

defect_detector = DefectDetector.DefectDetector('mask_rcnn_models\\mask_rcnn_model.resnet101.h5')

if __name__ == '__main__':
    img = cv2.imread('test0.jpg')
    t0 = time.time()
    img_re, result = defect_detector.detect(cv2.resize(img, (600, 600)))
    t1 = time.time()
    print(result)

    print(t1 - t0)

    timing = []
    for _ in range(100):
        t0 = time.time()
        img_re, result = defect_detector.detect(cv2.resize(img, (600, 600)))
        t1 = time.time()
        timing.append(t1 - t0)

    print(timing)
    cv2.imwrite('test_formatted.jpg', img)


