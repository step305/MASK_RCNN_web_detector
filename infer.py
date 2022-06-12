from backend import DefectDetector
import cv2

defect_detector = DefectDetector.DefectDetector('mask_rcnn_models\\mask_rcnn_model.resnet101.h5')

if __name__ == '__main__':
    img = cv2.imread('test0.jpg')
    img_re, result = defect_detector.detect(cv2.resize(img, (600, 600)))
    print(result)
    cv2.imwrite('test_formatted.jpg', img)


