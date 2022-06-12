import time

from backend import DefectDetector
import cv2
import os

defect_detector = DefectDetector.DefectDetector('mask_rcnn_models\\mask_rcnn_model.resnet101.h5')


if __name__ == '__main__':
    image_dir = os.path.abspath(os.path.join(os.path.curdir, 'dataset\\test'))
    files_list = os.listdir(image_dir)
    images_list = []
    for file_name in files_list:
        if '.jpg' in file_name.lower():
            images_list.append(os.path.abspath(os.path.join(os.path.curdir, 'dataset\\test', file_name)))

    image_dir = os.path.abspath(os.path.join(os.path.curdir, 'dataset\\test_0'))
    files_list = os.listdir(image_dir)
    for file_name in files_list:
        if '.jpg' in file_name.lower():
            images_list.append(os.path.abspath(os.path.join(os.path.curdir, 'dataset\\test_0', file_name)))

    image_dir = os.path.abspath(os.path.join(os.path.curdir, 'dataset\\test_1'))
    files_list = os.listdir(image_dir)
    for file_name in files_list:
        if '.jpg' in file_name.lower():
            images_list.append(os.path.abspath(os.path.join(os.path.curdir, 'dataset\\test_1', file_name)))

    image_dir = os.path.abspath(os.path.join(os.path.curdir, 'dataset\\train'))
    files_list = os.listdir(image_dir)
    for file_name in files_list:
        if '.jpg' in file_name.lower():
            images_list.append(os.path.abspath(os.path.join(os.path.curdir, 'dataset\\train', file_name)))

    image_counter = 0
    file_counter = 0
    N = len(images_list)
    t0 = time.time()
    for image_path in images_list:
        file_counter += 1
        if image_counter % 20 == 0:
            print('Processed {:d} images of {:d}'.format(file_counter, N))
        img = cv2.imread(image_path)
        img_re, result = defect_detector.detect(cv2.resize(img, (600, 600)))
        coords = result['coords']
        if coords:
            for coord in coords:
                x1, y1, x2, y2 = coord
                defect_image_path = os.path.abspath(os.path.join(os.path.curdir,
                                                                 'dataset\\classification\\images',
                                                                 '{:d}.jpg'.format(image_counter)))
                image_counter += 1
                defect_image = img_re[y1:y2, x1:x2]
                defect_image = cv2.resize(defect_image, (64, 64))
                cv2.imwrite(defect_image_path, defect_image)
    te = time.time()
    print('Done in {:.1f}sec with {:.1f} images per second'.format(te - t0, file_counter / (te - t0)))
