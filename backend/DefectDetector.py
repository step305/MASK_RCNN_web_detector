import pixellib
from pixellib.instance import custom_segmentation
import os
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np

IMAGE_SIZE = (64, 64)


class DefectDetector:
    def __init__(self, model_file):
        self.model = os.path.abspath(model_file)
        self.segment_image = custom_segmentation()
        self.segment_image.inferConfig(num_classes=1, class_names=["background", "defect"])
        self.segment_image.load_model(self.model)
        self.class_model = load_model('classification_model.h5')
        self.labels = ['Type_1', 'Type_2', 'Type_3', 'Type_4', 'Type_5']

    def detect(self, img):
        new_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        results = self.segment_image.model.detect([new_img])
        result = results[0]
        coords = []
        for class_id, score, roi in zip(result['class_ids'], result['scores'], result['rois']):
            y1, x1, y2, x2 = roi
            coords.append((x1, y1, x2, y2))

        types = []
        for coord in coords:
            class_type = 'Unknown'
            x1, y1, x2, y2 = coord
            image_class = cv2.resize(img[y1:y2, x1:x2], IMAGE_SIZE)
            image_class = img_to_array(image_class)
            image_arr = np.expand_dims(image_class, axis=0)
            probs = self.class_model.predict(image_arr)[0]
            proba = max(probs)
            label = self.labels[np.argmax(probs)]
            if proba > 0.4:
                class_type = label
            types.append(class_type)

        for coord, score, class_type in zip(coords, result['scores'], types):
            x1, y1, x2, y2 = coord
            cv2.rectangle(img,
                          (x1, y1),
                          (x2, y2),
                          color=(0, 0, 255),
                          thickness=2)
            cv2.putText(img,
                        '{:.2f}% {:}'.format(score * 100.0, class_type),
                        (x1, y1 - 10),
                        fontFace=cv2.FONT_ITALIC,
                        fontScale=0.4,
                        thickness=1,
                        color=(255, 0, 0))

        report = {'scores': result['scores'],
                  'coords': coords, 'types': types}
        return img, report
