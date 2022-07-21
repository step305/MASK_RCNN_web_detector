import datetime
from backend import DefectDetector
import cv2

defect_detector = DefectDetector.DefectDetector('mask_rcnn_models\\mask_rcnn_model.resnet101.h5')


class DefectFrame:
    def __init__(self, image=None, boxes=(), scores=(), types=()):
        self.image = image
        self.boxes = boxes
        self.scores = scores
        self.types = types

    def detect(self):
        found = False
        if self.image.size > 0:
            img_re, result = defect_detector.detect(cv2.resize(self.image, (600, 600)))
            if len(result['coords']) > 0:
                self.boxes = result['coords']
                self.scores = result['scores']
                self.types = result['types']
                self.image = img_re
                found = True
            else:
                self.image = None
        return found


class AirCraftDefectsList:
    def __init__(self, serial_num='0000', name='plane'):
        self.defects = []
        self.serial_num = serial_num
        self.name = name
        self.date = datetime.datetime.now()

    def add(self, image):
        defect = DefectFrame(image)
        if defect.detect():
            self.defects.append(defect)
