from flask import Flask, request, Response
import numpy as np
import cv2
import codecs
import pickle
import jsonpickle
from backend import image_utils
import time

import multiprocessing as mp
import sys


def detector_process(inQueue: mp.Queue, outQueue: mp.Queue, stop):
    from backend import DefectDetector
    defect_detector = DefectDetector.DefectDetector('mask_rcnn_models\\mask_rcnn_model.resnet101.h5')
    print('started')
    sys.stdout.flush()
    while not stop.is_set():
        if inQueue.empty():
            time.sleep(0.2)
        else:
            res = inQueue.get()
            print('got')
            sys.stdout.flush()
            result_img, result = defect_detector.detect(res)
            outQueue.put((result_img, result))


app = Flask(__name__)
stopEvent = mp.Event()
stopEvent.clear()
image_Queue = mp.Queue()
repoort_Queue = mp.Queue()
detector_proc = mp.Process(target=detector_process, args=(image_Queue, repoort_Queue, stopEvent))


@app.route('/api/find-defect', methods=['POST'])
def recognize_request():
    global defect_detector
    report = {'defects_coords': [], 'defects_types': [], 'scores': [], 'image': None}
    req = request
    original_img = image_utils.image_decode(req.data)
    w, h, _ = original_img.shape
    small_img = image_utils.image_prepare(original_img)
    ws, hs, _ = small_img.shape
    scale_x = w / ws
    scale_y = h / hs
    image_Queue.put(small_img)
    result_img, result = repoort_Queue.get(timeout=20)
    if len(result['scores']) > 0:
        coords = []
        for pt in result['coords']:
            x1, y1, x2, y2 = pt
            x1 = int(scale_x * x1)
            x2 = int(scale_x * x2)
            y1 = int(scale_y * y1)
            y2 = int(scale_y * y2)
            coords.append((x1, y1, x2, y2))
        report['scores'] = result['scores']
        report['defects_coords'] = coords
        report['image'] = cv2.imencode('.jpg', result_img)

    msg = codecs.encode(pickle.dumps(report), "base64").decode()
    response = {'message': msg}
    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")


if __name__ == '__main__':
    detector_proc.start()
    app.run()
    stopEvent.set()
    detector_proc.join()
