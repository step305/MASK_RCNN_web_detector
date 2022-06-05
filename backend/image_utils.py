import cv2
import pickle
import codecs
import numpy as np

camMatrix = np.array([[800, 0, 320],
                      [0, 800, 240],
                      [0, 0, 1]], dtype=np.float32)
camDistCoeffs = np.array([0, 0, 0, 0, 0], dtype=np.float32)
RCNN_FRAME_SIZE = (600, 600)


def image_decode(base64_string):
    # use to send:
    # _, img_jpeg = cv2.imencode('.jpg', img)
    # img_pickle = pickle.dumps(img_jpeg)
    # msg = codecs.encode(img_pickle, 'base64').decode()
    img_array = pickle.loads(codecs.decode(base64_string, 'base64'))
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


def image_prepare(img):
    # undistorted_frame = cv2.undistort(img, camMatrix, camDistCoeffs)
    return cv2.resize(img, RCNN_FRAME_SIZE)
