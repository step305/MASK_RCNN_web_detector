import cv2
import requests
import pickle
import jsonpickle
import codecs


if __name__ == '__main__':
    image = cv2.imread('test.jpg')
    content_type = 'image/jpeg'
    headers = {'content-type': content_type}

    try:
        _, img_jpeg = cv2.imencode('.jpg', image)
        img_pickle = pickle.dumps(img_jpeg)
        msg = codecs.encode(img_pickle, 'base64').decode()
        response = requests.post('http://127.0.0.1:5000/api/find-defect', data=msg, headers=headers)
        response_unpickled = jsonpickle.decode(response.text)['message'].encode()
        msg = codecs.decode(response_unpickled, "base64")
        report = pickle.loads(msg)
        print(report['scores'])
        print(report['defects_coords'])
        print(report['defects_types'])
        _, img_result_jpeg = report['image']
        print(type(img_result_jpeg))
        image_report = cv2.imdecode(img_result_jpeg, cv2.IMREAD_COLOR)
        cv2.imwrite('test_service_image.jpg', image_report)
    except Exception as e:
        print(e)
        print('Failed to send frame to server.')
