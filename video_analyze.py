import cv2
import sys
import os
import shutil
import subprocess


if __name__ == '__main__':
    print('Parameters: airplane name, aircraft serial, path to video divided by space')
    aircraft_name = sys.argv[1]
    aircraft_serial = sys.argv[2]
    vid = cv2.VideoCapture(sys.argv[3])
    if os.path.exists('temp_video_frames'):
        shutil.rmtree('temp_video_frames', ignore_errors=True)
        os.mkdir('temp_video_frames')
    cnt = 0
    cnt_skip = 1
    while True:
        ret, frame = vid.read()
        if ret:
            if cnt % cnt_skip == 0:
                cv2.imwrite('temp_video_frames\\{}.jpg'.format(cnt), frame)
            cnt += 1
        else:
            break
    vid.release()
    if sys.platform.startswith('win'):
        cmd = ['python', 'image_analyze.py', '"{}"'.format(aircraft_name), '"{}"'.format(aircraft_serial),
               'temp_video_frames']
    else:
        cmd = ['python3', 'image_analyze.py', '"{}"'.format(aircraft_name), '"{}"'.format(aircraft_serial),
               'temp_video_frames']
    im_analyzer = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    for line in im_analyzer.stdout:
        print(line)
    im_analyzer.wait()
