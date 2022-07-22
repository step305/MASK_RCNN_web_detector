from backend import defects
from backend import defect_base
import cv2
import sys
import os

if __name__ == '__main__':
    print('Parameters: airplane name, aircraft serial, list of images paths divided by space')
    aircraft_name = sys.argv[1]
    aircraft_serial = sys.argv[2]
    images_paths = []
    for im_p in sys.argv[3:]:
        if os.path.isfile(im_p) and os.path.exists(im_p):
            images_paths.append(im_p)
        elif os.path.isdir(im_p) and os.path.exists(im_p):
            images = [os.path.join(im_p, f) for f in os.listdir(im_p)
                      if (os.path.isfile(os.path.join(im_p, f)) and '.jpg' in f.lower())]
            for img in images:
                images_paths.append(img)

    images = [cv2.resize(cv2.imread(fn), (1920, 1080)) for fn in images_paths]
    air_craft = defects.AirCraftDefectsList(aircraft_serial, aircraft_name)
    for img in images:
        air_craft.add(img)

    db = defect_base.DefectsBase()
    db.add(air_craft)
    db.report(aircraft_name, aircraft_serial)
