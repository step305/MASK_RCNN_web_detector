from backend import defects
from backend import defect_base
import cv2


if __name__ == '__main__':
    images = [cv2.imread('test.JPG'), cv2.imread('test0.jpg'), cv2.imread('test2.JPG'), cv2.imread('test3.png'),
              cv2.imread('test4.jpg')]
    air_craft = defects.AirCraftDefectsList('#1', 'самолет')
    for img in images:
        air_craft.add(img)

    db = defect_base.DefectsBase()
    db.add(air_craft)
    db.report('самолет', '#1')
