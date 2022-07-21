from backend import defects
from backend import defect_base
import cv2


if __name__ == '__main__':
    images = [cv2.imread('test_car\\IMG_20220622_162225.jpg'),
              cv2.imread('test_car\\IMG_20220622_162243.jpg'),
              cv2.imread('test_car\\IMG_20220622_162218.jpg'),
              cv2.imread('test_car\\IMG_20220622_162253.jpg'),
              cv2.imread('test_car\\IMG_20220622_162313.jpg'),
              cv2.imread('test0.jpg')]
    air_craft = defects.AirCraftDefectsList('#1', 'самолет')
    for img in images:
        air_craft.add(img)

    db = defect_base.DefectsBase()
    db.add(air_craft)
    db.report('самолет', '#1')
