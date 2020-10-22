from plate_recognition import *
import cv2
import time
os.environ['CUDA_VISIBLE_DEVICES']='0'

PR = PlateRecog()
names = 'Bing_0704.jpeg'
img = cv2.imread(names)
text = PR.plate_recognition_with_xywh(names)
for i in range(len(text['plate_info'])):
    plate = text['plate_info'][i]['words_result']
    loc = text['plate_info'][i]['location']
    l, t, w, h = loc['left'], loc['top'], loc['width'], loc['height']
    draw_0 = cv2.rectangle(img, (l, t), (l + w, t + h), (0, 255, 0), 2)
    cv2.putText(img, plate, (l, t-10), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)
cv2.imwrite("res"+names, draw_0)
print('plate_recognition_with_xywh: ', text)
