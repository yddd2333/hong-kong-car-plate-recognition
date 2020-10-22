# -*- coding: utf-8 -*-
'''
@Time          : 20/04/25 15:49
@Author        : huguanghao
@File          : demo.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :
'''

# import sys
# import time
# from PIL import Image, ImageDraw
# from models.tiny_yolo import TinyYoloNet
import os, sys
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import argparse
import json
import cv2

print(sys.path)
# from tqdm import tqdm

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""hyper parameters"""
use_cuda = True


def cal_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def detect_cv2(cfgfile, weightfile, imgfolder):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/plate.names'
    class_names = load_class_names(namesfile)
    cnt = 0
    for file in tqdm(os.listdir(imgfolder)):
        if file[-4:] == 'json' or file[-3:] == 'txt':
            continue
        imgfile = os.path.join(imgfolder, file)
        img = cv2.imread(imgfile)
        sized = cv2.resize(img, (m.width, m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
        with open(os.path.join(imgfolder, file.split('.')[0] + '.json'), 'r') as fr:
            js = json.load(fr)
        gt_boxes = []
        h = js['imageHeight']
        w = js['imageWidth']
        for i in range(len(js['shapes'])):
            x0 = min(js['shapes'][i]['points'][0][0], js['shapes'][i]['points'][1][0])
            y0 = min(js['shapes'][i]['points'][0][1], js['shapes'][i]['points'][1][1])
            x1 = max(js['shapes'][i]['points'][0][0], js['shapes'][i]['points'][1][0])
            y1 = max(js['shapes'][i]['points'][0][1], js['shapes'][i]['points'][1][1])
            gt_boxes.append(([x0 / w, y0 / h, x1 / w, y1 / h], js['shapes'][i]['label'],
                             js['shapes'][i]['group_id']))

        for i in range(2):
            start = time.time()
            boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
            finish = time.time()
            # if i == 1:
            # print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

        true_boxes = []
        for i in range(len(boxes[0])):
            for j in range(len(gt_boxes)):
                if cal_iou(gt_boxes[j][0], boxes[0][i][:4]) > 0.9:
                    true_boxes.append(
                        (boxes[0][i][:4], gt_boxes[j][0], gt_boxes[j][1], gt_boxes[j][2]))
                    break
        for box in true_boxes:
            if box[-1] == 3:
                continue
            det = img[int(h * box[0][1]):int(h * box[0][3]),
                  int(w * box[0][0]):int(w * box[0][2])]
            gt = img[int(h * box[1][1]):int(h * box[1][3]),
                 int(w * box[1][0]):int(w * box[1][2])]
            cnt += 1
            try:
                cv2.imwrite(
                    os.path.join(
                        '/home/yang.deng/4t/datasets/HK_plate_dataset/predicted_box_iou90',
                        box[2] + '_' + str(cnt) + '_det.' + file.split('.')[-1]), det)
                cv2.imwrite(
                    os.path.join(
                        '/home/yang.deng/4t/datasets/HK_plate_dataset/predicted_box_iou90',
                        box[2] + '_' + str(cnt) + '_gt.' + file.split('.')[-1]), gt)
            except:
                continue


def to_darknet_format(boxes, h, w):
    if not boxes[0]:
        return []
    res = []
    for box in boxes[0]:
        r0 = (box[0] + box[2]) / 2 * w
        r1 = (box[1] + box[3]) / 2 * h
        r2 = (box[2] - box[0]) * w
        r3 = (box[3] - box[1]) * h
        res.append(('plate', box[4], (r0, r1, r2, r3)))
    return res


def detect_cv2_img(cfgfile, weightfile, img_file):
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    # print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    img = cv2.imread(img_file)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
    boxes_darknet_format = to_darknet_format(boxes, img.shape[0], img.shape[1])
    print(boxes[0])
    print(boxes_darknet_format)
    return boxes[0]
    # [[0.1712202, 0.67228127, 0.24706267, 0.7257899, 0.9979943, 0.9979943, 0], [0.8395154, 0.6372847, 0.8891141, 0.67440283, 0.99708986, 0.99708986, 0]]
    # [('plate', 0.7031520009040833, (66.77825927734375, 32.39432144165039, 102.37455749511719, 47.708003997802734))]


def detect_from_loaded_img(model, img):
    sized = cv2.resize(img, (model.width, model.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    boxes = do_detect(model, sized, 0.4, 0.6, use_cuda)
    boxes_darknet_format = to_darknet_format(boxes, img.shape[0], img.shape[1])
    # print(boxes_darknet_format)
    return boxes_darknet_format


def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str, default='./hk_plate.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str, default='./hk_plate.weights',
                        help='path of trained model.', dest='weightfile')
    parser.add_argument('-imgfolder', type=str,
                        default='/home/yang.deng/4t/datasets/HK_plate_dataset/ALL',
                        help='path of your image file.', dest='imgfolder')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    # detect_cv2_img(args.cfgfile, args.weightfile, '../3.jpeg')
    m = Darknet(args.cfgfile)
    m.print_network()
    m.load_weights(args.weightfile)
    m.cuda()
    img = cv2.imread('../3.jpeg')
    res = detect_from_loaded_img(m, img)
    print(res)
