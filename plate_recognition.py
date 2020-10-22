# coding:utf-8

import sys, os, cv2, json, time
import numpy as np
import collections
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import torch
from torchvision import transforms


sys.path.append(os.path.join(os.getcwd(), 'pytorch_darknet'))
from detect import detect_from_loaded_img
from tool.darknet2pytorch import *
from tool.torch_utils import *

sys.path.append(os.path.join(os.getcwd(), 'crnn'))
import keys_HK_chepai as keys
import densenet as crnn_net

sys.path.append(os.path.join(os.getcwd(), 'direct'))
from keras.models import model_from_json
from label import Label
from utils import getWH, nms
from projection_utils import getRectPts, find_T_matrix

sys.path.append(os.path.join(os.getcwd(), 'classify'))
from classify import classify_from_cv2_imgs


class DLabel(Label):
    def __init__(self, cl, pts, prob):
        self.pts = pts
        tl = np.amin(pts, 1)
        br = np.amax(pts, 1)
        Label.__init__(self, cl, tl, br, prob)


class PlateRecog():
    def __init__(self):
        self.input_height = 80
        self.input_width = 320
        self.darknet_cfg = os.path.join(os.getcwd(), 'pytorch_darknet/hk_plate.cfg')
        self.darknet_weight = os.path.join(os.getcwd(),
                                           'pytorch_darknet/hk_plate.weights')
        # self.darknet_data = os.path.join(os.getcwd(), 'darknet', 'models', 'plate.data')
        self.crnn_weight = os.path.join(os.getcwd(), 'crnn', 'models',
                                        'CNN_pre--35--0.335.hdf5')
        self.direct_weight = os.path.join(os.getcwd(), 'direct', 'wpod-net.h5')
        self.direct_json = os.path.join(os.getcwd(), 'direct', 'wpod-net.json')
        self.classify_weight = os.path.join(os.getcwd(), 'classify/epoch_15.t7')

        # --------------- 车牌检测-------------------------
        # 加载yolo模型，车牌检测
        self.darknet_model = Darknet(self.darknet_cfg)
        self.darknet_model.load_weights(self.darknet_weight)
        self.darknet_model.cuda()

        # --------------ocr crnn model字符识别---------------------
        # 加载字符识别模型
        self.crnn_dict = keys.alphabet[:]
        self.crnn_dict_num = len(self.crnn_dict) + 1
        # input = Input(shape=(self.input_height, self.input_width, 1), name='the_input')
        # y_pred = crnn_net.dense_cnn(input, self.crnn_dict_num)
        self.crnn_model = crnn_net.get_Model_noLSTM(False, shape=(
            self.input_height, self.input_width, 1))
        self.crnn_model.load_weights(self.crnn_weight)

        # --------------- 双行车牌识别-------------------------
        # 加载yolo模型，车牌检测
        self.classify_model = torch.load(self.classify_weight)['net'].cuda()
        self.classify_model.eval()
        self.classify_transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1 / 3.2, 1 / 3.2, 1 / 3.2))
        ])

        # 车牌校正模型,扭曲平面检测网络（WPOD NET）
        josn_file = open(self.direct_json, 'r')
        model_json = josn_file.read()
        self.direct_model = model_from_json(model_json, custom_objects={})
        self.direct_graph = tf.get_default_graph()
        with self.direct_graph.as_default():
            self.direct_model.load_weights(self.direct_weight)

    def detect(self, names):
        img = cv2.imread(names)
        if (img is None):
            # print('img is None')
            return [], 0, 0, 0
        height, width, _ = img.shape
        plate = detect_from_loaded_img(self.darknet_model, img)
        return plate, img, height, width

    # 车牌校正所需函数
    def reconstruct(self, Iorig, I, Y, out_size, threshold=.9):
        net_stride = 2 ** 4
        side = ((208. + 40.) / 2.) / net_stride
        Probs = Y[..., 0]
        Affines = Y[..., 2:]
        xx, yy = np.where(Probs > threshold)
        WH = getWH(I.shape)
        MN = WH / net_stride
        vxx = vyy = 0.5
        base = lambda vx, vy: np.matrix(
            [[-vx, -vy, 1.], [vx, -vy, 1.], [vx, vy, 1.], [-vx, vy, 1.]]).T
        labels = []
        for i in range(len(xx)):
            y, x = xx[i], yy[i]
            affine = Affines[y, x]
            prob = Probs[y, x]
            mn = np.array([float(x) + .5, float(y) + .5])
            A = np.reshape(affine, (2, 3))
            A[0, 0] = max(A[0, 0], 0.)
            A[1, 1] = max(A[1, 1], 0.)
            pts = np.array(A * base(vxx, vyy))
            pts_MN_center_mn = pts * side
            pts_MN = pts_MN_center_mn + mn.reshape((2, 1))
            pts_prop = pts_MN / MN.reshape((2, 1))
            labels.append(DLabel(0, pts_prop, prob))

        final_labels = nms(labels, .1)
        TLps = []
        if len(final_labels):
            final_labels.sort(key=lambda x: x.prob(), reverse=True)
            for i, label in enumerate(final_labels):
                t_ptsh = getRectPts(0, 0, out_size[0], out_size[1])
                ptsh = np.concatenate(
                    (label.pts * getWH(Iorig.shape).reshape((2, 1)), np.ones((1, 4))))
                H = find_T_matrix(ptsh, t_ptsh)
                Ilp = cv2.warpPerspective(Iorig, H, out_size, borderValue=.0)
                TLps.append(Ilp)
        return final_labels, TLps

    def correct_plate(self, I, max_dim, net_step, out_size, threshold):
        min_dim_img = min(I.shape[:2])
        factor = float(max_dim) / min_dim_img
        w, h = (np.array(I.shape[1::-1], dtype=float) * factor).astype(int).tolist()
        w += (w % net_step != 0) * (net_step - w % net_step)
        h += (h % net_step != 0) * (net_step - h % net_step)
        Iresized = cv2.resize(I, (w, h))
        T = Iresized.copy()
        T = T.reshape((1, T.shape[0], T.shape[1], T.shape[2]))
        start = time.time()
        Yr = self.direct_model.predict(T)
        Yr = np.squeeze(Yr)
        elapsed = time.time() - start
        L, TLps = self.reconstruct(I, Iresized, Yr, out_size, threshold)
        return L, TLps, elapsed

    def im2single(self, I):
        assert (I.dtype == 'uint8')
        return I.astype('float32') / 255.

    def decode(self, pred):
        char_list = []
        conf_list = []
        pred_text = pred.argmax(axis=2)[0]
        for i in range(len(pred_text)):
            if pred_text[i] != self.crnn_dict_num - 1 and (
                    (not (i > 0 and pred_text[i] == pred_text[i - 1])) or (not
                                                                           i > 1 and
                                                                           pred_text[i] ==
                                                                           pred_text[
                                                                               i - 2])):
                conf_list.append(pred[0, i, pred_text[i]])
                char_list.append(self.crnn_dict[pred_text[i]])
        return u''.join(char_list), conf_list

    def recog_line(self, plate_vec):
        if (len(plate_vec) == 0):
            return ''
        text = ''
        for plate in plate_vec:
            plate = plate.astype('uint8')
            plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            height, width = plate_gray.shape
            scale = height * 1.0 / self.input_height
            width = int(width / scale)
            if (width > self.input_height):
                img = cv2.resize(plate_gray, (self.input_width, self.input_height))
            else:
                img = cv2.resize(plate_gray, (width, self.input_height))
                img = cv2.copyMakeBorder(img, 0, 0, 0, self.input_width - width,
                                         cv2.BORDER_CONSTANT, value=0)
            img = img.astype(np.float32) / 255.0 - 0.5
            self.input_height = 80
            X = img.reshape([1, self.input_height, self.input_width, 1])
            y_pred = self.crnn_model.predict(X)
            y_pred = y_pred[:, :, :]
            text_temp, conf = self.decode(y_pred)
            flag = self.judge_plate(text_temp, conf)
            # if (flag == True):
            text = text + text_temp + ';'
        return text

    # 车牌校正主函数，输入检测到的车牌和原图信息，返回矫正后的车牌
    def correct_xywh(self, plate, img, height, width):
        if (len(plate) == 0):
            return [], []
        plate_vec = []
        plate_coordinate = []
        for i in range(len(plate)):
            bbox = plate[i][2]
            left = max(0, int((bbox[0] - bbox[2])))
            right = min(width, int((bbox[0] + bbox[2])))
            top = max(0, int((bbox[1] - bbox[3])))
            bot = min(height, int((bbox[1] + bbox[3])))

            plate_img = img[top:bot, left:right]
            boader_distant = max(plate_img.shape) // 2
            Ivehicle = cv2.copyMakeBorder(plate_img, boader_distant, boader_distant,
                                          boader_distant, boader_distant,
                                          cv2.BORDER_CONSTANT, 0)
            ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
            side = int(ratio * 288.0)
            bound_dim = min(side + (side % (2 ** 4)), 608)

            _, LlpImgs, _ = self.correct_plate(self.im2single(Ivehicle), bound_dim,
                                               2 ** 4, (320, 160), 0.5)

            left1 = max(0, int((bbox[0] - bbox[2] / 2.0 - bbox[2] / 10.0)))
            top1 = max(0, int((bbox[1] - bbox[3] / 2.0 - bbox[3] / 10.0)))
            right1 = min(width, int((bbox[0] + bbox[2] / 2.0 + bbox[2] / 10.0)))
            bot1 = min(height, int((bbox[1] + bbox[3] / 2.0 + bbox[3] / 10.0)))
            plate_coordinate.append([left1, top1, right1 - left1, bot1 - top1])

            if (len(LlpImgs) == 0):
                plate_img2 = img[top1:bot1, left1:right1]
                plate_vec.append(plate_img2)
                continue
            Ilp = LlpImgs[0]
            Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
            Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
            Ilp = Ilp * 255
            plate_vec.append(Ilp)
        return plate_vec, plate_coordinate

    def plate_recognition_with_xywh(self, names):
        # 车牌检测
        plate, img, height, width = self.detect(names)
        # 车牌矫正
        img_plate, plate_coordinate = self.correct_xywh(plate, img, height, width)

        # Classify
        clses = classify_from_cv2_imgs(net=self.classify_model, imgs=img_plate,
                                       transform=self.classify_transform)

        # test
        # img_plate.append(img_plate[0])
        # clses.append([1,100])
        # plate_coordinate.append(plate_coordinate[0])

        # 车牌识别
        plate_coordinate2, plate_coordinate2_single, plate_coordinate2_double = [], [], []
        img_plate_single, img_plate_double = [], []
        for idx, cls in enumerate(clses):
            if cls[0] == 0:
                img_plate_single.append(img_plate[idx])
                plate_coordinate2_single.append(plate_coordinate[idx])
            elif cls[0] == 1:
                img_plate_double.append(
                    img_plate[idx][:int(img_plate[idx].shape[0] / 1.9)])
                img_plate_double.append(
                    img_plate[idx][int((0.9 * img_plate[idx].shape[0]) / 1.9):])
                plate_coordinate2_double.append(plate_coordinate[idx])
        print('Number of single-line plate is {} and double-line plate is {}'.format(
            len(img_plate_single), len(img_plate_double)))
        textS = self.recog_line(img_plate_single)
        textD = ';'.join([' '.join(
            self.recog_line(img_plate_double).strip().split(';')[2 * i:2 * i + 2]) for i
                             in range(len(img_plate_double) // 2)] + [''])

        text_list = textS.split(';')[:-1] + textD.split(';')
        plate_coordinate2 = plate_coordinate2_single + plate_coordinate2_double
        # text_list, plate_coordinate2 = self.recog_line_with_xywh(img_plate,
        #                                                          plate_coordinate)

        # 输出状态信息
        detect_plate_num = len(plate)
        recognize_plate_num = len(text_list)
        if detect_plate_num == 0:
            status = -2
        elif recognize_plate_num == 0:
            status = -1
        else:
            status = 0

        # 返回结果
        result = collections.OrderedDict()
        result['detect_plate_num'] = detect_plate_num
        result['recognize_plate_num'] = recognize_plate_num
        result['status'] = status
        output = []
        if text_list:
            for i in range(len(plate_coordinate2)):
                output_ = collections.OrderedDict()
                output_['words_result'] = text_list[i]
                location = collections.OrderedDict()
                location['left'] = (plate_coordinate2[i])[0]
                location['top'] = (plate_coordinate2[i])[1]
                location['width'] = (plate_coordinate2[i])[2]
                location['height'] = (plate_coordinate2[i])[3]
                output_['location'] = location
                output.append(output_)
        result['plate_info'] = output
        # result = json.dumps(result,ensure_ascii=False,indent=4)
        return result


if __name__ == '__main__':
    a = PlateRecog()

    names = 'Baidu_0123.jpeg'
    text = a.plate_recognition_with_xywh(names)
    print('plate_recognition_with_xywh', text)
