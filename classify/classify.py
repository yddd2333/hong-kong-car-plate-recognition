import os, cv2
import numpy as np
import shutil
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def classify_from_file(
        img_path='/home/yang.deng/4t/datasets/HK_plate_classify/2/00 U_1194.jpeg'):
    val_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (1 / 3.2, 1 / 3.2, 1 / 3.2))
    ])

    checkpoint = torch.load(
        os.path.join('/home/yang.deng/4t/datasets/HK_plate_classify/checkpoints_nni',
                     'jOSnu',
                     'epoch_15.t7'))
    net = checkpoint['net']
    best_f1 = checkpoint['f1']
    start_epoch = checkpoint['epoch'] + 1

    net = net.cuda()
    net.eval()
    img = val_transform(Image.open(img_path).convert('RGB'))
    img = img.unsqueeze(0)
    img = torch.autograd.Variable(img.cuda())
    output = net(img)
    output = torch.nn.functional.softmax(output, 1)
    _, predicted = torch.max(output.data, 1)
    print(
        'Predict class is %1d with probability %.5f%%' % (int(predicted), float(_) * 100))


def classify_from_img(net, img, val_transform):
    img = val_transform(img)
    img = img.unsqueeze(0)
    img = torch.autograd.Variable(img.cuda())
    output = net(img)
    output = torch.nn.functional.softmax(output, 1)
    _, predicted = torch.max(output.data, 1)
    print(
        'Predict class is %1d with probability %.5f%%' % (
            int(predicted), float(_) * 100))
    return int(predicted)


def classify_from_img(net, img, val_transform):
    img = val_transform(img)
    img = img.unsqueeze(0)
    img = torch.autograd.Variable(img.cuda())
    output = net(img)
    output = torch.nn.functional.softmax(output, 1)
    _, predicted = torch.max(output.data, 1)
    print(
        'Predict class is %1d with probability %.5f%%' % (
            int(predicted), float(_) * 100))
    return int(predicted)


def classify_from_cv2_imgs(net, imgs: list, transform) -> list:
    res = []
    for idx, cv2_img in enumerate(imgs):
        img = transform(
            Image.fromarray(np.array(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB),
                                     dtype='uint8'))).unsqueeze(0)
        img = torch.autograd.Variable(img.cuda())
        output = net(img)
        output = torch.nn.functional.softmax(output, 1)
        _, predicted = torch.max(output.data, 1)
        # print(
        #     'Predict class is %1d with probability %.5f%%' % (
        #         int(predicted), float(_) * 100))
        # if predicted==1:
        #     imgs[idx] = np.concatenate((imgs[idx][:imgs[idx].shape[0]//2], imgs[idx][imgs[idx].shape[0]//2:]), axis=-2)
        res.append([int(predicted), float(_) * 100])
    return res


# def classify_folder(folder='/home/yang.deng/4t/datasets/HK_plate_dataset/predicted_box_iou90',
#                     new_folder='/home/yang.deng/4t/datasets/HK_plate_dataset/classify_folder'):
#     if not os.path.exists(new_folder):
#         os.makedirs(new_folder)
#     val_transform = transforms.Compose([
#         transforms.Resize((112, 112)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (1 / 3.2, 1 / 3.2, 1 / 3.2))
#     ])
#
#     checkpoint = torch.load(
#         os.path.join('/home/yang.deng/4t/datasets/HK_plate_classify/checkpoints_nni', 'jOSnu',
#                      'epoch_15.t7'))
#     net = checkpoint['net']
#     best_f1 = checkpoint['f1']
#     start_epoch = checkpoint['epoch'] + 1
#
#     net = net.cuda()
#     net.eval()
#     cnt = 0
#     prob = []
#     for file in tqdm(os.listdir(folder)):
#         img_path = os.path.join(folder, file)
#         img = val_transform(Image.open(img_path).convert('RGB'))
#         img = img.unsqueeze(0)
#         img = torch.autograd.Variable(img.cuda())
#         output = net(img)
#         output = torch.nn.functional.softmax(output, 1)
#         _, predicted = torch.max(output.data, 1)
#         if float(_) * 100 > 95:
#             # shutil.copy(img_path, os.path.join(new_folder, str(int(predicted)), file))
#         if int(predicted) != (len(file.strip().split(' ')) - 1):
#             print(file)
#             print('Predict class is %1d with probability %.5f%%' % (int(predicted), float(_) * 100))
#             prob.append(float(_) * 100)
#             cnt += 1
#     print(cnt)

def test_folder(
        folder='/home/yang.deng/4t/datasets/plate/dataset/crnn_trainset/plate_realandfake',
        new_folder='/home/yang.deng/4t/datasets/plate/dataset/crnn_trainset/deleted'):
    val_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (1 / 3.2, 1 / 3.2, 1 / 3.2))
    ])

    checkpoint = torch.load(
        os.path.join('/home/yang.deng/4t/datasets/HK_plate_classify/checkpoints_nni',
                     'jOSnu',
                     'epoch_15.t7'))
    net = checkpoint['net']
    best_f1 = checkpoint['f1']
    start_epoch = checkpoint['epoch'] + 1

    net = net.cuda()
    net.eval()
    cnt = 0
    prob = []
    for file in tqdm(os.listdir(folder)):
        img_path = os.path.join(folder, file)
        img = val_transform(Image.open(img_path).convert('RGB'))
        img = img.unsqueeze(0)
        img = torch.autograd.Variable(img.cuda())
        output = net(img)
        output = torch.nn.functional.softmax(output, 1)
        _, predicted = torch.max(output.data, 1)
        # print(predicted)
        if output.data[0][0] < 0.9:
            print(file)
            print(output)
            print('\n')
            cnt += 1
            shutil.move(os.path.join(folder, file), os.path.join(new_folder, file))
    print(cnt)
    print(cnt / len(os.listdir(folder)))


if __name__ == '__main__':
    # classify_from_file(
    #     img_path='/home/yang.deng/4t/datasets/HK_plate_dataset/predicted_box_iou90/0UTATIME_683_gt.jpeg')
    # classify_folder(folder='/home/yang.deng/4t/datasets/HK_plate_dataset/predicted_box_iou90',
    #                 new_folder='/home/yang.deng/4t/datasets/HK_plate_dataset/classify_folder')
    # test_folder()
    net = torch.load('epoch_15.t7')['net'].cuda()
    net.eval()
    val_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (1 / 3.2, 1 / 3.2, 1 / 3.2))
    ])
    classify_from_img(net=net, img=Image.open(
        '/home/yang.deng/4t/datasets/HK_plate_classify/2/00 U_1194.jpeg').convert('RGB'),
                      val_transform=val_transform)
