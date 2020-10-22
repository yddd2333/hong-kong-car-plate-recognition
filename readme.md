## 香港车牌识别
主要步骤为：检测车牌->车牌校正->车牌分类(用于区分单双行车牌)->识别

检测模型基于YOLOV4，校正模型使用WPODNET，分类模型为mobilenet，识别模型为去除RNN部分的CRNN
### 运行方法
直接运行test_img.py

### 运行环境
tensorflow-gpu=1.9.0，keras=2.2.2，matplotlib=3.0.3，opencv，easydict，zhon，pillow,cython

### 文件内容：
crnn用于字符识别，darknet用于车牌检测，pytorch_direct用于车牌校正

主要函数位于plate_recognition.py

车牌检测模型文件：
链接：https://pan.baidu.com/s/1XguRFMPF-DnNmrS6V6qgoQ 
提取码：zf29 
