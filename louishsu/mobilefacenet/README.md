# Reproduce Mobilefacenet

## Prepare data
### 检测
1. 下载CASIA数据集`CASIA_WebFace`，解压到`data/CASIA-WebFace`
2. 标签文件`data/CASIA_labels.txt`
3. 运行`prepare_data/detect.py`，检测结果保存在`data/CASIA_detect.txt`

### 剪裁与对齐
1. 裁剪

2. 对齐
    注意，不只是利用关键点进行对齐`cv2.warpAffine`，应使用重映射`cv.remap`函数，
    > 对应C语言函数为`convertMaps`