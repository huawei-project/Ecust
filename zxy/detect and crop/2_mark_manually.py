import cv2
import os
import numpy as np

ptnames = ["左上角", "右下角", "左眼", "右眼", "鼻尖", "左嘴角", "右嘴角"]
WINNAME = "image unknown {}"

def square(bbox):
    """
    Params:
        bbox: tuple(int)
    Returns:
        bbox: tuple(int)
    """
    (x1, y1, x2, y2) = bbox
    h = y2 - y1; w = x2 - x1
    max_side = np.maximum(h, w)
    x1 = x1 + w * 0.5 - max_side * 0.5
    y1 = y1 + h * 0.5 - max_side * 0.5
    x2 = x1 + max_side - 1
    y2 = y1 + max_side - 1
    return (int(x1), int(y1), int(x2), int(y2))

def mouse_callback(event, x, y, flags, param):
    """

    Notes:
        - 双击左右键点击切换点
        - 单击保存点位置
    """
    if event == cv2.EVENT_LBUTTONDBLCLK:
        param['dict'][ptnames[param["ptidx"]]] = [x, y]
        print(param['dict'])

        param['ptidx'] += 1
        param['ptidx'] %= 7

        cv2.circle(param["image"], (x, y), 2, (255, 0, 0))
        cv2.imshow(param["winname"], param["image"])
        cv2.waitKey(10)

def mark_rgb():

    with open(rgbtxtpath, 'r') as f:
        dict = eval(f.read())

    marked = []

    for key, value in dict.items():

        if key.split('/')[2] != 'rgb': continue
        if value == (None,None,None):
            ## 记录图片
            marked += [key + '\n']
            winname = WINNAME.format(len(marked))

            ## 读取图片
            imgpath = '{}{}.jpg'.format(rootdir, key)
            assert os.path.exists(imgpath), "rgb file does not exist"
            image = cv2.imread(imgpath, cv2.IMREAD_COLOR)

            ## 初始化变量
            ptidx = 0
            dict_ptname_locate = {ptname: [-1, -1] for ptname in ptnames}
            param = {"ptidx": ptidx, "dict": dict_ptname_locate, "image": image, "winname": winname}

            ## 创建窗口和回调函数
            cv2.namedWindow(winname)
            cv2.setMouseCallback(winname, mouse_callback, param)
            cv2.imshow(winname, image)
            cv2.waitKey(0)
            cv2.destroyWindow(winname)

            ## 计算在原图中的坐标
            for k, v in dict_ptname_locate.items():
                x, y = dict_ptname_locate[k]
                dict_ptname_locate[k][0] = int(x)
                dict_ptname_locate[k][1] = int(y)

            ## 保存结果
            score = 1.0
            bbox = dict_ptname_locate['左上角'] + dict_ptname_locate['右下角']
            # bbox = square(bbox) 这个不需要？
            landmark = dict_ptname_locate['左眼'] + dict_ptname_locate['右眼'] + \
                        dict_ptname_locate['鼻尖'] + \
                    dict_ptname_locate['左嘴角'] + dict_ptname_locate['右嘴角']
            dict[key] = (score, bbox, landmark)

    with open(rgbtxtpath, 'w') as f:
        f.write(str(dict))

    with open(manualrgb, 'w') as f:
        f.writelines(marked)

def mark_multi():
    
    with open(multitxtpath, 'r') as f:
        dict = eval(f.read())

    marked = []

    for key, value in dict.items():
        
        if key.split('/')[2] != 'multi': continue

        if value == (None,None,None):
            ## 记录图片
            marked += [key + '\n']
            winname = WINNAME.format(len(marked))

            ## 读取图片
            imgdir = '{}/{}'.format(rootdir, key)
            bmp = '{}/{}'.format(imgdir, os.listdir(imgdir)[0])
            image = cv2.imread(bmp, cv2.IMREAD_GRAYSCALE)

            ## 初始化变量
            ptidx = 0
            dict_ptname_locate = {ptname: [-1, -1] for ptname in ptnames}
            param = {"ptidx": ptidx, "dict": dict_ptname_locate, "image": image, "winname": winname}

            ## 创建窗口和回调函数
            cv2.namedWindow(winname)
            cv2.setMouseCallback(winname, mouse_callback, param)
            cv2.imshow(winname, image)
            cv2.waitKey(0)
            cv2.destroyWindow(winname)

            ## 计算在原图中的坐标
            for k, v in dict_ptname_locate.items():
                x, y = dict_ptname_locate[k]
                dict_ptname_locate[k][0] = int(x)
                dict_ptname_locate[k][1] = int(y)

            ## 保存结果
            score = 1.0
            bbox = dict_ptname_locate['左上角'] + dict_ptname_locate['右下角']
            # bbox = square(bbox)？？？？
            landmark = dict_ptname_locate['左眼'] + dict_ptname_locate['右眼'] + \
                        dict_ptname_locate['鼻尖'] + \
                    dict_ptname_locate['左嘴角'] + dict_ptname_locate['右嘴角']
            dict[key] = (score, bbox, landmark)
        
        else:
            continue

    with open(multitxtpath, 'w') as f:
        f.write(str(dict))

    with open(manualmulti, 'a') as f:
        f.writelines(marked)


if __name__ == "__main__":
    rootdir = 'E:/Desktop/Outdoor20190810'

    rgbtxtpath = rootdir + '/rgbdetect.txt'
    multitxtpath = rootdir + "/multidetect.txt"
    manualrgb = rootdir + "/rgbmanual.txt"
    manualmulti = rootdir + "/multimanual.txt"
    mark_rgb()
    mark_multi()