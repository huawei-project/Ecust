import os
import sys
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from models.mtcnn.detectors import Detector, FcnDetector, MtcnnDetector
from models.mtcnn.models import P_Net, R_Net, O_Net
from load_data import load_rgb, load_multi, show_result
from utiles import getTime, getVol, getWavelen
from processbar import ProcessBar
from noise import gaussianNoise, spNoise, signal_to_noise_ratio

ORIGINSIZE = (1648, 1236)
DATAPATH   = "/home/louishsu/Work/Workspace/ECUST2019_rename"
# DATAPATH   = "/home/louishsu/Work/Workspace/ECUST2019"
DIRNAME    = "DATA{volidx}/{subidx}/{datatype}/{illumtype}/{datatype}_{posidx}_W1_{glass}"

def init_detector():
    thresh = [0.9, 0.6, 0.7]
    min_face_size = 12
    stride = 2
    slide_window = False
    detectors = [None, None, None]
    prefix = ['./models/mtcnn/modelfile/PNet/PNet', 
                './models/mtcnn/modelfile/RNet/RNet', 
                './models/mtcnn/modelfile/ONet/ONet']
    epoch = [18, 14, 16]
    model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
    PNet = FcnDetector(P_Net, model_path[0]);       detectors[0] = PNet
    RNet = Detector(R_Net, 24, 1, model_path[1]);   detectors[1] = RNet
    ONet = Detector(O_Net, 48, 1, model_path[2]);   detectors[2] = ONet
    mtcnn_detector = MtcnnDetector(detectors=detectors,
                                    min_face_size=min_face_size,
                                    stride=stride, 
                                    threshold=thresh, 
                                    slide_window=slide_window)
    return mtcnn_detector

def _square(bbox):
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

def _detect_c3(detector, img_c3):
    """
    Params:
        detector:   {mtcnn_detector}
        img_c3:     {ndarray(H, W, 3)}
    Returns:
        score:      {float}
        bbox:       {ndarray(4)}
        landmarks:  {ndarray(10)}
    Notes:
        - 只检测最中心的人脸
        # - 只检测最大的人脸
    # """
    for i in range(3):
        img_c3[:, :, i] = cv2.equalizeHist(img_c3[:, :, i])     # 直方图均衡化
    boxes_c, landmarks = detector.detect(img_c3)

    n = boxes_c.shape[0]
    if n==0: return None, None, None
    scores = boxes_c[:, -1]
    bboxes = boxes_c[:, :-1]                                    # shape(n_faces, 4), [x1, y1, x2, y2]

    # 面积
    # areas  = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    # idx = np.argmax(areas)
    
    # 最中心
    imgct = np.array(img_c3.shape[:2]) / 2
    xx = 0.5 * (bboxes[:, 2] + bboxes[:, 0]).reshape((-1, 1))
    yy = 0.5 * (bboxes[:, 3] + bboxes[:, 1]).reshape((-1, 1))
    centers = np.c_[yy, xx]
    vect = centers - imgct
    dists = np.linalg.norm(vect, axis=1)
    idx = np.argmin(dists)
    
    score = scores[idx]
    bbox = bboxes[idx].astype('int')
    landmark = landmarks[idx].astype('int')
    return score, bbox, landmark

def detect_multi(detector, multi):
    """
    Params:
        detector:   {mtcnn_detector}
        multi:      {ndarray(H, W, C)}
    """
    c = multi.shape[-1]
    scores = []
    bboxes = []
    landmarks = []
    for i in range(c):
        img = multi[:, :, i]
        img = np.stack([img, img, img], axis=2)
        score, bbox, landmark = _detect_c3(detector, img)
        if score is None: continue
        scores.append(score)
        bboxes.append(list(bbox))
        landmarks.append(list(landmark))
        # show_result(img, score.reshape([-1]), bbox.reshape([1, -1]), landmark.reshape([1, -1]))
    if len(scores)==0: return None, None, None
    
    score     = np.mean(scores,    axis=0)
    bbox      = np.mean(bboxes,    axis=0, dtype='int')
    landmark  = np.mean(landmarks, axis=0, dtype='int')
    # show_result(img, score.reshape([-1]), bbox.reshape([1, -1]), landmark.reshape([1, -1]))
    return score, bbox, landmark

def listFiles():
    """
    Note:
        1. 对每张图片进行检测， 即多光谱每个通道结果均保存
    """
    filelist = []
    for subidx in range(1, 71):
        volidx = getVol(subidx)
        for illumtype in ['normal', 'illum1', 'illum2']:
            for posidx in range(10):
                for glass in range(10):
                    for datatype in ['Multi', 'RGB']:
                        dirname = DIRNAME.format(volidx=volidx, subidx=subidx, 
                                datatype=datatype, illumtype=illumtype, posidx=posidx, glass=glass)
                        
                        if datatype == 'Multi':
                            dirname_ = os.path.join(DATAPATH, dirname)
                            if not os.path.exists(dirname_):
                                continue
                            filelist += list(map(lambda x: os.path.join(dirname, x), os.listdir(dirname_)))
                        elif datatype == 'RGB':
                            filename = "{}.JPG".format(dirname)
                            filename_ = os.path.join(DATAPATH, filename)
                            if not os.path.exists(filename_):
                                continue
                            filelist += [filename]
    return filelist

def detect_size(detector, filelist, dsize):
    """
    Params:
        detector:   {MtcnnDetector}
        filelist:   {list[str]}
        dsize:      {tuple(w: int, h: int)}
    """
    annodir = './anno'
    if not os.path.exists(annodir):
        os.mkdir(annodir)
    annofile = os.path.join(annodir, '{}x{}.txt'.format(dsize[0], dsize[1]))
    
    bar = ProcessBar(total_step=len(filelist), title='{}x{} Almost D'.format(dsize[0], dsize[1]))
    f = open(annofile, 'w')
    
    for filename in filelist:
        bar.step()

        ## read image
        image = cv2.imread(os.path.join(DATAPATH, filename), cv2.IMREAD_ANYCOLOR)
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.concatenate([image, image, image], axis=2)
        image = cv2.resize(image, dsize)
        
        ## detect
        score, bbox, landmark = _detect_c3(detector, image)
        score = "" if score is None else str(score)
        bbox = "" if bbox is None else ' '.join(map(str, list(bbox)))
        landmark = "" if landmark is None else ' '.join(map(str, list(landmark)))
        line = "{} {} {} {}\n".format(filename, score, bbox, landmark)
        
        ## save result
        f.write(line)

    f.close()

def detect_noise(detector, filelist, dsize, noise_rate):
    """
    Params:
        detector:   {MtcnnDetector}
        filelist:   {list[str]}
        dsize:      {tuple(w: int, h: int)}
    """

    annodir = './anno'
    if not os.path.exists(annodir):
        os.mkdir(annodir)
    annofile = os.path.join(annodir, '{}x{}_{}.txt'.format(dsize[0], dsize[1], noise_rate))
    filelist = list(filter(lambda x: x.split('/')[2]=='Multi', filelist))

    bar = ProcessBar(total_step=len(filelist), title='{:.2f} Almost D'.format(noise_rate))
    f = open(annofile, 'w')
    snr = []
    
    for filename in filelist:
        bar.step()

        ## read image
        img = cv2.imread(os.path.join(DATAPATH, filename), cv2.IMREAD_ANYCOLOR)
        img = cv2.resize(img, dsize)

        ## add noise
        # image = gaussianNoise(img, 0, 75, noise_rate)
        image = spNoise(img, noise_rate)
        # cv2.imshow("", image); cv2.waitKey(1)
        snr += [signal_to_noise_ratio(img, image)]

        image = image[:, :, np.newaxis]
        image = np.concatenate([image, image, image], axis=2)
        
        ## detect
        score, bbox, landmark = _detect_c3(detector, image)
        score = "" if score is None else str(score)
        bbox = "" if bbox is None else ' '.join(map(str, list(bbox)))
        landmark = "" if landmark is None else ' '.join(map(str, list(landmark)))
        line = "{} {} {} {}\n".format(filename, score, bbox, landmark)
        
        ## save result
        f.write(line)

    f.write("SNR: {:.6f}".format(np.mean(np.array(snr))))
    f.close()

def detect_statistic_size(dsize):
    """
    Params:
        dsize:      {tuple(w: int, h: int)}
    """
    annofile = "./anno/{}x{}.txt".format(dsize[0], dsize[1])

    with open(annofile, 'r') as f:
        anno_all = f.readlines()
    anno_all = list(map(lambda x: x.strip().split(' '), anno_all))

    ## 按图片数统计
    anno_multi_image = list(filter(lambda x: x[0].split('/')[2]=='Multi', anno_all))
    anno_rgb_image   = list(filter(lambda x: x[0].split('/')[2]=='RGB',   anno_all))
    anno_multi_image_detected = list(filter(lambda x: len(x)!=1, anno_multi_image))
    anno_rgb_image_detected   = list(filter(lambda x: len(x)!=1, anno_rgb_image  ))

    n_multi = len(anno_multi_image)
    n_rgb   = len(anno_rgb_image)
    n_multi_detected = len(anno_multi_image_detected)
    n_rgb_detected   = len(anno_rgb_image_detected)
    multi_image_ratio = n_multi_detected / n_multi
    rgb_image_ratio   = n_rgb_detected / n_rgb

    ## 按张量统计
    anno_multi_tensor = list(set(map(lambda x: '/'.join(x[0].split('/')[:-1]), anno_multi_image)))
    anno_multi_tensor_detected = list(set(map(lambda x: '/'.join(x[0].split('/')[:-1]), anno_multi_image_detected)))

    n_multi = len(anno_multi_tensor)
    n_multi_detected = len(anno_multi_tensor_detected)
    multi_tensor_ratio = n_multi_detected / n_multi
    rgb_tensor_ratio = rgb_image_ratio

    return multi_image_ratio, rgb_image_ratio, multi_tensor_ratio, rgb_tensor_ratio

def detect_statistic_noise(dsize, noise_rate):
    annofile = "./anno/{}x{}_{}.txt".format(dsize[0], dsize[1], noise_rate)

    with open(annofile, 'r') as f:
        anno_all = f.readlines()
        print(anno_all[-1])
        anno_all = anno_all[:-1]
    anno_all = list(map(lambda x: x.strip().split(' '), anno_all))

    ## 按图片数统计
    anno_multi_image = list(filter(lambda x: x[0].split('/')[2]=='Multi', anno_all))
    anno_multi_image_detected = list(filter(lambda x: len(x)!=1, anno_multi_image))

    n_multi = len(anno_multi_image)
    n_multi_detected = len(anno_multi_image_detected)
    multi_image_ratio = n_multi_detected / n_multi

    ## 按张量统计
    anno_multi_tensor = list(set(map(lambda x: '/'.join(x[0].split('/')[:-1]), anno_multi_image)))
    anno_multi_tensor_detected = list(set(map(lambda x: '/'.join(x[0].split('/')[:-1]), anno_multi_image_detected)))

    n_multi = len(anno_multi_tensor)
    n_multi_detected = len(anno_multi_tensor_detected)
    multi_tensor_ratio = n_multi_detected / n_multi

    return multi_image_ratio, multi_tensor_ratio

def detect_statistic_spectral_resolution(dsize):
    annofile = "./anno/{}x{}.txt".format(dsize[0], dsize[1])

    with open(annofile, 'r') as f:
        anno_all = f.readlines()
    anno_all = map(lambda x: x.strip().split(' '), anno_all)

    ## 按图片数统计
    anno_multi_image = list(filter(lambda x: x[0].split('/')[2]=='Multi', anno_all))
    anno_multi_image_detected = list(filter(lambda x: len(x)!=1, anno_multi_image))
    anno_multi_image = list(map(lambda x: x[0], anno_multi_image))
    anno_multi_image_detected = list(map(lambda x: x[0], anno_multi_image_detected))

    CHANNEL_SORT = [550 + 20*i for i in range(23)]
    channelsList = [CHANNEL_SORT[::i+1] for i in range(22)]
    dict_saved = dict()
    for channels in channelsList:

        anno_multi_image_sub = list(filter(lambda x: int(x.split('.')[0].split('_')[-1]) in channels, anno_multi_image))
        anno_multi_image_detected_sub = list(filter(lambda x: int(x.split('.')[0].split('_')[-1]) in channels, anno_multi_image_detected))

        ## 按张量统计
        anno_multi_tensor = list(set(map(lambda x: '/'.join(x.split('/')[:-1]), anno_multi_image_sub)))
        anno_multi_tensor_detected = list(set(map(lambda x: '/'.join(x.split('/')[:-1]), anno_multi_image_detected_sub)))

        n_multi = len(anno_multi_tensor)
        n_multi_detected = len(anno_multi_tensor_detected)
        multi_tensor_ratio = n_multi_detected / n_multi
        
        dict_saved[channels[1] - channels[0]] = multi_tensor_ratio
    
    return dict_saved

if __name__ == "__main__":
    # detector = init_detector()
    # filelist = listFiles()
    # detect_size(detector, filelist, ORIGINSIZE)
    # detect_statistic((400, 300))
    
    resolutions = [(40+20*i, 30+15*i) for i in range(9)] + [(400+200*i, 300+150*i) for i in range(6)] + [(1648, 1236)]
    for resolution in resolutions:
        
        res = detect_statistic_spectral_resolution(resolution)

        print(resolution)
        res = sorted(res.items(), key = lambda x: x[0])
        for k, v in res:
            print(v)

        print()