# coding:utf-8
import os
import sys
import cv2
import numpy as np

from detectors import Detector, FcnDetector, MtcnnDetector
from models import P_Net, R_Net, O_Net


def init_detector():
    thresh = [0.9, 0.6, 0.7]
    min_face_size = 24
    stride = 2
    slide_window = False
    shuffle = False
    detectors = [None, None, None]
    prefix = ['./models/mtcnn/modelfile/PNet/PNet', './models/mtcnn/modelfile/RNet/RNet', './models/mtcnn/modelfile/ONet/ONet']
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

def detect_image(mtcnn_detector, image):
    """
    Params:
        image: {ndarray(H, W, 3)}
    """
    boxes_c, landmarks = mtcnn_detector.detect(image)

    for i in range(boxes_c.shape[0]):
        bbox = boxes_c[i, :4]
        score = boxes_c[i, 4]

        corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        cv2.rectangle(image, (corpbbox[0], corpbbox[1]),
                        (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
        cv2.putText(image, '{:.3f}'.format(score), (corpbbox[0], corpbbox[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
    for i in range(landmarks.shape[0]):
        for j in range(int(len(landmarks[i])/2)):
            cv2.circle(image, (int(landmarks[i][2*j]),int(int(landmarks[i][2*j+1]))), 2, (0,0,255))

    cv2.imshow("", image)
    cv2.waitKey(0)

if __name__ == "__main__":
    detector = init_detector()
    # image = cv2.imread("/home/louishsu/Pictures/joker.jpg", cv2.IMREAD_COLOR)
    image = cv2.imread("/media/louishsu/备份/ECUST2019/DATA1/9/RGB/normal/RGB_4_W1_1.JPG", cv2.IMREAD_COLOR)
    image = cv2.resize(image, (800, 600))
    detect_image(detector, image)