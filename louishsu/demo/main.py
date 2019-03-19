import os
import cv2
import numpy as np
from PIL import Image

import mtcnn  as D
import deepid as V

def init_detect():
    pnet, rnet, onet = D.init_detecter()
    detect = lambda image: D.detect_faces(Image.fromarray(image), pnet, rnet, onet, min_face_size=100)
    return detect

def init_verify():
    net = V.init_verifier('deepid_vgg11_bn_3chs_64feats')
    proto = V.init_proto()
    verify = lambda face: V.verify(face, net, proto, thresh=0.0)
    return verify

def main():
    detect = init_detect()
    verify = init_verify()
    video_capture = cv2.VideoCapture(0)
    cv2.namedWindow("")

    while True:

        ret, frame = video_capture.read()
        # frame = np.array(frame)

        if ret:
            bboxes, landmarks = detect(frame)

            if bboxes.shape[0] != 0: 
                bboxes_square = D.convert_to_square(bboxes).astype('int')
                bboxes = bboxes.astype('int')

                for i in range(bboxes.shape[0]):

                    x1, y1, x2, y2, _ = bboxes_square[i]
                    face = cv2.resize(frame[y1: y2, x1: x2], (96, 96))
                    name = verify(face)
                    
                    x1, y1, x2, y2, _ = bboxes[i]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255))

                    # landmark = landmarks[i]
                    # for j in range(5):
                    #     cv2.circle(frame, tuple(landmark[j: j+1]), 2, (255, 0, 0))

                    cv2.putText(frame, name, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 255, 0))

            cv2.imshow("", frame)
            cv2.waitKey(50)
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()