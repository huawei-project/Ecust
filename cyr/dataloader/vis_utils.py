import numpy as np
import cv2


def draw_bbbox(image, score, bbox, landmarks=None):
    """
    Params:
        image:      {ndarray(H, W, 3)}
        score:      {ndarray(n_faces)}
        bbox:       {ndarray(n_faces, 4)}
        landmarks:  {ndarray(n_faces, 10)}
    """
    n_faces = bbox.shape[0]
    for i in range(n_faces):
        corpbbox = [int(bbox[i, 0]), int(bbox[i, 1]),
                    int(bbox[i, 2]), int(bbox[i, 3])]
        cv2.rectangle(image, (corpbbox[0], corpbbox[1]),
                      (corpbbox[2], corpbbox[3]),
                      (255, 0, 0), 10)
        cv2.putText(image, '{:.3f}'.format(score[i]),
                    (corpbbox[0], corpbbox[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 5,
                    (0, 255, 0), 10)
        if landmarks is not None:
            for j in range(int(len(landmarks[i]) / 2)):
                cv2.circle(image, (int(landmarks[i][2 * j]), int(landmarks[i][2 * j + 1])),
                           16, (255, 0, 0), -1)
    return image


def show_result(image, score=None, bbox=None, landmarks=None, winname="", waitkey=0):
    """
    Params:
        image:      {ndarray(H, W, 3)}
        score:      {ndarray(n_faces)}
        bbox:       {ndarray(n_faces, 4)}
        landmarks:  {ndarray(n_faces, 10)}
        winname:    {str}
    """
    if bbox is not None:
        n_faces = bbox.shape[0]
        for i in range(n_faces):
            corpbbox = [int(bbox[i, 0]), int(bbox[i, 1]),
                        int(bbox[i, 2]), int(bbox[i, 3])]
            cv2.rectangle(image, (corpbbox[0], corpbbox[1]),
                          (corpbbox[2], corpbbox[3]),
                          (255, 0, 0), 1)
            cv2.putText(image, '{:.3f}'.format(score[i]),
                        (corpbbox[0], corpbbox[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
            if landmarks is not None:
                for j in range(int(len(landmarks[i]) / 2)):
                    cv2.circle(image, (int(
                        landmarks[i][2 * j]), int(int(landmarks[i][2 * j + 1]))), 2, (0, 0, 255))
    cv2.imshow(winname, image)
    cv2.waitKey(waitkey)
    cv2.destroyWindow(winname)
