import os
import sys
import cv2
import numpy as np

def load_rgb(imgpath, dsize=None):
    """
    Params:
        imgpath: {str}
        dsize:   {tuple(W, H)}
    Returns:
        imgs:   {ndarray(H, W, 3)}
    """
    assert os.path.exists(imgpath), "rgb file does not exist!"
    img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    if dsize is not None:
        img = cv2.resize(img, dsize)
    return img

def load_multi(imgdir, dsize=None):
    """
    Params:
        imgdir: {str}
        dsize:   {tuple(W, H)}
    Returns:
        imgs:   {ndarray(H, W, C)}
    """
    assert os.path.exists(imgdir), "multi directory does not exist!"
    imgfiles = os.listdir(imgdir)
    
    # 根据波长排序
    wavelength = []
    for imgfile in imgfiles:
        wavelength += [int(imgfile.split('.')[0].split('_')[-1])]
    imgfiles = np.array(imgfiles); wavelength = np.array(wavelength)
    imgfiles = imgfiles[np.argsort(wavelength)]
    imgfiles = list(imgfiles)
    
    # 载入图像
    c = len(imgfiles)
    if dsize is None:
        img = cv2.imread(os.path.join(imgdir, imgfiles[0]), cv2.IMREAD_GRAYSCALE)
        (h, w) = img.shape
    else:
        (w, h) = dsize
    imgs = np.zeros(shape=(h, w, c), dtype='uint8')
    for i in range(c):
        img = cv2.imread(os.path.join(imgdir, imgfiles[i]), cv2.IMREAD_GRAYSCALE)
        imgs[:, :, i] = img if (dsize is None) else cv2.resize(img, dsize)
    return imgs 

def getBbox(imgpath):
    """
    Params:
        imgpath:    {str}
    Returns:
        score:      {float}
        bbox:       {ndarray(4)}
        landmark:   {ndarray(10)}
    Notes:
        - 可返回字典
    """
    idx = imgpath.find("DATA") + 5
    txtfile = os.path.join(imgpath[: idx], "detect.txt")
    imgname = imgpath[idx: ].split('.')[0]
    with open(txtfile, 'r') as f:
        dict_save = eval(f.read())
    score, bbox, landmark = dict_save[imgname]
    assert (score is not None), "detect error! "

    score = np.array(score)
    bbox = np.array(bbox, dtype='int')
    landmark = np.array(landmark, dtype='int')
    return score, bbox, landmark

def show_result(image, score, bbox, landmarks, winname="", waitkey=0):
    """
    Params:
        image:      {ndarray(H, W, 3)}
        score:      {ndarray(n_faces)}
        bbox:       {ndarray(n_faces, 4)}
        landmarks:  {ndarray(n_faces, 10)}
        winname:    {str}
    """
    n_faces = bbox.shape[0]
    for i in range(n_faces):
        corpbbox = [int(bbox[i, 0]), int(bbox[i, 1]), int(bbox[i, 2]), int(bbox[i, 3])]
        cv2.rectangle(image, (corpbbox[0], corpbbox[1]),
                        (corpbbox[2], corpbbox[3]), 
                        (255, 0, 0), 1)
        cv2.putText(image, '{:.3f}'.format(score[i]), 
                        (corpbbox[0], corpbbox[1] - 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
        for j in range(int(len(landmarks[i])/2)):
            cv2.circle(image, (int(landmarks[i][2*j]),int(int(landmarks[i][2*j+1]))), 2, (0,0,255))
    cv2.imshow(winname, image)
    cv2.waitKey(waitkey)
    cv2.destroyWindow(winname)

if __name__ == "__main__":
    # Multi
    imgpath = "../../ECUST2019/DATA1/9/Multi/normal/Multi_4_W1_1"
    image = load_multi(imgpath)
    score, bbox, landmark = getBbox(imgpath)
    show_result(np.array(image[:, :, :3]), 
                            score.reshape([-1]), 
                            bbox.reshape([1, -1]), 
                            landmark.reshape([1, -1]),
                            winname=imgpath, waitkey=0)
    # RGB
    imgpath = "../../ECUST2019/DATA1/9/RGB/normal/RGB_4_W1_1.JPG"
    image = load_rgb(imgpath)
    score, bbox, landmark = getBbox(imgpath)
    show_result(image, 
                            score.reshape([-1]), 
                            bbox.reshape([1, -1]), 
                            landmark.reshape([1, -1]),
                            winname=imgpath, waitkey=0)
