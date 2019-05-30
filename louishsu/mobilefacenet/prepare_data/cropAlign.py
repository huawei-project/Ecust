import cv2
import numpy as np
from cp2tform import cp2tform, warpImage, warpCoordinate

# ALIGNED = [30.2946, 51.6963,    # xx1, yy1
#             65.5318, 51.5014,   # xx2, yy2
#             48.0252, 71.7366,   # xx3, yy3
#             33.5493, 92.3655,   # xx4, yy4
#             62.7299, 92.2041]   # xx5, yy5

# ALIGNED = [110., 105.,
#             140., 105.,
#             125., 125.,
#             110., 145.,
#             140., 145.]

ALIGNED = [ 1./3, 1./3,
            2./3, 1./3,
            1./2, 1./2,
            1./3, 2./3,
            2./3, 2./3]

def drawCoordinate(im, coord):
    """
    Params:
        im:  {ndarray(H, W, 3)}
        coord: {ndarray(n, 2)}
    Returns:
        im:  {ndarray(H, W, 3)}
    """
    coord = coord.astype('int')
    for i in range(coord.shape[0]):
        cv2.circle(im, tuple(coord[i]), 1, (255, 255, 255), 3)
    return im

def imageAlignCrop(im, bbox, landmark, dsize=(112, 96), return_unaligned=False):
    """
    Params:
        im:         {ndarray(H, W, 3)}
        bbox:       {ndarray(4/5)}
        landmark:   {ndarray(10)}
        dsize:      {tuple/list(H, W)}
        return_unaligned:   {bool}
    Notes:
        - 考虑实际应用，先人脸后再进行矫正
        - 产生图像会有黑边
    """
    ## 先截取人脸
    x1, y1, x2, y2 = bbox[:4].astype('int')
    w = x2 - x1 + 1; h = y2 - y1 + 1
    ro = h / w; rd = dsize[0] / dsize[1]
    if ro < rd: ## h为较短边
        w = int(h / rd)
        x1 = (x1 + x2 - w) // 2
        x2 = x1 + w
    else:       ## w为较短边
        h = int(w * rd)
        y1 = (y1 + y2 - h) // 2
        y2 = y1 + h

    cropedImage = im[y1: y2, x1: x2]
    box = bbox[:4].reshape(-1, 2) - bbox[:2]    ## [[x1, y1], [x2, y2]]
    src = landmark.reshape(-1, 2) - bbox[:2]    ## [[x1, y1], [x2, y2], ..., [x5, y5]]
    ## 以图像尺寸计算对齐后的关键点位置
    dst = (np.array([w, h]*5) * np.array(ALIGNED)).reshape(-1, 2)

    ## 变换矩阵
    M = cp2tform(src, dst)
    ## 用矩阵变换图像
    warpedImage = warpImage(cropedImage, M)
    ## 用矩阵变换坐标
    # x1, y1, x2, y2 = box.reshape(-1)
    # coord = np.r_[np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]]), src]
    # warpedCoord = warpCoordinate(coord, M)
    cv2.imshow("warpedImage", warpedImage); cv2.imshow("cropedImage", cropedImage); cv2.waitKey(0)

    ## 缩放
    warpedImage = cv2.resize(warpedImage, dsize[::-1])
    if return_unaligned:
        cropedImage = cv2.resize(cropedImage, dsize[::-1])
        return warpedImage, cropedImage
    else:
        return warpedImage
