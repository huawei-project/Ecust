import numpy as np
from cp2tform import cp2tform, warpImage, warpCoordinate

# ALIGNED = [30.2946, 51.6963,    # xx1, yy1
#             65.5318, 51.5014,   # xx2, yy2
#             48.0252, 71.7366,   # xx3, yy3
#             33.5493, 92.3655,   # xx4, yy4
#             62.7299, 92.2041]   # xx5, yy5

ALIGNED = [110., 105.,
            140., 105.,
            125., 125.,
            110., 145.,
            140., 145.]

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

def imageAlignCrop(im, bbox, landmark, a=15, b=20, dsize=(112, 96)):
    """
    Params:
        im:         {ndarray(H, W, 3)}
        bbox:       {ndarray(4/5)}
        landmark:   {ndarray(10)}
        dsize:      {tuple/list(H, W)}
    """
    ## 先截取人脸
    x1, y1, x2, y2 = bbox[:4].astype('int')
    im = im[y1: y2, x1: x2]
    box = bbox[:4].reshape(-1, 2) - bbox[:2]    ## [[x1, y1], [x2, y2]]
    src = landmark.reshape(-1, 2) - bbox[:2]    ## [[x1, y1], [x2, y2], ..., [x5, y5]]

    ## 以图像中心为中心，计算变换后的坐标
    centroid = np.mean(box, axis=0).reshape(1, -1)
    dst = np.repeat(centroid, 5, axis=0) + np.array(
                [[-a, -b], [ a, -b], [ 0,  0], [-a,  b], [ a,  b]])

    ## 变换矩阵
    M = cp2tform(src, dst)
    ## 用矩阵变换图像
    warpedImage = warpImage(im, M)
    ## 用矩阵变换坐标
    x1, y1, x2, y2 = box.reshape(-1)
    coord = np.r_[np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]]), src]
    warpedCoord = warpCoordinate(coord, M)

    ### 按尺寸和比例，裁剪人脸
    h, w = dsize
    # TODO: ratio
    x1, x2, y1, y2 = (np.repeat(centroid.reshape(-1), 2) + ratio*np.array(
        [-w/2, w/2, -h/2, h/2])).astype('int')
    croped = warpedImage[y1: y2, x1: x2]

    ## 显示
    cv2.imshow("im", drawCoordinate(im, coord))
    # cv2.imshow("im", drawCoordinate(drawCoordinate(im, coord), warpedCoord))
    cv2.imshow("warped", drawCoordinate(warpedImage, warpedCoord))
    cv2.imshow("croped", croped)
    cv2.waitKey(0)



    ...

if __name__ == "__main__":
    
    import cv2
    # '1982597/308.jpg 49.52332992240136 24.06662541083861 202.37791485006906 233.98998851107498 0.9853229522705078 103.38385577776721 107.99730841372376 152.95072083710804 94.07187004712003 129.71222954219695 134.1165372861388 125.4369617285715 165.95677781823343 162.23256272370804 154.00411073077635\n'
    im = cv2.imread("/home/louishsu/Desktop/308.jpg")
    box = np.array([49.52332992240136, 24.06662541083861, 
                    202.37791485006906, 233.98998851107498, 
                    0.9853229522705078])
    landmark = np.array([103.38385577776721, 107.99730841372376, 
                            152.95072083710804, 94.07187004712003, 
                            129.71222954219695, 134.1165372861388, 
                            125.4369617285715, 165.95677781823343, 
                            162.23256272370804, 154.00411073077635])
    # offset = box[:, :2] # (x1, y1) 1x2
    # landmark -= offset

    # aligned = np.array(ALIGNED).reshape(-1, 2) # 5x2

    # cv2.remap(im, )

    imageAlignCrop(im, np.array(box[:-1]), landmark, dsize=(112, 96))

    ...