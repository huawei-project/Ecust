import cv2
import gdal
import numpy as np
from scipy import misc

def readBsq(bsqfile):
    data = gdal.Open(bsqfile)
    ...

def parse(raw):
    """
    Params:
        raw: {ndarray(H, W)}
    Returns:
        img: {ndarray(25, h, w)}
    """
    h, w = raw.shape
    h = (h // 5) * 5
    w = (w // 5) * 5
    raw = raw[:h, :w]

    img = []
    for c in range(25):
        i, j = c // 5, c % 5
        img += [raw[i::5, j::5][np.newaxis]]
    img = np.concatenate(img, axis=0)

    return img

# raw = misc.imread('../images/image.tif')
# bsq = readBsq('SSM5x5-NIR_18880754_20190630094951.bsq')
raw = cv2.imread('../images/SSM5x5-NIR_18880754_20190630094951.jpg', cv2.IMREAD_GRAYSCALE)

# raw = raw / 255 * 65535; raw = raw.astype(np.uint16)
clahe = cv2.createCLAHE(tileGridSize=(8,8)); raw = clahe.apply(raw)

image = parse(raw)

cv2.imshow("raw", raw)
for c in range(25):
    cv2.imshow("%d" % c, image[c])
cv2.waitKey(0)