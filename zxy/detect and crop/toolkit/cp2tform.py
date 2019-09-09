import cv2
import numpy as np

# see [Chen-Zhihui/cvn - Github](https://github.com/Chen-Zhihui/cvn/tree/093672ed4a890ce6bd240c51a068bca8a3597bde/src/Cvutil/include/Cvn/Cvutil)

def _norm2(m):
    """
    Params:
        m: {ndarray(n, 2)}
    Returns:
        ret:{float} largest singular value
    Notes:
        求解矩阵2范数，即最大奇异值
    """
    u, s, vh = np.linalg.svd(m)
    ret = np.max(s)
    return ret

def _stitch(xy):
    """
    Params:
        xy:     {ndarray( n, 2)}
    Returns:
        ret:    {ndarray(2n, 4)}
    Notes：
        return
            x  y 1 0
            y -x 0 1
    """
    x, y = np.hsplit(xy, indices_or_sections=2)
    ones = np.ones_like(x)
    zeros = np.zeros_like(x)

    ret = np.r_[np.c_[x,  y, ones, zeros], 
                np.c_[y, -x, zeros, ones]]

    return ret

def tformfwd(M, uv):
    """
    Params:
        M:  {ndarray(2, 3)}
        uv: {ndarray(n, 2)}
    Returns:
        ret: {ndarray(n, 2)}
    Notes:
        ret = [uv, 1] * M^T
    """
    ones = np.ones(shape=(uv.shape[0], 1))  # n x 2
    UV = np.c_[uv, ones]                    # n x 3
    ret = UV.dot(M.T)                       # n x 2
    return ret                              # n x 2

def findNonreflectiveSimilarity(uv, xy):
    """
    Params:
        uv: {ndarray(n, 2)}
        xy: {ndarray(n, 2)}
    Returns:
        M:  {ndarray(2, 3)}
    Notes:
        - Xr = U   ===>  r = (X^T X + \lambda I)^{-1} X^T U
        - r = [r1 r2 r3 r4]^T
        - M
            [r1 -r2 0
             r2  r1 0
             r3  r4 1]^{-1}[:, :2].T
    """
    X = _stitch(xy)
    U = uv.T.reshape(-1)
    r = np.linalg.pinv(X).dot(U)
    M = np.array(
        [[r[0], -r[1], 0],
         [r[1],  r[0], 0],
         [r[2],  r[3], 1]]
    )
    M = np.linalg.inv(M)
    return M[:, :2].T

def findReflectiveSimilarity(uv, xy):
    """
    Params:
        uv: {ndarray(n, 2)}
        xy: {ndarray(n, 2)}
    Returns:
        M:  {ndarray(2, 3)}
    """
    xyR = xy.copy(); xyR[:, 0] *= -1

    M1 = findNonreflectiveSimilarity(uv, xy)
    M2 = findNonreflectiveSimilarity(uv, xyR)
    
    M2[:, 0] *= -1

    xy1 = tformfwd(M1, uv)
    xy2 = tformfwd(M2, uv)

    norm1 = _norm2(xy1 - xy)
    norm2 = _norm2(xy2 - xy)

    return M1 if norm1 < norm2 else M2

def cp2tform(src, dst, mode = 'similarity'):
    """
    Params:
        src: {ndarray(n, 2)}
        dst: {ndarray(n, 2)}
        mode:{str} `similarity` or `noreflective`
    Returns:
        M:  {ndarray(2, 3)}
    """
    assert src.shape == dst.shape

    M = None
    if mode == 'similarity':
        M = findReflectiveSimilarity(src, dst)
    elif mode == 'noreflective':
        M = findNonreflectiveSimilarity(src, dst)
    else:
        print("Unsupported mode!")
    
    return M

def warpCoordinate(coord, M):
    """
    Params:
        coord: {ndarray(n, 2)}
        M:   {ndarray(2, 3)}
    """
    coord = np.c_[coord, np.ones(coord.shape[0])]
    coord = M.dot(coord.T).T
    return coord

def warpImage(im, M):
    return cv2.warpAffine(im, M, im.shape[:2][::-1])

def drawCoordinate(im, coord):
    """
    Params:
        im:  {ndarray(H, W, 3)}
        coord: {ndarray(n, 2)}
    Returns:
        im:  {ndarray(H, W, 3)}
    """
    for i in range(coord.shape[0]):
        cv2.circle(im, tuple(coord[i]), 1, (255, 255, 255), 3)
    return im

if __name__ == "__main__":
    src1 = [
        [30.48, 40.64] ,
        [66.65, 37.26] ,
        [55.68, 60.99] ,
        [37.64, 78.67] ,
        [67.00, 76.07],
    ]
    src2 = [
        [33.05, 36.37 ],
        [73.20, 36.19 ],
        [61.15, 58.91 ],
        [38.22, 79.78] ,
        [70.07, 79.63],
    ]

    dst = [
        [30.29, 51.70 ],
        [65.53, 51.50 ],
        [48.03, 71.74 ],
        [33.55, 92.37 ],
        [62.73, 92.20 ],
    ]
    cp2tform(np.array(src1), np.array(dst))
    cp2tform(np.array(src2), np.array(dst))