import os
import sys
import cv2
import numpy as np
from toolkit.cp2tform import cp2tform, warpImage, warpCoordinate

# ALIGNED = [ 30.2946, 51.6963,
#             65.5318, 51.5014,
#             48.0252, 71.7366,
#             33.5493, 92.3655,
#             62.7299, 92.2041]

ALIGNED = [ 28.2946, 41.6963,
            67.5318, 41.5014,
            48.0252, 71.7366,
            31.5493, 92.3655,
            65.7299, 92.2041]

dsize = (112,96)


def imageAlignCrop(im, landmark, dsize=(112, 96)):
    """
    Params:
        im:         {ndarray(H, W, 3)}
        landmark:   {ndarray(5, 2)}
        dsize:      {tuple/list(H, W)}
    Returns:
        dstImage:   {ndarray(h, w, 3)}
    Notes:
        对齐后裁剪
    """
    ## 变换矩阵
    M = cp2tform(landmark, np.array(ALIGNED).reshape(-1, 2))
    
    ## 用矩阵变换图像
    warpedImage = warpImage(im, M)
    
    ## 裁剪固定大小的图片尺寸
    h, w = dsize
    dstImage = warpedImage[:h, :w]
    
    return dstImage

def croprgbimage():
    
    dict = None
    with open(rgbtxtpath, 'r') as f:
        dict = eval(f.read())

    for key, values in dict.items():
        landmark = values[2]
        landmark = np.array(landmark).reshape(5,2)
        picpath = rootdir +'/'+key+'.jpg'
        newpicpath = SAVEPATH+'/'+key+'.jpg'
        newpicdir=os.path.dirname(newpicpath)
        if not os.path.exists(newpicdir):
            os.makedirs(newpicdir)
        im = cv2.imread(picpath, cv2.IMREAD_COLOR)       
        # cv2.imshow('b',im)
        cropimage = imageAlignCrop(im,landmark)
        # cv2.imread(cropimage,cv2.IMREAD_COLOR)        
        # cv2.imshow('a',cropimage)
        # cv2.waitKey(0)
        cv2.imwrite(newpicpath,cropimage)

def cropmultimage():
    
    dict = None
    with open(multitxtpath, 'r') as f:
        dict = eval(f.read())

    for key, values in dict.items():
        landmark = values[2]
        landmark = np.array(landmark).reshape(5,2)
        picdir = rootdir+key
        newpicdir = SAVEPATH+key
        if not os.path.exists(newpicdir):
            os.makedirs(newpicdir)
        for i in os.listdir(picdir):
            picpath = os.path.join(picdir,i)
            im = cv2.imread(picpath, cv2.IMREAD_GRAYSCALE)
            cropimage = imageAlignCrop(im,landmark)
            # cv2.imread(cropimage,cv2.IMREAD_COLOR)        
            # cv2.imshow('a',cropimage)
            # cv2.waitKey(0)
            newpicpath = os.path.join(newpicdir,i)
            cv2.imwrite(newpicpath,cropimage)
        

if __name__ == "__main__":

    rootdir  = 'E:/Desktop/Outdoor20190810'
    SAVEPATH  = 'E:/Desktop/Outdoor20190810crop1'

    rgbtxtpath = rootdir + '/rgbdetect.txt'
    multitxtpath = rootdir + '/multidetect.txt'
    croprgbimage()
    cropmultimage()
