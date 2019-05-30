import os
import cv2
import time
import numpy as np
from cropAlign import imageAlignCrop

def line2kv(line):
    line = line.strip().split(' ')
    filename = line[0]
    coords   = list(map(float, line[1:]))
    return [filename, coords]

def parseCoord(coords):
    box, score, landmark = coords[:4], coords[4], coords[5:]
    return np.array(box), score, np.array(landmark)

def crop_casia(prefix='../data/CASIA-WebFace', detected='../data/CASIA_detect.txt', 
        unaligned='../data/CASIA-WebFace-Unaligned', aligned='../data/CASIA-WebFace-Croped-Aligned', dsize=(112, 96)):
    if not os.path.exists(unaligned): os.mkdir(unaligned)
    if not os.path.exists(aligned): os.mkdir(aligned)
    
    ## 载入已检测的结果
    start_time = time.time()
    print('\033[2J\033[1;1H')
    print('\033[1;1HLoading detections...\033[s')
    with open(detected, 'r') as f:
        detect = f.readlines()
    detect = list(map(lambda x: line2kv(x), detect))
    detect = {k: v for k, v in detect}
    print('\033[uOK! >>> {:.2f}s'.format(time.time() - start_time))

    i = 0; n = len(detect)
    elapsed_time = 0
    start_time = time.time()

    for subdir in os.listdir(prefix):
        subdirSrc      = os.path.join(prefix, subdir)
        subdirUnWarped = os.path.join(unaligned, subdir)
        subdirWarped   = os.path.join(aligned, subdir)
        if not os.path.exists(subdirUnWarped): os.mkdir(subdirUnWarped)
        if not os.path.exists(subdirWarped): os.mkdir(subdirWarped)
        
        for imidx in os.listdir(subdirSrc):
            key = '/'.join([subdir, imidx])
            if key not in detect.keys(): continue

            i += 1
            duration = time.time() - start_time
            elapsed_time += duration
            start_time = time.time()
            
            print('\033[2;1H[{:6d}]/[{:6d}] FPS: {:.4f}  Elapsed: {:.4f}h Left: {:.4f}h'.\
                            format(i, n, 1./duration, elapsed_time/3600, (duration*n - elapsed_time)/3600))
            
            ## 读取原图
            srcImg = cv2.imread(os.path.join(subdirSrc, imidx), cv2.IMREAD_COLOR)
            ## 剪裁
            box, score, landmark = parseCoord(detect[key])
            warpedImage, unWarpedImage = imageAlignCrop(srcImg, box, landmark, dsize, return_unaligned=True)

            ## 保存结果
            cv2.imwrite(os.path.join(subdirUnWarped, imidx), unWarpedImage)
            cv2.imwrite(os.path.join(subdirWarped, imidx), warpedImage)

if __name__ == "__main__":
    crop_casia()