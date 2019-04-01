import os
import cv2
import numpy




# 修改以下即可
dsize = 64
srcdir = '/home/louishsu/Work/Workspace/ECUST2019_rename'
dstdir = '/home/louishsu/Work/Workspace/ECUST2019_{}x{}'.format(dsize, dsize)
if not os.path.exists(dstdir): os.makedirs(dstdir)










get_vol = lambda i: (i - 1) // 10 + 1

subjects = [i+1 for i in range(63)]
imgtypes = ['Multi', 'RGB']
lights   = ['normal', 'illum1', 'illum2']
positions = [i+1 for i in range(7)]
sessions  = [i+1 for i in range(10)]

for subject in subjects:
    vol = get_vol(subject)
    with open('{}/DATA{}/detect.txt'.format(srcdir, vol), 'r') as f:
        dict = eval(f.read())

    for imgtype in imgtypes:
        if imgtype == 'Multi': continue

        for light in lights:
            
            for position in positions:
                for session in sessions:

                    srcfile = '{}/DATA{}/{}/{}/{}/{}_{}_W1_{}'.\
                                format(srcdir, vol, subject, imgtype, light, imgtype, position, session)
                    dstfile = '{}/DATA{}/{}/{}/{}/{}_{}_W1_{}'.\
                                format(dstdir, vol, subject, imgtype, light, imgtype, position, session)
                    key = '/{}/{}/{}/{}_{}_W1_{}'.format(subject, imgtype, light, imgtype, position, session)

                    if imgtype == 'RGB':
                        srcfile = srcfile + '.JPG'
                        dstfile = dstfile + '.JPG'

                        if os.path.exists(srcfile):
                            
                            image = cv2.imread(srcfile, cv2.IMREAD_ANYCOLOR)
                            x1, y1, x2, y2 = dict[key][1]

                            h, w = image.shape[:-1]
                            x1 = 0 if x1 < 0 else x1
                            y1 = 0 if y1 < 0 else y1
                            x2 = w - 1 if x2 > w - 1 else x2
                            y2 = h - 1 if y2 > h - 1 else y2

                            image = image[y1: y2, x1:x2]
                            image = cv2.resize(image, (dsize, dsize))

                            dstdir = '/'.join(dstfile.split('/')[:-1])
                            os.makedirs(dstdir)
                            cv2.imwrite(dstfile, image)

                    else:

                        if os.path.exists(srcfile):
                            if not os.path.exists(dstfile): os.makedirs(dstfile)
                            
                            for bmp in os.listdir(srcfile):
                                
                                image = cv2.imread('{}/{}'.format(srcfile, bmp), cv2.IMREAD_GRAYSCALE)
                                x1, y1, x2, y2 = dict[key][1]

                                h, w = image.shape
                                x1 = 0 if x1 < 0 else x1
                                y1 = 0 if y1 < 0 else y1
                                x2 = w - 1 if x2 > w - 1 else x2
                                y2 = h - 1 if y2 > h - 1 else y2

                                image = image[y1: y2, x1:x2]
                                image = cv2.resize(image, (dsize, dsize))

                                jpg = bmp.split('.')[0] + '.JPG'
                                cv2.imwrite('{}/{}'.format(dstfile, jpg), image)
