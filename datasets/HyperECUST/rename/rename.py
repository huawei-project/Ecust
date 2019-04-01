import os
import shutil
import numpy as np

srcdir = '/home/louishsu/Work/Workspace/ECUST2019'
dstdir = '/home/louishsu/Work/Workspace/ECUST2019_rename'
if not os.path.exists(dstdir): os.makedirs(dstdir)

get_vol = lambda i: (i - 1) // 10 + 1

subjects = [i+1 for i in range(63)]
imgtypes = ['Multi', 'RGB']
lights   = ['non-obtructive', 'obtructive/ob1', 'obtructive/ob2']
lights_replace = {
    'non-obtructive': 'normal', 
    'obtructive/ob1': 'illum1', 
    'obtructive/ob2': 'illum2',
    }
positions = [i+1 for i in range(7)]
sessions  = [i+1 for i in range(10)]

for subject in subjects:
    vol = get_vol(subject)

    for imgtype in imgtypes:
        for light in lights:
            
            for position in positions:
                for session in sessions:

                    srcfile = '{}/DATA{}/{}/{}/{}/{}_{}_W1_{}'.\
                                format(srcdir, vol, subject, imgtype, light, imgtype, position, session)
                    dstfile = '{}/DATA{}/{}/{}/{}/{}_{}_W1_{}'.\
                                format(dstdir, vol, subject, imgtype, lights_replace[light], imgtype, position, session)
                    
                    if imgtype == 'RGB':
                        srcfile = srcfile + '.JPG'
                        dstfile = dstfile + '.JPG'

                        if os.path.exists(srcfile):
                            dir = '/'.join(dstfile.split('/')[:-1])
                            if not os.path.exists(dir): os.makedirs(dir)
                            shutil.copy(srcfile, dstfile)
                    
                    else:

                        if os.path.exists(srcfile):
                            shutil.copytree(srcfile, dstfile)
