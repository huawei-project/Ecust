import os
import shutil
import numpy as np

srcdir = '/home/louishsu/Work/Workspace/ECUST2019'

volumes = [i+1 for i in range(7)]
lights_replace = {
    'non-obtructive': 'normal', 
    'obtructive/ob1': 'illum1', 
    'obtructive/ob2': 'illum2',
    }

for volume in volumes:
    detect_txt = '{}/DATA{}/detect.txt'.format(srcdir, volume)
    with open(detect_txt, 'r') as f:
        dict_old = eval(f.read())

    dict_new = dict()

    for filename, detect in dict_old.items():
        filename_split = filename.split('/')
        if len(filename_split) == 5:
            filename_split[3] = lights_replace[filename_split[3]]
        elif len(filename_split) == 6:
            filename_split[3] = lights_replace['/'.join(filename_split[3:5])]
            filename_split.pop(4)
        filename_new = '/'.join(filename_split)
        dict_new[filename_new] = dict_old[filename]
    
    with open(detect_txt, 'w') as f:
        f.write(str(dict_new))