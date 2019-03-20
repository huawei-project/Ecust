import os
import shutil

"""
拍照顺序
```
for pos in positions:
    for ob in obtypes:
        # get images
```        
- 位置
    4, 3, 2, 1, 5, 6, 7

- 灯光
    non-ob
    ob/ob1: 白色大的灯
    ob/ob2: 黄色小的灯
"""

def dirtime(dirname):
    """
    Params:
        dirname:    {str}
    """
    h, m, s = [int(t) for t in dirname.split('_')[3:]]
    return h*3600 + m*60 + s

def sort_multi(srcdir, dstdir, index):
    """
    Params:
        srcdir: {str} 
        dstdir: {str} 
        index:  {list[index1, index2]}
    """

    dirs = os.listdir(srcdir)
    dirs = [dir for dir in dirs if dir.split('.')[-1]!='txt']
    dirs = sorted(dirs, key=lambda dir: dirtime(dir))

    pos = [4, 3, 2, 1, 5, 6, 7]
    obs = ['non-obtructive', 'obtructive/ob1', 'obtructive/ob2']

    for i_dir in range(len(dirs)//2):
        srcdir1, srcdir2 = srcdir + '/' + dirs[i_dir*2], srcdir + '/' + dirs[i_dir*2+1]

        ob = obs[i_dir % (len(index)*len(obs)) % len(obs)]
        po = pos[i_dir // (len(obs)*len(index)//2)]

        dstdir1, dstdir2 = '{}/{}/Multi/{}/Multi_{}_W1_1'.format(dstdir, index[0], ob, po), '{}/{}/Multi/{}/Multi_{}_W1_1'.format(dstdir, index[1], ob, po)

        shutil.copytree(srcdir1, dstdir1)
        shutil.copytree(srcdir2, dstdir2)

def jpgidx(jpgfile):
    """
    Params:
        jpgfile: {str} xxx_index.jpg
    """
    index = int(jpgfile.split('.')[0].split('_')[-1])
    return index

def sort_rgb(srcdir, dstdir, index):
    """
    Params:
        srcdir: {str} 
        dstdir: {str} 
        index:  {list[index1, index2]}
    """

    jpgs = os.listdir(srcdir)
    jpgs = [jpg for jpg in jpgs if jpg.split('.')[-1]!='txt']
    jpgs = sorted(jpgs, key=lambda jpg: jpgidx(jpg))

    pos = [4, 3, 2, 1, 5, 6, 7]
    obs = ['non-obtructive', 'obtructive/ob1', 'obtructive/ob2']

    for i_jpg in range(len(jpgs)//2):
        srcjpg1, srcjpg2 = srcdir + '/' + jpgs[i_jpg*2], srcdir + '/' + jpgs[i_jpg*2+1]

        ob = obs[i_jpg % (len(index)*len(obs)) % len(obs)]
        po = pos[i_jpg // (len(obs)*len(index)//2)]

        dstdir1 = '{}/{}/RGB/{}'.format(dstdir, index[0], ob)
        if not os.path.exists(dstdir1): os.makedirs(dstdir1)
        dstdir2 = '{}/{}/RGB/{}'.format(dstdir, index[1], ob)
        if not os.path.exists(dstdir2): os.makedirs(dstdir2)

        dstjpg1, dstjpg2 = '{}/RGB_{}_W1_1.jpg'.format(dstdir1, po), '{}/RGB_{}_W1_1.jpg'.format(dstdir2, po)

        shutil.copy(srcjpg1, dstjpg1)
        shutil.copy(srcjpg2, dstjpg2)



if __name__ == "__main__":
    sort_multi("./multi/34 35", "E:/sorted", [34, 35])
    sort_multi("./multi/36 37", "E:/sorted", [36, 37])
    sort_multi("./multi/38 39", "E:/sorted", [38, 39])
    sort_multi("./multi/40 41", "E:/sorted", [40, 41])

    sort_rgb("./rgb/34 35", "E:/sorted", [34, 35])
    sort_rgb("./rgb/36 37", "E:/sorted", [36, 37])
    sort_rgb("./rgb/38 39", "E:/sorted", [38, 39])
    sort_rgb("./rgb/40 41", "E:/sorted", [40, 41])