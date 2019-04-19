import os
import cv2
from scipy import io
import numpy as np

dict_illum = {
    'non-obtructive': 'normal',
    'obtructive/ob1': 'illum1',
    'obtructive/ob2': 'illum2',
}
DATAPATH = "/home/louishsu/Work/Workspace/ECUST2019_rename"

def get_mtcnn_result():

    def parse(line):
        line = line.strip().split(' ')
        filename = line[0]
        landmark = np.array(list(map(float, line[6:])))
        return filename, landmark

    with open('{}/anno.txt'.format(DATAPATH), 'r') as f:
        detectlist = f.readlines()
    detectlist = list(map(parse, detectlist))
    detectdict = dict(detectlist)

    return detectdict


def get_manually_result():
    def parse_path(path):

        path = path.replace('\\', '/')

        path = path[path.find('DATA'):]
        path = path.split('/')

        if len(path) == 7:
            path.pop(3)
            path[3] = 'obtructive/' + path[3]
        path[3] = dict_illum[path[3]]

        jpg = path[-1]
        dir = '/'.join(path[:-1])

        return [dir, jpg]

    markdict = dict()

    for volidx in [4, 6, 7]:

        ptvol = 'point{:d}'.format(volidx)
        ptlist = list(io.loadmat('{}.mat'.format(ptvol))[ptvol][0])

        ## 解析
        ptlist = list(map(list, ptlist))
        ptlist = list(filter(lambda x: x[1].size!=0, ptlist))       # 滤除空数组
        ptlist = list(map(lambda x: [str(x[0][0]), x[1].reshape(-1)], ptlist))
        ptlist = list(map(lambda x: parse_path(x[0]) + [x[1]], ptlist))

        ## 文件夹标注集中
        markdict_vol = dict()
        for dirname, jpgname, landmark in ptlist:
            try:
                markdict_vol[dirname]
            except KeyError:
                markdict_vol[dirname] = []
            
            landmark = landmark.astype(np.float)
            landmark = landmark.reshape((2, -1))[:, 2:].T
            landmark = landmark.reshape((1, -1))
            markdict_vol[dirname] += [landmark]
        
        ## 取各通道均值
        for dirname in markdict_vol.keys():        
            markdict_vol[dirname] = np.mean(np.concatenate(markdict_vol[dirname], axis=0), axis=0)
        markdict.update(markdict_vol)

    return markdict

def show_landmark(dirname, landmark):

    dir = os.path.join(DATAPATH, dirname)
    image = cv2.imread(os.path.join(dir, os.listdir(dir)[0]), cv2.IMREAD_GRAYSCALE)

    landmark = landmark.reshape((5, 2)).astype('int')

    for i in range(5):
        cv2.circle(image, tuple(landmark[i]), 3, (255, 255, 255), thickness=3)
    cv2.imshow("", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    
    mtcnn_result = get_mtcnn_result()
    manually_result = get_manually_result()

    error = []
    for dirname in manually_result.keys():
        
        manres = manually_result[dirname]
        cnnres = mtcnn_result[dirname]

        # print(list(manres))
        # print(list(cnnres))
        # show_landmark(dirname, cnnres)
        # show_landmark(dirname, manres)

        error += [list(cnnres - manres)]
    
    error = np.array(error)

    error_mse = np.sqrt(np.mean(error**2, axis=0))
    pass
