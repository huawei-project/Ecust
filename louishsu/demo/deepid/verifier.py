import torch
from   torchvision.transforms import ToTensor

import cv2
import numpy as np
from PIL import Image
from .models import VGGFeatures, DeepId2Features, DeepIdModel

def init_verifier(modelname):
    modelstatepath = './deepid/weights/{}.pkl'.format(modelname + '_state')
    modelstate = torch.load(modelstatepath)
    
    # 打上补丁。。。
    def modify(k):
        k_split = k.split('.')
        if k_split[0] == 'features':
            k_split[1] = 'base' if k_split[1] == 'features' else 'vect'
        k = '.'.join(k_split)
        return k
    modelstate = {modify(k): v for k, v in modelstate.items()}

    model = DeepIdModel(lambda n1, n2: VGGFeatures(n1, n2, 'vgg11_bn'), 3, 64)
    model.load_state_dict(modelstate)
    return model

def init_proto(usedch=[30, 31, 32]):
    proto = dict()
    proto['xyb'] = np.load('./deepid/data/xyb.npy')[:, :, usedch]
    proto['zhao'] = np.load('./deepid/data/zht.npy')[:, :, usedch]
    proto['lzy'] = np.load('./deepid/data/lzy.npy')[:, :, usedch]
    proto['zsm'] = np.load('./deepid/data/zsm.npy')[:, :, usedch]
    proto['zxy'] = np.load('./deepid/data/zxy.npy')[:, :, usedch]
    proto = {k: cv2.resize(v, (96, 96)) for k, v in proto.items()}
    proto = {k: ToTensor()(v).unsqueeze(0) for k, v in proto.items()}
    return proto

def verify(face, net, proto, thresh=0.7):
    """
    Params:
        face:   {ndarray(H, W, 3)}
        net:    {deepid model}
        proto:  {dict{name: face}}
    """
    face = ToTensor()(face).unsqueeze(0)
    face_score = {name: 0.0 for name in proto.keys()}
    for name, X in proto.items():
        _, _, score = net(X, face)
        face_score[name] = score.detach().numpy()[0]
    score = np.array(list(face_score.values()))
    score[score < thresh] = -1.0
    index = np.argmax(score)
    if score[index] == -1.0:
        name = 'unknown'
    else:
        name = list(face_score.keys())[index]
    return name