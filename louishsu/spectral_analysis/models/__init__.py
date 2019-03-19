from .vgg import VGG
from .spectral_analysis import *
from .lstm import *

import sys
sys.path.append('../')
from config import configer
n_channels = configer.n_channels
n_classes = configer.n_classes
input_size = configer.facesize[0]

_models = {
    'vgg11': VGG(n_channels, n_classes, 'VGG11', batch_norm=True),
    
    'analysis_vgg11':  SpectralAnalysis(n_channels, n_classes, VGG(1, n_classes, 'VGG11', batch_norm=True, finetune=True)),
    'analysis_bilstm': SpectralAnalysisBiLSTM(1, n_classes, n_channels, 
                            lambda in_channels, n_classes: VGG(in_channels, n_classes, 'VGG11_tiny')),
}
