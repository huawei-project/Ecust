from .vggnet import VGG
from .shufflenet_v1 import ShuffleNet
from .shufflenet_v2 import ShuffleNetV2
from .squeezenet import SqueezeNet
from .mobilenet_v2 import MobileNetV2
from .densenet import DenseNet
from .lstm import ConvLSTM, ConvLSTMNet
from .resnet34 import ResNet
from .inceptionv3 import InceptionV3

modeldict = {
    'recognize_vgg11_bn':       lambda inp, outp, size: VGG(inp, outp, 'VGG11', batch_norm=True),
    'recognize_shufflev2':      lambda inp, outp, size: ShuffleNetV2(inp, outp, size),
    'recognize_squeeze_v11':    lambda inp, outp, size: SqueezeNet(inp, outp, version=1.1),
    'recognize_dense121':       lambda inp, outp, size: DenseNet(in_channels=inp, num_classes=outp),
    'recognize_convlstm':       lambda inp, outp, size: ConvLSTM(1, outp, size, inp),
    'recognize_convlstm2':      lambda inp, outp, size: ConvLSTMNet(1, outp, size, inp),
    'recognize_resnet34':       lambda inp, outp, size: ResNet(in_channels=inp,num_classes=outp),
    'recognize_inceptionv3':    lambda inp, outp, size: InceptionV3(in_channels=inp, num_classes=outp),
}