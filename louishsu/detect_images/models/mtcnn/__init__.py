from .detectors import Detector, FcnDetector, MtcnnDetector, py_nms
from .models import P_Net, R_Net, O_Net

prefix = ['./models/mtcnn/modelfile/PNet/PNet', './models/mtcnn/modelfile/RNet/RNet', './models/mtcnn/modelfile/ONet/ONet']
epoch = [18, 14, 16]
model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]

detectors = [None, None, None]
PNet = FcnDetector(P_Net,     model_path[0]);   detectors[0] = PNet
RNet = Detector(R_Net, 24, 1, model_path[1]);   detectors[1] = RNet
ONet = Detector(O_Net, 48, 1, model_path[2]);   detectors[2] = ONet

mtcnn_detector = MtcnnDetector(detectors=detectors,
                                min_face_size=48,
                                stride=2, 
                                threshold=[0.9, 0.6, 0.6], 
                                slide_window=False)