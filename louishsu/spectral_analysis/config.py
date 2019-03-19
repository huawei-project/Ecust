from easydict import EasyDict

configer = EasyDict()

configer.datapath = "/home/louishsu/Work/Workspace/ECUST2019"
configer.logspath = "/home/louishsu/Work/Workspace/HUAWEI/pytorch/logs"
configer.mdlspath = "/home/louishsu/Work/Workspace/HUAWEI/pytorch/modelfiles"

configer.splitmode = 'split_1'

configer.facesize = (64, 64)
configer.n_channels = 46
configer.n_classes = 27

# configer.modelname = "analysis_bilstm"  
configer.modelname = "analysis_vgg11"
configer.lossname  = "multi_crossent"
configer.learningrate = 0.0001
configer.batchsize = 5                  # 10
configer.n_epoch = 300
configer.earlystopping = False