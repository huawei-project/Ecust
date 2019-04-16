import os
import numpy as np

def filter_condition(filelist, illum: str, position: int, glasses: int):
    """ filter condition
    """
    if illum != 'none':
        idx_illum = np.array(list(map(lambda x: int(-1 != x.find(illum)), filelist)), dtype='bool')
    else:
        idx_illum = np.ones(shape=len(filelist), dtype='bool')
    if position != 'none':
        idx_position = np.array(list(map(lambda x: int(-1 != x.find('{}_W1'.format(position))), filelist)), dtype='bool')
    else:
        idx_position = np.ones(shape=len(filelist), dtype='bool')
    if glasses != 'none':
        idx_glasses = np.array(list(map(lambda x: int(-1 != x.find('W1_{}'.format(glasses))), filelist)), dtype='bool')
    else:
        idx_glasses = np.ones(shape=len(filelist), dtype='bool')
    index = idx_illum & idx_position & idx_glasses
    cond = "{} {} {}".format(illum, position, glasses)

    return cond, index



def analysis(configer):

    softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=0)
    accuracy = lambda pred, gt: np.mean(pred==gt)
    ## get test files and labels
    if configer.datatype == 'Multi':
        txtfile = 'test'
    elif configer.datatype == 'RGB':
        txtfile = 'test_rgb'
    with open('./split/{}/{}.txt'.format(configer.splitmode, txtfile), 'r') as f:
        testfiles = f.readlines()
    y_true_label = np.array(list(map(lambda x: int(x.split('/')[2])-1, testfiles)))

    ## get output
    log_modelname_dir = os.path.join(configer.logspath, configer.modelname)
    testout = np.load(os.path.join(log_modelname_dir, 'test_out.npy'))
    y_pred_proba = softmax(testout)
    y_pred_label = np.argmax(y_pred_proba, axis=1)

    while True:
        
        illum = None
        position = None
        glasses = None

        while illum not in ['normal', 'illum1', 'illum2', 'none']:
            illum = input("please input illumination: <normal/illum1/illum2 or none>")
        while position not in [str(i+1) for i in range(7)] + ['none']:
            position = input("please input position: <1~7 or none>")
        while glasses not in ['1', '5'] + ['none']:
            glasses = input("please input glass condition: <1/5 or none>")

        cond, index = filter_condition(testfiles, illum, position, glasses)
        print('-----------------------------------------------------')
        
        if np.sum(index) == 0:
            print('no such condition: ', cond)
            continue
        else:
            print('get condition: {}, number of samples: {}'.\
                        format(cond, index[index==True].shape[0]))
        print('-----------------------------------------------------')
        
        y_pred_proba_filt = y_pred_proba[index]
        y_pred_label_filt = y_pred_label[index]
        y_true_label_filt = y_true_label[index]

        acc = accuracy(y_pred_label_filt, y_true_label_filt)

        print('accuracy score is: {}'.format(acc))
        print('=====================================================')

if __name__ == "__main__":
    from config import configer
    analysis(configer)