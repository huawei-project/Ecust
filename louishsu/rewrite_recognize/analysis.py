import os
import numpy as np

def filter_condition(filelist, imgtype: str, illum: str, positon: int, glasses: int):
    """ filter condition
    """
    cond = "{}/{}_{}_W1_{}".format(illum, imgtype, positon, glasses)
    index = np.array(list(map(lambda x: int(-1 != x.find(cond)), filelist)), dtype='bool')

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

        illum = input("please input illumination: <normal/illum1/illum2>")
        positon = input("please input position: <1~7>")
        glasses = input("please input glass condition: <1/5>")
        cond, index = filter_condition(testfiles, configer.datatype, illum, positon, glasses)
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