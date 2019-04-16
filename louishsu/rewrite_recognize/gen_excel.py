import os
import numpy as np
import xlwt

def write_excel():
    f = xlwt.Workbook()
    sheet1 = f.add_sheet('学生',cell_overwrite_ok=True)
    row0 = ["姓名","年龄","出生日期","爱好"]
    colum0 = ["张三","李四","恋习Python","小明","小红","无名"]
    #写第一行
    for i in range(0,len(row0)):
        sheet1.write(0,i,row0[i],set_style('Times New Roman',220,True))
    #写第一列
    for i in range(0,len(colum0)):
        sheet1.write(i+1,0,colum0[i],set_style('Times New Roman',220,True))

    sheet1.write(1,3,'2006/12/12')
    sheet1.write_merge(6,6,1,3,'未知')#合并行单元格
    sheet1.write_merge(1,2,3,3,'打游戏')#合并列单元格
    sheet1.write_merge(4,5,3,3,'打篮球')

    f.save('test.xls')

def gen_out_excel(configer):

    def write_title(sheet):
        title = ['number', 'filename', 'image type', 'illumination', 'position', 'glasses', 
                            'predicted probability', 'predicted label', 'ground truth label']
        width = [1, 1, 1, 1, 1, 1, 63, 1, 1]
        x1 = 0; y1 = 0
        for i in range(len(title)):
            x2 = x1 + width[i] - 1; y2 = y1 
            sheet.write_merge(y1, y2, x1, x2, title[i])
            x1 = x2 + 1

    def get_property(filename):
        imtype = lambda x: x.split('/')[-3]
        illum  = lambda x: x.split('/')[-2]
        position = lambda x: int(x.split('/')[-1].split('_')[-3])
        glasses = lambda x: int(x.split('/')[-1].split('_')[-1])
        groundtruth = lambda x: int(x.split('/')[2]) - 1

        return imtype(filename), illum(filename), position(filename), glasses(filename), groundtruth(filename)

    def get_predict(output):
        """ output: {ndarray(63)} """
        pred_prob = np.exp(output) / np.sum(np.exp(output))
        pred_label = int(np.argmax(pred_prob, axis=0))
        pred_prob = list(pred_prob); pred_prob = list(map(float, pred_prob))
        return pred_prob, pred_label

    ## get test files and labels
    if configer.datatype == 'Multi':
        txtfile = 'test'
    elif configer.datatype == 'RGB':
        txtfile = 'test_rgb'
    with open('./split/{}/{}.txt'.format(configer.splitmode, txtfile), 'r') as f:
        testfiles = f.readlines()
    
    ## get output
    log_modelname_dir = os.path.join(configer.logspath, configer.modelname)
    testout = np.load(os.path.join(log_modelname_dir, 'test_out.npy'))

    ## excel
    xls = xlwt.Workbook()
    sheet = xls.add_sheet('output', cell_overwrite_ok=True)

    ## write title
    write_title(sheet)

    ## write body
    for i in range(len(testfiles)):
        imtype, illum, position, glasses, groundtruth = get_property(testfiles[i])
        pred_prob, pred_label = get_predict(testout[i])
        line = [i, testfiles[i], imtype, illum, position, glasses] + pred_prob + [pred_label, groundtruth]
        for j in range(len(line)):
            sheet.write(i+1, j, line[j])
        
    xls.save(os.path.join(log_modelname_dir, 'test.xls'))

    pass

if __name__ == "__main__":
    
    from config import configer
    
    gen_out_excel(configer)