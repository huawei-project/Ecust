'''
@Description: 
@Version: 1.0.0
@Auther: louishsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-08-10 10:36:18
@LastEditTime: 2019-09-17 13:52:47
@Update: 
'''
import os
import time
import torch
import numpy as np

getTime     = lambda: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

class ImageAttributes():
    """ Gather attributes of given image path

    Examples:
        ```
        paths = [
            '1/multi/illum3/Multi_3_W1_1/2/22.jpg',
            '1/multi/illum3/Multi_3_W1_5/22.jpg',
            '1/multi/illum3/Multi_3_W1_1/2/',
            '1/multi/illum3/Multi_3_W1_5/',
            '1/rgb/illum3/RGB_3_W1_1/2.jpg',
            '1/rgb/illum3/RGB_3_W1_5.jpg',
        ]
        for path in paths:
            print(ImageAttributes(path))
        ```
    """

    def __init__(self, path, return_channel_index=False):
        """
        Params:
            path:   {str}
                - 若为多光谱，则路径形如
                    - '1/multi/illum3/Multi_3_W1_1/2/22.jpg'；
                    - '1/multi/illum3/Multi_3_W1_5/22.jpg'；
                    - '1/multi/illum3/Multi_3_W1_1/2/'；
                    - '1/multi/illum3/Multi_3_W1_5/'；
                - 若为可见光，则路径形如
                    - '1/rgb/illum3/RGB_3_W1_1/2.jpg'；
                    - '1/rgb/illum3/RGB_3_W1_5.jpg'；
        """
        
        self.path = path
        self.label, self.image_type, self.illum_type, \
            self.position, self.glass_type, \
                self.image_index, self.channel_index = self.parse(path, return_channel_index)

    def __repr__(self):

        return """
        path:           {},
        label:          {},
        image_type:     {}, 
        illum_type:     {}, 
        position:       {}, 
        glass_type:     {}, 
        image_index:    {}, 
        channel_index:  {}
        """.format(self.path, self.label, self.image_type, self.illum_type,
                self.position, self.glass_type, self.image_index, self.channel_index)

    @staticmethod
    def parse(path, return_channel_index):

        path = path.strip('/').split('/')

        label = int(path[0])
        illum_type = path[2]

        image_type_position_W1_glass_type = path[3].split('_')
        image_type = image_type_position_W1_glass_type[0]
        position   = int(image_type_position_W1_glass_type[1])
        glass_type = int(image_type_position_W1_glass_type[3]) if image_type == "Multi" \
            else int(image_type_position_W1_glass_type[3].split('.')[0])

        image_index = channel_index = None

        if image_type == "Multi":
            
            if glass_type != 5:
                image_index = int(path[4])
                channel_index = int(path[5].split('.')[0]) if return_channel_index else None
            else:
                channel_index = int(path[4].split('.')[0]) if return_channel_index else None

        return label, image_type, illum_type, position, glass_type, image_index, channel_index


def get_path_by_attr(label, image_type, illum_type, position, glass_type, 
                            image_index=None, channel_index=None):
    """
    Params:

        label:      {int} 1 - 92
        image_type: {str} "Multi", "RGB"
        illum_type: {str} "illum1", "illum2", "illum3", "normal"
        position:   {int} 1 - 7
        glass_type: {int} 1, 5, 6
        image_index:    {int} 1 - 4
        channel_index:  {int} 1 - 25 or 0
    
    Returns:

        path: {str} "[label]/[image_type.lower()]/[illum_type]/[image_type]_[position]_W1_[glass_type]/[jpg_file]"
            - 若为多光谱，则路径形如
                - '1/multi/illum3/Multi_3_W1_1/2/22.jpg'；
                - '1/multi/illum3/Multi_3_W1_5/22.jpg'；
            - 若为可见光，则路径形如
                - '1/rgb/illum3/RGB_3_W1_1/2.jpg'；
                - '1/rgb/illum3/RGB_3_W1_5.jpg'；

    Notes:

        if image_type == "Multi":
                    
            if glass_type == 5:
                jpg_file = "[channel_index].jpg"
            else:
                jpg_file = "[image_index]/[channel_index].jpg"
            
            if channel_index == 0:
                jpg_file = "[image_index]/"

        else == "RGB":
            
            if glass_type == 5:
                jpg_file = ".jpg"
            else:
                jpg_file = "[image_index].jpg"
    """
    jpg_file = None
    

    if image_type == "Multi":
        if channel_index is None:
            raise ValueError("`channel_index` should not be `None`")

        if glass_type == 5:
            jpg_file = "/{:d}.jpg".format(channel_index)

        else:
            if image_index is None:
                raise ValueError("`image_index` should not be `None`")
            jpg_file = "/{:d}/{:d}.jpg".format(image_index, channel_index)

        if channel_index == 0:
            jpg_file = "/".join(jpg_file.split('/')[:-1])

    else:

        if glass_type == 5:
            jpg_file = ".jpg"
        else:
            if image_index is None:
                raise ValueError("`image_index` should not be `None`")
            jpg_file = "/{:d}.jpg".format(image_index)

    path = "{:d}/{:s}/{:s}/{:s}_{:d}_W1_{:d}{:s}".format(
        label, image_type.lower(), illum_type,
        image_type, position, glass_type,
        jpg_file
    )

    return path

def is_with_no_sun_glasses(path):
    """
    Params:
        path:   {str}
            - 若为多光谱，则路径形如
                - '1/multi/illum3/Multi_3_W1_1/2'；
                - '1/multi/illum3/Multi_3_W1_5'；
            - 若为可见光，则路径形如
                - '1/rgb/illum3/RGB_3_W1_1/2.jpg'；
                - '1/rgb/illum3/RGB_3_W1_5.jpg'；
    """
    path = path.split('/')[3]
    glasses = int(path.split('.')[0].split('_')[-1])

    return not glasses == 6

def is_with_no_glasses(path):
    """
    Params:
        path:   {str}
            - 若为多光谱，则路径形如
                - '1/multi/illum3/Multi_3_W1_1/2'；
                - '1/multi/illum3/Multi_3_W1_5'；
            - 若为可见光，则路径形如
                - '1/rgb/illum3/RGB_3_W1_1/2.jpg'；
                - '1/rgb/illum3/RGB_3_W1_5.jpg'；
    """
    path = path.split('/')[3]
    glasses = int(path.split('.')[0].split('_')[-1])

    return glasses == 1

def is_with_no_sunglasses(path):
    """
    Params:
        path:   {str}
            - 若为多光谱，则路径形如
                - '1/multi/illum3/Multi_3_W1_1/2'；
                - '1/multi/illum3/Multi_3_W1_5'；
            - 若为可见光，则路径形如
                - '1/rgb/illum3/RGB_3_W1_1/2.jpg'；
                - '1/rgb/illum3/RGB_3_W1_5.jpg'；
    """
    path = path.split('/')[3]
    glasses = int(path.split('.')[0].split('_')[-1])

    return glasses == 1 or glasses == 5

def accuracy(y_pred_prob, y_true):
    """
    Params:
        y_pred_prob:{tensor(N, n_classes) or tensor(N, C, n_classes)}
        y_true:     {tensor(N)}
    Returns:
        acc:        {tensor(1)}
    """
    y_pred = torch.argmax(y_pred_prob, 1)
    acc = torch.mean((y_pred==y_true).float())
    return acc

def gen_markdown_table_2d(head_name, rows_name, cols_name, data):
    """
    Params:
        head_name: {str} 表头名， 如"count\比例"
        rows_name, cols_name: {list[str]} 项目名， 如 1,2,3
        data: {ndarray(H, W)}
    
    Returns:
        table: {str}

    Example:
        H = 5; W = 6
        data = np.arange(H*W).reshape(H, W)
        head_name = "行\列"
        rows_name = [str(i + 1) for i in range(H)]
        cols_name = [str(i + 10) for i in range(W)]

        gen_markdown_table_2d(head_name, rows_name, cols_name, data)

        [out]:
            | 行\列 | 10 | 11 | 12 | 13 | 14 | 15 |
            | ---: | --: | --: | --: | --: | --: | --: |
            | 1 | 0 | 1 | 2 | 3 | 4 | 5 |
            | 2 | 6 | 7 | 8 | 9 | 10 | 11 |
            | 3 | 12 | 13 | 14 | 15 | 16 | 17 |
            | 4 | 18 | 19 | 20 | 21 | 22 | 23 |
            | 5 | 24 | 25 | 26 | 27 | 28 | 29 |
    """
    ELEMENT = " {} |"

    H, W = data.shape
    LINE = "|" + ELEMENT * W
    
    lines = []

    ## 表头部分
    lines += ["| {} | {} |".format(head_name, ' | '.join(cols_name))]

    ## 分割线
    SPLIT = "{}:"
    line = "| {} |".format(SPLIT.format('-'*len(head_name)))
    for i in range(W):
        line = "{} {} |".format(line, SPLIT.format('-'*len(cols_name[i])))
    lines += [line]
    
    ## 数据部分
    for i in range(H):
        d = list(map(str, list(data[i])))
        lines += ["| {} | {} |".format(rows_name[i], ' | '.join(d))]

    table = '\n'.join(lines)

    return table

def parse_log(log):
    """
    Params:
        log: {str} e.g. "2019-08-10 07:17:03 || test | acc: 96.25%, loss: 0.2208"
    Returns:
        acc, loss: {float}
    """
    log = log.strip().split(' ')
    acc = float(log[-3].split('%')[0]) / 100
    loss = float(log[-1])
    
    return acc, loss

if __name__ == '__main__':

    # parse_log("2019-08-10 07:17:03 || test | acc: 96.25%, loss: 0.2208")

    H = 5; W = 6
    data = np.arange(H*W).reshape(H, W)
    head_name = "行\列"
    rows_name = [str(i + 1) for i in range(H)]
    cols_name = [str(i + 10) for i in range(W)]

    table = gen_markdown_table_2d(head_name, rows_name, cols_name, data)
    print(table)
    
