import os
import time
import torch
import numpy as np

getTime     = lambda: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
getWavelen  = lambda path: int(path.split('.')[0].split('_')[-1])
getLabel    = lambda path: int(path[path.find('DATA') + len('DATAx/'):].split('/')[0])

class ImageAttributes():

    def __init__(self, path):
        """
        Params:
            path:   {str}
                - 若为多光谱，则路径形如
                    - '1/multi/illum3/Multi_3_W1_1/2/22.jpg'；
                    - '1/multi/illum3/Multi_3_W1_5/22.jpg'；
                - 若为可见光，则路径形如
                    - '1/rgb/illum3/RGB_3_W1_1/2.jpg'；
                    - '1/rgb/illum3/RGB_3_W1_5.jpg'；
        """
        
        self.path = path
        self.label, self.image_type, self.illum_type, \
            self.position, self.glass_type, \
                self.image_index, self.channel_index = self.parse(path)

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
    def parse(path):

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
                channel_index = int(path[5].split('.')[0])
            else:
                channel_index = int(path[4].split('.')[0])

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