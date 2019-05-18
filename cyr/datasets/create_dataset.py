"""
# Author: Yuru Chen
# Time: 2019 03 20
"""
import os
import sys
import glob
import cv2
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


def get_id(x): return int(x.split('/')[1])


def get_band(x): return int(x.split('_')[-1].split('.')[0])


def get_vol(i): return (i - 1) // 10 + 1


def get_path_from_condition(data_list, condition):
    return list(filter(lambda x: condition in x, data_list))


def get_path_from_id(data_list, ID):
    return list(filter(lambda x: int(x.split('/')[1]) == ID, data_list))


def get_path_from_band(data_list, band):
    return list(filter(lambda x: int(x.split('_')[-1].split('.')[0]) == band, data_list))


def getDicts(dataset_path):
    dicts = dict()
    for vol in ["DATA%d" % _ for _ in range(1, 6)]:
        txtfile = os.path.join(dataset_path, vol, "detect.txt")
        with open(txtfile, 'r') as f:
            dicts[vol] = eval(f.read())
    return dicts


def show_result(image, score=None, bbox=None, landmarks=None, winname="", waitkey=0):
    """
    Params:
        image:      {ndarray(H, W, 3)}
        score:      {ndarray(n_faces)}
        bbox:       {ndarray(n_faces, 4)}
        landmarks:  {ndarray(n_faces, 10)}
        winname:    {str}
    """
    if bbox is not None:
        n_faces = bbox.shape[0]
        for i in range(n_faces):
            corpbbox = [int(bbox[i, 0]), int(bbox[i, 1]),
                        int(bbox[i, 2]), int(bbox[i, 3])]
            cv2.rectangle(image, (corpbbox[0], corpbbox[1]),
                          (corpbbox[2], corpbbox[3]),
                          (255, 0, 0), 1)
            cv2.putText(image, '{:.3f}'.format(score[i]),
                        (corpbbox[0], corpbbox[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
            if landmarks is not None:
                for j in range(int(len(landmarks[i]) / 2)):
                    cv2.circle(image, (int(
                        landmarks[i][2 * j]), int(int(landmarks[i][2 * j + 1]))), 2, (0, 0, 255))
    cv2.imshow(winname, image)
    cv2.waitKey(waitkey)
    cv2.destroyWindow(winname)


def convert_to_square(bbox):
    """Convert bbox to square

    Parameters:
    ----------
    bbox: numpy array , shape n x 5
        input bbox

    Returns:
    -------
    square bbox
    """
    square_bbox = bbox.copy()

    h = bbox[:, 3] - bbox[:, 1] + 1
    w = bbox[:, 2] - bbox[:, 0] + 1
    max_side = np.maximum(h, w)
    square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
    square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
    square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
    return square_bbox


def filename_parse(filename):
    parse_dict = dict()
    parse_dict['image_type'] = 'Multi' if 'Multi' in filename else 'RGB'
    if '/ob1' in filename:
        parse_dict['ob_type'] = 'ob1'
    elif '/ob2' in filename:
        parse_dict['ob_type'] = 'ob2'
    else:
        parse_dict['ob_type'] = 'nonob'
    parse_dict['pos_type'] = filename.split('_')[1]
    parse_dict['makeup_type'] = filename.split('_')[3].split('/')[0]
    parse_dict['band'] = get_band(filename)
    parse_dict['id'] = get_id(filename)
    return parse_dict


def split_dataset(DATASET_PATH):
    multi_paths1 = glob.glob(os.path.join(
        DATASET_PATH, 'DATA*/*/Multi/non-obtructive/*/*.bmp'))
    multi_paths2 = glob.glob(os.path.join(
        DATASET_PATH, 'DATA*/*/Multi/obtructive/*/*/*.bmp'))
    multi_paths = multi_paths1 + multi_paths2
    index = multi_paths[0].find('DATA')
    multi_paths = sorted([x[index:] for x in multi_paths])

    rgb_paths1 = glob.glob(os.path.join(
        DATASET_PATH, 'DATA*/*/RGB/non-obtructive/*.JPG'))
    rgb_paths2 = glob.glob(os.path.join(
        DATASET_PATH, 'DATA*/*/RGB/obtructive/*/*.JPG'))
    rgb_paths = rgb_paths1 + rgb_paths2
    index = rgb_paths[0].find('DATA')
    rgb_paths = sorted([x[index:] for x in rgb_paths])

    print('Multispectral images: {}, rgb images: {}'.format(
        len(multi_paths), len(rgb_paths)))

    trainset = get_path_from_condition(multi_paths, 'non-ob')
    print('Multispectral non-ob images: {}'.format(len(trainset)))
    validset = get_path_from_condition(multi_paths, '/ob')
    print('Multispectral ob images: {}'.format(len(validset)))

    return trainset, validset


class HyperECUST(Dataset):
    def __init__(self, data_path, filenames, facesize=None, cropped_by_bbox=True, mode='train'):
        """
        Params:
            facesize:   {tuple/list[H, W]}
            mode:       {str} 'train', 'valid'
        """
        self.mode = mode
        self.data_path = data_path
        self.filenames = filenames
        self.facesize = tuple(facesize)
        self.cropped_by_bbox = cropped_by_bbox
        self.dicts = getDicts(data_path)

    def get_bbox(self, filename):
        image_attr = filename_parse(filename)
        # get bbox
        vol = "DATA%d" % get_vol(image_attr['id'])
        imgname = filename[filename.find("DATA") + 5:]
        dirname = '/'.join(imgname.split('/')[:-1])
        bbox = self.dicts[vol][dirname][1]
        square_bbox = convert_to_square(np.array([bbox]))
        return square_bbox

    def get_image(self, filename):
        # load image array
        if self.cropped_by_bbox:
            x1, y1, x2, y2 = self.get_bbox(filename)[0]
            image = cv2.imread(os.path.join(
                self.data_path, filename), cv2.IMREAD_GRAYSCALE)[y1: y2, x1: x2]
        else:
            image = cv2.imread(os.path.join(self.data_path, filename),
                               cv2.IMREAD_GRAYSCALE)
        if self.facesize is not None:
            image = cv2.resize(image, self.facesize[::-1])
        image = image[:, :, np.newaxis]
        return image

    def __getitem__(self, index):
        filename = self.filenames[index].strip()
        image_attr = filename_parse(filename)
        #print('image attribute {}, image path {}'.format(), image_attr, filename)
        image = self.get_image(filename)
        label = image_attr['id'] - 1
        # RandomFlipLeftRight(image, self.mode)
        image = np.concatenate((image, image, image), axis=-1)
        image = ToTensor()(image)
        return image, label, image_attr

    def __len__(self):
        return len(self.filenames)


class HyperECUST_SingleChannel(Dataset):
    def __init__(self, data_path, filenames, band=550, facesize=None, cropped_by_bbox=True, mode='train'):
        """
        Params:
            facesize:   {tuple/list[H, W]}
            mode:       {str} 'train', 'valid'
        """
        self.mode = mode
        self.data_path = data_path
        self.filenames = filenames
        self.band = band
        self.facesize = tuple(facesize)
        self.cropped_by_bbox = cropped_by_bbox
        self.dicts = getDicts(data_path)

    def get_bbox(self, filename):
        image_attr = filename_parse(filename)
        # get bbox
        vol = "DATA%d" % get_vol(image_attr['id'])
        imgname = filename[filename.find("DATA") + 5:]
        dirname = '/'.join(imgname.split('/')[:-1])
        bbox = self.dicts[vol][dirname][1]
        square_bbox = convert_to_square(np.array([bbox]))
        return square_bbox

    def get_image(self, filename):
        # load image array
        if self.cropped_by_bbox:
            x1, y1, x2, y2 = self.get_bbox(filename)[0]
            image = cv2.imread(os.path.join(
                self.data_path, filename), cv2.IMREAD_GRAYSCALE)[y1: y2, x1: x2]
        else:
            image = cv2.imread(os.path.join(self.data_path, filename),
                               cv2.IMREAD_GRAYSCALE)
        if self.facesize is not None:
            image = cv2.resize(image, self.facesize[::-1])
        image = image[:, :, np.newaxis]
        return image

    def __getitem__(self, index):
        filename = self.filenames[index].strip()
        filepath = os.listdir(os.path.join(self.data_path, filename))
        filepath = [os.path.join(filename, x) for x in filepath]
        filepath = get_path_from_band(filepath, self.band)[0]
        image_attr = filename_parse(filepath)
        #print('image attribute {}, image path {}'.format(), image_attr, filename)
        image = self.get_image(filepath)
        label = image_attr['id'] - 1
        # RandomFlipLeftRight(image, self.mode)
        image = np.concatenate((image, image, image), axis=-1)
        image = ToTensor()(image)
        return image, label, image_attr

    def __len__(self):
        return len(self.filenames)


if __name__ == '__main__':
    DATASET_PATH = '/home/lilium/myDataset/ECUST/'  # Your HyperECUST dataset path

    n_channels = 46
    waveLen = [550 + 10 * i for i in range(46)]

    # Example of constructing the trainset and validset
    batch_size = 8
    #train_list, valid_list = split_dataset(DATASET_PATH)
    train_path = '/home/lilium/yrc/myFile/huaweiproj/code/louishsu/recognize/split/split_1/train.txt'
    valid_path = '/home/lilium/yrc/myFile/huaweiproj/code/louishsu/recognize/split/split_1/valid.txt'
    with open(train_path, 'r') as f:
        train_list = f.readlines()
    with open(valid_path, 'r') as f:
        valid_list = f.readlines()

    trainset = HyperECUST_SingleChannel(DATASET_PATH, train_list, 550,
                                        facesize=(128, 128), cropped_by_bbox=False, mode='train')
    validset = HyperECUST_SingleChannel(DATASET_PATH, valid_list, 550,
                                        facesize=(128, 128), cropped_by_bbox=False, mode='valid')
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    validloader = DataLoader(validset, batch_size=1, shuffle=True)

    image, label, image_attr = trainset[10]
    print('trainset length {} validset length {}'.format(
        len(trainset), len(validset)))
    print('image attribute: {}'.format(image_attr))
    print(image.shape)
    print(label)
    show_result(image.numpy().transpose(1, 2, 0))
