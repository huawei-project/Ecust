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


def get_id(x): return int(x.split('/')[2])


def get_band(x): return int(x.split('_')[-1].split('.')[0])


def get_vol(i): return (i - 1) // 10 + 1


def get_wavelen(bmp): return int(bmp.split('.')[0].split('_')[-1])


def get_path_from_condition(data_list, condition):
    return list(filter(lambda x: condition in x, data_list))


def get_path_from_id(data_list, ID):
    return list(filter(lambda x: int(x.split('/')[2]) == ID, data_list))


def get_path_from_band(data_list, band):
    return list(filter(lambda x: int(x.split('_')[-1].split('.')[0]) == band, data_list))


def get_path_from_band_range(data_list, low, high):
    return list(filter(lambda x: low <= int(x.split('_')[-1].split('.')[0]) <= high, data_list))


def getDicts(dataset_path):
    dicts = dict()
    for vol in ["DATA%d" % _ for _ in range(1, 5)]:
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


def split_dataset_2input():
    multi_paths1 = glob.glob(os.path.join(
        DATASET_PATH, 'DATA*/*/Multi/non-obtructive/*/*.bmp'))
    multi_paths2 = glob.glob(os.path.join(
        DATASET_PATH, 'DATA*/*/Multi/obtructive/*/*/*.bmp'))
    multi_paths = multi_paths1 + multi_paths2
    index = multi_paths[0].find('/DATA')
    multi_paths = sorted([x[index:] for x in multi_paths])

    rgb_paths1 = glob.glob(os.path.join(
        DATASET_PATH, 'DATA*/*/RGB/non-obtructive/*.JPG'))
    rgb_paths2 = glob.glob(os.path.join(
        DATASET_PATH, 'DATA*/*/RGB/obtructive/*/*.JPG'))
    rgb_paths = rgb_paths1 + rgb_paths2
    index = rgb_paths[0].find('/DATA')
    rgb_paths = sorted([x[index:] for x in rgb_paths])

    print('Multispectral images: {}, rgb images: {}'.format(
        len(multi_paths), len(rgb_paths)))

    nonob_set = get_path_from_condition(multi_paths, 'non-ob')
    print('Multispectral non-ob images: {}'.format(len(nonob_set)))
    trainset1 = []
    trainset2 = []
    for i in range(1, n_subject + 1):
        temp_set = get_path_from_id(nonob_set, i)
        for j in range(len(temp_set) // n_channels):
            sample = temp_set[j * 46:j * 46 + 46]
            trainset1.append(get_path_from_band_range(sample, 700, 790))
            trainset2.append(get_path_from_band_range(sample, 800, 890))
    print('trainset1: {}, trainset2: {}'.format(len(trainset1), len(trainset2)))

    ob_set = get_path_from_condition(multi_paths, '/ob')
    print('Multispectral ob images: {}'.format(len(ob_set)))
    validset1 = []
    validset2 = []
    for i in range(1, n_subject + 1):
        temp_set = get_path_from_id(ob_set, i)
        for j in range(len(temp_set) // n_channels):
            sample = temp_set[j * 46:j * 46 + 46]
            validset1.append(get_path_from_band_range(sample, 700, 790))
            validset2.append(get_path_from_band_range(sample, 800, 890))
    print('validset1: {}, validset2: {}'.format(len(validset1), len(validset2)))

    return trainset1, trainset2, validset1, validset2


class HyperECUST_2input(Dataset):
    def __init__(self, data_path, filenames1, filenames2, facesize=None, mode='train'):
        """
        Params:
            facesize:   {tuple/list[H, W]}
            mode:       {str} 'train', 'valid'
        """
        self.mode = mode
        self.data_path = data_path
        self.filenames1 = filenames1
        self.filenames2 = filenames2
        self.facesize = tuple(facesize)
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

    def get_multi_image(self, image_list, cropped_by_bbox=True):
        """
        Params:
                filename 
        Return:
                image: {np.ndarray} shape H x W x 1
        """
        # 载入图像
        c = len(image_list)
        if self.facesize is None:
            img = cv2.imread(os.path.join(
                self.data_path, image_list[0][1:]), cv2.IMREAD_GRAYSCALE)
            (h, w) = img.shape
        else:
            (h, w) = self.facesize
        imgs = np.zeros(shape=(h, w, c))
        for i in range(c):
            if cropped_by_bbox:
                x1, y1, x2, y2 = self.get_bbox(image_list[i])[0]
                img = cv2.imread(os.path.join(
                    self.data_path, image_list[i][1:]), cv2.IMREAD_GRAYSCALE)[y1:y2, x1:x2]
            else:
                img = cv2.imread(os.path.join(
                    self.data_path, image_list[i][1:]), cv2.IMREAD_GRAYSCALE)
            imgs[:, :, i] = img if (
                self.facesize is None) else cv2.resize(img, self.facesize)
        return imgs

    def __getitem__(self, index):
        image_list1 = self.filenames1[index]
        image_list2 = self.filenames2[index]
        # load image array
        image1 = self.get_multi_image(image_list1)
        image2 = self.get_multi_image(image_list2)
        image1 = ToTensor()(image1)
        image2 = ToTensor()(image2)

        label = get_id(image_list1[0])
        return image1, image2, label, image_list1, image_list2

    def __len__(self):
        return len(self.filenames1)


if __name__ == '__main__':

    DATASET_PATH = '/home/lilium/myDataset/ECUST/'  # Your HyperECUST dataset path

    n_channels = 46
    waveLen = [550 + 10 * i for i in range(46)]
    n_subject = 33

    # Example of constructing the trainset and validset whose input includes 2 images of 10 channels
    batch_size = 8
    train_list1, train_list2, valid_list1, valid_list2 = split_dataset_2input()
    # train set
    trainset = HyperECUST_2input(DATASET_PATH, train_list1, train_list2,
                                 facesize=(128, 128), mode='train')
    validset = HyperECUST_2input(DATASET_PATH, valid_list1, valid_list2,
                                 facesize=(128, 128), mode='valid')

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    validloader = DataLoader(validset, batch_size=1, shuffle=True)

    image1, image2, label, image_list1, image_list2 = trainset[10]
    print('image1 shape: {}, image2 shape: {}, label: {}'.format(
        image1.shape, image2.shape, label))
    print('\n')
    print('image1 path: {}'.format(image_list1))
    print('\n')
    print('image2 path: {}'.format(image_list2))
    # show_result(image)
