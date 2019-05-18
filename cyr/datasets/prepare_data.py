import os
import sys
import numpy as np
import glob
import cv2


def get_id(x): return int(x.split('/')[6])


def get_band(x): return int(x.split('_')[-1].split('.')[0])


def get_vol(i): return (i - 1) // 10 + 1


def getDicts(dataset_path):
    dicts = dict()
    for vol in ["DATA%d" % _ for _ in range(1, 10)]:
        txtfile = os.path.join(dataset_path, vol, "detect.txt")
        if os.path.isfile(txtfile):
            with open(txtfile, 'r') as f:
                dicts[vol] = eval(f.read())
    return dicts


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
        parse_dict['ob'] = 'nonob'
    parse_dict['pose'] = filename.split('_')[1]
    parse_dict['glasses'] = filename.split('_')[3].split('/')[0]
    parse_dict['band'] = get_band(filename)
    parse_dict['id'] = get_id(filename)
    return parse_dict


def get_bbox(filename):
    image_attr = filename_parse(filename)
    # get bbox
    vol = "DATA%d" % get_vol(image_attr['id'])
    imgname = filename[filename.find("DATA") + 5:]
    if 'RGB' in filename:
        dirname = imgname.split('.')[0]
    else:
        dirname = '/'.join(imgname.split('/')[:-1])
    bbox = dicts[vol][dirname][1]
    square_bbox = convert_to_square(np.array([bbox]))
    return square_bbox


def get_image(filename, facesize, cropped_by_bbox=True):
    # load image array
    if 'RGB' in filename:
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if cropped_by_bbox:
        x1, y1, x2, y2 = get_bbox(filename)[0]
        h, w = image.shape[:2]
        x1, x2 = max(0, x1), min(x2, w)
        y1, y2 = max(0, y1), min(y2, h)
        image = image[y1: y2, x1: x2]
    if facesize is not None:
        image = cv2.resize(image, facesize[::-1])
    return image


if __name__ == '__main__':
    DATASET_PATH = '/home/lilium/myDataset/ECUST/'  # Your HyperECUST dataset path

    multi_paths = glob.glob(os.path.join(
        DATASET_PATH, 'DATA*/*/Multi/*/**/*.bmp'), recursive=True)
    rgb_paths = glob.glob(os.path.join(
        DATASET_PATH, 'DATA*/*/RGB/**/*.JPG'), recursive=True)

    facesize = (112, 96)
    dicts = getDicts(DATASET_PATH)

    data_list = multi_paths + rgb_paths
    data_list = sorted(data_list)
    for i in range(len(data_list)):
        image = get_image(data_list[i], facesize=facesize)
        save_path = data_list[i].replace(
            'ECUST', 'ECUST_{}x{}'.format(*facesize))
        save_path = save_path.replace('bmp', 'JPG')
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cv2.imwrite(save_path, image)
        print('image {}, saved in {}'.format(i, save_path))
