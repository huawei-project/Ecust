import os

from detect import init_detector, detect_size, listFiles, detect_statistic

def main_size():
    detector = init_detector()
    filelist = listFiles()

    for dsize in [(1648, 1236), (1400, 1050), (1200, 900), (1000, 750), (800, 600), (600, 450), (400, 300), (200, 150)]:
        print("dsize: {}x{}".format(dsize[0], dsize[1]))
        # detect_size(detector, filelist, dsize)
        multi_image_ratio, rgb_image_ratio, multi_tensor_ratio, rgb_tensor_ratio = detect_statistic(dsize)
        print("{:.4%}\n{:.4%}\n{:.4%}\n{:.4%}\n".format(multi_image_ratio, rgb_image_ratio, multi_tensor_ratio, rgb_tensor_ratio))

if __name__ == "__main__":
    main_size()
    