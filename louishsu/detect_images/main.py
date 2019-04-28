import os

from detect import init_detector, detect_size, listFiles

def main_size():
    detector = init_detector()
    filelist = listFiles()

    for dsize in [(1648, 1236), (1400, 1050), (1200, 900), (1000, 750), (800, 600), (600, 450), (400, 300), (200, 150)]:
        detect_size(detector, filelist, dsize)

if __name__ == "__main__":
    main_size()
    