import os

def gen_casia_label(prefix='../data/CASIA-WebFace', label = '../data/CASIA_label.txt'):
    """
    Notes:
        结果保存在`../data/CASIA_label.txt`，格式为
        `filepath label`
    """
    index = sorted(list(set(map(int, os.listdir(prefix)))))
    index = list(map(lambda x: '{:07d}'.format(x), index))
    index_dict = dict(zip(index, range(len(index))))

    f = open(label, 'w')
    for k, v in index_dict.items():
        subdir = os.path.join(prefix, k)
        for filename in os.listdir(subdir):
            line = '{:s} {:d}\n'.format('/'.join([k, filename]), v)
            f.write(line)
    f.close()

if __name__ == "__main__":
    gen_casia_label()
    