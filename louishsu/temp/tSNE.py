import numpy as np
from matplotlib import pyplot as plt
from scipy import io
from sklearn.manifold import TSNE

getId = lambda x: int(x.split('/')[1]) 

matfile = 'valid_result.mat'

result = io.loadmat(matfile)

featureLs = result['featureLs']
featureRs = result['featureRs']
filenameLs = result['filenameLs']
filenameRs = result['filenameRs']

features = np.r_[featureLs, featureRs]
filenames = np.r_[filenameLs, filenameRs]

_, index = np.unique(filenames, return_index=True)
features = features[index]
filenames = filenames[index]
labels = np.array(list(map(getId, filenames)))

tsne = TSNE()
features2D = tsne.fit_transform(features)

plt.figure()
plt.scatter(features2D[:, 0], features2D[:, 1], c=labels)
plt.show()
