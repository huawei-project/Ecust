import numpy as np
from collections import OrderedDict
from sklearn.decomposition import PCA

class NDarrayPCA(object):
    """ PCA for ndarray

    Attributes:
        n_dims:         {int}               number of dimension of input data
        n_components:   {list[int/None]}    number of components of each dimension
        decomposers:    {OrderedDict}       index: PCA
    """
    def __init__(self, n_components=None):

        self.n_dims = None
        self.n_components = [] if n_components is None else n_components
        self.decomposers = OrderedDict()

    def fit(self, X):
        """
        Params:
            X: {ndarray(n_samples, d0, d1, d2, ..., dn-1)} n-dim array
        """

        self.n_dims = len(X.shape) - 1
        idx = [i for i in range(len(X.shape))]   # index of dimensions

        for i_dim in range(self.n_dims):
            self.decomposers[i_dim] = PCA(n_components=self.__n_components(i_dim))
            
            ## transpose tensor
            idx[-1], idx[i_dim + 1] = idx[i_dim + 1], idx[-1]
            X = X.transpose(idx)
            shape = list(X.shape)

            # 1-dim pca
            X = X.reshape((-1, shape[-1]))
            X = self.decomposers[i_dim].fit_transform(X)

            ## transpose tensor
            X = X.reshape(shape[:-1]+[X.shape[-1]])
            X = X.transpose(idx)
            idx[-1], idx[i_dim + 1] = idx[i_dim + 1], idx[-1]

    def transform(self, X):
        """
        Params:
            X: {ndarray(n_samples, d0, d1, d2, ..., dn-1)} n-dim array
        Returns:
            X: {ndarray(n_samples, d0, d1, d2, ..., dn-1)} n-dim array
        """

        assert self.n_dims == len(X.shape) - 1, 'please check input dimension! '
        idx = [i for i in range(len(X.shape))]   # index of dimensions

        for i_dim in range(self.n_dims):
            
            ## transpose tensor
            idx[-1], idx[i_dim + 1] = idx[i_dim + 1], idx[-1]
            X = X.transpose(idx)
            shape = list(X.shape)

            # 1-dim pca
            X = X.reshape((-1, shape[-1]))
            X = self.decomposers[i_dim].transform(X)

            ## transpose tensor
            X = X.reshape(shape[:-1]+[X.shape[-1]])
            X = X.transpose(idx)
            idx[-1], idx[i_dim + 1] = idx[i_dim + 1], idx[-1]
        
        return X
    
    def fit_transform(self, X):
        self.fit(X)
        X = self.transform(X)
        return X
    
    def __n_components(self, index):
        
        try:
            return self.n_components[index]
        except IndexError:
            return None


if __name__ == "__main__":
    import cv2
    from matplotlib import pyplot as plt

    X = np.random.randn(5, 64, 64, 23)

    # show first image
    plt.figure(0)
    plt.imshow(X[0, :, :, 0])

    decomposer = NDarrayPCA(n_components=[None, None, 15])
    decomposer.fit(X)

    X_transformed = decomposer.transform(X)

    # show first decomposed image
    plt.figure(1)
    plt.imshow(X_transformed[0, :, :, 0])

    plt.show()