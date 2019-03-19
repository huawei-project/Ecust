import torch
import torch.nn as nn

class SpectralAnalysis(nn.Module):
    def __init__(self, in_channels, n_classes, model):
        super(SpectralAnalysis, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.model = model
    def forward(self, x):
        """
        Params:
            x:   {tensor(N, C, H, W)}
        Returns:
            out: {tensor(N, C, n_classes)}
        """
        (N, C) = x.shape[:2]
        out = torch.zeros([N, C, self.n_classes])
        for c in range(C):
            _x = torch.unsqueeze(x[:, c], 1)
            out[:, c] = self.model(_x)
        return out


def summary_out(out):
    """
    Params:
        out:    {tensor(N, C, n_classes)}
    Returns:
        mean:   {ndarray(N, C)}
        var:    {ndarray(N, C)}
        loss:   {ndarray()}
    """
    pass