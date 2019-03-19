import torch.nn as nn

class MultiCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MultiCrossEntropyLoss, self).__init__()
        self.crossent = nn.CrossEntropyLoss()
    def forward(self, y_pred_prob, y_true):
        """
        Params:
            y_pred_prob:    {tensor(N, C, n_classes)}
            y_true:         {tensor(N,    n_classes)}
        Returns:
            out:            {tensor(1)}
        """
        C = y_pred_prob.shape[1]
        out = 0
        for c in range(C):
            y_pred = y_pred_prob[:, c]
            out += self.crossent(y_pred, y_true)
        out /= C
        return out

        