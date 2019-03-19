import torch.nn as nn
from .multi_crossentropy import MultiCrossEntropyLoss

_losses = {
    'crossent': nn.CrossEntropyLoss(),
    'multi_crossent': MultiCrossEntropyLoss(),
}