import os
import numpy as np

import torch
import torch.nn as nn


metrics = {
    "crossent": nn.CrossEntropyLoss(),
}