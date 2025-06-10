import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableWeightScalingLoss(nn.Module):
    def __init__(self, num_classes=7):
        super(LearnableWeightScalingLoss, self).__init__()
        self.learned_norm = nn.Parameter(torch.ones(1, num_classes))

    def forward(self, x):
        return self.learned_norm * x
