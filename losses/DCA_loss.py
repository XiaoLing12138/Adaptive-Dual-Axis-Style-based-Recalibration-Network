import torch
from torch import nn
import torch.nn.functional as F


class DCALoss(nn.Module):
    """
    This is for DCA loss
    """
    def __init__(self):
        super(DCALoss, self).__init__()

    # here we implemented step by step for corresponding to our formula
    def forward(self, inputs, targets):
        batch_size = inputs.size(0)
        predicted_values = F.softmax(inputs, 1)

        probability, predict_label = torch.max(predicted_values, dim=1)

        accuracies = predict_label.eq(targets)
        dca = torch.abs(torch.mean(probability) - torch.sum(accuracies)/batch_size)

        return dca










