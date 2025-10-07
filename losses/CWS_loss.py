import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class CWSLoss(nn.Module):
    def __init__(self):
        super(CWSLoss, self).__init__()

    def forward(self, inputs, targets):
        batch_size = inputs.size(0)
        class_num = inputs.size(1)
        predicted_values = F.softmax(inputs, 1)

        # one-hot encoding
        y_zerohot = torch.zeros(batch_size, class_num).scatter_(1, targets.view(batch_size, 1).data.cpu(), 1)
        temp_output = torch.zeros(class_num)

        for i in range(class_num):
            if torch.sum(y_zerohot[:, i]) > 0:
                temp_output[i] = -torch.log(
                    torch.sum((predicted_values[:, i] + 1e-7) * y_zerohot[:, i].cuda()) / torch.sum(y_zerohot[:, i]))

        loss = temp_output.mean()

        return loss

