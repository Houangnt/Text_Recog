import torch
import torch.nn as nn

__all__ = ['ctc']


class CTC(nn.Module):
    def __init__(self, input_size, num_classes, device=torch.device('cuda')):
        super(CTC, self).__init__()
        self.net = nn.Linear(input_size, num_classes)

    def forward(self, x, text=None, **kwargs):
        return self.net(x.contiguous())


def ctc(input_size, num_classes, device=torch.device('cuda')):
    return CTC(input_size, num_classes, device=device)
