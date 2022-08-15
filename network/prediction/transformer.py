import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['transformer']


class TransformerPredictor(nn.Module):
    def __init__(self, prediction_layer, device=torch.device('cuda')):
        """
        :param prediction_layer: Ouput layer
        :param device: CPU or GPU
        """
        super(TransformerPredictor, self).__init__()
        self.prediction_layer = prediction_layer
        self.device = device

    def forward(self, x, **kwargs):
        """
        :param x: the input
        :return:
        """
        ltr_input, rtl_input = x

        if rtl_input is None:
            ltr = self.prediction_layer(x)
            return ltr, None
        else:
            ltr = self.prediction_layer(ltr_input)
            rtl = self.prediction_layer(rtl_input)

            return ltr, rtl


class PredictionLayer(nn.Module):
    """
    Define standard linear feed forward + softmax for final prediction layer.
    """

    def __init__(self, d_model, vocab):
        """

        :param d_model: model dimensionality
        :param vocab: output vocabulary size
        """

        super(PredictionLayer, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x, **kwargs):
        """

        :param x: input tensor
        :return: output distribution
        """
        return F.log_softmax(self.proj(x), dim=-1)


def transformer(opt):
    return TransformerPredictor(PredictionLayer(opt.sequence_modeling.d_model, opt.num_classes), opt.device)
