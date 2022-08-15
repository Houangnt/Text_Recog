import torch
import torch.nn as nn
import torch.nn.init as init

from .bilstm import BidirectionalLSTM

__all__ = ['cascade_rnn']


class CascadeRNN(nn.Module):
    """ Cascade Recurrent Structure"""

    def __init__(self, rnn_modules, repeat=None, mode=1, ):
        """

        Args:
            rnn_modules (list): stack structure(RNN|LSTM)
            repeat (int): repeat times
            mode (int): use mode,  0 - without mask, output dim [batch_size x C x height x width]
                                   1 - with mask, output dim [batch_size x T x output_size]

        """
        super(CascadeRNN, self).__init__()
        if repeat:
            assert len(rnn_modules) == 1 or len(rnn_modules) == repeat
            modules = []
            for _ in range(repeat):
                # build the rnn_modules
                modules.append(rnn_modules[0])
        else:
            # build the rnn_module
            modules = rnn_modules

        self.SequenceModeling = nn.Sequential(*modules)
        self.mode = mode

    def init_weights(self, pretrained=None):
        """

        Args:
            pretrained (str): model path of the pre_trained model

        Returns:

        """
        if isinstance(pretrained, str):
            checkpoint = torch.load(pretrained)
            self.load_state_dict(checkpoint['model_state_dict'])
        elif pretrained is None:
            for name, param in self.SequenceModeling.named_parameters():
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)

        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, inputs, mask=None, **kwargs):
        """

        Args:
            inputs (torch.Tensor): input feature
            mask (torch.Tensor): input mask, according to the structure and specify the corresponding mask
        Returns:
            torch.Tensor: output feature of the stack structure

        """

        if self.mode:
            output = self.SequenceModeling(inputs)
            if mask is not None:
                output = inputs
                for module in self.SequenceModeling:
                    output = module(output, mask=mask)
            return output

        output = self.SequenceModeling(inputs.contiguous().view(inputs.size(0),
                                                                inputs.size(1), -1).permute(0, 2, 1))
        return output.permute(0, 2, 1).unsqueeze(2)


class CascadeCNN(nn.Module):
    """ Cascade Convolution Structure"""

    def __init__(self, cnn_modules, repeat=None, mode=0):
        """

        Args:
            cnn_modules (list): stack structure(CNN)
            repeat (int): repeat times
            mode (int): use mode,  0 - without mask, output dim [batch_size x C x height x width]
                                   1 - with mask, output dim [batch_size x T x output_size]
        """
        super(CascadeCNN, self).__init__()

        if repeat:
            assert len(cnn_modules) == 1 or len(cnn_modules) == repeat
            modules = []
            for _ in range(repeat):
                modules.append(cnn_modules[0])
        else:
            modules = cnn_modules

        self.SequenceModeling = nn.Sequential(*modules)
        self.mode = mode

    def init_weights(self, pretrained=None):
        """

        Args:
            pretrained (str): model path of the pre_trained model

        Returns:

        """
        if isinstance(pretrained, str):
            checkpoint = torch.load(pretrained)
            self.load_state_dict(checkpoint['model_state_dict'])
        elif pretrained is None:
            for name, param in self.SequenceModeling.named_parameters():
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    if 'bn' in name:
                        init.constant_(param, 1.0)
                    else:
                        init.kaiming_normal_(param)

        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, input_feature):
        """

        Args:
            (torch.Tensor): input feature

        Returns:
            torch.Tensor: output feature of the stack structure

        """

        if self.mode:
            output = self.SequenceModeling(input_feature.unsqueeze(-1).permute(0, 2, 1, 3))
            return output.permute(0, 2, 1, 3).squeeze(3)

        output = self.SequenceModeling(input_feature)
        return output


def cascade_rnn(opt, **kargs):
    rnn_modules = [
        BidirectionalLSTM(input_size=opt.hidden_channel, hidden_size=256, output_size=256, with_linear=True,
                          bidirectional=True),
        BidirectionalLSTM(input_size=256, hidden_size=256, output_size=opt.hidden_channel, with_linear=True,
                          bidirectional=True)]
    return CascadeRNN(rnn_modules=rnn_modules)
