import torch.nn as nn
import torch.nn.init as init

import network.transformation.get_module as trans
import network.feature_extraction.get_module as fe
import network.sequence_modeling.get_module as sm
import network.prediction.get_module as prd


class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        model_opt = validated_config(opt)

        self.transformation = getattr(trans, model_opt.transformation.name, none_module)(model_opt)
        self.feature_extraction = getattr(fe, model_opt.feature_extraction.name)(model_opt)
        self.sequence_modeling = getattr(sm, model_opt.sequence_modeling.name, none_module)(model_opt)
        self.prediction = getattr(prd, model_opt.prediction.name)(model_opt)

    def forward(self, x, prediction=True, **kwargs):
        out = self.transformation(x)
        out = self.feature_extraction(out)
        out = self.sequence_modeling(out, **kwargs)
        if prediction:
            out = self.prediction(out, batch_max_length=self.opt.batch_max_length, **kwargs)
        return out

    def weight_initialization(self):
        for name, param in self.named_parameters():
            if 'localization_fc2' in name:
                print(f'Skip {name} as it is already initialized')
                continue
            try:
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)
            except Exception as e:  # for batchnorm.
                if 'weight' in name:
                    param.data.fill_(1)
                continue


def validated_config(opt):
    model_opt = opt.model
    assert model_opt.transformation.name.lower() in trans.list_modules, \
        'transformation module is not supported <{}>'.format(model_opt.transformation.name)
    assert model_opt.feature_extraction.name.lower() in fe.list_modules, \
        'feature_extraction module is not supported <{}>'.format(model_opt.feature_extraction.name)
    assert model_opt.sequence_modeling.name.lower() in sm.list_modules, \
        'sequence_modeling module is not supported <{}>'.format(model_opt.sequence_modeling.name)
    assert model_opt.prediction.name.lower() in prd.list_modules, \
        'prediction module is not supported <{}>'.format(model_opt.prediction.name)
    model_opt.device = opt.device
    return model_opt


def none_module(opt):
    return nn.Sequential()
