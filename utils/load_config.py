import os
import yaml

import torch


class Config:
    def __init__(self, in_dict: dict):
        assert isinstance(in_dict, dict)

        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
                setattr(self, key, [Config(x) if isinstance(x, dict) else x for x in val])
            else:
                setattr(self, key, Config(val) if isinstance(val, dict) else val)


def read_yaml(file_path):
    assert os.path.isfile(file_path), 'yaml file is not exist: {}'.format(file_path)
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def load_config(yaml_path):
    config_dict = read_yaml(yaml_path)
    config_dict['device'] = torch.device(config_dict.get('device', 'cpu'))
    config_dict['model']['device'] = config_dict['device']
    config_dict['model']['img_size'] = config_dict['imgH'], config_dict['imgW']
    return Config(config_dict)
