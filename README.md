# Text Recognition
Training flow and programs structure are based on [ClovaAI's four-stage STR framework](https://github.com/clovaai/deep-text-recognition-benchmark) implementation.

## Getting Started
.

## Configurations
### General configuration
...
### Network modules configuration
#### Transformation
- SPIN: {name: spin, k:...}
- TPS: {name: tps, num_fiducial:...}
- SPIN w TPS: Coming soon

#### Feature Extraction
- VGG: vgg11, vgg16, vgg19.
- ResNet: resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202
- RCNN: rcnn

#### Sequence Modeling
- CascadeRNN BiLSTM: cascade_rnn
- Transformer: transformer

#### Prediction
- Attention: attn
- CTC: ctc
- Transformer: transformer


## Training
```
python train.py \
--train_data=<train_data_dir> --valid_data=<val_data_dir> \
--select_data=<data_dir_name> --batch_ratio=1.0 \
--exp_name=<exp_name> \
--num_iter=100000 --batch_size=500 \
--model_config=./configs/<config_file>
```
