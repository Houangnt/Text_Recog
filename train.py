import os
import sys
import time
import random
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np

from utils.load_config import load_config
from utils.dataset import hierarchical_dataset, AlignCollate, BatchBalancedDataset
from utils.logging import Logging
from utils.utils import concat_ltr_rtl
import utils.converter as converter_pgk
import network.optimizer as losses_pkg
from network.model import Model
from validation import validation


def get_data_loader(args, opt, logger):
    train_dataset = BatchBalancedDataset(args, opt)

    collage_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.pad)
    valid_dataset, valid_dataset_log = hierarchical_dataset(root=args.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=int(args.workers), collate_fn=collage_valid, pin_memory=True)
    logger.save_data_log(valid_dataset_log)
    logger.save_args_log(args)
    return train_dataset, valid_loader


def save_model(model, save_path, current_score=None, best_score=None):
    if current_score is not None:
        if current_score >= best_score:
            best_score = current_score
            torch.save(model.state_dict(), save_path)
        return best_score
    else:
        torch.save(model.state_dict(), save_path)


def training(opt, loss_compute, model, image, text, length, batch_size):
    assert opt.model.prediction.name in ['transformer', 'ctc', 'baidu_ctc', 'attn'], \
        print(f'predicter is not supported: {opt.model.prediction.name}')
    if 'transformer' in opt.model.prediction.name:
        ltr_target, rtl_target, target_y, target_mask, targets_embedding_mask = text
        target_mask = concat_ltr_rtl(target_mask, target_mask)
        ltr, rtl = model(image, src_mask=None, tgt_embedding_mask=targets_embedding_mask, ltr_targets=ltr_target,
                         rtl_targets=rtl_target)
        out = concat_ltr_rtl(ltr, rtl)
        train_loss = loss_compute(out, target_y, tgt_mask=target_mask)
    else:
        if 'ctc' in opt.model.prediction.name:
            preds = model(image)
        else:  # 'attn' in opt.model.prediction.name:
            preds = model(image, prediction = True, text=text[:, :-1], is_train=True)  # align with Attention.forward
        train_loss = loss_compute(preds, text, batch_size, length)

    return train_loss


def train(args, opt):
    logger = Logging(args.exp_name)
    train_dataset, valid_loader = get_data_loader(args, opt, logger)

    # Model configuration
    converter = getattr(converter_pgk, opt.model.prediction.name)(opt)
    opt.model.num_classes = len(converter.character)

    if opt.rgb:
        opt.model.nc = 3
    model = Model(opt)

    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).to(opt.device)
    model.train()
    if args.saved_model != '':
        print(f'loading pretrained model from {args.saved_model}')
        if args.FT:
            model.load_state_dict(torch.load(args.saved_model), strict=False)
        else:
            model.load_state_dict(torch.load(args.saved_model))

    # Setup loss
    criterion = getattr(losses_pkg, opt.model.prediction.name)(opt)

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda x: x.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))

    # setup optimizer
    if opt.train.optimizer == 'adam':
        optimizer = optim.Adam(filtered_parameters, lr=opt.train.lr, betas=(opt.train.beta1, 0.999))
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.train.lr, rho=opt.train.rho, eps=opt.train.eps)

    if 'transformer' in opt.model.prediction.name:
        loss_compute = losses_pkg.TransformerLossCompute(criterion, optimizer, opt.train)
    else:
        loss_compute = losses_pkg.LossCompute(opt.model.prediction.name, criterion, optimizer)

    print("Optimizer:", optimizer)

    # Start training
    start_iter = 0
    if args.saved_model != '':
        try:
            start_iter = int(args.saved_model.split('_')[-1].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except ValueError:
            print(f'cannot get start_iter from {args.saved_model}, set start_iter = 0')

    start_time = time.time()
    best_accuracy = -1
    best_norm_ed = -1

    for iteration in range(start_iter, args.num_iter):
        # train part
        image_tensors, labels = train_dataset.get_batch()
        image = image_tensors.to(opt.device)
        target, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
        train_loss = training(opt, loss_compute, model, image, target, length, image.size(0))
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.train.grad_clip)  # gradient clipping with 5 (Default)

        # validation part
        if (iteration + 1) % args.valInterval == 0 or iteration == 0:
            elapsed_time = time.time() - start_time

            model.eval()
            with torch.no_grad():
                valid_loss, current_acc, current_norm_ed, preds, confs, labels, _, _ = \
                    validation(model, criterion, valid_loader, converter, opt)
            model.train()

            # keep best accuracy model (on valid dataset)
            best_accuracy = save_model(model, f'{args.save_dir}/best_accuracy.pth', current_acc, best_accuracy)
            best_norm_ed = save_model(model, f'{args.save_dir}/best_norm_ED.pth', current_norm_ed, best_norm_ed)
            loss_model_log = logger.write_training_log(iteration, args.num_iter, train_loss, valid_loss,
                                                       elapsed_time, current_acc, current_norm_ed, best_accuracy,
                                                       best_norm_ed)
            # show some predicted results
            predicted_result_log = logger.write_validate_log(labels, preds, confs, opt.model.prediction.name)

            print(loss_model_log)
            print(predicted_result_log)

        # save model per 1e+5 iter.
        if (iteration + 1) % 1e+5 == 0:
            save_model(model, f'{args.save_dir}/iter_{iteration + 1}.pth')

    print('end the training')
    sys.exit()


def main():
    # Model Architecture
    args = parser.parse_args()
    args.select_data = args.select_data.split('-')
    args.batch_ratio = args.batch_ratio.split('-')

    os.makedirs(f'./saved_models/{args.exp_name}', exist_ok=True)
    args.save_dir = f'./saved_models/{args.exp_name}'

    opt = load_config(args.model_config)

    # Vocab / character number configuration
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    # Seed and GPU setting
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    torch.cuda.manual_seed(args.manual_seed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    args.num_gpu = torch.cuda.device_count()

    if args.num_gpu > 1:
        args.workers = args.workers * args.num_gpu
        args.batch_size = args.batch_size * args.num_gpu

    train(args, opt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', required=True, help='Where to store logs and models')
    parser.add_argument('--train_data', required=True, help='path to training dataset')
    parser.add_argument('--valid_data', required=True, help='path to validation dataset')
    parser.add_argument('--model_config', required=True, help='path to yaml model_config')

    parser.add_argument('--manual_seed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=100000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=2000, help='Interval between each validation')
    parser.add_argument('--saved_model', default='', help="path to model to continue training")
    parser.add_argument('--FT', action='store_true', help='whether to do fine-tuning')
    parser.add_argument('--select_data', type=str, default='MJ-ST',
                        help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
    parser.add_argument('--batch_ratio', type=str, default='0.5-0.5',
                        help='assign ratio for each selected data in the batch')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    main()
