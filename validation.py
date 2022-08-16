import time
import re

import torch
import torch.utils.data
import torch.nn.functional as F
from nltk.metrics.distance import edit_distance

from network.optimizer import LossesAverager

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def validation(model, criterion, evaluation_loader, converter, opt):
    """ validation or evaluation """
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = LossesAverager()

    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        image = image_tensors.to(device)
        length_of_data += image_tensors.size(0)
        start_time = time.time()
        if 'transformer' in opt.model.prediction.name:
            memories = model(image, src_mask=None, decode=False, prediction=False)
            preds_str = []
            preds_max_prob = []
            for memory in memories:
                memory = memory.unsqueeze(0)
                ltr_targets, rtl_targets = converter.get_init_target(), converter.get_init_target()
                ltr_result, rtl_result = '', ''
                ltr_probability, rtl_probability = 1, 1
                for char_index in range(opt.batch_max_length):
                    out = model.module.sequence_modeling(memory, src_mask=None, ltr_targets=ltr_targets,
                                     rtl_targets=rtl_targets,
                                     ltr_tgt_mask=converter.subsequent_mask(ltr_targets.shape[-1]).to(device),
                                     rtl_tgt_mask=converter.subsequent_mask(rtl_targets.shape[-1]).to(device),
                                     encode=False)
                    ltr, rtl = model.module.prediction(out)

                    ltr_result, ltr_probability, ltr_targets, ltr_ended = converter.greedy_decode(ltr, ltr_targets,
                                                                                                  ltr_result,
                                                                                                  ltr_probability)
                    rtl_result, rtl_probability, rtl_targets, rtl_ended = converter.greedy_decode(rtl, rtl_targets,
                                                                                                  rtl_result,
                                                                                                  rtl_probability)
                    if ltr_ended and rtl_ended:
                        break

                preds_str.append(ltr_result if ltr_probability >= rtl_probability else rtl_result[::-1])
                preds_max_prob.append(max(ltr_probability, rtl_probability))
            forward_time = time.time() - start_time
            cost = torch.tensor([0])
        else:
            # For max length prediction
            batch_size = image_tensors.size(0)
            length_of_data = length_of_data + batch_size
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)
            if 'ctc' in opt.model.prediction.name:
                preds = model(image, text_for_pred, is_train=False)
                forward_time = time.time() - start_time
                # Calculate evaluation loss for CTC deocder.
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                cost = criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)

                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index.data, preds_size.data)

            else:  # 'attn' in opt.model.prediction.name:
                preds = model(image, prediction = True, text = text_for_pred, is_train=False)
                forward_time = time.time() - start_time

                preds = preds[:, :text_for_loss.shape[1] - 1, :]
                target = text_for_loss[:, 1:]  # without [GO] Symbol
                cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)
                labels = converter.decode(text_for_loss[:, 1:], length_for_loss)

            # calculate accuracy & confidence score
            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)

        infer_time += forward_time
        valid_loss_avg.add(cost)

        confidence_score_list = []
        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            if 'attn' in opt.model.prediction.name:
                gt = gt[:gt.find('[s]')]
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

            # To evaluate 'case sensitive model' with alphanumeric and case insensitve setting.
            if opt.sensitive:
                pred = pred.lower()
                gt = gt.lower()
                alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
                out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
                pred = re.sub(out_of_alphanumeric_case_insensitve, '', pred)
                gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)

            if pred == gt:
                n_correct += 1

            # ICDAR2019 Normalized Edit Distance
            if len(gt) == 0 or len(pred) == 0:
                norm_ED += 0
            elif len(gt) > len(pred):
                norm_ED += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ED += 1 - edit_distance(pred, gt) / len(pred)

            # calculate confidence score (= multiply of pred_max_prob)
            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
            confidence_score_list.append(confidence_score)
            # print(pred, gt, pred==gt, confidence_score)

    accuracy = n_correct / float(length_of_data) * 100
    norm_ED = norm_ED / float(length_of_data)  # ICDAR2019 Normalized Edit Distance

    return valid_loss_avg.val(), accuracy, norm_ED, preds_str, confidence_score_list, labels, infer_time, length_of_data
