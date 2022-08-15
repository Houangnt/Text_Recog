class Logging(object):
    def __init__(self, exp_name):
        super(Logging, self).__init__()
        self.exp_name = exp_name
        self.data_log_file = f'./saved_models/{exp_name}/log_dataset.txt'
        self.args_log_file = f'./saved_models/{exp_name}/opt.txt'
        self.training_log_file = f'./saved_models/{exp_name}/log_train.txt'

    def save_data_log(self, valid_dataset_log):
        log = open(self.data_log_file, 'a')
        log.write(valid_dataset_log)
        print('-' * 80)
        log.write('-' * 80 + '\n')
        log.close()

    def save_args_log(self, args):
        with open(self.args_log_file, 'a') as opt_file:
            opt_log = '------------ Options -------------\n'
            save_args = vars(args)
            for k, v in save_args.items():
                opt_log += f'{str(k)}: {str(v)}\n'
            opt_log += '---------------------------------------\n'
            opt_file.write(opt_log)

    def write_training_log(self, _iter, n_iter, train_loss, val_loss, elps_time, acc, norm_ed, best_accuracy,
                           best_norm_ed):
        with open(self.training_log_file, 'a') as log:
            loss_log = f'[{_iter + 1}/{n_iter}] Train loss: {train_loss:0.5f}, Valid loss: {val_loss:0.5f}, ' \
                       f'Elapsed_time: {elps_time:0.5f}'
            current_model_log = f'{"Current_accuracy":17s}: {acc:0.3f}, {"Current_norm_ED":17s}: {norm_ed:0.2f}'
            best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ed:0.2f}'
            loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
            log.write(loss_model_log + '\n')
        return loss_model_log

    def write_validate_log(self, labels, preds, confs, pred_type):
        dashed_line = '-' * 80
        head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
        predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
        for gt, pred, confidence in zip(labels[:5], preds[:5], confs[:5]):
            if 'attn' in pred_type:
                gt = gt[:gt.find('[s]')]
                pred = pred[:pred.find('[s]')]

            predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
        predicted_result_log += f'{dashed_line}'
        with open(self.training_log_file, 'a') as log:
            log.write(predicted_result_log + '\n')
        return predicted_result_log
