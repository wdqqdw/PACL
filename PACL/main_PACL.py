import os
from datetime import datetime, timezone, timedelta

import argparse
import torch
import numpy as np
import random
import torch.nn
import torch.nn.modules as nn

import warnings

import PACL.data_process_PACL as data_process
import PACL.train_PACL as train
import PACL.pretrain_PACL as pretrain
from PACL.config_PACL import Config
from model.PACL import PACL
from model.clip import ClipEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default='0',
                        help='which GPU to run')
    parser.add_argument('-dataset_name', type=str, default='TumEmo',
                        help='support HFM/TumEmo/Emotic/FI/WEBEmo')
    parser.add_argument('-model_name', type=str, default='PACL',
                        help='Support PACL')
    parser.add_argument('-save_model', type=str, default='0',
                        help='create dir and save model?')
    parser.add_argument('-remark', type=str, default='Default',
                        help='Remark added to the saving dir')
    parser.add_argument('-cluster', type=str, default= '0',
                        help='Using text cluster to guide image-text contrast')
    parser.add_argument('-process_id', type=int, default=0,
                        help='To identify process running on the same device')
    parser.add_argument('-adaptive', type=str, default='0',
                        help='Use the adaptive pretraining or not')
    parser.add_argument('-pretrain', type=str, default='0',
                        help='Pretraining or not')
    parser.add_argument('-loss', type=str, default='all',
                        help='which to backward (logits/itm)')
    #pretrain 0: irrelevant to pretraining
    #pretrain 1: repretrain the model and save
    #pretrain 2: load pretrained model's image model for downstream task
    #pretrain 3: load the whole pretrain model for downstream task
    #pretrain 4: pretrain the whole model and don't save
    parser.add_argument('-epoch', type=str, default='0',
                        help='State number of epoch')
    parser.add_argument('-pretrain_dataset', type=str, default='UNSET',
                        help='Name of pretrain dataset')
    parser.add_argument('-pretrain_lr', type=float, default=1e-3,
                        help='Init pretrain lr')
    parser.add_argument('-pretrain_weight_decay', type=float, default=0.02,
                        help='Init pretrain weight_decay')
    parser.add_argument('-pretrain_temp', type=float, default=0.07,
                        help='Init pretrain temperature')
    parser.add_argument('-weight_clip', type=str, default=-1,
                        help='the ratio of itm loss')

    return parser.parse_args()

def setup_seed(args, seed):
    args.seed = seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_current_time():
    beijing = timezone(timedelta(hours=8))
    utc = timezone.utc
    utc_time = datetime.utcnow().replace(tzinfo=utc)
    time_beijing = utc_time.astimezone(beijing)
    return time_beijing

def main_PACL(args):
    warnings.filterwarnings('ignore')
    torch.set_num_threads(1)
    dt = get_current_time()

    cfg = Config()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed = 110
    args.seed = None
    setup_seed(args, seed)
    print("Random seed set to " + str(seed))

    model_cfg = cfg.get_model_config()
    dataset_cfg = cfg.get_dataset_config()
    args.infer_mode = cfg.infer_mode
    if args.pretrain_dataset == 'UNSET':
        args.pretrain_dataset = cfg.pretrain_dataset
    args.downstream_mode = cfg.downstream_mode

    args.AA_dataset = dataset_cfg['AA_dataset']
    args.MSA_dataset = dataset_cfg['MSA_dataset']
    args.VSA_dataset = dataset_cfg['VSA_dataset']
    args.ML_dataset = dataset_cfg['ML_dataset']
    args.TO_dataset = dataset_cfg['TO_dataset']
    args.label_map = dataset_cfg[args.dataset_name]['label_map']
    if args.dataset_name not in args.ML_dataset:
        if args.dataset_name == 'WEBEmo':
            args.text_prompt = [[cfg.text_prompt.replace('[label]', l) for l in lm.keys()] for lm in args.label_map]
        else:
            args.text_prompt = []
            for l in args.label_map.keys():
                args.text_prompt.append(cfg.text_prompt.replace('[label]', l))
    args.text_encoder = model_cfg['commonParas']['text_encoder']
    args.image_encoder = model_cfg['commonParas']['image_encoder']
    args.cluster = model_cfg['commonParas']['cluster']
    if args.weight_clip == -1:
        args.weight_clip = model_cfg['commonParas']['weight_clip']
    args.weight_a234 = model_cfg['commonParas']['weight_a234']
    args.itm_target = model_cfg['commonParas']['itm_target']
    args.clip_temperature = model_cfg['commonParas']['clip_temperature']
    args.text_max_length = model_cfg['commonParas']['text_max_length']


    model = PACL(args, cfg)
    model.to(args.device)
    print("Model instance with textual encoder(%s) and visual encoder(%s) created and set to:" %
          (args.text_encoder, args.image_encoder), 'cuda')

    save_dir = model_cfg['commonParas']['save_dir']

    if args.epoch == '0':
        epoch_num = model_cfg['commonParas']['epoch_num']
    else:
        epoch_num = int(args.epoch)
    args.epoch = epoch_num
    args.warmup_epoch1 = int(epoch_num / 3)
    args.warmup_epoch2 = int(2 * epoch_num / 3)
    pretrain_epoch_num = model_cfg['commonParas']['pretrain_epoch_num']

    text_tokenizer = model.get_text_tokenizor()

    pre_train_loader = None
    pre_test_loader = None
    pre_dev_loader = None
    if args.pretrain == '1' or args.pretrain == '4':
        pre_test_loader, args.num_pretest_sample, _ = data_process.create_dataloader(args, ['test_' + args.task], text_tokenizer, True)
        pre_dev_loader, args.num_predev_sample, _ = data_process.create_dataloader(args, ['dev_' + args.task], text_tokenizer, True)
        if args.adaptive == '0':
            pre_train_loader, args.num_pretrain_sample, _ = data_process.create_dataloader(args, ['train_' + args.task], text_tokenizer, True)
        else:
            task_list = ['train_fs_' + args.task, 'train_nfs_' + args.task, 'train_fns_' + args.task, 'train_nfns_' + args.task]
            pre_train_loader, args.num_pretrain_sample, _ = data_process.create_dataloader(args, task_list, text_tokenizer, True, adaptive=True)

    train_loader, args.num_train_sample, _ = data_process.create_dataloader(args, ['train_' + args.task, 'dev_' + args.task, 'test_' + args.task], text_tokenizer)
    test_loader, args.num_test_sample, _ = data_process.create_dataloader(args, ['test_' + args.task], text_tokenizer)

    dev_loader = None
    if args.dataset_name not in args.TO_dataset:
        dev_loader, args.num_dev_sample, _ = data_process.create_dataloader(args, ['dev_' + args.task], text_tokenizer)

    hard_loader = None
    if args.dataset_name in args.AA_dataset:
        hard_loader, args.num_hard_sample, _ = data_process.create_dataloader(args, ['hard_' + args.task], text_tokenizer)

    args.image_encoder = model_cfg['commonParas']['image_encoder']
    args.acc_grad = model_cfg['commonParas']['acc_grad']
    args.warm_up_epoch1 = model_cfg['commonParas']['warm_up_epoch1']
    args.warm_up_epoch2 = model_cfg['commonParas']['warm_up_epoch2']
    args.batch_size = model_cfg['commonParas']['batch_size']

    folder_name = dt.strftime('%Y%m%d-%H%M-') + args.task + '-' + args.remark + '-' + str(epoch_num)
    if args.pretrain == '1':
        folder_name += '-' + str(pretrain_epoch_num)
    args.save_dir = save_dir + '/' + args.dataset_name + '/' + folder_name

    os.makedirs(args.save_dir)
    args.log_name = args.save_dir + '/' + 'train_log.txt'
    args.arg_save_path = args.save_dir + '/' + 'args.txt'
    args.cfg_save_path = args.save_dir + '/' + 'config.json'
    args.error_log = args.save_dir + '/' + 'error_log.txt'

    args.metric_dir = args.save_dir + '/' + args.dataset_name + '_metric'
    os.makedirs(args.metric_dir)

    args.full_model_path = dataset_cfg['PACL'][args.gpu][args.process_id]

    with open(args.arg_save_path, 'w', encoding='utf-8') as f:
        for k in list(vars(args).keys()):
            f.write('%s: %s \n' % (k, vars(args)[k]))
    cfg.output_cfg(args, args.cfg_save_path)
    print("Arguments and config saved to " + args.save_dir)

    loss_nce = nn.CrossEntropyLoss()
    loss_nce.to(args.device)
    loss_mse = nn.MSELoss()
    loss_mse.to(args.device)
    loss = [loss_nce, loss_mse]

    # Do the complete pretrain
    if args.pretrain == '1':
        optimizer, scheduler = pretrain.set_optim(args, model, model_cfg, pretrain_epoch_num)
        #print("Optimizer setted, use optimizer:", optimizer_name)
        pretrain.train_model(args, pretrain_epoch_num, model, [pre_train_loader, pre_dev_loader, pre_test_loader], loss, optimizer, scheduler)
        model.load_full_model(args, args.full_model_path)
    # load pretrained full model
    elif args.pretrain == '3':
        model.load_full_model(args, args.full_model_path)
    elif args.pretrain == '4':
        optimizer, scheduler = pretrain.set_optim(args, model, model_cfg, pretrain_epoch_num)
        pretrain.train_model(args, pretrain_epoch_num, model, [pre_train_loader, pre_dev_loader, pre_test_loader], loss, optimizer, scheduler)
        return

    loss = nn.CrossEntropyLoss()
    loss.to(args.device)
    if args.dataset_name in args.ML_dataset:
        loss = nn.BCEWithLogitsLoss()
        loss.to(args.device)

    optimizer, scheduler = train.set_optim(args, model, model_cfg, epoch_num)
    if args.pretrain != '0' and args.infer_mode == 'zero_shot':
        train.zero_shot_infer(args, model, [train_loader, dev_loader, test_loader, hard_loader])
        error_msg = 'Zero-shot inference processed successfully.\n'
    else:
        train.train_model(args, epoch_num, model, [train_loader, dev_loader, test_loader, hard_loader], loss, optimizer, scheduler)
        error_msg = 'Pretraining and training processed successfully.\n'
    train.output_to_error_log(args, error_msg)

if __name__ == "__main__":
    args = parse_args()
    main_PACL(args)