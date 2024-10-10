import argparse
import copy
from PACL.main_sentiCLIP import main_sentiCLIP
from CLIP.main_CLIP import main_CLIP
from CLIP.main_AssessSenti import main_AssessSenti

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default='0',
                        help='which GPU to run')
    parser.add_argument('-dataset_name', type=str, default='MVSA-single',
                        help='support MVSA-single/MVSA-multiple')
    parser.add_argument('-model_name', type=str, default='MACL',
                        help='support MACL')
    parser.add_argument('-save_model', type=str, default='0',
                        help='create dir and save model?')
    parser.add_argument('-remark', type=str, default='Default',
                        help='Remark added to the saving dir')
    parser.add_argument('-task', type=str, default='vtsa',
                        help='Visual-Textual Sentiment Analysis & Ambiguity Discrimination')
    parser.add_argument('-cluster', type=str, default= '0',
                        help='Using text cluster to guide image-text contrast')
    parser.add_argument('-process_id', type=int, default=0,
                        help='To identify process running on the same device')
    parser.add_argument('-adaptive', type=str, default='0',
                        help='Use the adaptive pretraining or not')
    parser.add_argument('-loss', type=str, default='all',
                        help='which to backward (logits/itm/all)')
    parser.add_argument('-pretrain', type=str, default='0',
                        help='Pretraining or not')
    parser.add_argument('-method', type=str, default='CLIP',
                        help='STACL or SentiCLIP or CLIP or AssessSenti')
    parser.add_argument('-epoch', type=str, default='0',
                        help='State number of epoch')
    parser.add_argument('-pretrain_dataset', type=str, default='UNSET',
                        help='Name of pretrain dataset')
    parser.add_argument('-grid_search_lr', type=str, default='0',
                        help='1 stand for search optim hyper parameters')
    parser.add_argument('-pretrain_lr', type=float, default=-1,
                        help='Init pretrain lr')
    parser.add_argument('-pretrain_weight_decay', type=float, default=0.02,
                        help='Init pretrain weight_decay')
    parser.add_argument('-grid_search_weight_decay', type=str, default='0',
                        help='1 stand for search optim hyper parameters')
    parser.add_argument('-pretrain_temp', type=float, default=0.07,
                        help='Init pretrain temperature')
    parser.add_argument('-grid_search_temp', type=str, default='0',
                        help='1 stand for search optim hyper parameters')
    parser.add_argument('-senti_target', type=str, default='UNSET',
                        help='text/image')
    parser.add_argument('-weight_clip', type=str, default=-1,
                        help='the ratio of clip loss')

    return parser.parse_args()

def run_grid_search(args):
    pass

def run_all_dataset(args, type, main_func):
    if type == 'all_msa':
        dataset_name_list = ['MVSA-single', 'MVSA-multiple', 'HFM', 'TumEmo']
    else:
        dataset_name_list = ['WEBEmo', 'FI', 'Emotion6', 'UnbiasedEmo', 'Emotic']
    if args.pretrain == '0':
        pretrain_set_list = ['0', '0', '0', '0', '0']
    elif args.pretrain in ['1', '2', '3']:
        pretrain_set_list = [args.pretrain, '3', '3', '3', '3']
    else:
        pretrain_set_list = ['4', '4', '4', '4', '4']

    for i in range(len(dataset_name_list)):
        tmp_args = copy.deepcopy(args)
        tmp_args.dataset_name = dataset_name_list[i]
        tmp_args.pretrain = pretrain_set_list[i]
        main_func(tmp_args)


if __name__ == "__main__":
    args = parse_args()
    if args.dataset_name in ['all_msa', 'all_vsa']:
        run_all_dataset(args, args.dataset_name, main_sentiCLIP)
    else:
        main_sentiCLIP(args)



