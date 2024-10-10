import os
import json

class Config():
    def __init__(self):
        # hyper parameters for models
        self.hyper_model_map = self.__model_params()
        # hyper parameters for datasets
        self.hyper_dataset_map = self.__dataset_params()

        #["MVSA-single", "MVSA-multiple", "HFM", "TumEmo"]
        self.pretrain_dataset = 'TumEmo'

        #["image only", "text only", "image-text pretrain"]
        self.test_model = 'image-text pretrain'

        #["zero_shot", "linear_probe"]
        self.infer_mode = 'linear_probe'

        #["finetune", "freeze"]
        self.downstream_mode = 'finetune'

        # [single/both, [sentiment/scene/attention]/[attention/addition]]
        self.image_model_type = ['single', 'scene']
        # [single/both, [sentiment/general/clip]/[attention/addition]]
        self.text_model_type = ['single', 'general']

        self.text_prompt = 'A photo with [label] emotion' #'What a [label] day! We feel so  [label]! A photo with [label] emotion'


    def __dataset_params(self):
        vsa_root_dataset_dir = '/change_to_your_path/VSA_dataset'
        msa_root_dataset_dir = '/change_to_your_path/MSA_dataset'
        tsa_root_dataset_dir = '/change_to_your_path/TSA_dataset'
        usp_root_dataset_dir = '/change_to_your_path/pretrain_dataset'
        pretrain_model_dir = '/change_to_your_path/pretrain_models'
        timm_model_dir = '/change_to_your_path/timm'
        PACL_model_dir = '/change_to_your_path/PACL/pretrain_models/PACL'
        #download these datasets if need
        dataset_config = {
            'AA_dataset': ['MVSA-single', 'MVSA-multiple'],
            'VSA_dataset': ['Emotic', 'WEBEmo', 'FI', 'Emotion6', 'UnbiasedEmo'],
            'MSA_dataset': ['MVSA-single', 'MVSA-multiple', 'HFM', 'TumEmo'],
            'TSA_dataset': ['IMDB'],
            #Unsupervised
            'UNS_dataset': ['CC3M/Valid_split'],
            #Multiple label
            'ML_dataset': ['Emotic'],
            #Test set only (do not have Dev set)
            'TO_dataset': ['WEBEmo', 'IMDB'],
            'pretrain_models':{
                #change directory name according to your file path
                'bert-base-uncased': os.path.join(pretrain_model_dir, 'bert-base-uncased'),
                'bertweet-base-sentiment-analysis': os.path.join(pretrain_model_dir, 'bertweet-base-sentiment-analysis'),
                'clip-vit-base-patch32': os.path.join(pretrain_model_dir, 'clip-vit-base-patch32'),
                'resnet50': os.path.join(timm_model_dir, 'resnet50_a1_0-14fe96d1.pth'),
                'roberta-large-ernie2-skep-en': os.path.join(pretrain_model_dir, 'roberta-large-ernie2-skep-en'),
                'vit-base': os.path.join(timm_model_dir, 'vit_base_patch16_224.augreg2_in21k_ft_in1k/pytorch_model.bin'),
                'swin-base': os.path.join(timm_model_dir, 'swin_s3_base_224/pytorch_model.bin'),
            },
            'PACL':{
                '0': [os.path.join(PACL_model_dir, 'model_0_0.pth'), os.path.join(PACL_model_dir, 'model_0_1.pth')],
                '1': [os.path.join(PACL_model_dir, 'model_1_0.pth'), os.path.join(PACL_model_dir, 'model_1_1.pth')],
                '2': [os.path.join(PACL_model_dir, 'model_2_0.pth'), os.path.join(PACL_model_dir, 'model_2_1.pth')],
                '3': [os.path.join(PACL_model_dir, 'model_3_0.pth'), os.path.join(PACL_model_dir, 'model_3_1.pth')],
                '4': [os.path.join(PACL_model_dir, 'model_4_0.pth'), os.path.join(PACL_model_dir, 'model_4_1.pth')],
                '5': [os.path.join(PACL_model_dir, 'model_5_0.pth'), os.path.join(PACL_model_dir, 'model_5_1.pth')],
                '6': [os.path.join(PACL_model_dir, 'model_6_0.pth'), os.path.join(PACL_model_dir, 'model_6_1.pth')],
                '7': [os.path.join(PACL_model_dir, 'model_7_0.pth'), os.path.join(PACL_model_dir, 'model_7_1.pth')],
            },
            'MVSA-single':{
                'class_num': 3,
                'label_map': {"Positive": 0, "Neutral": 1, "Negative": 2},
                'text_prompt': ["The sentiment of image is positive",
                                "The sentiment of image is neutral",
                                "The sentiment of image is negetive"]
            },
            'MVSA-multiple':{
                'class_num': 3,
                'label_map': {"Positive": 0, "Neutral": 1, "Negative": 2},
                'text_prompt': ["The sentiment of image is positive",
                                "The sentiment of image is neutral",
                                "The sentiment of image is negetive"]
            },
            'HFM': {
                'class_num': 2,
                'label_map': {"Positive": 0, "Negative": 1},
                'text_prompt': ["The sentiment of image is positive",
                                "The sentiment of image is negetive"]
            },
            'TumEmo': {
                # generate data files in json format and save them in these pathes
                'train_vtsa': os.path.join(msa_root_dataset_dir, 'TumEmo/10-fold-1/train_2.json'),
                'train_cluster_vtsa': os.path.join(msa_root_dataset_dir, 'TumEmo/10-fold-1/train_1_cluster.json'),
                'train_fs_vtsa': os.path.join(msa_root_dataset_dir, 'TumEmo/10-fold-1/train_2_fs.json'),
                'train_fns_vtsa': os.path.join(msa_root_dataset_dir, 'TumEmo/10-fold-1/train_2_fns.json'),
                'train_nfs_vtsa': os.path.join(msa_root_dataset_dir, 'TumEmo/10-fold-1/train_2_nfs.json'),
                'train_nfns_vtsa': os.path.join(msa_root_dataset_dir, 'TumEmo/10-fold-1/train_2_nfns.json'),
                'dev_vtsa': os.path.join(msa_root_dataset_dir, 'TumEmo/10-fold-1/dev_2.json'),
                'test_vtsa': os.path.join(msa_root_dataset_dir, 'TumEmo/10-fold-1/test_2.json'),
                'hard_vtsa': os.path.join(msa_root_dataset_dir, 'TumEmo/10-fold-1/test_2.json'),
                'image': os.path.join(msa_root_dataset_dir, 'TumEmo/dataset_image'),
                'label': os.path.join(msa_root_dataset_dir, 'TumEmo/labelProcessed.json'),
                'class_num': 7,
                'label_map': {"Love": 0, "Happy": 1, "Calm": 2, "Bored": 3, "Sad": 4, "Angry": 5, "Fear": 6},
                'text_prompt': ["The sentiment of image is Love",
                                "The sentiment of image is Happy",
                                "The sentiment of image is Calm",
                                "The sentiment of image is Bored",
                                "The sentiment of image is Sad",
                                "The sentiment of image is Angry",
                                "The sentiment of image is Fear"]
            },
            'Emotic':{
                'train_vtsa': os.path.join(vsa_root_dataset_dir, 'Emotic/10-fold-1/train.json'),
                'test_vtsa': os.path.join(vsa_root_dataset_dir, 'Emotic/10-fold-1/test.json'),
                'dev_vtsa': os.path.join(vsa_root_dataset_dir, 'Emotic/10-fold-1/val.json'),
                'image': os.path.join(vsa_root_dataset_dir, 'Emotic/dataset_image'),
                'class_num': 26,
                'label_map':{"Affection": 0, "Anger": 1, "Annoyance": 2, "Anticipation": 3, "Aversion": 4, "Confidence": 5,
                             "Disapproval": 6, "Disconnection": 7, "Disquietment": 8, "Doubt/Confusion": 9, "Embarrassment": 10,
                             "Engagement": 11, "Esteem": 12, "Excitement": 13, "Fatigue": 14, "Fear": 15, "Happiness": 16,
                             "Pain": 17, "Peace": 18, "Pleasure": 19, "Sadness": 20, "Sensitivity": 21, "Suffering": 22,
                             "Surprise": 23, "Sympathy": 24, "Yearning": 25},
                'text_prompt': []
            },
            'WEBEmo': {
                'train_vtsa': os.path.join(vsa_root_dataset_dir, 'WEBEmo/10-fold-1/train_2.json'),
                'test_vtsa': os.path.join(vsa_root_dataset_dir, 'WEBEmo/10-fold-1/test_2.json'),
                'image': os.path.join(vsa_root_dataset_dir, 'WEBEmo/dataset_image'),
                'class_num': [2, 7, 25],
                'label_map': [
                    {'positive': 0, 'negative': 1},
                    {'love': 0, 'joy': 1, 'surprise': 2, 'confusion': 3, 'sadness': 4, 'anger': 5, 'fear': 6},
                    {'affection': 0, 'cheerfullness': 1, 'confusion': 2, 'contentment':3, 'disappointment': 4,
                     'disgust': 5, 'enthrallment': 6, 'envy': 7, 'exasperation': 8, 'gratitude': 9, 'horror': 10,
                     'irritabilty': 11, 'lust': 12, 'neglect': 13, 'nervousness': 14, 'optimism': 15, 'pride': 16,
                     'rage': 17, 'relief': 18, 'sadness': 19, 'shame': 20, 'suffering': 21, 'surprise': 22,
                     'sympathy': 23, 'zest': 24}
                ],
                'text_prompt': []
            },
            'FI': {
                'train_vtsa': os.path.join(vsa_root_dataset_dir, 'FI/10-fold-1/train_1.json'),
                'test_vtsa': os.path.join(vsa_root_dataset_dir, 'FI/10-fold-1/test_1.json'),
                'dev_vtsa': os.path.join(vsa_root_dataset_dir, 'FI/10-fold-1/dev_1.json'),
                'image': os.path.join(vsa_root_dataset_dir, 'FI/dataset_image'),
                'label': os.path.join(vsa_root_dataset_dir, 'FI/labelProcessed.json'),
                'class_num': 8,
                'label_map': {'amusement': 1, 'awe': 2, 'anger': 6, 'contentment': 3,
                              'disgust': 5, 'excitement': 0, 'fear': 7, 'sadness': 4},
                'text_prompt': []
            },
            'Emotion6': {
                'train_vtsa': os.path.join(vsa_root_dataset_dir, 'Emotion6/10-fold-1/train_1.json'),
                'test_vtsa': os.path.join(vsa_root_dataset_dir, 'Emotion6/10-fold-1/test_1.json'),
                'dev_vtsa': os.path.join(vsa_root_dataset_dir, 'Emotion6/10-fold-1/dev_1.json'),
                'image': os.path.join(vsa_root_dataset_dir, 'Emotion6/dataset_image'),
                'class_num': 6,
                'label_map': {'anger': 4, 'fear': 5, 'joy': 1, 'love': 0, 'sadness': 3, 'surprise': 2},
                'text_prompt': []
            },
            'UnbiasedEmo': {
                'train_vtsa': os.path.join(vsa_root_dataset_dir, 'UnbiasedEmo/10-fold-1/train_1.json'),
                'test_vtsa': os.path.join(vsa_root_dataset_dir, 'UnbiasedEmo/10-fold-1/test_1.json'),
                'dev_vtsa': os.path.join(vsa_root_dataset_dir, 'UnbiasedEmo/10-fold-1/dev_1.json'),
                'image': os.path.join(vsa_root_dataset_dir, 'UnbiasedEmo/dataset_image'),
                'class_num': 6,
                'label_map': {'anger': 4, 'fear': 5, 'joy': 1, 'love': 0, 'sadness': 3, 'surprise': 2},
                'text_prompt': []
            },
            'IMDB':{
                'train_vtsa': os.path.join(tsa_root_dataset_dir, 'IMDB/10-fold-1/train_1.json'),
                'test_vtsa': os.path.join(tsa_root_dataset_dir, 'IMDB/10-fold-1/test_1.json'),
                'class_num': 2,
                'label_map': {'Positive': 0, 'Negative': 1},
                'text_prompt': []
            },
            'CC3M/Valid_split':{
                'train_vtsa': os.path.join(usp_root_dataset_dir, 'CC3M/Valid_split/10-fold-1/train_1.json'),
                'test_vtsa': os.path.join(usp_root_dataset_dir, 'CC3M/Valid_split/10-fold-1/test_1.json'),
                'dev_vtsa': os.path.join(usp_root_dataset_dir, 'CC3M/Valid_split/10-fold-1/dev_1.json'),
                'image': os.path.join(usp_root_dataset_dir, 'CC3M/Valid_split/dataset_image'),
                'class_num': 0,
                'text_prompt': []
            }
        }
        return dataset_config

    def __model_params(self):
        model_config = {
            'commonParas':{
                'optimizer': 'AdamW',
                'image_encoder_channel_size': 49,
                #'image_encoder_channel_size': 196,
                'ocr_enhancement': False,
                'root_dir': '/change_to_your_path/PACL/',
                'save_dir': '/change_to_your_path/PACL/checkpoint',
                'projector_dim': '768',
                'text_max_length': 77,
                'num_workers': 4,
                'cluster': "2",
                'weight_clip': 0.1,
                'weight_a234': [0.6, 0.6, 0.1],
                'itm_target': [1, 1, 1, 1],
                'clip_temperature': 0.07,

                'lr_scene_image': 1e-3,
                'lr_scene_image_ft': 2e-4,
                'lr_senti_image': 1e-3,
                'lr_senti_image_ft': 2e-4,
                'lr_general_text': 0,
                'lr_general_text_ft': 0,
                'lr_senti_text': 0,
                'lr_senti_text_ft': 0,
                'lr_other': 1e-3,
                'lr_other_ft': 1e-3,
                'lr_classify': 1e-2,
                'lr_classify_ft': 1e-2,
                'lr_fusion': 1e-3,

                #'image_encoder': 'resnet-50',
                # 'image_encoder': 'vit',
                # 'image_encoder': 'swin',
                #['resnet-50', 'clip', 'vit', 'swin']
                'image_encoder': 'resnet-50',
                #['bert', 'bertweet', 'skep', 'clip-text']
                'text_encoder': 'skep',

                'batch_size': 32,
                'acc_grad': 8,
                'epoch_num': 10,
                'pretrain_epoch_num': 30,
                'hidden_size': 768,
                'pooling_output_size': 1,
                'warm_up_epoch1': 0,
                'warm_up_epoch2': 0,
                'image_size': [224, 224],
                'cls_dropout': 0,
            },
            # dataset
            'datasetParas':{
            },

        }
        return model_config

    def get_model_config(self):
        return self.hyper_model_map

    def get_dataset_config(self):
        return self.hyper_dataset_map

    def get_model_type(self):
        return self.test_model

    def output_cfg(self, args,  path):
        cfg_content = {"test_model": self.test_model,
                       "infer_mode": self.infer_mode,
                       "downstream_mode": self.downstream_mode,
                       "pretrain_dataset": args.pretrain_dataset,
                       "image_model_type": self.image_model_type,
                       "text_model_type": self.text_model_type,
                       "dataset_param": self.get_dataset_config(),
                       "model_param": self.get_model_config(),}

        json_data = json.dumps(cfg_content, indent=2)
        with open(path, 'w') as f:
            f.write(json_data)