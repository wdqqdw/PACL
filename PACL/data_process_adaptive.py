import json
import torch
import torch.nn.utils.rnn as run_utils
import re
import numpy as np

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
from PACL.config_PACL import Config

class TextImageData(Dataset):
    def __init__(self, args, cfg, sample_path_list, tokenizor, for_pretrain = False):
        if for_pretrain:
            self.dataset_name = args.pretrain_dataset
        else:
            self.dataset_name = args.dataset_name
        dataset_config = cfg.get_dataset_config()[self.dataset_name]
        model_config = cfg.get_model_config()

        self.text_encoder = args.text_encoder
        self.image_encoder = args.image_encoder
        self.image_path = dataset_config['image']
        self.ocr_enhancement = model_config['commonParas']['ocr_enhancement']
        self.text_max_length = model_config['commonParas']['text_max_length']
        self.image_size = model_config['commonParas']['image_size']
        self.statistics = {}
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.id_list = []
        self.text_list = []
        self.ocr_list = []
        self.label_list = []
        self.cluster_list = []
        self.image_list = []
        sample_content = []
        for sample_path in sample_path_list:
            with open(sample_path, 'r', encoding='utf-8') as sample_read:
                sample_content += json.load(sample_read)

        for data in sample_content:

            self.id_list.append(data['id'])


            self.cluster_list.append(data.get('cluster'))
            text = data['text_with_ocr'] if self.ocr_enhancement else data['text']
            self.text_list.append(text)

            self.label_list.append(data['emotion_label'])
            if self.statistics.get(data['emotion_label']) == None:
                self.statistics[data['emotion_label']] = 1
            else:
                self.statistics[data['emotion_label']] += 1

            if len(data['ocr']) == 0:
                self.ocr_list.append("")
            else:
                self.ocr_list.append(" ".join(data['ocr']))

        self.text_list = tokenizor(self.text_list, padding=True, return_tensors="pt", truncation=True, max_length=self.text_max_length)
        self.ocr_list = tokenizor(self.ocr_list, padding=True, return_tensors="pt", truncation=True, max_length=self.text_max_length)

        if for_pretrain:
            print("Pretraining dataloader created for", ",".join(sample_path_list))
        else:
            print("Dataloader created for", ",".join(sample_path_list))

    def get_data_statistics(self):
        return self.statistics

    def get_data_id_list(self):
        return self.id_list

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        text_item = {k:self.text_list[k][index] for k in self.text_list.keys()}
        ocr_item = {k:self.ocr_list[k][index] for k in self.ocr_list.keys()}

        image_index_path = self.image_path + '/' + str(self.id_list[index]) + '.jpg'
        image_read = Image.open(image_index_path)
        image_read.load()
        image_read = self.image_transforms(image_read)

        return self.id_list[index], text_item, image_read, self.label_list[index], \
            ocr_item, self.cluster_list[index]



class CollateImageText():
    def __init__(self, args, cfg, tokenizer):
        self.device = args.device
        model_config = cfg.get_model_config()
        self.image_mask_num = model_config['commonParas']['image_encoder_channel_size']
        self.tokenizer = tokenizer

    def __call__(self, batch_data):
        return self._collate(batch_data)

    def _collate(self, batch_data):
        text_keys = list(batch_data[0][1].keys())
        batch_id_list = []
        batch_text_list = [[] for _ in range(len(text_keys))]
        batch_ocr_list = [[] for _ in range(len(text_keys))]
        batch_image_list = []
        batch_label_list = []
        batch_cluster_list = []

        for b in batch_data:
            batch_id_list.append(b[0])
            for i in range(len(text_keys)):
                batch_text_list[i].append(b[1][text_keys[i]].view(1, -1))
                batch_ocr_list[i].append(b[4][text_keys[i]].view(1, -1))
            batch_text_list.append(b[1])
            batch_image_list.append(np.array(b[2]))
            batch_label_list.append(b[3])
            batch_cluster_list.append(b[5])

        batch_text_list = {text_keys[i]: torch.cat(batch_text_list[i], dim=0) for i in range(len(text_keys))}
        batch_ocr_list = {text_keys[i]: torch.cat(batch_ocr_list[i], dim=0) for i in range(len(text_keys))}
        batch_image_list = torch.FloatTensor(np.array(batch_image_list))
        batch_label_list = torch.LongTensor(np.array(batch_label_list))

        return batch_id_list, batch_text_list, batch_image_list, batch_label_list, batch_ocr_list, batch_cluster_list



#data_type: indicate the usage of dataloader (train dev test)
def create_dataloader(args, data_type_list, tokenizor, for_pretrain = False, for_retrieval = False):
    cfg = Config()
    dataset_para = cfg.get_dataset_config()
    model_para = cfg.get_model_config()
    num_workers = model_para['commonParas']['num_workers']

    if for_pretrain:
        dataset_name = args.pretrain_dataset
    else:
        dataset_name = args.dataset_name

    sample_path_list = []
    for data_type in data_type_list:
        sample_path_list.append(dataset_para[dataset_name][data_type])
    batch_size = model_para['commonParas']['batch_size']

    dataset = TextImageData(args, cfg, sample_path_list, tokenizor, for_pretrain)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, collate_fn
            = CollateImageText(args, cfg, tokenizor), shuffle=True)
    return data_loader, len(dataset), dataset.statistics