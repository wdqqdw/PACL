import json
import torch
import torch.nn.utils.rnn as run_utils
import re
import math
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.sampler import RandomSampler
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
        self.senti_cluster_list = []
        self.fact_cluster_list = []
        self.fs_match_list = []
        self.image_list = []
        sample_content = []
        for sample_path in sample_path_list:
            with open(sample_path, 'r', encoding='utf-8') as sample_read:
                sample_content += json.load(sample_read)

        for data in sample_content:

            self.id_list.append(data['id'])

            self.senti_cluster_list.append(data.get('senti_cluster'))
            self.fact_cluster_list.append(data.get('fact_cluster'))
            self.fs_match_list.append(data.get('fact_senti_match'))
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
            ocr_item, self.senti_cluster_list[index], self.fact_cluster_list[index], self.fs_match_list[index]


class ImageData(Dataset):
    def __init__(self, args, cfg, sample_path_list, for_pretrain = False):
        if for_pretrain:
            self.dataset_name = args.pretrain_dataset
        else:
            self.dataset_name = args.dataset_name
        dataset_config = cfg.get_dataset_config()[self.dataset_name]
        model_config = cfg.get_model_config()

        self.image_path = dataset_config['image']
        self.label_map = dataset_config['label_map']

        sample_content = []
        for sample_path in sample_path_list:
            with open(sample_path, 'r', encoding='utf-8') as sample_read:
                sample_content += json.load(sample_read)

        self.image_size = model_config['commonParas']['image_size']
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.statistics = {}
        if self.dataset_name == 'Emotic':
            self.image_list = []
            self.category_list = []
            self.vad_list = []

            for data in sample_content:
                self.image_list.append(data['file_dir'])
                category = [0]*len(self.label_map)
                for i in data['category']:
                    category[self.label_map[i]] = 1
                    if self.statistics.get(i) == None:
                        self.statistics[i] = 1
                    else:
                        self.statistics[i] += 1

                self.category_list.append(category)
                self.vad_list.append([data['vad'][k] for k in data['vad'].keys()])

        elif self.dataset_name == 'WEBEmo':
            self.statistics = {
                'label2': {k: 0 for k in self.label_map[0].keys()},
                'label7': {k: 0 for k in self.label_map[1].keys()},
                'label25': {k: 0 for k in self.label_map[2].keys()},
            }
            self.image_list = []
            self.label2_list = []
            self.label7_list = []
            self.label25_list = []

            for data in sample_content:
                self.image_list.append(data['image_name'])
                self.label2_list.append(self.label_map[0][data['label_1']])
                self.statistics['label2'][data['label_1']] += 1
                self.label7_list.append(self.label_map[1][data['label_2']])
                self.statistics['label7'][data['label_2']] += 1
                self.label25_list.append(self.label_map[2][data['label_3']])
                self.statistics['label25'][data['label_3']] += 1

        elif self.dataset_name in ['FI', 'Emotion6', 'UnbiasedEmo']:
            reversed_label_map = {self.label_map[k]:k for k in self.label_map.keys()}
            self.statistics = {k:0 for k in self.label_map.keys()}
            self.image_list = []
            self.label_list = []

            for data in sample_content:
                self.image_list.append(data['id'])
                self.label_list.append(data['label'])
                self.statistics[reversed_label_map[data['label']]] += 1

        else:
            raise TypeError('Dataset can not be loaded.')
        self.text_encoder = args.text_encoder
        self.image_encoder = args.image_encoder

        if for_pretrain:
            print("Pretraining dataloader created for", ",".join(sample_path_list))
        else:
            print("Dataloader created for", ",".join(sample_path_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_index_path = self.image_path + '/' + self.image_list[index]
        image_read = Image.open(image_index_path)
        image_read.load()
        image_read = self.image_transforms(image_read)
        if self.dataset_name == 'Emotic':
            return image_read, self.category_list[index], self.vad_list[index]

        elif self.dataset_name == 'WEBEmo':
            return image_read, self.label2_list[index], self.label7_list[index], self.label25_list[index]

        elif self.dataset_name in ['FI', 'Emotion6', 'UnbiasedEmo']:
            return self.image_list[index], image_read, self.label_list[index]

        else:
            raise AssertionError('Cannot be reached')


class TextData(Dataset):
    def __init__(self, args, cfg, sample_path_list, tokenizor, for_pretrain = False):
        if for_pretrain:
            self.dataset_name = args.pretrain_dataset
        else:
            self.dataset_name = args.dataset_name
        dataset_config = cfg.get_dataset_config()[self.dataset_name]
        model_config = cfg.get_model_config()

        self.label_map = dataset_config['label_map']
        self.text_encoder = args.text_encoder
        self.image_encoder = args.image_encoder
        self.ocr_enhancement = model_config['commonParas']['ocr_enhancement']
        self.text_max_length = model_config['commonParas']['text_max_length']
        self.statistics = {}
        self.id_list = []
        self.text_list = []
        self.label_list = []
        sample_content = []
        for sample_path in sample_path_list:
            with open(sample_path, 'r', encoding='utf-8') as sample_read:
                sample_content += json.load(sample_read)

        for data in sample_content:
            self.id_list.append(data['id'])
            self.text_list.append(data['text'])


            self.label_list.append(self.label_map[data['label']])
            if self.statistics.get(data['label']) == None:
                self.statistics[data['label']] = 1
            else:
                self.statistics[data['label']] += 1

        self.text_list = tokenizor(self.text_list, padding=True, return_tensors="pt", truncation=True, max_length=self.text_max_length)

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

        return self.id_list[index], text_item, self.label_list[index]


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
        batch_senti_cluster_list = []
        batch_fact_cluster_list = []
        batch_fs_match_list = []

        for b in batch_data:
            batch_id_list.append(b[0])
            for i in range(len(text_keys)):
                batch_text_list[i].append(b[1][text_keys[i]].view(1, -1))
                batch_ocr_list[i].append(b[4][text_keys[i]].view(1, -1))
            batch_text_list.append(b[1])
            batch_image_list.append(np.array(b[2]))
            batch_label_list.append(b[3])
            batch_senti_cluster_list.append(b[5])
            batch_fact_cluster_list.append(b[6])
            batch_fs_match_list.append(b[7])

        batch_text_list = {text_keys[i]: torch.cat(batch_text_list[i], dim=0) for i in range(len(text_keys))}
        batch_ocr_list = {text_keys[i]: torch.cat(batch_ocr_list[i], dim=0) for i in range(len(text_keys))}
        batch_image_list = torch.FloatTensor(np.array(batch_image_list))
        batch_label_list = torch.LongTensor(np.array(batch_label_list))

        return batch_id_list, batch_text_list, batch_image_list, batch_label_list, batch_ocr_list, \
            batch_senti_cluster_list, batch_fact_cluster_list, batch_fs_match_list

class CollateImage():
    def __init__(self, args, cfg):
        #Image dataset will not used for pretraining
        self.dataset_name = args.dataset_name

    def __call__(self, batch_data):
        return self._collate(batch_data)

    def _collate(self, batch_data):
        if self.dataset_name == 'Emotic':
            batch_image_list = []
            batch_category_list = []
            batch_vad_list = []

            for b in batch_data:
                batch_image_list.append(np.array(b[0]))
                batch_category_list.append(b[1])
                batch_vad_list.append(b[2])

            batch_image_list = torch.FloatTensor(np.array(batch_image_list))
            batch_category_list = torch.LongTensor(np.array(batch_category_list))
            batch_vad_list = torch.LongTensor(np.array(batch_vad_list))

            return batch_image_list, batch_category_list, batch_vad_list

        elif self.dataset_name == 'WEBEmo':
            batch_image_list = []
            batch_label2_list = []
            batch_label7_list = []
            batch_label25_list = []

            for b in batch_data:
                batch_image_list.append(np.array(b[0]))
                batch_label2_list.append(b[1])
                batch_label7_list.append(b[2])
                batch_label25_list.append(b[3])

            batch_image_list = torch.FloatTensor(np.array(batch_image_list))
            batch_label2_list = torch.LongTensor(np.array(batch_label2_list))
            batch_label7_list = torch.LongTensor(np.array(batch_label7_list))
            batch_label25_list = torch.LongTensor(np.array(batch_label25_list))

            return batch_image_list, batch_label2_list, batch_label7_list, batch_label25_list

        elif self.dataset_name in ['FI', 'Emotion6', 'UnbiasedEmo']:
            batch_id_list = []
            batch_image_list = []
            batch_label_list = []

            for b in batch_data:
                batch_id_list.append(b[0])
                batch_image_list.append(np.array(b[1]))
                batch_label_list.append(b[2])

            batch_image_list = torch.FloatTensor(np.array(batch_image_list))
            batch_label_list = torch.LongTensor(np.array(batch_label_list))

            return batch_id_list, batch_image_list, batch_label_list

        else:
            raise AssertionError('Cannot be reached')


class CollateText():
    def __init__(self, args, cfg, tokenizer):
        self.device = args.device
        self.tokenizer = tokenizer

    def __call__(self, batch_data):
        return self._collate(batch_data)

    def _collate(self, batch_data):
        text_keys = list(batch_data[0][1].keys())
        batch_id_list = []
        batch_text_list = [[] for _ in range(len(text_keys))]
        batch_label_list = []

        for b in batch_data:
            batch_id_list.append(b[0])
            for i in range(len(text_keys)):
                batch_text_list[i].append(b[1][text_keys[i]].view(1, -1))
            batch_text_list.append(b[1])
            batch_label_list.append(b[2])

        batch_text_list = {text_keys[i]: torch.cat(batch_text_list[i], dim=0) for i in range(len(text_keys))}
        batch_label_list = torch.LongTensor(np.array(batch_label_list))

        return batch_id_list, batch_text_list, batch_label_list


class BatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    """
    iterate over tasks and provide a random batch per task in each mini-batch
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.largest_dataset_size = max([len(cur_dataset) for cur_dataset in dataset.datasets])

    def __len__(self):
        return self.batch_size * math.ceil(self.largest_dataset_size / self.batch_size) * len(self.dataset.datasets)

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = RandomSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size * self.number_of_datasets
        samples_to_grab = self.batch_size
        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        epoch_samples = self.largest_dataset_size * self.number_of_datasets

        final_samples_list = []  # this is a list of indexes from the combined dataset
        for _ in range(0, epoch_samples, step):
            for i in range(self.number_of_datasets):
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []
                for _ in range(samples_to_grab):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        # got to the end of iterator - restart the iterator and continue to get samples
                        # until reaching "epoch_samples"
                        sampler_iterators[i] = samplers_list[i].__iter__()
                        cur_batch_sampler = sampler_iterators[i]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                final_samples_list.extend(cur_samples)

        return iter(final_samples_list)



#data_type: indicate the usage of dataloader (train dev test)
def create_dataloader(args, data_type_list, tokenizor, for_pretrain = False, for_retrieval = False, adaptive = False):
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
    if for_retrieval == True:
        batch_size = 64
    else:
        batch_size = model_para['commonParas']['batch_size']

    if dataset_name in args.VSA_dataset:
        dataset = ImageData(args, cfg, sample_path_list, for_pretrain)
        data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, collate_fn
            = CollateImage(args, cfg), shuffle=True)
        return data_loader, len(dataset), dataset.statistics
    elif dataset_name in args.MSA_dataset:
        if adaptive == False:
            dataset = TextImageData(args, cfg, sample_path_list, tokenizor, for_pretrain)
            data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, collate_fn
                = CollateImageText(args, cfg, tokenizor), shuffle=True)
            return data_loader, len(dataset), dataset.statistics
        else:
            dataset_list = []
            for sample_path in sample_path_list:
                dataset = TextImageData(args, cfg, [sample_path], tokenizor, for_pretrain)
                dataset_list.append(dataset)
            dataset = ConcatDataset(dataset_list)
            data_loader = DataLoader(dataset, batch_size=batch_size*len(dataset_list), num_workers=num_workers, pin_memory=True, collate_fn
                = CollateImageText(args, cfg, tokenizor), sampler=BatchSchedulerSampler(dataset=dataset, batch_size=batch_size))
            return data_loader, len(dataset), [cur_dataset.statistics for cur_dataset in dataset.datasets]


    else:
        dataset = TextData(args, cfg, sample_path_list, tokenizor, for_pretrain)
        data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, collate_fn
            = CollateText(args, cfg, tokenizor), shuffle=True)
        return data_loader, len(dataset), dataset.statistics