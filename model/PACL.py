import copy

import torch
import torch.nn as nn
import numpy as np
import argparse

from model.fuse import FusionModel, CrossAttention
from model.contrast import SupConLoss
from model.text import UniTextEncoder
from model.image import UniImageEncoder
from model.param import ModelParam
import torch.nn.functional as F

class PACL(nn.Module):
    def __init__(self, args, cfg):
        super(PACL, self).__init__()
        model_cfg = cfg.get_model_config()
        dataset_cfg = cfg.get_dataset_config()

        self.device = args.device
        self.image_model = SentiVisualModel(args, cfg)
        self.text_model = SentiTextualModel(args, cfg)

        self.model_type = cfg.test_model
        self.infer_mode = cfg.infer_mode
        self.clip_temperature = args.clip_temperature
        self.downstream_mode = args.downstream_mode
        self.class_num = dataset_cfg[args.dataset_name]['class_num']
        self.pretrain_class_num = dataset_cfg[args.pretrain_dataset]['class_num']
        self.hidden_size = model_cfg['commonParas']['hidden_size']
        self.queue_size = model_cfg['commonParas']['queue_size']
        self.projector_dim = [int(d) for d in model_cfg['commonParas']['projector_dim'].split('-')]

        self.pooling_output_size = model_cfg['commonParas']['pooling_output_size']  # 1
        self.text_output_dim = self.text_model.get_output_dim()
        self.image_output_dim = self.image_model.get_output_dim()
        self.image_channel_dim = self.image_model.get_channel_dim()

        self.logit_scale = 1 / args.clip_temperature
        #self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / args.pretrain_temp))

        self.avg_pool = nn.AdaptiveAvgPool1d(self.pooling_output_size)
        self.max_pool = nn.AdaptiveMaxPool1d(self.pooling_output_size)

        sizes = [self.image_output_dim] + self.projector_dim
        layer = []
        for i in range(len(sizes) - 2):
            layer.append(nn.Linear(sizes[i], sizes[i+1], bias=False))
            layer.append(nn.BatchNorm1d(sizes[i+1]))
            layer.append(nn.ReLU(inplace=True))
        layer.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        #layer.append(nn.BatchNorm1d(sizes[-1], affine=False))
        self.image_to_embed_projection = nn.Sequential(*layer)

        sizes = [self.text_output_dim] + self.projector_dim
        layer = []
        for i in range(len(sizes) - 2):
            layer.append(nn.Linear(sizes[i], sizes[i+1], bias=False))
            layer.append(nn.BatchNorm1d(sizes[i+1]))
            layer.append(nn.ReLU(inplace=True))
        layer.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        #layer.append(nn.BatchNorm1d(sizes[-1], affine=False))
        self.text_to_embed_projection = nn.Sequential(*layer)

        self.text_2_q = nn.Linear(self.text_output_dim, self.hidden_size)
        self.image_2_k = nn.Linear(self.image_output_dim, self.hidden_size)

        self.itm_output = np.array(args.itm_target).max()+1
        self.itm_classify = nn.Sequential(
            nn.Linear(self.image_channel_dim, self.image_channel_dim),
            nn.ReLU(),
            nn.Linear(self.image_channel_dim, self.itm_output)
        )

        if args.senti_target == 'image':
            self.create_classify_head('WEBEmo', [2,7,25])
        elif args.senti_target == 'text':
            self.create_classify_head('IMDB', 2)
        elif args.pretrain == '0':
            self.create_classify_head(args.dataset_name)

        #self.initialize_parameters()
        self.for_downstream = False

        self.register_buffer("text_queue", torch.randn(self.text_output_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

    def forward(self, param: ModelParam, dataset_name = 'None'):
        if dataset_name == 'WEBEmo':
            return self.forward_webemo(param)

        if self.for_downstream:
            if self.infer_mode == 'zero_shot':
                return self.forward_zeroshot(param)
            if self.infer_mode == 'linear_probe':
                return self.forward_sentivisual(param)

        if self.model_type == 'image only':
            return self.forward_sentivisual(param)
        if self.model_type == 'text only':
            return self.forward_sentitextual(param)
        if self.model_type == 'image-text pretrain':
            return self.forward_clip(param)
        raise TypeError("Model type is not assigned")

    def forward_sentivisual(self, param: ModelParam):
        image_feature = self.image_model(param.images)#(batch_size, 49, 2048)
        if isinstance(image_feature, tuple):
            image_embed = image_feature[1]
            image_feature = image_feature[0]
        else:
            image_feature = image_feature.transpose(1, 2)
            image_embed = self.avg_pool(image_feature)
            image_embed = image_embed.view(image_embed.size(0), -1)
            image_feature = image_feature.transpose(1, 2)

        output = self.classify_forward(image_embed, self.image_classify, self.downstream_mode)
        return output, image_embed

    def forward_sentitextual(self, param: ModelParam):
        text_encoder = self.text_model(param.texts)
        text_encoder = text_encoder.transpose(1, 2)
        text_cls = self.avg_pool(text_encoder)
        text_cls = text_cls.view(text_cls.size(0), -1)
        output = self.classify_forward(text_cls, self.text_classify, self.downstream_mode)
        return output

    def forward_webemo(self, param: ModelParam):
        image_feature = self.image_model(param.images)#(batch_size, 49, 2048)
        if isinstance(image_feature, tuple):
            image_embed = image_feature[1]
            image_feature = image_feature[0]
        else:
            image_feature = image_feature.transpose(1, 2)
            image_embed = self.avg_pool(image_feature)
            image_embed = image_embed.view(image_embed.size(0), -1)
            image_feature = image_feature.transpose(1, 2)

        output1 = self.classify_forward(image_embed, self.image_classify, self.downstream_mode)
        output2 = self.classify_forward(image_embed, self.image_classify7, self.downstream_mode)
        output3 = self.classify_forward(image_embed, self.image_classify25, self.downstream_mode)
        return output1, output2, output3

    def forward_clip(self, param: ModelParam):
        image_feature = self.image_model(param.images)  # (batch_size, 49, 2048)
        if isinstance(image_feature, tuple):
            image_embed = image_feature[1]
            image_feature = image_feature[0]
        else:
            image_feature = image_feature.transpose(1, 2)
            image_embed = self.avg_pool(image_feature)
            image_embed = image_embed.view(image_embed.size(0), -1)
            image_feature = image_feature.transpose(1, 2)
        image_cov = F.normalize(self.image_to_embed_projection(image_embed), dim=0)
        image_embed = F.normalize(self.image_to_embed_projection(image_embed), dim=1)
        #image_embed = F.normalize(self.image_2_k(image_embed), dim=1)

        with torch.no_grad():
            text_feature = self.text_model(param.texts)
            text_embed = text_feature[:,0,:]
        text_cov = F.normalize(self.text_to_embed_projection(text_embed), dim=0)
        text_embed = F.normalize(self.text_to_embed_projection(text_embed), dim=1)
        #text_embed = F.normalize(self.text_2_q(text_embed), dim=1)

        sim_per_image = torch.matmul(image_embed, text_embed.T)
        sim_per_text = torch.matmul(text_embed, image_embed.T)

        #For image text contrastive
        logits_per_image = sim_per_image * self.logit_scale
        logits_per_text = sim_per_text * self.logit_scale

        bs = image_feature.size(0)
        text_q = self.text_2_q(text_feature.clone().detach())
        #image_k = self.image_2_k(image_feature.clone().detach())
        image_k = self.image_2_k(image_feature)
        weight_pos = torch.einsum('btd,bid->bit', text_q, image_k)
        weight_pos = self.max_pool(weight_pos).view(bs, -1)

        with torch.no_grad():
            weights_i2t = F.softmax(logits_per_image[:, :bs], dim=1)
            weights_t2i = F.softmax(logits_per_text[:, :bs], dim=1)

            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)
            orig_weights_i2t = weights_i2t.clone()
            orig_weights_t2i = weights_t2i.clone()

            if param.use_senti_mask == True:
                senti_mask = self.create_cluster_map(param.senti_cluster)
                senti_mask = senti_mask.to(self.device)
                weights_i2t = weights_i2t.mul(senti_mask)
                weights_t2i = weights_t2i.mul(senti_mask)
            elif param.use_fact_mask == True:
                fact_mask = self.create_cluster_map(param.fact_cluster)
                fact_mask = fact_mask.to(self.device)
                weights_i2t = weights_i2t.mul(fact_mask)
                weights_t2i = weights_t2i.mul(fact_mask)

        # select a negative image for each text
        image_feature_neg = []
        for b in range(bs):
            try:
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            except:
                neg_idx = torch.multinomial(orig_weights_t2i[b], 1).item()
            image_feature_neg.append(image_feature[neg_idx])
        image_feature_neg = torch.stack(image_feature_neg, dim=0)
        #image_k_neg = self.image_2_k(image_feature_neg.clone().detach())
        image_k_neg = self.image_2_k(image_feature_neg)
        weight_i_neg = torch.einsum('btd,bid->bit', text_q, image_k_neg)
        weight_i_neg = self.max_pool(weight_i_neg).view(bs, -1)

        text_feature_neg = []
        for b in range(bs):
            try:
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            except:
                neg_idx = torch.multinomial(orig_weights_i2t[b], 1).item()
            text_feature_neg.append(text_feature[neg_idx])
        text_feature_neg = torch.stack(text_feature_neg, dim=0)
        text_q_neg = self.text_2_q(text_feature_neg.clone().detach())
        weight_t_neg = torch.einsum('btd,bid->bit', text_q_neg, image_k)
        weight_t_neg = self.max_pool(weight_t_neg).view(bs, -1)

        itm_feature = torch.cat([weight_pos, weight_i_neg, weight_t_neg], dim=0)
        itm_output = self.itm_classify(itm_feature)

        #Calculate gram matric distillation loss
        text_cov_matrix = torch.matmul(text_cov.T, text_cov)
        image_cov_matrix = torch.matmul(image_cov.T, image_cov)

        return logits_per_text, logits_per_image, text_cov_matrix, image_cov_matrix, itm_output

        #return logits_per_text, logits_per_image, itm_output

    def forward_zeroshot(self, param: ModelParam):
        image_feature = self.image_model(param.images)  # (batch_size, 49, 2048)
        if isinstance(image_feature, tuple):
            image_embed = image_feature[1]
            image_feature = image_feature[0]
        else:
            image_feature = image_feature.transpose(1, 2)
            image_embed = self.avg_pool(image_feature)
            image_embed = image_embed.view(image_embed.size(0), -1)
            image_feature = image_feature.transpose(1, 2)
        image_cov = F.normalize(self.image_to_embed_projection(image_embed), dim=0)
        image_embed = F.normalize(self.image_to_embed_projection(image_embed), dim=1)
        #image_embed = F.normalize(self.image_2_k(image_embed), dim=1)

        with torch.no_grad():
            text_feature = self.text_model(param.texts)
            text_embed = text_feature[:,0,:]
        text_cov = F.normalize(self.text_to_embed_projection(text_embed), dim=0)
        text_embed = F.normalize(self.text_to_embed_projection(text_embed), dim=1)
        #text_embed = F.normalize(self.text_2_q(text_embed), dim=1)

        sim_per_image = torch.matmul(image_embed, text_embed.T)
        sim_per_text = torch.matmul(text_embed, image_embed.T)

        #For image text contrastive
        logits_per_image = sim_per_image * self.logit_scale
        logits_per_text = sim_per_text * self.logit_scale

        return logits_per_text, logits_per_image


    def create_cluster_map(self, label):
        bs = len(label)
        mask = torch.ones((bs, bs))
        for i in range(bs):
            for j in range(bs):
                if label[i] == label[j]:
                    mask[i, j] = 0
        mask.fill_diagonal_(0)
        return mask


    def initialize_parameters(self):
        for name, param in self.named_parameters():
            if 'scene_image_model' in name or 'general_text_model' in name or 'senti_text_model' in name:
                continue
            elif 'classify' in name:
                nn.init.normal_(param, std=0.02)
            elif 'cross_model' in name:
                std = self.image_model.cross_model.get_output_dim() ** -0.5
                nn.init.normal_(param, std=std)
            elif 'senti_image_model' in name:
                std = self.image_model.senti_image_model.get_output_dim() ** -0.5
                nn.init.normal_(param, std=std)
            else:
                std = self.hidden_size ** -0.5
                nn.init.normal_(param, std=std)

    def _dequeue_and_enqueue(self, text_embed):
        batch_size = text_embed.shape[0]

        ptr = int(self.queue_ptr)

        # for simplicity
        if self.queue_size % batch_size != 0:
            return

        # replace the keys at ptr (dequeue and enqueue)
        self.text_queue[:, ptr:ptr + batch_size] = text_embed.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr


    def get_text_tokenizor(self):
        tokenizer = self.text_model.get_tokenizer()
        return tokenizer

    def load_image_model(self, args, checkpoint_path = None):
        if checkpoint_path != None:
            checkpoint = torch.load(checkpoint_path)
            self.image_model.load_state_dict(checkpoint, strict=False)
        self.for_downstream = True
        self.create_classify_head(args)
        self.to(args.device)

    def load_full_model(self, args, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint, strict=False)
        self.for_downstream = True
        self.create_classify_head(args.dataset_name)
        self.to(args.device)

    def create_classify_head(self, dataset_name, class_num=None):
        if class_num != None:
            tmp = self.class_num
            self.class_num = class_num

        if dataset_name == 'WEBEmo':
            assert isinstance(self.class_num, list)
            assert len(self.class_num) == 3
            self.image_classify = nn.Sequential(
                nn.Linear(self.image_output_dim, self.image_output_dim),
                nn.ReLU(),
                nn.Linear(self.image_output_dim, self.class_num[0])
            )
            self.image_classify7 = nn.Sequential(
                nn.Linear(self.image_output_dim, self.image_output_dim),
                nn.ReLU(),
                nn.Linear(self.image_output_dim, self.class_num[1])
            )
            self.image_classify25 = nn.Sequential(
                nn.Linear(self.image_output_dim, self.image_output_dim),
                nn.ReLU(),
                nn.Linear(self.image_output_dim, self.class_num[2])
            )
        elif dataset_name == 'Emotic':
            self.image_classify = nn.Sequential(
                nn.Linear(self.image_output_dim, self.image_output_dim),
                nn.ReLU(),
                nn.Linear(self.image_output_dim, 3)
            )
        else:
            self.image_classify = nn.Sequential(
                nn.Linear(self.image_output_dim, self.image_output_dim),
                nn.ReLU(),
                nn.Linear(self.image_output_dim, self.class_num)
            )
            self.text_classify = nn.Sequential(
                nn.Linear(self.text_output_dim, self.text_output_dim),
                nn.ReLU(),
                nn.Linear(self.text_output_dim, self.class_num)
            )
        if class_num != None:
            self.class_num = tmp

    def classify_forward(self, x, classifier, mode):
        if mode == 'freeze':
            return classifier(x.clone().detach())
        else:
            assert mode == 'finetune'
            return classifier(x)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, text_feat):
        batch_size = text_feat.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.text_queue[:, ptr:ptr + batch_size] = text_feat.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

class SentiVisualModel(nn.Module):
    def __init__(self, args, cfg):
        super(SentiVisualModel, self).__init__()
        #[single/both, [sentiment/scene]/[attention/addition]]
        self.model_type = cfg.image_model_type
        dataset_config = cfg.get_dataset_config()[args.dataset_name]

        if self.model_type[0] == 'both' or self.model_type[1] == 'scene':
            self.scene_image_model = UniImageEncoder(args, cfg, True, True)
        if self.model_type[0] == 'both' or self.model_type[1] == 'sentiment':
            self.senti_image_model = UniImageEncoder(args, cfg, True, False)
        if self.model_type[1] == 'attention':
            self.cross_model = FusionModel(self.scene_image_model.get_output_dim())

    def get_output_dim(self):
        if self.model_type[1] == 'attention':
            return self.cross_model.get_output_dim()
        else:
            if self.model_type[0] == 'both' or self.model_type[1] == 'scene':
                return self.scene_image_model.get_output_dim()
            elif self.model_type[1] == 'sentiment':
                return self.senti_image_model.get_output_dim()
            else:
                raise TypeError('Model type out of range')

    def get_channel_dim(self):
        if self.model_type[1] == 'attention':
            raise ValueError('Should not be reached')
        else:
            if self.model_type[0] == 'both' or self.model_type[1] == 'scene':
                return self.scene_image_model.get_channel_dim()
            elif self.model_type[1] == 'sentiment':
                return self.senti_image_model.get_channel_dim()
            else:
                raise TypeError('Model type out of range')


    def forward(self, image):
        if self.model_type[0] == 'single':
            visual_encoder = self.forward_single_branch(image, self.model_type[1])
        else:
            assert self.model_type[0] == 'both'
            if self.model_type[1] == 'attention':
                visual_encoder = self.forward_self_attention(image)
            elif self.model_type[1] == 'addition':
                visual_encoder =self.forward_addition(image)
            else:
                raise TypeError("Visual model type [1] undefined.")
        return visual_encoder

    def forward_single_branch(self, image, type):
        if type == 'sentiment':
            visual_encoder, visual_cls = self.senti_image_model(image)
            return visual_encoder, visual_cls
        elif type == 'scene':
            visual_encoder, visual_cls = self.scene_image_model(image)
            return visual_encoder, visual_cls
        elif type == 'attention':
            visual_encoder, visual_cls = self.senti_image_model(image)
            visual_encoder = self.cross_model(visual_encoder)
            return visual_encoder
        else:
            raise TypeError("Visual model type [1] undefined.")


    def forward_self_attention(self, image):
        scene_encoder, scene_cls = self.scene_image_model(image)
        senti_encoder, senti_cls = self.senti_image_model(image)
        visual_encoder = self.cross_model(senti_encoder, scene_encoder)
        return visual_encoder

    def forward_addition(self, image):
        scene_encoder, scene_cls = self.scene_image_model(image)
        senti_encoder, senti_cls = self.senti_image_model(image)
        visual_encoder = scene_encoder + senti_encoder
        return visual_encoder

class SentiTextualModel(nn.Module):
    def __init__(self, args, cfg):
        super(SentiTextualModel, self).__init__()
        # [single/both, [sentiment/general]/[attention/addition]]
        self.model_type = cfg.text_model_type
        dataset_config = cfg.get_dataset_config()[args.dataset_name]

        if self.model_type[0] == 'both' or self.model_type[1] == 'general':
            self.general_text_model = UniTextEncoder(args, cfg, True, False)
        if self.model_type[0] == 'both' or self.model_type[1] == 'sentiment':
            self.senti_text_model = UniTextEncoder(args, cfg, True, True)
        if self.model_type[1] == 'attention':
            self.cross_model = FusionModel(self.general_text_model.get_output_dim())


    def get_tokenizer(self):
        if self.model_type[0] == 'both' or self.model_type[1] == 'general':
            return self.general_text_model.get_tokenizer()
        elif self.model_type[1] == 'sentiment':
            return self.senti_text_model.get_tokenizer()
        else:
            raise TypeError('Model type out of range')

    def get_output_dim(self):
        if self.model_type[1] == 'attention':
            return self.cross_model.get_output_dim()
        else:
            if self.model_type[0] == 'both' or self.model_type[1] == 'general':
                return self.general_text_model.get_output_dim()
            elif self.model_type[1] == 'sentiment':
                return self.senti_text_model.get_output_dim()
            else:
                raise TypeError('Model type out of range')

    def forward(self, text):
        if self.model_type[0] == 'single':
            textual_encoder = self.forward_single_branch(text, self.model_type[1])
        else:
            assert self.model_type[0] == 'both'
            if self.model_type[1] == 'attention':
                textual_encoder = self.forward_self_attention(text)
            elif self.model_type[1] == 'addition':
                textual_encoder = self.forward_addition(text)
            else:
                raise TypeError("Textual model type [1] undefined.")
        return textual_encoder

    def forward_single_branch(self, text, type):
        if type == 'sentiment':
            textual_encoder, textual_cls = self.senti_text_model(text)
            return textual_encoder
        elif type == 'general':
            textual_encoder, textual_cls = self.general_text_model(text)
            return textual_encoder
        else:
            raise TypeError("Textual model type [1] undefined.")

    def forward_self_attention(self, text):
        general_encoder, general_cls = self.general_text_model(text)
        senti_encoder, senti_cls = self.senti_text_model(text)
        textual_encoder = self.cross_model(senti_encoder, general_encoder)
        return textual_encoder

    def forward_addition(self, text):
        general_encoder, general_cls = self.general_text_model(text)
        senti_encoder, senti_cls = self.senti_text_model(text)
        textual_encoder = torch.cat((general_encoder, senti_encoder), dim=1)
        return textual_encoder