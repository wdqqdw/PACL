import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from transformers import AutoProcessor, AutoTokenizer, CLIPProcessor, \
    CLIPModel, CLIPTextModel, CLIPVisionModel, CLIPTextConfig, CLIPVisionConfig


class ClipEncoder(nn.Module):
    def __init__(self, args, cfg, requires_grad=True):
        super(ClipEncoder, self).__init__()
        dataset_config = cfg.get_dataset_config()

        model_dir = dataset_config['pretrain_models']['clip-vit-base-patch32']
        self.model = CLIPModel.from_pretrained(model_dir)
        self.processor = CLIPProcessor.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.device = args.device

        for param in self.model.parameters():
            param.requires_grad = requires_grad

        self.output_dim = 768

    def get_output_dim(self):
        return self.output_dim

    def forward(self, text, image):
        input = {}
        input['input_ids'] = text['input_ids'].to(self.device)
        input['attention_mask'] = text['attention_mask'].to(self.device)
        image = [i.view(1, i.size(0), i.size(1), i.size(2)) for i in image]
        input['pixel_values'] = torch.cat(image, dim=0).to(self.device)

        outputs = self.model(**input)
        #Find text according to image
        logits_per_image = outputs.logits_per_image
        probs_per_image = logits_per_image.softmax(dim=1)
        logits_per_text = outputs.logits_per_text
        probs_per_text = logits_per_text.softmax(dim=1)
        #return probs_per_image, probs_per_text
        return probs_per_image, probs_per_text

    def get_text_tokenizor(self):
        return self.tokenizer

    def get_text_features(self, text):
        text_features = self.model.get_text_features(**text)
        return text_features

    def get_image_features(self, image):
        inputs = self.processor(image = image, return_tensors="pt")
        image_features = self.model.get_image_features(**inputs)
        return image_features

class ClipTextEncoder(nn.Module):
    def __init__(self, args, cfg, requires_grad=True):
        super(ClipTextEncoder, self).__init__()
        model_cfg = cfg.get_model_config()
        dataset_config = cfg.get_dataset_config()
        self.text_max_length = model_cfg['commonParas']['text_max_length']

        model_dir = dataset_config['pretrain_models']['clip-vit-base-patch32']

        self.config_text = CLIPTextConfig()
        self.model = CLIPTextModel.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        for param in self.model.parameters():
            param.requires_grad = requires_grad

        self.device = args.device
        self.output_dim = self.model.config.hidden_size

    def get_output_dim(self):
        return self.output_dim

    def forward(self, batch_text):
        for key in batch_text.keys():
            batch_text[key] = batch_text[key].long().to(self.device)

        text_encoder = self.model(**batch_text)
        text_cls = text_encoder.pooler_output
        text_encoder = text_encoder.last_hidden_state
        return text_encoder, text_cls

    def get_tokenizer(self):
        return self.tokenizer


class ClipImageEncoder(nn.Module):
    def __init__(self, args, cfg, requires_grad=True):
        super(ClipImageEncoder, self).__init__()

        model_cfg = cfg.get_model_config()
        dataset_config = cfg.get_dataset_config()

        model_dir = dataset_config['pretrain_models']['clip-vit-base-patch32']

        self.config_text = CLIPVisionConfig()
        self.model = CLIPVisionModel.from_pretrained(model_dir)
        self.processor = AutoProcessor.from_pretrained(model_dir)

        for param in self.model.parameters():
            param.requires_grad = requires_grad

        self.device = args.device
        self.output_dim = self.model.config.hidden_size
        self.channel_dim = 50

    def get_output_dim(self):
        return self.output_dim

    def get_channel_dim(self):
        return self.channel_dim

    def forward(self, image):
        image = [i.view(1, i.size(0), i.size(1), i.size(2)) for i in image]
        input = {}
        input['pixel_values'] = torch.cat(image, dim=0).to(self.device)


        # input = self.processor(images=input, return_tensors="pt")
        # input['pixel_values'] = input['pixel_values'].to('cuda')
        image_encoder = self.model(**input)
        image_cls = image_encoder.pooler_output
        image_encoder = image_encoder.last_hidden_state
        return image_encoder, image_cls


if __name__ == "__main__":
    model_dir = '/data3/wdq/STACL/pretain_models/clip-vit-base-patch32'
    model = CLIPVisionModel.from_pretrained(model_dir)
    processor = AutoProcessor.from_pretrained(model_dir)

    image_path = '/data3/wdq/MACL/data/MVSA-single/dataset_image/1.jpg'
    image = Image.open(image_path)

    inputs = processor(images=image, return_tensors="pt")

    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    pooled_output = outputs.pooler_output  # pooled CLS states