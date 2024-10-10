import torch
import torch.nn as nn

import timm
from model.clip import ClipImageEncoder
#from transformers import AutoImageProcessor, ViTModel
#from transformers import AutoImageProcessor, ResNetModel

class UniImageEncoder(nn.Module):
    def __init__(self, args, cfg, requires_grad = True, from_pretrained = True):
        super(UniImageEncoder, self).__init__()
        model_cfg = cfg.get_model_config()
        self.image_encoder_type = model_cfg['commonParas']['image_encoder']

        if self.image_encoder_type == 'resnet-50':
            self.image_model = ResNetImageEncoder(args, cfg, requires_grad, from_pretrained)
        elif self.image_encoder_type == 'vit':
            self.image_model = ViTImageEncoder(args, cfg, requires_grad, from_pretrained)
        elif self.image_encoder_type == 'swin':
            self.image_model = SwinImageEncoder(args, cfg, requires_grad, from_pretrained)
        elif self.image_encoder_type == 'clip':
            self.image_model = ClipImageEncoder(args, cfg, requires_grad)
        else:
            raise TypeError("Image Encoder type undefined.")

    def get_output_dim(self):
        return self.image_model.get_output_dim()

    def get_channel_dim(self):
        return self.image_model.get_channel_dim()

    def forward(self, images):
        return self.image_model(images)


class ResNetImageEncoder(nn.Module):
    def __init__(self, args, cfg, requires_grad = True, from_pretrained = True):
        super(ResNetImageEncoder, self).__init__()

        model_cfg = cfg.get_model_config()
        dataset_cfg = cfg.get_dataset_config()
        checkpoint_path = dataset_cfg['pretrain_models']['resnet50']
        self.resnet = timm.create_model('resnet50', num_classes=0)
        if from_pretrained:
            checkpoint = torch.load(checkpoint_path)
            self.resnet.load_state_dict(checkpoint, strict=False)

        self.resnet_encoder = nn.Sequential(*(list(self.resnet.children())[:-2]))
        self.resnet_avgpool = nn.Sequential(list(self.resnet.children())[-2])
        self.output_dim = 2048

        image_size = model_cfg['commonParas']['image_size']
        if image_size == [224, 224]:
            self.channel_dim = 49
        elif image_size == [448, 448]:
            self.channel_dim = 196
        else:
            self.channel_dim = -1

        for param in self.resnet.parameters():
            param.requires_grad = requires_grad

    def get_output_dim(self):
        return self.output_dim

    def get_channel_dim(self):
        return self.channel_dim


    def forward(self, images):
        image_encoder = self.resnet_encoder(images)  #[batch_size, 2048, 7, 7]
        # image_encoder = self.conv_output(image_encoder)
        image_cls = self.resnet_avgpool(image_encoder)
        image_cls = torch.flatten(image_cls, 1)  #[batch_size, 2048]
        image_encoder = image_encoder.view(image_encoder.size(0), -1, image_encoder.size(1))#(batch_size, 49, 2048)
        return image_encoder, image_cls


class TransformerResNetImageEncoder(nn.Module):
    def __init__(self, args, cfg, requires_grad = True, from_pretrained = True):
        super(TransformerResNetImageEncoder, self).__init__()

        model_cfg = cfg.get_model_config()
        dataset_cfg = cfg.get_dataset_config()
        checkpoint_path = dataset_cfg['pretrain_models']['resnet50']
        self.resnet = timm.create_model('resnet50', num_classes=0)
        if from_pretrained:
            checkpoint = torch.load(checkpoint_path)
            self.resnet.load_state_dict(checkpoint, strict=False)

        self.resnet_encoder = nn.Sequential(*(list(self.resnet.children())[:-2]))
        self.resnet_avgpool = nn.Sequential(list(self.resnet.children())[-2])
        self.output_dim = 2048

        image_size = model_cfg['commonParas']['image_size']
        if image_size == [224, 224]:
            self.channel_dim = 49
        elif image_size == [448, 448]:
            self.channel_dim = 196
        else:
            self.channel_dim = -1

        for param in self.resnet.parameters():
            param.requires_grad = requires_grad

    def get_output_dim(self):
        return self.output_dim

    def get_channel_dim(self):
        return self.channel_dim


    def forward(self, images):
        image_encoder = self.resnet_encoder(images)  #[batch_size, 2048, 7, 7]
        # image_encoder = self.conv_output(image_encoder)
        image_cls = self.resnet_avgpool(image_encoder)
        image_cls = torch.flatten(image_cls, 1)  #[batch_size, 2048]
        image_encoder = image_encoder.view(image_encoder.size(0), -1, image_encoder.size(1))#(batch_size, 49, 2048)
        return image_encoder, image_cls


class ViTImageEncoder(nn.Module):
    def __init__(self, args, cfg, requires_grad = True, from_pretrained = True):
        super(ViTImageEncoder, self).__init__()

        model_cfg = cfg.get_model_config()
        dataset_cfg = cfg.get_dataset_config()
        checkpoint_path = dataset_cfg['pretrain_models']['vit-base']

        self.vit = timm.create_model('vit_base_patch16_224', num_classes=0)
        if from_pretrained:
            checkpoint = torch.load(checkpoint_path)
            self.vit.load_state_dict(checkpoint, strict=False)

        #self.vit = timm.create_model('vit_base_patch32_224_clip_laion2b', pretrained=True, num_classes=0)

        #self.vit = nn.Sequential(*(list(self.vit.children())))

        for param in self.vit.parameters():
            param.requires_grad = requires_grad

        self.output_dim = 768
        self.channel_dim = 196

    def get_output_dim(self):
        return self.output_dim

    def get_channel_dim(self):
        return self.channel_dim

    def forward(self, images):
        image_encoder = self.vit.forward_features(images)  # (batch_size, 49, 768)
        image_cls = self.vit.forward_head(image_encoder)
        image_encoder = image_encoder[:,0:-1,:]

        return image_encoder, image_cls



class SwinImageEncoder(nn.Module):
    def __init__(self, args, cfg, requires_grad = True, from_pretrained = True):
        super(SwinImageEncoder, self).__init__()

        model_cfg = cfg.get_model_config()
        dataset_cfg = cfg.get_dataset_config()
        checkpoint_path = dataset_cfg['pretrain_models']['swin-base']

        self.swin = timm.create_model('swin_s3_base_224', num_classes=0)
        if from_pretrained:
            checkpoint = torch.load(checkpoint_path)
            self.swin.load_state_dict(checkpoint, strict=False)

        #self.swin_encoder = nn.Sequential(*(list(self.swin.children())))

        for param in self.swin.parameters():
            param.requires_grad = requires_grad

        self.output_dim = 768
        self.channel_dim = 49

    def get_output_dim(self):
        return self.output_dim

    def get_channel_dim(self):
        return self.channel_dim

    def forward(self, images):
        image_encoder = self.swin.forward_features(images)#(batch_size, 49, 768)
        image_cls = self.swin.forward_head(image_encoder)
        image_encoder = image_encoder.view(image_encoder.size(0), -1, image_encoder.size(3))

        return image_encoder, image_cls