import os
import torch
import torch.nn as nn
import torch.nn.functional as F


from transformers import BertTokenizer, BertModel, BertConfig, AutoTokenizer

class FusionModel(nn.Module):
    def __init__(self, input_size = 768, num_attention_heads = 12, requires_grad=True):
        super(FusionModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = 768
        self.config = BertConfig(num_hidden_layers = 1, num_attention_heads = num_attention_heads)
        self.model = BertModel(self.config)

        for param in self.model.parameters():
            param.requires_grad = requires_grad

        self.projector = nn.Linear(self.input_size, 768)

        self.model = nn.Sequential(*(list(self.model.children())))[1:-1]
        self.output_dim = self.hidden_size


    def get_output_dim(self):
        return self.output_dim

    def get_config(self):
        return self.config

    def forward(self, *input):
        if len(input) == 2:
            return self.forward_multi(input[0], input[1])
        if len(input) == 1:
            return self.forward_single(input[0])
        else:
            raise TypeError("Forward can not take more then 2 augments.")

    def forward_single(self, input):
        input = self.projector(input)
        output = self.model(input)
        output = output.last_hidden_state
        return output

    def forward_multi(self, input1, input2):
        input_cat = torch.cat((input1, input2), dim = 1)
        input_cat = self.projector(input_cat)
        output = self.model(input_cat)
        output = output.last_hidden_state
        return output

class CrossAttention(nn.Module):
    def __init__(self, args, cfg, requires_grad=True):
        super(CrossAttention, self).__init__()
        dataset_config = cfg.get_dataset_config()

        model_dir = dataset_config['pretrain_models']['bert-base-uncased']
        self.config = BertConfig(is_decoder=True, add_cross_attention=True, num_hidden_layers=1)
        self.model = BertModel.from_pretrained(model_dir, self.config)
        for param in self.model.parameters():
            param.requires_grad = requires_grad


    def forward(self, text_feature, image_feature):
        image_mask = torch.ones(image_feature.size()[:-1],dtype=torch.long).cuda()
        output = self.model(inputs_embeds = text_feature,
                            encoder_hidden_states = image_feature,
                            encoder_attention_mask = image_mask)
        return output.last_hidden_state, output.pooler_output

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    model = FusionModel()