import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel, BertConfig, AutoTokenizer, RobertaModel, RobertaTokenizer
from model.clip import ClipTextEncoder
from torch.nn.utils.rnn import pad_sequence

class UniTextEncoder(nn.Module):
    def __init__(self, args, cfg, requires_grad = True, senti_pretrained = False):
        super(UniTextEncoder, self).__init__()
        model_cfg = cfg.get_model_config()
        self.text_encoder_type = model_cfg['commonParas']['text_encoder']

        if self.text_encoder_type == 'bert':
            assert senti_pretrained == False
            self.text_model = BertTextEncoder(args, cfg, requires_grad)
        elif self.text_encoder_type in ['bertweet', 'skep']:
            self.text_model = RobertaTextEncoder(args, cfg, requires_grad)
        elif self.text_encoder_type == 'clip':
            self.text_model = ClipTextEncoder(args, cfg, requires_grad)
        else:
            raise TypeError("Text Encoder type undefined.")

    def get_output_dim(self):
        return self.text_model.get_output_dim()

    def get_tokenizer(self):
        return self.text_model.get_tokenizer()

    def get_channel_dim(self):
        return self.text_model.get_channel_dim()

    def forward(self, input):
        return self.text_model(input)




class BertTextEncoder(nn.Module):
    def __init__(self, args, cfg, requires_grad = True):
        super(BertTextEncoder, self).__init__()
        dataset_config = cfg.get_dataset_config()
        model_cfg = cfg.get_model_config()
        self.text_max_length = model_cfg['commonParas']['text_max_length']

        model_dir = dataset_config['pretrain_models']['bert-base-uncased']
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = BertModel.from_pretrained(model_dir)

        if not requires_grad:
            for param in self.model.parameters():
                param.requires_grad = requires_grad

        self.device = args.device
        self.output_dim = self.model.config.hidden_size

    def get_output_dim(self):
        return self.output_dim

    def get_channel_dim(self):
        return self.text_max_length

    def get_tokenizer(self):
        return self.tokenizer
    
    def forward(self, batch_text):
        for key in batch_text.keys():
            batch_text[key] = batch_text[key].long().to(self.device)

        text_encoder = self.model(**batch_text)
        text_cls = text_encoder.pooler_output
        text_encoder = text_encoder.last_hidden_state
        return text_encoder, text_cls


class RobertaTextEncoder(nn.Module):
    def __init__(self, args, cfg, requires_grad=True):
        super(RobertaTextEncoder, self).__init__()
        dataset_config = cfg.get_dataset_config()

        if args.text_encoder == 'bertweet':
            model_dir = dataset_config['pretrain_models']['bertweet-base-sentiment-analysis']
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        else:
            assert args.text_encoder == 'skep'
            model_dir = dataset_config['pretrain_models']['roberta-large-ernie2-skep-en']
            self.tokenizer = RobertaTokenizer.from_pretrained(model_dir)
        self.model = RobertaModel.from_pretrained(model_dir)

        if not requires_grad:
            for param in self.model.parameters():
                param.requires_grad = requires_grad

        self.device = args.device
        self.output_dim = self.model.config.hidden_size


    def get_output_dim(self):
        return self.output_dim

    def get_tokenizer(self):
        return self.tokenizer

    def get_channel_dim(self):
        return self.text_max_length

    def get_config(self):
        return self.config

    def forward(self, batch_text):
        for key in batch_text.keys():
            batch_text[key] = batch_text[key].long().to(self.device)

        text_encoder = self.model(**batch_text)
        text_cls = text_encoder.pooler_output
        text_encoder = text_encoder.last_hidden_state
        return text_encoder, text_cls
    

#generate by new bing
class CrossAttention(nn.Module):
    def __init__(self, hidden_size1=768, hidden_size2=768):
        super(CrossAttention, self).__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2

        self.query1 = nn.Linear(hidden_size1, hidden_size2)
        self.key1 = nn.Linear(hidden_size2, hidden_size2)
        self.value1 = nn.Linear(hidden_size2, hidden_size2)

        self.query2 = nn.Linear(hidden_size1, hidden_size2)
        self.key2 = nn.Linear(hidden_size2, hidden_size2)
        self.value2 = nn.Linear(hidden_size2, hidden_size2)

        self.query3 = nn.Linear(hidden_size1, hidden_size2)
        self.key3 = nn.Linear(hidden_size2, hidden_size2)
        self.value3 = nn.Linear(hidden_size2, hidden_size2)

        #self.ln = nn.LayerNorm(3*hidden_size2)
        self.project_multihead = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_size2*3 , self.hidden_size2),
        )


    def get_output_dim(self):
        return self.hidden_size2



    #multihead attention (3)
    def forward(self, tensor1, tensor2, attention_mask=None):
        # tensor1 shape: (batch_size=32, seq_length=49, hidden_size=2048)
        # tensor2 shape: (batch_size=32, seq_length=word_length, hidden_size=768)
        mixed_query_layer1 = self.query1(tensor1)
        mixed_key_layer1 = self.key1(tensor2)
        mixed_value_layer1 = self.value1(tensor2)
        attention_scores1 = torch.bmm(mixed_query_layer1, mixed_key_layer1.transpose(1, 2))
        if attention_mask is not None:
            attention_scores1 += attention_mask

        # Compute attention weights
        attention_weights1 = nn.Softmax(dim=-1)(attention_scores1)  # shape: (batch_size=32 , seq_len1=49 , seq_len2=word_length)
        # Compute weighted average of value
        context1 = torch.bmm(attention_weights1, mixed_value_layer1)


        mixed_query_layer2 = self.query2(tensor1)
        mixed_key_layer2 = self.key2(tensor2)
        mixed_value_layer2 = self.value2(tensor2)
        attention_scores2 = torch.bmm(mixed_query_layer2, mixed_key_layer2.transpose(1, 2))
        if attention_mask is not None:
            attention_scores2 += attention_mask

        # Compute attention weights
        attention_weights2 = nn.Softmax(dim=-1)(attention_scores2)  # shape: (batch_size=32 , seq_len1=49 , seq_len2=word_length)
        # Compute weighted average of value
        context2 = torch.bmm(attention_weights2, mixed_value_layer2)


        mixed_query_layer3 = self.query3(tensor1)
        mixed_key_layer3 = self.key3(tensor2)
        mixed_value_layer3 = self.value3(tensor2)
        attention_scores3 = torch.bmm(mixed_query_layer3, mixed_key_layer3.transpose(1, 2))
        if attention_mask is not None:
            attention_scores3 += attention_mask

        # Compute attention weights
        attention_weights3 = nn.Softmax(dim=-1)(attention_scores3)  # shape: (batch_size=32 , seq_len1=49 , seq_len2=word_length)
        # Compute weighted average of value
        context3 = torch.bmm(attention_weights3, mixed_value_layer3)

        multi_head_output = torch.cat((context1, context2, context3), dim=2)
        output = self.project_multihead(multi_head_output)
        return output

