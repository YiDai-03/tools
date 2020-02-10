import torch
import torch.autograd as autograd
import torch.nn as nn
from transformers import *
from ..layers.crf import CRF
from ..layers.bilstm import BILSTM
import time

class BERT_LSTM(nn.Module):

    def __init__(self, num_classes, model_config, device,bert_route='bert-base-chinese',  num_layers=1):
        super(BERT_LSTM, self).__init__()
        self.tagset_size = num_classes     # num of tags for final softmax layer

        if (model_config['crf'] == False):
            classes = num_classes
        else:
            classes = num_classes + 2
        self.bert_encoder = BertModel.from_pretrained(bert_route)


        # also input dim of LSTM
        self.bert_out_dim = self.bert_encoder.config.hidden_size
        # LSTM layer
        self.lstm = BILSTM(input_size = self.bert_out_dim,
                           hidden_size= model_config['hidden_size'],
                           num_layer  = model_config['num_layer'],
                           bi_tag     = model_config['bi_tag'],
                           dropout_p  = model_config['dropout_p'],
                           num_classes= classes)

        self.crf = CRF(device = device,tagset_size=num_classes, have_crf = model_config['crf'])


    def forward(self, sent, length, masks = None, gaz =None):
        # sent,tags,masks: (batch * seq_length)
        bert_out = self.bert_encoder(sent, masks)[0]
        # bert_out: (batch * seq_length * bert_hidden=768)
        x = self.lstm(bert_out,length)

        
        return x
