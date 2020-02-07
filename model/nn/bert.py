import torch
import torch.autograd as autograd
import torch.nn as nn
from transformers import *
from ..layers.crf import CRF
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
        self.lstm = nn.LSTM(self.bert_out_dim, model_config['hidden_size'] // 2, batch_first=True,
                            num_layers=num_layers, bidirectional=True)
        # map LSTM output to tag space
        self.linear = nn.Linear(model_config['hidden_size'], classes)
        nn.init.xavier_uniform(self.linear.weight)
        self.crf = CRF(device = device,tagset_size=num_classes, have_crf = model_config['crf'])


    def forward(self, sent, masks):
        # sent,tags,masks: (batch * seq_length)
        bert_out = self.bert_encoder(sent, masks)[0]
        # bert_out: (batch * seq_length * bert_hidden=768)
        lstm_out, _ = self.lstm(bert_out)
        # lstm_out: (batch * seq_length * lstm_hidden)
        feats = self.linear(lstm_out)
        # feats: (batch * seq_length * tag_size)
        
        return feats

    def eval(self):
        self.bert_encoder.eval()

    def train(self):
        self.bert_encoder.train()