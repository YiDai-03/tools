#encoding:utf-8
import torch.nn as nn
from ..layers.embed_layer import Embed_Layer
from ..layers.crf import CRF
from ..layers.bilstm import BILSTM

class BiLSTM(nn.Module):
    def __init__(self,model_config,
                 embedding_dim,
                 num_classes,
                 vocab_size,
                 embedding_weight,
                 device):
        super(BiLSTM ,self).__init__()
        self.embedding = Embed_Layer(vocab_size = vocab_size,
                                     embedding_weight = embedding_weight,
                                     embedding_dim = embedding_dim,
                                     dropout_emb=model_config['dropout_emb'],
                                     training=True)
        if (model_config['crf'] == False):
            classes = num_classes
        else:
            classes = num_classes + 2
        self.lstm = BILSTM(input_size = embedding_dim,
                           hidden_size= model_config['hidden_size'],
                           num_layer  = model_config['num_layer'],
                           bi_tag     = model_config['bi_tag'],
                           dropout_p  = model_config['dropout_p'],
                           num_classes= classes)
        self.crf = CRF(device = device,tagset_size=num_classes, have_crf = model_config['crf'])

    def forward(self, inputs,length, masks = None, gaz = None):
        x = self.embedding(inputs)
        x = self.lstm(x,length)
        return x

