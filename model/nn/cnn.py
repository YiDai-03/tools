#encoding:utf-8
import torch.nn as nn
from ..layers.embed_layer import Embed_Layer
from ..layers.crf import CRF

class CNN(nn.Module):
    def __init__(self,model_config,
                 embedding_dim,
                 num_classes,
                 vocab_size,
                 embedding_weight,
                 device):
        super(CNN ,self).__init__()
        self.embedding = Embed_Layer(vocab_size = vocab_size,
                                     embedding_weight = embedding_weight,
                                     embedding_dim = embedding_dim,
                                     dropout_emb=model_config['dropout_emb'],
                                     training=True)
        if (model_config['crf'] == False):
            classes = num_classes
        else:
            classes = num_classes + 2
            
        self.Conv1d = nn.Conv1d(in_channels = 80,out_channels = 80,kernel_size=3,stride = 1,padding = 0) #batch_size:256
        self.relu = nn.ReLU(inplace=True)
        print("embedding_dim",embedding_dim)
        self.dl = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(296, classes))
        self.crf = CRF(device = device,tagset_size=num_classes, have_crf = model_config['crf'])

    def forward(self, inputs,length):
        x = self.embedding(inputs)
        x = self.relu(self.Conv1d(x))


        h = self.Conv1d(x)

        #print (x.shape)
        h = self.dl(h)

        return h