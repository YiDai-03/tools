#encoding:utf-8
import torch.nn as nn
from ..layers.embed_layer import Embed_Layer
from ..layers.crf import CRF
from ..layers.lattice import LatticeLSTM
import torch

class Lattice(nn.Module):
    def __init__(self,model_config,
                 embedding_dim,
                 num_classes,
                 vocab_size,
                 embedding_weight,
                 dict_size, #word dict
                 pretrain_dict_embedding, #word_dict
                 device,
):
        super(Lattice ,self).__init__()
        self.embedding = Embed_Layer(vocab_size = vocab_size,
                                     embedding_weight = embedding_weight,
                                     embedding_dim = embedding_dim,
                                     dropout_emb=model_config['dropout_emb'],
                                     training=True)
        if (model_config['crf'] == False):
            classes = num_classes
        else:
            classes = num_classes + 2
        self.forward_lstm = LatticeLSTM(input_dim = embedding_dim,
                           hidden_dim = model_config['hidden_size']//2,
                           word_drop = 0.5,
                           word_alphabet_size = dict_size,
                           word_emb_dim  = 300,
                           pretrain_word_emb = pretrain_dict_embedding,
                           left2right = True,
                           fix_word_emb=True, 
                           gpu=False, 
                           use_bias=True
                           )
        self.backward_lstm = LatticeLSTM(input_dim = embedding_dim,
                           hidden_dim = model_config['hidden_size']//2,
                           word_drop = 0.5,
                           word_alphabet_size = dict_size,
                           word_emb_dim  = 300,
                           pretrain_word_emb = pretrain_dict_embedding,
                           left2right = False,
                           fix_word_emb=True, 
                           gpu=False, 
                           use_bias=True
                           )
        self.linear = nn.Linear(in_features=model_config['hidden_size'], out_features= classes)
        nn.init.xavier_uniform(self.linear.weight)
        self.crf = CRF(device = device,tagset_size=num_classes, have_crf = model_config['crf'])

    def forward(self, inputs, gazs, length):
        x = self.embedding(inputs) #batch,len,emb_size
        batch_size = inputs.size(0)
        sent_len = inputs.size(1)
        #word_embs = self.word_embeddings(word_inputs)

        # packed_words = pack_padded_sequence(word_embs, word_seq_lengths.cpu().numpy(), True)
        hidden = None
        lstm_out, hidden = self.forward_lstm(x, gazs, hidden)

        backward_hidden = None
        backward_lstm_out, backward_hidden = self.backward_lstm(x, gazs, backward_hidden)
        lstm_out = torch.cat([lstm_out, backward_lstm_out], 2)
        # lstm_out, _ = pad_packed_sequence(lstm_out)
        #lstm_out = self.droplstm(lstm_out)
        lstm_out = self.linear(lstm_out)
        return lstm_out
        
        

