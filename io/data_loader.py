#encoding:utf-8
import json
from torchtext import data
from ..utils.gazetteer import Gazetteer
from ..config.basic_config import configs as config
import numpy

class CreateDataset(data.Dataset):
    def __init__(self,
                 data_path,
                 text_field,
                 label_field,
                 x_var,y_var,
                 skip_header,
                 is_train_mode,
                 ):
        fields = [(x_var, text_field), (y_var, label_field)]
        examples = []
        if is_train_mode:
            with open(data_path,'r') as fr:
                for i,line in enumerate(fr):

                    if i == 0 and skip_header:
                        continue
                    df = json.loads(line)
                    sentence = df[x_var]
                    label = df[y_var]
                    examples.append(data.Example.fromlist([sentence, label], fields))
        else:
            with open(data_path,'r') as fr:
                for i,line in enumerate(fr):
                    if i == 0 and skip_header:
                        continue
                    df = json.loads(line)
                    sentence = df[x_var]
                    label = df[y_var]
                    examples.append(data.Example.fromlist([sentence,label], fields))
        super(CreateDataset,self).__init__(examples,fields)

class DataLoader(object):
    def __init__(self,
                 data_path,
                 batch_size,
                 logger,
                 x_var,
                 y_var,
                 gaz,
                 skip_header = False,
                 is_train_mode = True,
                 max_sentence_length = None,

                 device = 'cpu'
                 ):

        self.logger           = logger
        self.device           = device
        self.batch_size       = batch_size
        self.data_path        = data_path
        self.x_var            = x_var
        self.y_var            = y_var
        self.is_train_mode    = is_train_mode
        self.skip_header      = skip_header
        self.max_sentence_len = max_sentence_length
        self.gaz = gaz

    def make_iter(self):

        TEXT = data.Field(sequential      = True,
                          use_vocab       = False,
                          tokenize        = lambda x: [int(c) for c in x.split()],  
                          fix_length      = self.max_sentence_len, 
                          pad_token       = 0, 
                          batch_first     = True,
                          eos_token       = None,
                          init_token      = None,
                          include_lengths = True
                          )

        LABEL = data.Field(sequential      = True,
                           use_vocab       = False,
                           tokenize        = lambda x: [int(c)+1 for c in x.split()],
                           batch_first     = True,
                           fix_length      = self.max_sentence_len,
                           eos_token       = None,
                           init_token      = None,
                           include_lengths = False,
                           pad_token       = 0  
                           )

        dataset = CreateDataset(data_path = self.data_path,
                               text_field  = TEXT,
                               label_field = LABEL,
                               x_var  = self.x_var,
                               y_var  = self.y_var,
                               skip_header = self.skip_header,
                               is_train_mode = self.is_train_mode
                              )

        if self.is_train_mode:
            data_iter = data.BucketIterator(dataset = dataset,
                                            batch_size = self.batch_size,
                                            shuffle = True,
                                            repeat=False,
                                            sort_within_batch = True,
                                            sort_key = lambda x:len(getattr(x,self.x_var)),
                                            device = self.device)
        else:
            data_iter = data.Iterator(dataset = dataset,
                                      batch_size = self.batch_size,
                                      shuffle = False,
                                      sort = False,
                                      repeat = False,
                                      device = self.device)
        return BatchWrapper(data_iter, self.gaz, self.batch_size, x_var=self.x_var, y_var=self.y_var)

class BatchWrapper(object):
    def __init__(self, dl, gaz, batch_size, x_var, y_var):
        self.dl, self.x_var, self.y_var = dl, x_var, y_var
        self.gaz = gaz
        self.batchsz = batch_size

    def __iter__(self):
        for batch in self.dl:
            x = getattr(batch, self.x_var)
            target = getattr(batch, self.y_var)
            source = x[0]
            length = x[1]
            print("length",length)
            gaz_list = []
            gaz = []
            text = source.cpu().numpy().tolist()
            lens = length.cpu().numpy().tolist()
            for i,sent in enumerate(text):
                sent_len = lens[i]
                for idx in range(sent_len):
                    match_out = self.gaz.enumerateMatchList(sent[idx:sent_len])
                    if (len(match_out[0])==0):
                        match_out = []
                    gaz.append(match_out)
                while (len(gaz)<config['max_length']):
                    gaz.append([])
                gaz_list.append(gaz)
            yield (source, gaz_list, target, length)
    def __len__(self):
        return len(self.dl)
