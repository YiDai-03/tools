#encoding:utf-8
import os
import random
import operator
import numpy as np
from tqdm import tqdm
from collections import Counter
from ..utils.utils import pkl_read,pkl_write,text_write
from ..utils.gazetteer import Gazetteer


class DataTransformer(object):

    def __init__(self,
                 vocab_path,
                 logger,
                 max_features = None,
                 min_freq = 3,
                 all_data_path = None,
                 label_to_id = None,
                 train_file = None,
                 valid_file = None,
                 test_file = None,
                 valid_size = None,
                 skip_header = False,
                 is_train_mode = True,
                 seed=1024,
                 ):

        self.seed = seed
        self.logger = logger
        self.valid_size = valid_size
        self.min_freq = min_freq
        self.train_file = train_file
        self.valid_file = valid_file
        self.test_file = test_file
        self.all_data_path = all_data_path
        self.vocab_path = vocab_path
        self.word_vocab = None
        self.skip_header = skip_header
        self.max_features = max_features
        self.label_to_id = label_to_id
        self.is_train_mode = is_train_mode


    def _split_sent(self,line):

        res = line.strip('\n').split()
        return res

    def _word_to_id(self,word, vocab):

        return vocab[word] if word in vocab else vocab['<unk>']

    def train_val_split(self,X, y, valid_size=0.3, random_state=2018, shuffle=True):

        self.logger.info('train val split')
        data = []
        for data_x, data_y in tqdm(zip(X, y), desc='Merge'):
            data.append((data_x, data_y))
        del X, y
        N = len(data)
        test_size = int(N * valid_size)
        if shuffle:
            random.seed(random_state)
        random.shuffle(data)
        valid = data[:test_size]
        train = data[test_size:]
        return train,valid

    def build_vocab(self):
        if os.path.isfile(self.vocab_path):
            self.vocab = pkl_read(self.vocab_path)
        else:
            count = Counter()
            with open(self.all_data_path, 'r') as fr:
                self.logger.info('Building word vocab')
                for i,line in enumerate(fr):

                    if i==0 and self.skip_header:
                        continue
                    words = self._split_sent(line)
                    count.update(words)
            count = {k: v for k, v in count.items()}
            count = sorted(count.items(), key=operator.itemgetter(1))

            all_words = [w[0] for w in count if w[1] >= self.min_freq]
            if self.max_features:
                all_words = all_words[:self.max_features]

            flag_words = ['<pad>', '<unk>']
            all_words = flag_words + all_words
            self.logger.info('vocab_size is %d' % len(all_words))

            word2id = {k: v for k, v in zip(all_words, range(0, len(all_words)))}
            assert word2id['<pad>'] == 0, "ValueError: '<pad>' id is not 0"

            pkl_write(data = word2id,filename=self.vocab_path)
            self.vocab = word2id

    def sentence2id(self,raw_data_path=None,raw_target_path  =None,x_var = None,y_var = None):

        self.logger.info('sentence to id')
        if self.is_train_mode:
            if os.path.isfile(self.train_file) and os.path.isfile(self.valid_file):
                return True
            sentences, labels = [], []
            with open(raw_data_path, 'r') as fr_x,open(raw_target_path,'r') as fr_y:
                for i,(sent,target) in enumerate(zip(fr_x,fr_y)):
                    if i==0 and self.skip_header:
                        continue
                    words = self._split_sent(sent)
                    label = self._split_sent(target)
                    if len(words) ==0 or len(label) ==0:
                        continue
                    sent2id = [self._word_to_id(word=word, vocab=self.vocab) for word in words]
                    label = [self.label_to_id[x] for x in label]
                    sentences.append(sent2id)
                    labels.append(label)

            if self.valid_size:
                train, val = self.train_val_split(X = sentences, y = labels,
                                                  valid_size=self.valid_size,
                                                  random_state=self.seed,
                                                  shuffle=True)
                text_write(self.train_file, train,x_var = x_var,y_var = y_var)
                text_write(self.valid_file, val,x_var = x_var,y_var = y_var)
        else:
            if os.path.isfile(self.test_file):
                return True
            sentences,labels = [],[]
            with open(raw_data_path, 'r') as fr_x:
                for i,sent in enumerate(fr_x):
                    if i==0 and self.skip_header:
                        continue
                    words = self._split_sent(sent)
                    if len(words) ==0:
                        continue
                    sent2id = [self._word_to_id(word=word, vocab=self.vocab) for word in words]
                    label    = [-1 for _ in range(len(sent2id))]
                    sentences.append(sent2id)
                    labels.append(label)
            text_write(self.test_file,zip(sentences,labels),x_var = x_var,y_var = y_var)

    def build_embedding_matrix(self,embedding_path,dict_path, emb_mean = None,emb_std = None):

        self.logger.info("initializer embedding matrix")
        embeddings_index = self._load_embedding(embedding_path)
        all_embs = np.stack((embeddings_index.values()))
        if emb_mean is None or emb_std is None:
            emb_mean = all_embs.mean()
            emb_std  = all_embs.std()
        embed_size = all_embs.shape[1]
        nb_words = len(self.vocab)
        gaz = Gazetteer(False)
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
        for word, id in self.vocab.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[id] = embedding_vector
                #-------------------------------------
        word_embedding_matrix = None
        if dict_path != None:
            words_embedding_index = self._load_embedding(dict_path)
            words = [k for k in words_embedding_index.keys() if len(k)>1]
            word_num = len(words)
            all_embs = np.stack((embeddings_index.values()))
            if emb_mean is None or emb_std is None:
                emb_mean = all_embs.mean()
                emb_std  = all_embs.std()
            embed_size = all_embs.shape[1]
            print("Word embedding size is:",embed_size)
            
            
            word_embedding_matrix = np.random.normal(emb_mean,emb_std,(word_num, embed_size))


            self.word_vocab = {k: v for k, v in zip(words, range(0, len(words)))}
            for word, id in self.word_vocab.items():
                char_list = []
                for char in word:
                    try:
                        char_list.append(self.vocab[char])
                    except:
                        char_list = []
                        break   
                if (char_list == []):
                    continue
                gaz.insert(char_list,id)
                embedding_vector = words_embedding_index.get(word)
                if embedding_vector is not None:
                    word_embedding_matrix[id] = embedding_vector
            
                            
        return embedding_matrix, word_embedding_matrix, gaz

    def _load_embedding(self, embedding_path):

        self.logger.info(" load emebedding weights")
        embeddings_index = {}
        f = open(embedding_path, 'r',errors='ignore',encoding = 'utf8')
        for line in f:
            values = line.rstrip().split(' ')
            if len(values)<10:
                continue
            try:
                word  = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            except:
                print("Error on ", values[:2])
        f.close()
        self.logger.info('Total %s word vectors.' % len(embeddings_index))
        return embeddings_index