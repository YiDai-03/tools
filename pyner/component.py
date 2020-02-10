import argparse
import torch
import numpy as np
import warnings
import os
from torch import optim
from .test.predicter import Predicter
from .test.predict_utils import test_write
from .train.metrics import F1_score
from .train.trainer import Trainer
from .train.train_utils import restore_checkpoint,model_device
from .io.data_loader import DataLoader
from .io.data_transformer import DataTransformer
from .utils.logginger import init_logger
from .utils.utils import seed_everything
from .config.basic_config import configs as config
from .callback.lrscheduler import ReduceLROnPlateau
from .model.nn.cnn import CNN
from .model.nn.bilstm_crf import BiLSTM
from .model.nn.latticelstm_crf import Lattice
from .model.nn.bert import BERT_LSTM
from .callback.modelcheckpoint import ModelCheckpoint
from .callback.trainingmonitor import TrainingMonitor
warnings.filterwarnings("ignore")

class Component():
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_config = config['model']
        self.logger = init_logger(log_name=self.model_config['name'], log_dir=config['log_dir'])
        self.logger.info("seed is %d"%config['seed'])
        seed_everything(seed = config['seed'])
        self.device = 'cuda:%d' % config['n_gpus'][0] if len(config['n_gpus']) else 'cpu'
        self.logger.info('starting load train data from disk')
        self.default_tk = True if ('bert' in self.model_config['name']) else False

    def train(self):
        self.data_transformer = DataTransformer(logger        = self.logger,
                                       is_train_mode = True,
                                       all_data_path = self.config['all_data_path'],
                                       vocab_path    = self.config['vocab_path'],
                                       rev_vocab_path    = self.config['rev_vocab_path'],
                                       max_features  = self.config['max_features'],
                                       label_to_id   = self.config['label_to_id'],
                                       train_file    = self.config['train_file_path'],
                                       valid_file      = self.config['valid_file_path'],
                                       valid_size      = self.config['valid_size'],
                                       min_freq      = self.config['min_freq'],
                                       seed          = self.config['seed'],
                                       default_token = self.default_tk)
        self.data_transformer.build_vocab()
        self.data_transformer.sentence2id(raw_data_path   = self.config['raw_train_path'],
                                 raw_target_path = self.config['raw_target_path'],
                                 x_var           = self.config['x_var'],
                                 y_var           = self.config['y_var'])
        embedding_weight, words_embedding, gaz_tree = self.data_transformer.build_embedding_matrix(embedding_path = config['embedding_weight_path'], dict_path = config['embedding_dict_path'])
        bs = self.config['batch_size']
        if ('lattice' in self.model_config['name']):
            bs = 1
        train_loader = DataLoader(logger         = self.logger,
                              vocab = self.data_transformer.vocab,
                              rev_vocab = self.data_transformer.rev_vocab, 
                              is_train_mode  = True,
                              x_var          = self.config['x_var'],
                              y_var          = self.config['y_var'],
                              skip_header    = False,
                              data_path      = self.config['train_file_path'],
                              batch_size     = bs,
                              max_sentence_length = self.config['max_length'],
                              gaz            = gaz_tree,
                              default_token = self.default_tk,
                              device = self.device)

        val_loader = DataLoader(logger        = self.logger,
                                  vocab = self.data_transformer.vocab,
                              rev_vocab = self.data_transformer.rev_vocab, 
                            is_train_mode = True,
                            x_var         = self.config['x_var'],
                            y_var         = self.config['y_var'],
                            skip_header   = False,
                            data_path     = self.config['valid_file_path'],
                            batch_size    = bs,
                            max_sentence_length = self.config['max_length'],
                            gaz           = gaz_tree,
                            device = self.device)
        train_iter = train_loader.make_iter()
        val_iter = val_loader.make_iter()
        self.logger.info("initializing model")
        arch = self.model_config['name']
        if (arch == 'cnn_crf' or arch == 'cnn'):
            self.model = CNN(num_classes      = self.config['num_classes'],
                      embedding_dim    = self.config['embedding_dim'],
                      model_config     = self.config['models'][arch],
                      embedding_weight = embedding_weight,
                      vocab_size       = len(self.data_transformer.vocab),
                      device           = self.device)
        elif (arch =='bilstm' or arch == 'bilstm_crf'):
            self.model = BiLSTM(num_classes      = self.config['num_classes'],
                      embedding_dim    = self.config['embedding_dim'],
                      model_config     = self.config['models'][arch],
                      embedding_weight = embedding_weight,
                      vocab_size       = len(self.data_transformer.vocab),
                      device           = self.device)    
        elif (arch == 'lattice_lstm'):
            self.model = Lattice(num_classes      = self.config['num_classes'],
                      embedding_dim    = self.config['embedding_dim'],
                      model_config     = self.config['models'][arch],
                      embedding_weight = embedding_weight,
                      vocab_size       = len(self.data_transformer.vocab),
                      dict_size = len(self.data_transformer.word_vocab),
                      pretrain_dict_embedding = words_embedding,
                      device           = self.device)
        elif (arch == 'bert_lstm'):
            self.model = BERT_LSTM(num_classes      = self.config['num_classes'],
                      model_config     = self.config['models'][arch],
                      device           = self.device)
        self.optimizer = optim.Adam(params = self.model.parameters(),lr = self.config['learning_rate'],
                           weight_decay = self.config['weight_decay'])
        self.logger.info("initializing callbacks")

        self.model_checkpoint = ModelCheckpoint(checkpoint_dir   = self.config['checkpoint_dir'],
                                       mode             = self.config['mode'],
                                       monitor          = self.config['monitor'],
                                       save_best_only   = self.config['save_best_only'],
                                       best_model_name  = self.config['best_model_name'],
                                       epoch_model_name = self.config['epoch_model_name'],
                                       arch             = arch,
                                       logger           = self.logger)

        self.train_monitor = TrainingMonitor(fig_dir  = self.config['figure_dir'],
                                    json_dir = self.config['log_dir'],
                                    arch     = arch)

        self.lr_scheduler = ReduceLROnPlateau(optimizer = self.optimizer,
                                     factor   = 0.5,
                                     patience = self.config['lr_patience'],
                                     min_lr   = 1e-9,
                                     epsilon  = 1e-5,
                                     verbose  = 1,
                                     mode     = self.config['mode'])

        self.logger.info('training model....')
        self.trainer = Trainer(model            = self.model,
                      model_name       = arch,
                      train_data       = train_iter,
                      val_data         = val_iter,
                      optimizer        = self.optimizer,
                      epochs           = self.config['epochs'],
                      label_to_id      = self.config['label_to_id'],
                      evaluate         = F1_score(num_classes=config['num_classes']),
                      logger           = self.logger,
                      model_checkpoint = self.model_checkpoint,
                      training_monitor = self.train_monitor,
                      resume           = self.config['resume'],
                      lr_scheduler     = self.lr_scheduler,
                      n_gpu            = self.config['n_gpus'],
                      avg_batch_loss   = True)

        self.trainer.summary()

        self.trainer.train()

        if len(self.config['n_gpus']) > 0:
            torch.cuda.empty_cache()

    def predict(self):
        arch  = self.config['model']['name']
        self.logger = init_logger(log_name=arch, log_dir=config['log_dir'])
        self.logger.info("seed is %d"%self.config['seed'])
        seed_everything(seed = self.config['seed'])

        self.checkpoint_path = os.path.join(self.config['checkpoint_dir'].format(arch =arch),
                                    self.config['best_model_name'].format(arch = arch))
        self.device = 'cuda:%d' % self.config['n_gpus'][0] if len(self.config['n_gpus']) else 'cpu'


        self.logger.info('starting load test data from disk')
        self.data_transformer = DataTransformer(
                     vocab_path    = self.config['vocab_path'],
                     rev_vocab_path    = self.config['rev_vocab_path'],
                      all_data_path = self.config['all_data_path'],
                     test_file     = self.config['test_file_path'],
                     logger        = self.logger,
                     skip_header   = False,
                     is_train_mode = False,
                     default_token = self.default_tk,
                     seed          = self.config['seed'])
        self.data_transformer.build_vocab()
        self.data_transformer.sentence2id(raw_data_path = self.config['raw_test_path'],
                                 x_var=self.config['x_var'],
                                 y_var=self.config['y_var']
                                 )
        embedding_weight, words_embedding, gaz_tree = self.data_transformer.build_embedding_matrix(embedding_path = self.config['embedding_weight_path'], dict_path = self.config['embedding_dict_path'])
        bs = self.config['batch_size']
        arch = self.model_config['name'] 
        if ('lattice' in arch):
   	        bs = 1
        test_loader = DataLoader(logger = self.logger,
                        is_train_mode=False,
                                                      vocab = self.data_transformer.vocab,
                              rev_vocab = self.data_transformer.rev_vocab, 
                        x_var = self.config['x_var'],
                        y_var = self.config['y_var'],
                        skip_header = False,
                        data_path   = self.config['test_file_path'],
                        batch_size  = bs,
                        max_sentence_length = self.config['max_length'],
                        gaz         = gaz_tree,
                        default_token = self.default_tk,
                        device = self.device)
        test_iter = test_loader.make_iter()

        self.logger.info("initializing model")
        if (arch == 'cnn_crf' or arch == 'cnn'):
            self.model = CNN(num_classes      = self.config['num_classes'],
                      embedding_dim    = self.config['embedding_dim'],
                      model_config     = self.config['models'][arch],
                      embedding_weight = embedding_weight,
                      vocab_size       = len(self.data_transformer.vocab),
                      device           = self.device)
        elif (arch =='bilstm' or arch == 'bilstm_crf'):
            self.model = BiLSTM(num_classes      = self.config['num_classes'],
                      embedding_dim    = self.config['embedding_dim'],
                      model_config     = self.config['models'][arch],
                      embedding_weight = embedding_weight,
                      vocab_size       = len(self.data_transformer.vocab),
                      device           = self.device)    
        elif (arch == 'lattice_lstm'):
            self.model = Lattice(num_classes      = self.config['num_classes'],
                      embedding_dim    = self.config['embedding_dim'],
                      model_config     = self.config['models'][arch],
                      embedding_weight = embedding_weight,
                      vocab_size       = len(self.data_transformer.vocab),
                      dict_size = len(self.data_transformer.word_vocab),
                      pretrain_dict_embedding = words_embedding,
                      device           = self.device)
        elif (arch == 'bert_lstm'):
            self.model = BERT_LSTM(num_classes      = self.config['num_classes'],
                      model_config     = self.config['models'][arch],
                      device           = self.device)

        self.logger.info('predicting model....')
        predicter = Predicter(model           = self.model,
                          model_name      = arch, 
                          logger          = self.logger,
                          n_gpu           = self.config['n_gpus'],
                          test_data       = test_iter,
                          checkpoint_path = self.checkpoint_path,
                          label_to_id     = self.config['label_to_id'])

        predictions = predicter.predict()
        test_write(data = predictions,filename = self.config['result_path'],raw_text_path=self.config['raw_test_path'])

        if len(config['n_gpus']) > 0:
            torch.cuda.empty_cache()

   