#encoding:utf-8
# a temporary file to test CNN, later merge
import argparse
import torch
import warnings
from torch import optim
from pyner.train.metrics import F1_score
from pyner.train.trainer import Trainer
from pyner.io.data_loader import DataLoader
from pyner.io.data_transformer import DataTransformer
from pyner.utils.logginger import init_logger
from pyner.utils.utils import seed_everything
from pyner.config.basic_config import configs as config
from pyner.callback.lrscheduler import ReduceLROnPlateau
from pyner.model.nn.cnn import CNN
from pyner.model.nn.bilstm_crf import BiLSTM
from pyner.model.nn.latticelstm_crf import Lattice
from pyner.callback.modelcheckpoint import ModelCheckpoint
from pyner.callback.trainingmonitor import TrainingMonitor
warnings.filterwarnings("ignore")

####
def main(arch):
    logger = init_logger(log_name=arch, log_dir=config['log_dir'])
    logger.info("seed is %d"%args['seed'])
    seed_everything(seed = args['seed'])
    device = 'cuda:%d' % config['n_gpus'][0] if len(config['n_gpus']) else 'cpu'

    logger.info('starting load train data from disk')


    data_transformer = DataTransformer(logger        = logger,
                                       is_train_mode = True,
                                       all_data_path = config['all_data_path'],
                                       vocab_path    = config['vocab_path'],
                                       max_features  = config['max_features'],
                                       label_to_id   = config['label_to_id'],
                                       train_file    = config['train_file_path'],
                                       valid_file      = config['valid_file_path'],
                                       valid_size      = config['valid_size'],
                                       min_freq      = config['min_freq'],
                                       seed          = args['seed'])

    data_transformer.build_vocab()

    data_transformer.sentence2id(raw_data_path   = config['raw_train_path'],
                                 raw_target_path = config['raw_target_path'],
                                 x_var           = config['x_var'],
                                 y_var           = config['y_var'])

    char_embedding_weight, words_embedding, gaz_tree = data_transformer.build_embedding_matrix(embedding_path = config['embedding_weight_path'], dict_path = config['embedding_dict_path'])
    # glove_embedding_weight = data_transformer.build_embedding_matrix(embedding_path = config['glove_weight_path'])
    # embedding_weight = np.concatenate((word2vec_embedding_weight,glove_embedding_weight),axis=1)
    embedding_weight = char_embedding_weight
    bs = config['batch_size']
    if ('lattice' in arch):
        bs = 1

    train_loader = DataLoader(logger         = logger,
                              is_train_mode  = True,
                              x_var          = config['x_var'],
                              y_var          = config['y_var'],
                              skip_header    = False,
                              data_path      = config['train_file_path'],
                              batch_size     = bs,
                              max_sentence_length = config['max_length'],
                              gaz            = gaz_tree,
                              device = device)

    val_loader = DataLoader(logger        = logger,
                            is_train_mode = True,
                            x_var         = config['x_var'],
                            y_var         =config['y_var'],
                            skip_header   = False,
                            data_path     = config['valid_file_path'],
                            batch_size    = config['batch_size'],
                            max_sentence_length = config['max_length'],
                            gaz           = gaz_tree,
                            device = device)

    train_iter = train_loader.make_iter()
    val_iter = val_loader.make_iter()



    logger.info("initializing model")
    if (arch == 'cnn_crf' or arch == 'cnn'):
        model = CNN(num_classes      = config['num_classes'],
                      embedding_dim    = config['embedding_dim'],
                      model_config     = config['models'][arch],
                      embedding_weight = embedding_weight,
                      vocab_size       = len(data_transformer.vocab),
                      device           = device)
    elif (arch =='bilstm' or arch == 'bilstm_crf'):
        model = BiLSTM(num_classes      = config['num_classes'],
                      embedding_dim    = config['embedding_dim'],
                      model_config     = config['models'][arch],
                      embedding_weight = embedding_weight,
                      vocab_size       = len(data_transformer.vocab),
                      device           = device)    
    elif (arch == 'lattice_lstm'):
        model = Lattice(num_classes      = config['num_classes'],
                      embedding_dim    = config['embedding_dim'],
                      model_config     = config['models'][arch],
                      embedding_weight = embedding_weight,
                      vocab_size       = len(data_transformer.vocab),
                      dict_size = len(data_transformer.word_vocab),
                      pretrain_dict_embedding = words_embedding,
                      device           = device)
    optimizer = optim.Adam(params = model.parameters(),lr = config['learning_rate'],
                           weight_decay = config['weight_decay'])



    logger.info("initializing callbacks")

    model_checkpoint = ModelCheckpoint(checkpoint_dir   = config['checkpoint_dir'],
                                       mode             = config['mode'],
                                       monitor          = config['monitor'],
                                       save_best_only   = config['save_best_only'],
                                       best_model_name  = config['best_model_name'],
                                       epoch_model_name = config['epoch_model_name'],
                                       arch             = arch,
                                       logger           = logger)

    train_monitor = TrainingMonitor(fig_dir  = config['figure_dir'],
                                    json_dir = config['log_dir'],
                                    arch     = arch)

    lr_scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                     factor   = 0.5,
                                     patience = config['lr_patience'],
                                     min_lr   = 1e-9,
                                     epsilon  = 1e-5,
                                     verbose  = 1,
                                     mode     = config['mode'])

    logger.info('training model....')
    trainer = Trainer(model            = model,
                      model_name       = arch,
                      train_data       = train_iter,
                      val_data         = val_iter,
                      optimizer        = optimizer,
                      epochs           = config['epochs'],
                      label_to_id      = config['label_to_id'],
                      evaluate         = F1_score(num_classes=config['num_classes']),
                      logger           = logger,
                      model_checkpoint = model_checkpoint,
                      training_monitor = train_monitor,
                      resume           = args['resume'],
                      lr_scheduler     = lr_scheduler,
                      n_gpu            = config['n_gpus'],
                      avg_batch_loss   = True)

    trainer.summary()

    trainer.train()

    if len(config['n_gpus']) > 0:
        torch.cuda.empty_cache()

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='PyTorch model training')
    ap.add_argument('-s',
                    '--seed',
                    default=2018,
                    type = int,
                    help = 'Seed for initializing training.')

    ap.add_argument('-r',
                    '--resume',
                    default = False,
                    type = bool,
                    help = 'Choose whether resume checkpoint model')
    args = vars(ap.parse_args())
    print('Training total of {} models'.format(len(config['models'])))
    for i, model_name in enumerate(config['models'].keys()):
        print('{}/{}: Training {} '.format(i + 1, len(config['models']), model_name))
        main(arch = model_name)
