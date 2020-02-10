#encoding:utf-8
import torch
from torch.autograd import Variable
from tqdm import tqdm
from .predict_utils import get_entity
from ..train.train_utils import restore_checkpoint,model_device
from ..train.trainer import batchify_with_label
from ..config.basic_config import configs as config

class Predicter(object):
    def __init__(self,
                 model,
                 model_name,
                 test_data,
                 logger,
                 label_to_id,
                 checkpoint_path,
                 n_gpu = 0):
        self.model           = model
        self.model_name      = model_name
        self.test_data       = test_data
        self.logger          = logger
        self.checkpoint_path = checkpoint_path
        self.n_gpu           = n_gpu
        self.id_to_label     = {value:tag for tag,value in label_to_id.items()}
        self._reset()

    def sequence_mask(self, sequence_length, max_len):   # sequence_length :(batch_size, )
        batch_size = sequence_length.size(0)            # �õ�batch_size
        seq_range = torch.arange(0, max_len).long()
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        seq_range_expand = Variable(seq_range_expand)
        if sequence_length.is_cuda:
            seq_range_expand = seq_range_expand.cuda()
        seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))
        return seq_range_expand < seq_length_expand 
        
    def _reset(self):
        self.batch_num = len(self.test_data)
        self.model, self.device = model_device(n_gpu=self.n_gpu, model=self.model, logger=self.logger)
        if self.checkpoint_path:
            self.logger.info("\nLoading checkpoint: {} ...".format(self.checkpoint_path))
            resume_list = restore_checkpoint(resume_path=self.checkpoint_path,model=self.model)
            self.model = resume_list[0]
            self.logger.info("\nCheckpoint '{}' loaded".format(self.checkpoint_path))

    # batch预测
    def _predict_batch(self,inputs,gaz,length):
        with torch.no_grad():
            masks = self.sequence_mask(length, config['max_length']) if ('bert' in self.model_name) else None
            outputs = self.model(inputs, length, masks, gaz)
            mask, _ = batchify_with_label(inputs=inputs, outputs=outputs,is_train_mode=False)
            _,predicts = self.model.crf(outputs, mask)

            batch_result = []
            for index,(text,path) in enumerate(zip(inputs,predicts)):
                if self.device != 'cpu':
                    path = path.cpu().numpy()
                result = get_entity(path = path,tag_map=self.id_to_label)
                batch_result.append(result)
            return batch_result

    #预测test数据�?
    def predict(self):
        self.model.eval()
        predictions = []
        for batch_idx,(inputs,gaz,_,length) in tqdm(enumerate(self.test_data),total=self.batch_num,desc='test_data'):
            inputs   = inputs.to(self.device)
            length   = length.to(self.device)
            y_pred_batch = self._predict_batch(inputs = inputs,length = length, gaz = gaz)
            predictions.extend(y_pred_batch)
        return predictions