import os
import torch

from models import GPT4TS #, BERT4TS, ALBERT4TS, ROBERTA4TS,ELECTRA4TS,distilbert4TS,deberta4TS




class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'GPT4TS': GPT4TS
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = '3,2,1,0'
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(
            #     self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
