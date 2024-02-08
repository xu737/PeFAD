import argparse
import os
import torch
# from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
# from exp.exp_imputation import Exp_Imputation
# from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from data_provider.shared_construct import *

# from exp.exp_classification import Exp_Classification
import random
import numpy as np
from data_provider.data_factory import data_provider


fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser()

parser.add_argument('--client_nums', type=int, required=True, default=28, help='num of clients')
parser.add_argument('--local_bs', type=int, required=True, default=256, help='local batch size')
parser.add_argument('--local_epoch', type=int, required=True, default=1, help='local training epoch')
parser.add_argument('--seq_len', type=int, default=100, help='input sequence length')

parser.add_argument('--patch_len', type=int, required=False, default=10, help='the length of patch')
parser.add_argument('--patch_stride', type=int, required=False, default=10, help='the stride of patch')

parser.add_argument('--mask_ratio', type=float, required=False, default=0.2, help='mask ratio of patch')

parser.add_argument('--vae_train_epochs', type=int, required=False, default=1, help='training epochs of vae')
parser.add_argument('--vae_local_epochs', type=int, required=False, default=10, help='local epochs of vae')

parser.add_argument('--latent_dim', type=int, required=False, default=16, help='latent dim of vae')
parser.add_argument('--gpt', type=str, required=False, default='True', help='gpt2')
parser.add_argument('--consis_loss_coef', type=int, required=False, default=15, help='the weight of consistency loss')
parser.add_argument('--mask_factor', type=float, required=False, default=2, help='the weight to be chosen for masking')
parser.add_argument('--full_tuning', type=int, required=False, default=0, help='fully fine-tune')
parser.add_argument('--effi_layer', type=int, required=False, default=4, help='nums of tuning layer')
parser.add_argument('--connection_ratio', type=float, required=False, default=0.9, help='the ratio of connection')
parser.add_argument('--percentile', type=int, required=False, default=10, help='latent dim of vae')
parser.add_argument('--continue_training', type=int, required=False, default=0, help='continue training')
parser.add_argument('--test_path', type=str, required=False, default='', help='the path of test file')
parser.add_argument('--weight_similarity', type=float, required=False, default=0.8, help='the weight of inter-')
parser.add_argument('--weight_residual', type=float, required=False, default=0.2, help='the weight of intra-')

parser.add_argument('--train_path', type=str, required=False, default='', help='the path of train file')

# parser.add_argument('--anf', type=str, required=False, default='False', help='anomaly transformer')

# parser.add_argument('--svdd', type=str, required=False, default='False', help='latent dim of vae')
# parser.add_argument('--tranad', type=str, required=False, default='False', help='latent dim of vae')
parser.add_argument('--shared_size', type=int, default=100, help='the length of synthesis time series')

# parser.add_argument('--mutual_tes', type=str, required=False, default='False', help='latent dim of vae')


# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, Transformer, TimesNet]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')


parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

# model define
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=16, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=2, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='3,2,1,0', help='device ids of multile gpus')

# de-stationary projector params
parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                    help='hidden layer dimensions of projector (List)')
parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

# patching
# parser.add_argument('--patch_size', type=int, default=1)
# parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--gpt_layers', type=int, default=6)
parser.add_argument('--ln', type=int, default=0)
parser.add_argument('--mlp', type=int, default=0)
parser.add_argument('--weight', type=float, default=0)
parser.add_argument('--percent', type=int, default=5)

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    print('devices:',device_ids)
    args.device_ids = [int(id_) for id_ in device_ids]
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    # args.gpu = args.device_ids[3]
    # print(f"Using  GPU with device ID: {args.gpu}")

print('Args in experiment:')
print(args)


Exp = Exp_Anomaly_Detection


if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}-gpt_{}-ratio_{}-full_tuning_{}-effi'.format(           
            args.model_id,  
            args.model,         
            args.gpt,
            args.mask_ratio,
            args.full_tuning,
            args.effi_layer,)

        exp = Exp(args)  # set experiments
       
       
        print('>>>>>>>forming dataset : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        
        shared_dataset_loader = exp.shared_data(setting)
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(shared_dataset_loader, setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

else:
    ii = 0
    setting = '{}_{}_{}-gpt_{}-ratio_{}-full_tuning_{}-effi'.format(           
            args.model_id,  
            args.model,         
            args.gpt,
            args.mask_ratio,
            args.full_tuning,
            args.effi_layer,)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
