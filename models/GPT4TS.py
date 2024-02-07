from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from einops import rearrange
# from data_provider.patch_mask import *

class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.is_ln = configs.ln
        self.task_name = configs.task_name
        # self.patch_size = configs.patch_len  #1
        # self.stride = configs.patch_stride
        self.seq_len = configs.seq_len
        self.patch_len = configs.patch_len
        self.d_ff = configs.d_ff
        self.gpu = configs.gpu
        self.gpt = configs.gpt
        self.full_tuning = configs.full_tuning
        self.effi_layer = configs.effi_layer
        # print('configs.full_tuning:',configs.full_tuning)
        # self.patch_num = (configs.seq_len + self.pred_len - self.patch_size) // self.stride + 1

        # self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        # self.patch_num += 1
        #
        # self.enc_embedding = DataEmbedding(configs.enc_in * self.patch_size, configs.d_model, configs.embed, configs.freq,
        #                                    configs.dropout)
        if self.gpt == 'True': 
            self.gpt2 = GPT2Model.from_pretrained('/home/data/xrh/FL/AD_FL/gpt2large', output_attentions=True, output_hidden_states=True)


        self.gpt2.h = self.gpt2.h[:configs.gpt_layers] 
     
        if self.full_tuning:
            print('full-tuning')
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                param.requires_grad = True
        else:
            print('efficient-tuning')
            if self.effi_layer == 7:
                print('---7---')
                is_top_layer_param = lambda name: any(layer_name in name for layer_name in ['h.1','h.2','h.3','h.4','h.5','h.6','h.7', 'ln_f'])
            elif self.effi_layer == 6:
                print('---6---')
                is_top_layer_param = lambda name: any(layer_name in name for layer_name in ['h.2','h.3','h.4','h.5','h.6','h.7', 'ln_f'])
            elif self.effi_layer == 5:
                print('---5---')
                is_top_layer_param = lambda name: any(layer_name in name for layer_name in ['h.3','h.4','h.5','h.6','h.7', 'ln_f'])
            elif self.effi_layer == 4:
                print('---4---')
                is_top_layer_param = lambda name: any(layer_name in name for layer_name in ['h.4','h.5','h.6','h.7', 'ln_f'])
            elif self.effi_layer == 3:
                print('---3---')
                is_top_layer_param = lambda name: any(layer_name in name for layer_name in ['h.5','h.6','h.7', 'ln_f'])
            elif self.effi_layer == 2:
                print('---2---')
                is_top_layer_param = lambda name: any(layer_name in name for layer_name in ['h.6','h.7', 'ln_f'])
            elif self.effi_layer == 1:
                print('---1---')
                is_top_layer_param = lambda name: any(layer_name in name for layer_name in ['h.7', 'ln_f'])

            is_special_param = lambda name: any(layer_name in name for layer_name in ['wpe'])

            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if is_top_layer_param(name) or is_special_param(name):
                    # pass
                    if 'ln' in name: #or 'wpe' in name:
                        # print('ln;')
                        param.requires_grad = False
            
                else:
                    # print('False',name)
                    param.requires_grad = False

        # self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
                
        self.ln_proj = nn.LayerNorm(configs.d_ff)
        self.out_layer = nn.Linear(
                configs.d_ff, 
                configs.c_out, 
                bias=True)
            
       

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask_bool=False,mask=None):
       
        dec_out = self.anomaly_detection(x_enc,mask_bool, self.gpt)  #anomaly_detection_withmask
        # dec_out = self.anomaly_detection_withmask(x_enc,mask_bool)  
        return dec_out  # [B, L, D]



    def anomaly_detection(self, x_enc,mask_bool,gpt):
        # B, L, M = x_enc.shape
        #  [batch_size, num_patch, nvars, patch_len]
        # x_enc = x_enc.permute(0, 1, 3, 2)
        bs, num_patch_patch_length, nvars = x_enc.shape
        # x_enc = x_enc.reshape(bs, -1, nvars)
   
        # Normalization from Non-stationary Transformer  ??先mask还是先patch
        seg_num =  self.patch_len
        x_enc = rearrange(x_enc, 'b (n s) m -> b n s m', s=seg_num)
        means = x_enc.mean(2, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=2, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        x_enc = rearrange(x_enc, 'b n s m -> b (n s) m')

        # means = x_enc.mean(1, keepdim=True).detach()
        # x_enc = x_enc - means
        # stdev = torch.sqrt(
        #     torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x_enc /= stdev

        ## enc_out = self.enc_embedding(x_enc, None)  # [B,T,C] 
        enc_out = torch.nn.functional.pad(x_enc, (0, 1280-x_enc.shape[-1]))
        
        outputs = self.gpt2(inputs_embeds=enc_out).last_hidden_state
        # print(outputs.shape)
        outputs = outputs[:, :, :self.d_ff]
        # outputs = self.ln_proj(outputs)
        dec_out = self.out_layer(outputs)

        # De-Normalization from Non-stationary Transformer

        dec_out = rearrange(dec_out, 'b (n s) m -> b n s m', s=seg_num)
        dec_out = dec_out * \
                  (stdev[:, :, 0, :].unsqueeze(2).repeat(
                      1, 1, seg_num, 1))
        dec_out = dec_out + \
                  (means[:, :, 0, :].unsqueeze(2).repeat(
                      1, 1, seg_num, 1))
        dec_out = rearrange(dec_out, 'b n s m -> b (n s) m')

        # dec_out = dec_out * \
        #           (stdev[:, 0, :].unsqueeze(1).repeat(
        #               1, self.pred_len + self.seq_len, 1))
        # dec_out = dec_out + \
        #           (means[:, 0, :].unsqueeze(1).repeat(
        #               1, self.pred_len + self.seq_len, 1))
        return dec_out

    
