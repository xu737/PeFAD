from data_provider.data_factory import data_provider
from data_provider.shared_construct import *
from exp.exp_basic import Exp_Basic
from utils.tools import  adjustment
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import torch.multiprocessing
import copy
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys


torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
from torch.utils import data as data_utils
import os
import time
import warnings
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import random

warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score,roc_auc_score
from thop import profile


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)
        

    def _build_model(self):
        if self.args.model_id == 'SMD':
            self.original_dim,feats = 38,38
        elif self.args.model_id == 'SMAP':
            self.original_dim,feats = 25,25
        elif self.args.model_id == 'MSL':
            self.original_dim,feats = 55,55
        elif self.args.model_id == 'PSM':
            self.original_dim ,feats= 25,25
        elif self.args.model_id == 'SWAT':
            self.original_dim,feats = 51,51
        
        # self.device=self.args.devices
        self.model = self.model_dict[self.args.model].Model(self.args).float()
        self.vae_local_model = VAE(self.original_dim, self.args.latent_dim,device=self.device).float().to(self.device) 
            
        if self.args.continue_training == 1:
            setting = self.args.train_path
            print('loading model..')
            print('continue training..')
            print(setting)
            # state_dict = torch.load(os.path.join(setting, 'checkpoint3_0.2.pth'))
            state_dict = torch.load(setting)
            
            self.model= state_dict
        if self.args.use_multi_gpu and self.args.use_gpu:
            self.model = nn.DataParallel(self.model, device_ids=self.args.device_ids)
            # self.vae_local_model = nn.DataParallel(self.vae_local_model, device_ids=self.args.device_ids)
            print('use_multi_gpu:',self.args.device_ids)

        self.global_model = copy.deepcopy(self.model)
        self.global_vae_model = copy.deepcopy(self.vae_local_model)
        return self.model

    def _get_data(self, flag,shared=False):
        # shared = args.shared_dataset_loader
        data_set, data_loader = data_provider(self.args, flag,shared)
        num_clients = len(list(data_loader.keys()))
        # print(f"The number of clients is: {num_clients}") 
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate) #, weight_decay=self.args.weight
        self.vae_local_optimizer = optim.Adam(self.vae_local_model.parameters(), lr=0.001)
        return model_optim,self.vae_local_optimizer

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        # print("======================VALI MODE======================")
        
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _, mask_bool) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
             
                outputs = self.model(batch_x, None, None, None, mask_bool)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        # print("======================VALI DONE======================")
        
        return total_loss

    def train(self,shared_dataset_loader, setting):
        print("======================TRAIN MODE======================")
        torch.cuda.empty_cache()
        train_start_time = time.time()
       
        train_data, train_loader = self._get_data(flag='train') 
        train_list = [item for item in train_loader]  

        vali_data, vali_loader = self._get_data(flag='val')
        # test_data, test_loader = self._get_data(flag='test')
        # print("val_dict:",vali_loader.keys())

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader) 

        model_optim,vae_local_optimizer = self._select_optimizer()
        criterion = self._select_criterion()
        
        data_nums = {} 
        model_dicts = [] 


        for global_epoch in range(self.args.train_epochs):           
                       
            epoch_time = time.time()
            
            if global_epoch == 0:
                self.global_model = copy.deepcopy(self.model)
                global_state_dict = self.global_model.state_dict()
            else:
                global_state_dict = self.global_model.state_dict()
    
            model_dicts = [] 

            print(f'\n | Global Training Round : {global_epoch + 1} |\n')
            # print(train_loader.keys())
            
            for client_id, data_loader in train_loader.items():

                data_nums[client_id] = len(data_loader)  

                self.model.train()
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        param.data.copy_(global_state_dict[name])

                # iter_count = 1
                for epoch in range(self.args.local_epoch):  
                    train_loss = []  
                    for i, (batch_x, batch_y,mask_bool) in enumerate(data_loader):     
                        
                        model_optim.zero_grad()
                        batch_x = batch_x.float().to(self.device)
                     
                        outputs = self.model(batch_x, None, None, None, mask_bool) ###

                        consistency_loss = 0.0
                        count = 0
                        
                        for shared_batch_x in shared_dataset_loader:
                            # print("-------",shared_batch_x.shape)
                            count+=1
                            
                            shared_batch_x = shared_batch_x.float().to(self.device)
                            
                            shared_outputs = self.model(shared_batch_x, None, None, None)
                            global_output = self.global_model(shared_batch_x, None, None, None)
                                # shared_outputs = shared_outputs[:, :, f_dim:]
                                # global_output = global_output[:, :, f_dim:]
                            # shared_f_dim = -1 if self.args.features == 'MS' else 0
                            # shared_outputs = shared_outputs[:, :, shared_f_dim:]
                            consistency_loss += criterion(shared_outputs, global_output)
                        consistency_loss/=count
                        
                        # outputs = self.model(batch_x, None, None, None, mask_bool) ###
                        f_dim = -1 if self.args.features == 'MS' else 0
                        
                        outputs = outputs[:, :, f_dim:]
                        criterion_loss = criterion(outputs, batch_x)
                        # if consistency_loss != 0.0:
                        
                        
                        loss = criterion_loss + consistency_loss*self.args.consis_loss_coef
                        train_loss.append(loss)

                        loss.backward()
                        model_optim.step()
                       

                    torch.cuda.empty_cache()
                    
                    stacked_tensor = torch.stack(train_loss)
                    client_train_loss = torch.mean(stacked_tensor)
                    # vali_loss = self.vali(vali_data, vali_loader, criterion)  
                    client_vali_loader = vali_loader[client_id]
                    client_vali_data = vali_data[client_id]       
                    # client_test_loader = test_loader[client_id]
                    # client_test_data = test_data[client_id]               
                    vali_loss = self.vali(client_vali_data, client_vali_loader, criterion) 
                    # test_loss = self.vali(client_test_data, client_test_loader, criterion)
                    
                 
                    print("{0}, Local Epoch: {1}, Steps: {2} | Train Loss: {3:.7f} Vali Loss: {4:.7f}".format(
                    client_id, epoch + 1, train_steps, client_train_loss, vali_loss))   
                    # if self.args.client_nums == 1 and self.args.gpt == 'True' and self.args.local_epoch>1:
                    #     print("Saving local model ...")            
                    #     torch.save(self.model, path + '/' + 'checkpoint'+ "localepoch" + str(epoch) + '_' + str(self.args.mask_ratio) + '.pth')

                # Upload trainable_params to the server
                trainable_params = {name: param for name, param in self.model.named_parameters() if param.requires_grad}
                model_dicts.append(trainable_params)
                

                # model_dicts.append(self.model.state_dict())
                # print('\n')
            # iter_count =1
            # averaging
            state_dicts = model_dicts
            total_data = sum(data_nums.values())
            data_ratio = {client_id: count / total_data for client_id, count in data_nums.items()}  

            # Initialize global parameters
            global_state_dict = {name: torch.zeros_like(param) for name, param in self.global_model.named_parameters() if param.requires_grad}

            # Aggregate trainable parameters
            for state_dict, client_id in zip(state_dicts, list(data_ratio.keys())):
                for name, param in state_dict.items():
                    if name in global_state_dict:
                        global_state_dict[name] += param * data_ratio[client_id]

            
            for name, param in self.global_model.named_parameters():
                if name in global_state_dict:
                    param.data.copy_(global_state_dict[name])

            # state_dicts = model_dicts
            # total_data = sum(data_nums.values())
            # # fed_weight = torch.tensor(data_nums, dtype=torch.float) / sum(data_nums)  
            # data_ratio = {client_id: count / total_data for client_id, count in data_nums.items()}  
            # global_state_dict = {key: state_dicts[0][key] * data_ratio[list(data_ratio.keys())[0]]  for key in state_dicts[0].keys()}
            # for state_dict, client_id in zip(state_dicts[1:], list(data_ratio.keys())[1:]):
            #     for key in state_dict.keys():
            #         global_state_dict[key] += state_dict[key] * data_ratio[client_id]

            # avg_state_dict = {}
            # fed_avg_freqs = fed_weight
            # for key in state_dicts[0].keys():
            #     avg_state_dict[key] = state_dicts[0][key] * fed_avg_freqs[0]  #先加入第一个client 初始化

            # state_dicts = state_dicts[1:]
            # fed_avg_freqs = fed_avg_freqs[1:]
            # for state_dict, freq in zip(state_dicts, fed_avg_freqs):
            #     for key in state_dict.keys():
            #         avg_state_dict[key] += state_dict[key] * freq
            # global_state_dict = avg_state_dict

            # self.global_model.load_state_dict(global_state_dict)            
                    
            print("Saving model ...")            
            torch.save(self.global_model, path + '/' + 'checkpoint'+ str(global_epoch) + 'global_' + str(self.args.mask_ratio) + '.pth')   
                        
        return self.global_model


    def test(self, setting, test=0):  
        print("======================TEST MODE======================")
        test_start_time = time.time()

        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        if test:
            print('loading model')

            state_dict_lis = []
            for i in range(2,3):  # 假设有1到4的pth文件
                pth_filename = f'checkpoint{i}_' + str(self.args.mask_ratio) +'.pth'
                pth_path = os.path.join(self.args.test_path, pth_filename)
                state_dict_lis.append(pth_path)
            # state_dict = torch.load(os.path.join(setting, 'checkpoint1_0.3.pth'))
         
            # self.model =  copy.deepcopy(state_dict)
            # self.model.to(self.device)
            # self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        else:
            global_state_dict = self.global_model.state_dict()

            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(global_state_dict[name])
            # self.model.load_state_dict(global_state_dict)
            
            # self.model = self.global_model
        
        folder_path = '../test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if test:
            for state_dict_itm in state_dict_lis:
                print(state_dict_itm)
                self.model =  copy.deepcopy(torch.load(state_dict_itm))
                self.model.to(self.device)

                self.model.eval()
                self.anomaly_criterion = nn.MSELoss(reduce=False)
                train_energy ={}
                # (1) stastic on the train set
                stastic_time = time.time()
                for client_id, data_loader in tqdm(train_loader.items(), desc="stastic on the train set", unit="client"):
                    energy = []
                    attens_energy = []
                    with torch.no_grad():
                        for i, (batch_x, batch_y,mask_bool) in enumerate(data_loader):
                            batch_x = batch_x.float().to(self.device)
                            # reconstruction
                          
                            outputs = self.model(batch_x, None, None, None, mask_bool)
                            # criterion
                            # print("batch_x: ",batch_x.shape,'---',"outputs: ",outputs.shape)
                            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                            score = score.detach().cpu().numpy()
                            energy.append(score)

                    attens_energy = np.concatenate(energy, axis=0).reshape(-1)

                    tmp_train_energy = np.array(attens_energy)
                
                    train_energy[client_id] = tmp_train_energy
                
                print("cost time: {}".format(time.time() - stastic_time))

                

                # (2) find the threshold
                threshold_time = time.time()
                attens_energy = []
                test_labels = {}
                test_energy = {}
                for client_id, tes_data_loader in tqdm(test_loader.items(), desc="find the threshold", unit="client"):
                    energy = []
                    tmp_labels = []
                    attens_energy = []
                    
                    for i, (batch_x, batch_y,mask_bool) in enumerate(tes_data_loader):
                        batch_x = batch_x.float().to(self.device)
                        # reconstruction
                       
                        outputs = self.model(batch_x, None, None, None, mask_bool)

                        # criterion
                        score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                        score = score.detach().cpu().numpy()
                        energy.append(score)
                        tmp_labels.append(batch_y)

                    tmpt_labels = np.concatenate(tmp_labels, axis=0).reshape(-1)
                    test_labels[client_id] = np.array(tmpt_labels)

                    attens_energy = np.concatenate(energy, axis=0).reshape(-1)
                    tmp_test_energy = np.array(attens_energy)
                    test_energy[client_id] = tmp_test_energy
                    
                # combined_energy = np.concatenate([train_energy, test_energy], axis=0)
                # threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
                combined_energy = {client_id: np.concatenate([train_energy[client_id], test_energy[client_id]], axis=0)
                        for client_id in train_energy.keys()}
                thresholds = {client_id: np.percentile(combined_energy[client_id], 100 - self.args.anomaly_ratio)
                    for client_id in combined_energy.keys()}
                
                # print("Thresholds :", thresholds)   
                print(f"{self.args.model_id}_Thresholds:{thresholds}")

                all_thresholds = list(thresholds.values())
                
                print("Maximum value:", np.max(all_thresholds),'--',"Minimum value:", np.min(all_thresholds),'--',"Mean value:", np.mean(all_thresholds))
                print("Variance:", np.var(all_thresholds))
                #visualization
                plt.figure(figsize=(10, 10))
                x_labels = [str(client.split('_')[1]) for client in thresholds.keys()]
                plt.bar(x_labels, all_thresholds, color='skyblue')
                plt.title('{}_Thresholds for Different Clients'.format(self.args.model_id))
                # plt.xlabel('Clients')
                
                plt.ylabel('Threshold Values')
                plt.xticks(rotation=45)
                plt.savefig('{}_all_thresholds.png'.format(self.args.model_id)) 
                
                median_threshold = np.median(all_thresholds)
                threshold = median_threshold
                print("Threshold:", median_threshold)
                
                print("cost time: {}".format(time.time() - threshold_time))
                

                # (3) evaluation on the test set
                evaluation_time = time.time()
                pred = {}
                gt = {}
                for client_id, tes_data_loader in tqdm(test_loader.items(), desc="evaluation", unit="client"):
                    # pred[client_id] = (test_energy[client_id] > threshold).astype(int)
                    pred[client_id] = (test_energy[client_id] > thresholds[client_id]).astype(int)
                
                    tmp_test_labels = test_labels[client_id]  
                    gt[client_id] = tmp_test_labels.astype(int)

                    # print(client_id, "_pred:   ", pred[client_id].shape)
                    # print(client_id, "_gt:     ", gt[client_id].shape)
                print("cost time: {}".format(time.time() - evaluation_time))

                # (4) detection adjustment
                adjustment_time = time.time()
                total_result = {}
                for client_id, tes_data_loader in gt.items():
                    
                    tmp_gt, tmp_pred = adjustment(gt[client_id], pred[client_id])

                    client_pred = np.array(tmp_pred)
                    client_gt = np.array(tmp_gt)
                    roc_auc = roc_auc_score(client_gt, client_pred) ####
                    accuracy = accuracy_score(client_gt, client_pred)
                    precision, recall, f_score, support = precision_recall_fscore_support(client_gt, client_pred, average='binary')
                    print(client_id, " --Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                        accuracy, precision,
                        recall, f_score))
                    total_result[client_id] = {
                        'Accuracy': accuracy,
                        'Precision': precision,
                        'Recall': recall,
                        'F-score': f_score,
                        'roc_auc':roc_auc
                    }
                # Calculate and print mean values
                mean_accuracy = np.mean([result['Accuracy'] for result in total_result.values()])
                mean_precision = np.mean([result['Precision'] for result in total_result.values()])
                mean_recall = np.mean([result['Recall'] for result in total_result.values()])
                mean_f_score = np.mean([result['F-score'] for result in total_result.values()])
                mean_roc_auc = np.mean([result['roc_auc'] for result in total_result.values()])
                print("Mean Accuracy: {:0.4f}, Mean Precision: {:0.4f}, Mean Recall: {:0.4f},Mean rou_auc: {:0.4f}  Mean F-score: {:0.4f}".format(
                    mean_accuracy, mean_precision, mean_recall, mean_roc_auc, mean_f_score))


                print("cost time: {}".format(time.time() - adjustment_time))
                print("Total test time: {}".format(time.time() - test_start_time))
        else:
            self.model.eval()
            self.anomaly_criterion = nn.MSELoss(reduce=False)
            train_energy ={}
            # (1) stastic on the train set
            stastic_time = time.time()
            for client_id, data_loader in tqdm(train_loader.items(), desc="stastic on the train set", unit="client"):
                energy = []
                attens_energy = []
                with torch.no_grad():
                    for i, (batch_x, batch_y,mask_bool) in enumerate(data_loader):
                        batch_x = batch_x.float().to(self.device)
                        # reconstruction
                       
                        outputs = self.model(batch_x, None, None, None, mask_bool)
                        # criterion
                        # print("batch_x: ",batch_x.shape,'---',"outputs: ",outputs.shape)
                        score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                        score = score.detach().cpu().numpy()
                        energy.append(score)
                attens_energy = np.concatenate(energy, axis=0).reshape(-1)

                tmp_train_energy = np.array(attens_energy)
            
                train_energy[client_id] = tmp_train_energy
            
            print("cost time: {}".format(time.time() - stastic_time))
            # torch.cuda.empty_cache()
            

            # (2) find the threshold
            threshold_time = time.time()
            attens_energy = []
            test_labels = {}
            test_energy = {}
            for client_id, tes_data_loader in tqdm(test_loader.items(), desc="find the threshold", unit="client"):
                energy = []
                tmp_labels = []
                attens_energy = []
                
                for i, (batch_x, batch_y,mask_bool) in enumerate(tes_data_loader):
                    batch_x = batch_x.float().to(self.device)
                    # reconstruction
                    outputs = self.model(batch_x, None, None, None, mask_bool)
                    # criterion
                    score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                    score = score.detach().cpu().numpy()
                    energy.append(score)
                    tmp_labels.append(batch_y)

                tmpt_labels = np.concatenate(tmp_labels, axis=0).reshape(-1)
                test_labels[client_id] = np.array(tmpt_labels)

                attens_energy = np.concatenate(energy, axis=0).reshape(-1)
                tmp_test_energy = np.array(attens_energy)
                test_energy[client_id] = tmp_test_energy
                
            # combined_energy = np.concatenate([train_energy, test_energy], axis=0)
            # threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
            combined_energy = {client_id: np.concatenate([train_energy[client_id], test_energy[client_id]], axis=0)
                    for client_id in train_energy.keys()}
            thresholds = {client_id: np.percentile(combined_energy[client_id], 100 - self.args.anomaly_ratio)
                for client_id in combined_energy.keys()}
            
            # print("Thresholds :", thresholds)   #不同客户端的阈值差别比较大？
            print(f"{self.args.model_id}_Thresholds:{thresholds}")

            all_thresholds = list(thresholds.values())
            
            print("Maximum value:", np.max(all_thresholds),'--',"Minimum value:", np.min(all_thresholds),'--',"Mean value:", np.mean(all_thresholds))
            print("Variance:", np.var(all_thresholds))
            #visualization
            plt.figure(figsize=(10, 10))
            x_labels = [str(client.split('_')[1]) for client in thresholds.keys()]
            plt.bar(x_labels, all_thresholds, color='skyblue')
            plt.title('{}_Thresholds for Different Clients'.format(self.args.model_id))
            # plt.xlabel('Clients')
            
            plt.ylabel('Threshold Values')
            plt.xticks(rotation=45)
            plt.savefig('{}_all_thresholds.png'.format(self.args.model_id)) 
            
            median_threshold = np.median(all_thresholds)
            threshold = median_threshold
            print("Threshold:", median_threshold)
            
            print("cost time: {}".format(time.time() - threshold_time))
            
            # torch.cuda.empty_cache()

            # (3) evaluation on the test set
            evaluation_time = time.time()
            pred = {}
            gt = {}
            for client_id, tes_data_loader in tqdm(test_loader.items(), desc="evaluation", unit="client"):
                # pred[client_id] = (test_energy[client_id] > threshold).astype(int)
                pred[client_id] = (test_energy[client_id] > thresholds[client_id]).astype(int)
            
                tmp_test_labels = test_labels[client_id]  
                gt[client_id] = tmp_test_labels.astype(int)

                # print(client_id, "_pred:   ", pred[client_id].shape)
                # print(client_id, "_gt:     ", gt[client_id].shape)
            print("cost time: {}".format(time.time() - evaluation_time))

            # (4) detection adjustment
            adjustment_time = time.time()
            total_result = {}
            for client_id, tes_data_loader in gt.items():
                
                tmp_gt, tmp_pred = adjustment(gt[client_id], pred[client_id])

                client_pred = np.array(tmp_pred)
                client_gt = np.array(tmp_gt)
                

                roc_auc = roc_auc_score(client_gt, client_pred) ####
                accuracy = accuracy_score(client_gt, client_pred)
                precision, recall, f_score, support = precision_recall_fscore_support(client_gt, client_pred, average='binary')
                print(client_id, " --Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                    accuracy, precision,
                    recall, f_score))
                total_result[client_id] = {
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F-score': f_score,
                    'roc_auc':roc_auc
                }
            # Calculate and print mean values
            mean_accuracy = np.mean([result['Accuracy'] for result in total_result.values()])
            mean_precision = np.mean([result['Precision'] for result in total_result.values()])
            mean_recall = np.mean([result['Recall'] for result in total_result.values()])
            mean_f_score = np.mean([result['F-score'] for result in total_result.values()])
            mean_roc_auc = np.mean([result['roc_auc'] for result in total_result.values()])
            print("Mean Accuracy: {:0.4f}, Mean Precision: {:0.4f}, Mean Recall: {:0.4f},Mean rou_auc: {:0.4f}  Mean F-score: {:0.4f}".format(
                mean_accuracy, mean_precision, mean_recall, mean_roc_auc, mean_f_score))


            print("cost time: {}".format(time.time() - adjustment_time))
            print("Total test time: {}".format(time.time() - test_start_time)) 
        return

    def aggregate_models(self, data_nums, client_models_dict):
        total_data = sum(data_nums.values())
        data_ratio = {client_id: count / total_data for client_id, count in data_nums.items()}  
        state_dicts = client_models_dict
        client_names = list(state_dicts.keys())
        global_state_dict = {key: state_dicts[client_names[0]][key] * data_ratio[client_names[0]] for key in state_dicts[client_names[0]].keys()}
        for client_name in client_names[1:]:
            for key in state_dicts[client_name].keys():
                global_state_dict[key] += state_dicts[client_name][key] * data_ratio[client_name]

        return global_state_dict


    def shared_data(self, setting):
   
        train_data, train_loader = self._get_data(flag='train',shared=True)
        criterion = self._select_criterion()
        # vae = VAE(original_dim, latent_dim)
        # optimizer = optim.Adam(vae.parameters(), lr=0.001)

        client_models = {}
        client_losses  = {}
        data_nums = {}
        # vae_local_model = VAE(original_dim, self.latent_dim).to(self.device)
        # vae_local_optimizer = optim.Adam(self.vae_local_model.parameters(), lr=0.001)
        _,vae_local_optimizer = self._select_optimizer()
        shared_dataset_start_time = time.time()
        vae_model_list = {}
        for global_epoch in range(self.args.vae_train_epochs):   #每轮
            if global_epoch == 0:
                    self.global_vae_model = self.vae_local_model
                    global_state_dict = self.global_vae_model.state_dict()
            else:
                global_state_dict = self.global_vae_model.state_dict()
            
            for client_id, data_loader in train_loader.items():
                self.vae_local_model.train()
                self.vae_local_model.load_state_dict(global_state_dict) #initialization
                alpha1 = 10
                train_loss = {}
                data_nums[client_id] = len(data_loader)
                for epoch in range(self.args.vae_local_epochs):
                 
                    for i, (batch_x, batch_y) in enumerate(data_loader): 

                        vae_local_optimizer.zero_grad()
                        # inputs = batch_x.view(-1,original_dim)
                        
                        inputs = batch_x.float().to(self.device)
                        
                        outputs, z_mean, z_log_var = self.vae_local_model(inputs)  #outputs:  torch.Size([32, 100, 38])
                        # outputs = outputs.float().to(self.device)
                        # print("outputs: ",outputs.shape) #torch.Size([32, 100, 38])
                        # print("z_mean: ",z_mean.shape) #[32, 100, 16]
                        # generated_sequence = vae.decoder(torch.randn_like(z_mean))  #(32,100,38)

                        vaeloss = vae_loss(self.vae_local_model, inputs, outputs, z_mean, z_log_var)
                        input_size = self.original_dim
                        distribute_loss = wasserstein_distance(inputs, input_size,outputs) #
                        # distribute_loss2 = nn.functional.kl_div(outputs.flatten().log(), inputs.flatten(), reduction='mean') #加了这个loss会变nan
                        
                        mutual_information = manual_info_los1s(inputs, outputs)
                      
                        loss = vaeloss + distribute_loss + mutual_information 

                        loss.backward()
                        vae_local_optimizer.step()

                        train_loss[client_id] = loss.item()

                # vae_model_list[client_id] = self.vae_local_model

                client_losses[client_id] = np.average(train_loss[client_id])
                client_models[client_id] = self.vae_local_model.state_dict()
                print(f'Shared data Epoch [{epoch + 1}/{self.args.vae_local_epochs}],{client_id}  Loss: {client_losses[client_id]:.4f}')
            # global_state_dict = self.aggregate_models(data_nums,client_models) #
            # self.global_vae_model = global_state_dict

    #Client synthesizes time series   
        local_sequences_per_client = {}
        shared_dataset = []
        shared_sequences_list = []
        num_samples = self.args.shared_size
        # vae_model = VAE(self.original_dim, self.args.latent_dim).to(self.device)
        with torch.no_grad():
            for client_id, data_loader in train_loader.items():
                latent_samples = torch.randn(num_samples, self.args.latent_dim).to(self.device)
                self.vae_local_model.load_state_dict(client_models[client_id])
                generated_sequences = self.vae_local_model.decoder(latent_samples).view(-1,self.original_dim)
                # print("****,",generated_sequences.shape) #torch.Size([100, 38])
                generated_sequence = generated_sequences.cpu().detach().numpy()
                local_sequences_per_client[client_id] = generated_sequence
                shared_sequences_list.append(generated_sequence)    

            # shared_dataset_np = np.vstack(shared_sequences_list)
            shared_dataset_np = np.array(shared_sequences_list)
            # shared_dataset_np = shared_dataset_np.reshape(-1, num_samples, self.original_dim)
            shared_dataset_tensor = torch.tensor(shared_dataset_np, dtype=torch.float32)
            # shared_dataset_tensor = torch.tensor(shared_dataset_np, dtype=torch.float32)
            # shared_dataset_tensor = torch.tensor(shared_dataset_np, dtype=torch.float32).view(-1, num_samples, self.original_dim)

            # shared_dataset_loader = data_utils.DataLoader(shared_dataset_tensor, batch_size=self.batch_size, shuffle=True)
            shared_dataset_loader = data_utils.DataLoader(shared_dataset_tensor, batch_size=self.args.local_bs, shuffle=True)
        
            print(shared_dataset_tensor.shape) #torch.Size([10, 100, 38])
        
        total_size_bytes = sys.getsizeof(shared_dataset_tensor.storage())
            # 将字节数转换为 MB 或 B
        total_size_mb = total_size_bytes / (1024 ** 2)  # 转换为 MB
        total_size_b = total_size_bytes  # 保留字节单位
        print(f"Total size of shared_dataset_tensor: {total_size_mb:.2f} MB or {total_size_b} bytes")

        print('cost time:', time.time() - shared_dataset_start_time)
        torch.cuda.empty_cache()
        return shared_dataset_loader
        

    