import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from data_provider.patch_mask import *
from data_provider.test import *
# from sktime.utils import load_data
import warnings

warnings.filterwarnings('ignore')


class PSMSegLoader(Dataset):
    def __init__(self, data, test_data, test_labels,shared, root_path, win_size, step=1, flag="train", patch_len=10, patch_stride=10,mask_ratio=0.2,mask_factor=1.5,connection_ratio=1,percentile=10,weight_similarity=0.8, weight_residual=0.2):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = test_labels
        self.shared = shared
        self.win_size = win_size 
        self.patch_len = patch_len
        self.stride = patch_stride
        self.mask_factor = mask_factor
        self.mask_ratio = mask_ratio
        self.connection_ratio = connection_ratio
        self.percentile = percentile
        self.weight_similarity = weight_similarity
        self.weight_residual = weight_residual

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        # print("get_items,index:",index)
        
        if self.flag == "train":
            sequence_window,label_window = np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            sequence_window,label_window = np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            sequence_window,label_window = np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            sequence_window,label_window = np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        
        if not self.shared:
            xb_patch, yb_patch, num_patch = create_patch(sequence_window, label_window, self.patch_len, self.stride)     # xb_patch: [bs x num_patch x n_vars x patch_len]
            # # print(xb_patch.shape)
            if (self.flag == 'test'): 
                num_patch, nvars, patch_length = xb_patch.shape
                xb_patch_permuted = xb_patch.permute(0, 2, 1)
                xb_patch = xb_patch_permuted.reshape(-1, nvars)
                return xb_patch,yb_patch,mask_values
            
            b = np.random.binomial(np.ones(1).astype(int), self.connection_ratio)

            if b :
                patch_similarity = calculate_patch_similarity(xb_patch)
                patch_residual = series_decomposition(xb_patch)
                anomalous_patches = get_anom_index(patch_similarity,patch_residual,self.percentile,self.weight_similarity, self.weight_residual)

                xb_mask, _, self.mask, _ = random_masking_with_anomalies(xb_patch, self.mask_ratio, anomalous_patches, self.mask_factor)   # xb_mask: [bs x num_patch x n_vars x patch_len]
            else:
                #random masking
                xb_mask, _, self.mask, _ = random_masking(xb_patch, self.mask_ratio)   # xb_mask: [bs x num_patch x n_vars x patch_len]
            
            num_patch, nvars, patch_length = xb_mask.shape
            xb_mask_permuted = xb_mask.permute(0, 2, 1)
            # print(num_patch, '--', patch_length, '--', nvars)
            xb_mask = xb_mask_permuted.reshape(-1, nvars)
            yb_patch = yb_patch.view(-1)

            # self.mask = self.mask.bool()    # mask: [bs x num_patch x n_vars]
            mask_bool_expanded = self.mask.repeat(1,patch_length) 
            mask_values = mask_bool_expanded.view((num_patch, patch_length, nvars))  #mask: [(num_patch*patch_len) x n_vars]
            mask_values = mask_values.view((num_patch * patch_length, nvars))
            # mask_values = mask_values.bool()
            # print("xb_mask: ", xb_mask.shape)
        
            return xb_mask, yb_patch, mask_values
        return sequence_window,label_window



class MSLSegLoader(Dataset):
    def __init__(self, data, test_data, test_labels,shared, root_path, win_size, step=1, flag="train", patch_len=10, patch_stride=10,mask_ratio=0.2,mask_factor=1.5,connection_ratio=1,percentile=10,weight_similarity=0.8, weight_residual=0.2):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = test_labels
        self.shared = shared
        self.win_size = win_size 
        self.patch_len = patch_len
        self.stride = patch_stride
        
        self.mask_factor = mask_factor
        self.mask_ratio = mask_ratio
        self.connection_ratio = connection_ratio
        self.percentile = percentile
        self.weight_similarity = weight_similarity
        self.weight_residual = weight_residual

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        # print("get_items,index:",index)
        
        if self.flag == "train":
            sequence_window,label_window = np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            sequence_window,label_window = np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            sequence_window,label_window = np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            sequence_window,label_window = np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        
        mask_values= 0
        if not self.shared:
            xb_patch, yb_patch, num_patch = create_patch(sequence_window, label_window, self.patch_len, self.stride)     # xb_patch: [bs x num_patch x n_vars x patch_len]
            # # print(xb_patch.shape)
            if (self.flag == 'test') or self.mask_ratio == 0:
                num_patch, nvars, patch_length = xb_patch.shape
                xb_patch_permuted = xb_patch.permute(0, 2, 1)
                xb_patch = xb_patch_permuted.reshape(-1, nvars)
                return xb_patch,yb_patch,mask_values
            
            b = np.random.binomial(np.ones(1).astype(int), self.connection_ratio)
            if b :
                patch_similarity = calculate_patch_similarity(xb_patch)
                patch_residual = series_decomposition(xb_patch)
                anomalous_patches = get_anom_index(patch_similarity,patch_residual,self.percentile,self.weight_similarity, self.weight_residual)
                xb_mask, _, self.mask, _ = random_masking_with_anomalies(xb_patch, self.mask_ratio, anomalous_patches, self.mask_factor)   # xb_mask: [bs x num_patch x n_vars x patch_len]
            else:
                #random masking
                xb_mask, _, self.mask, _ = random_masking(xb_patch, self.mask_ratio)   # xb_mask: [bs x num_patch x n_vars x patch_len]
            
            num_patch, nvars, patch_length = xb_mask.shape
            xb_mask_permuted = xb_mask.permute(0, 2, 1)
            # print(num_patch, '--', patch_length, '--', nvars)
            xb_mask = xb_mask_permuted.reshape(-1, nvars)
            yb_patch = yb_patch.view(-1)

            # self.mask = self.mask.bool()    # mask: [bs x num_patch x n_vars]
            mask_bool_expanded = self.mask.repeat(1,patch_length) 
            mask_values = mask_bool_expanded.view((num_patch, patch_length, nvars))  #mask: [(num_patch*patch_len) x n_vars]
            mask_values = mask_values.view((num_patch * patch_length, nvars))
            # mask_values = mask_values.bool()
            # print("xb_mask: ", xb_mask.shape)

            return xb_mask, yb_patch, mask_values
        return sequence_window,label_window

class SMDSegLoader(Dataset):
    def __init__(self, data, test_data, test_labels,shared, root_path, win_size, step=100, flag="train", patch_len=10, patch_stride=10,mask_ratio=0.3,mask_factor=1.5,connection_ratio=1,percentile=10,weight_similarity=0.8, weight_residual=0.2):
        self.flag = flag
        self.step = step 
        self.win_size = win_size 
        self.patch_len = patch_len
        self.stride = patch_stride
        self.mask_ratio = mask_ratio
        self.scaler = StandardScaler()
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = test_labels
        self.shared = shared
        self.mask_factor = mask_factor
        self.connection_ratio = connection_ratio
        self.percentile = percentile
        self.weight_similarity = weight_similarity
        self.weight_residual = weight_residual

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        # print("get_items,index:",index)
        if self.flag == "train":
            sequence_window,label_window = np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])            
        
        elif (self.flag == 'val'):
            sequence_window,label_window  = np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
           
        elif (self.flag == 'test'):
            sequence_window,label_window  = np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            sequence_window,label_window  = np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        
       
        if not self.shared:
            mask_values= 0
            xb_patch, yb_patch, num_patch = create_patch(sequence_window, label_window, self.patch_len, self.stride)     # xb_patch: [bs x num_patch x n_vars x patch_len]
            # # print(xb_patch.shape)
            if (self.flag == 'test') or self.mask_ratio == 0: 
                num_patch, nvars, patch_length = xb_patch.shape
                xb_patch_permuted = xb_patch.permute(0, 2, 1)
                xb_patch = xb_patch_permuted.reshape(-1, nvars)
                return xb_patch,yb_patch,mask_values
            
            b = np.random.binomial(np.ones(1).astype(int), self.connection_ratio)

            if b:
                patch_similarity = calculate_patch_similarity(xb_patch)
                patch_residual = series_decomposition(xb_patch)
                anomalous_patches = get_anom_index(patch_similarity,patch_residual,self.percentile,self.weight_similarity, self.weight_residual)

                xb_mask, _, self.mask, _ = random_masking_with_anomalies(xb_patch, self.mask_ratio, anomalous_patches, self.mask_factor)   # xb_mask: [bs x num_patch x n_vars x patch_len]
            
            else:
                #random masking
                xb_mask, _, self.mask, _ = random_masking(xb_patch, self.mask_ratio)   # xb_mask: [bs x num_patch x n_vars x patch_len]
                
            num_patch, nvars, patch_length = xb_mask.shape
            xb_mask_permuted = xb_mask.permute(0, 2, 1)
            # print(num_patch, '--', patch_length, '--', nvars)
            xb_mask = xb_mask_permuted.reshape(-1, nvars)
            yb_patch = yb_patch.view(-1)

            # self.mask = self.mask.bool()    # mask: [bs x num_patch x n_vars]
            mask_bool_expanded = self.mask.repeat(1,patch_length) 
            mask_values = mask_bool_expanded.view((num_patch, patch_length, nvars))  #mask: [(num_patch*patch_len) x n_vars]
            mask_values = mask_values.view((num_patch * patch_length, nvars))
            # mask_values = mask_values.bool()
            # print("xb_mask: ", xb_mask.shape)

            return xb_mask, yb_patch, mask_values
        return sequence_window,label_window

class SWATSegLoader(Dataset):
    def __init__(self, data, test_data, test_labels,shared, root_path, win_size, step=1, flag="train", patch_len=10, patch_stride=10,mask_ratio=0.2,mask_factor=1.5,connection_ratio=1,percentile=10,weight_similarity=0.8, weight_residual=0.2):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        train_data = data
        labels = test_labels
        self.shared = shared
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        self.val = test_data
        self.test_labels = labels

        self.patch_len = patch_len
        self.stride = patch_stride
        self.mask_ratio = mask_ratio

        self.mask_factor = mask_factor
        self.connection_ratio = connection_ratio
        self.percentile = percentile
        self.weight_similarity = weight_similarity
        self.weight_residual = weight_residual

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            tmp = (self.train.shape[0] - self.win_size) // self.step + 1
            return tmp
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        # print("get_items,index:",index)
        if self.flag == "train":
            sequence_window,label_window = np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            sequence_window,label_window = np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            sequence_window,label_window = np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            sequence_window,label_window = np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        
        
        if not self.shared:
            xb_patch, yb_patch, num_patch = create_patch(sequence_window, label_window, self.patch_len, self.stride)     # xb_patch: [bs x num_patch x n_vars x patch_len]
            # # print(xb_patch.shape)
            if (self.flag == 'test'): 
                num_patch, nvars, patch_length = xb_patch.shape
                xb_patch_permuted = xb_patch.permute(0, 2, 1)
                xb_patch = xb_patch_permuted.reshape(-1, nvars)
                return xb_patch,yb_patch,mask_values

            b = np.random.binomial(np.ones(1).astype(int), self.connection_ratio)

            if b:
                patch_similarity = calculate_patch_similarity(xb_patch)
                patch_residual = series_decomposition(xb_patch)
                anomalous_patches = get_anom_index(patch_similarity,patch_residual,self.percentile,self.weight_similarity, self.weight_residual)

                xb_mask, _, self.mask, _ = random_masking_with_anomalies(xb_patch, self.mask_ratio, anomalous_patches, self.mask_factor)   # xb_mask: [bs x num_patch x n_vars x patch_len]
            
            else:
                #random masking
                xb_mask, _, self.mask, _ = random_masking(xb_patch, self.mask_ratio)   # xb_mask: [bs x num_patch x n_vars x patch_len]
            
            
            num_patch, nvars, patch_length = xb_mask.shape
            xb_mask_permuted = xb_mask.permute(0, 2, 1)
            # print(num_patch, '--', patch_length, '--', nvars)
            xb_mask = xb_mask_permuted.reshape(-1, nvars)
            yb_patch = yb_patch.view(-1)

            # self.mask = self.mask.bool()    # mask: [bs x num_patch x n_vars]
            mask_bool_expanded = self.mask.repeat(1,patch_length) 
            mask_values = mask_bool_expanded.view((num_patch, patch_length, nvars))  #mask: [(num_patch*patch_len) x n_vars]
            mask_values = mask_values.view((num_patch * patch_length, nvars))
            # mask_values = mask_values.bool()
            # print("xb_mask: ", xb_mask.shape)
            

            return xb_mask, yb_patch, mask_values
        return sequence_window,label_window