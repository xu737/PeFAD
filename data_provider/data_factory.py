from data_provider.data_loader import  PSMSegLoader,MSLSegLoader, SMDSegLoader, SWATSegLoader
from torch.utils.data import DataLoader
import numpy as np
import os
import pandas as pd


data_dict = {
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
}

def Distribute_data(dataset_name, root_path, client_nums, flag):
    # Distribute data to each client
    if dataset_name == 'SMD':
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))
    # test_data.shape (708420, 38) test_labels.shape (708420,)
        
    elif dataset_name == 'MSL':
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        
    elif dataset_name == 'PSM':
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        test_data =  pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        
    elif dataset_name == 'SWAT':
        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        data = train_data.values[:, :-1]
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        test_labels = test_data.values[:, -1:]
        test_data = test_data.values[:, :-1]
    
    
    data_per_client = len(data) // client_nums
    remainder = len(data) % client_nums  # Calculate the remainder
    
    tes_data_per_client = len(test_data) // client_nums
    # print("tes_data_per_client",tes_data_per_client)
    tes_remainder = len(test_data) % client_nums  # Calculate the remainder
    
    client_data = {}
    client_test_data = {}
    client_test_labels = {}
    for i in range(client_nums):
        start_idx = i * data_per_client
        tes_start_idx = i * tes_data_per_client
        if i == client_nums - 1 and remainder > 0:
            end_idx = (i + 1) * data_per_client + remainder
            tes_end_idx = (i + 1) * tes_data_per_client + tes_remainder
        else:
            end_idx = (i + 1) * data_per_client
            tes_end_idx = (i + 1) * tes_data_per_client
            
        client_data[f'client_{i+1}'] = {'X': data[start_idx:end_idx]}  #client_data.items()
        client_test_data[f'client_{i+1}'] = {'test_data': test_data[tes_start_idx:tes_end_idx]}  
        client_test_labels[f'client_{i+1}'] = {'test_labels': test_labels[tes_start_idx:tes_end_idx]}  

    return client_data,client_test_data,client_test_labels



def data_provider(args, flag,shared):
    Data = data_dict[args.data]

    if flag == 'test':
        shuffle_flag = False
    else:
        shuffle_flag = True
   
    drop_last = False              
    # assign data_set to clienta,client form DataLoader(local_batch_size)
    # dataset_name, root_path, client_nums, flag
    client_data,client_test_data,client_test_labels = Distribute_data(dataset_name=args.model_id, root_path=args.root_path, client_nums=args.client_nums, flag=flag)
    # print('dataset_clients',dataset_clients.keys())
    client_dataset_dict = {}

    for client_id, client_train_data in client_data.items():
        client_X = client_data[client_id]['X']
        client_test_X = client_test_data[client_id]['test_data']
        client_test_L = client_test_labels[client_id]['test_labels']

        client_dataset_dict[client_id] = Data(
            data=client_X,
            test_data=client_test_X,
            test_labels=client_test_L,
            root_path=args.root_path,   #dataset/SMD
            win_size=args.seq_len,  
            shared = shared,
            flag=flag,
            patch_len = args.patch_len,
            patch_stride = args.patch_stride,
            mask_ratio = args.mask_ratio,
            mask_factor = args.mask_factor,
            connection_ratio = args.connection_ratio,
            percentile = args.percentile,
            weight_similarity = args.weight_similarity,
            weight_residual= args.weight_residual
            )
    
    client_loader_dict={}
    for client_id, client_train_data in client_data.items():
        # print('client_id',client_id)
        client_loader_dict[client_id] = DataLoader(
            client_dataset_dict[client_id],
            batch_size=args.local_bs,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )  

    return client_dataset_dict, client_loader_dict

    