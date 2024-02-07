import torch
import torch.nn.functional as F
import numpy as np
# from statsmodels.tsa.seasonal import STL
# from statsmodels.tsa.seasonal import seasonal_decompose
from pyts.decomposition import SingularSpectrumAnalysis

def min_max_scaling_list(score_list):
    tensor_list_tensor = torch.tensor(score_list, dtype=torch.float32)

    min_value = float('inf')
    max_value = float('-inf')
    
    for item in tensor_list_tensor:
        min_value = min(min_value, torch.min(item))
        max_value = max(max_value, torch.max(item))
    
    if min_value == max_value:
        return tensor_list_tensor
    

    scaled_tensor_list = [(tensor - min_value) / (max_value - min_value) for tensor in tensor_list_tensor]
    
    return scaled_tensor_list




def calculate_patch_similarity(xb_patch):

    num_patch, n_vars, patch_len = xb_patch.shape

    cos_similarity_list = []
   
    for i in range(num_patch - 1):
        
        patch_k1 = xb_patch[i, :, :]
        patch_k2 = xb_patch[i + 1, :, :]

        cos_similarity_sum = 0.0
        # cosine_sim = F.cosine_similarity(patch_k1, patch_k2, dim=1)
        # cos_similarity_avg = cosine_sim.mean().item()

        for j in range(n_vars):
            
            patch_k1_var = patch_k1[j, :]
            patch_k2_var = patch_k2[j, :]

            patch_k1_flat = patch_k1_var.reshape(1, -1)
            patch_k2_flat = patch_k2_var.reshape(1, -1)


            cos_similarity = F.cosine_similarity(patch_k1_flat, patch_k2_flat, dim=-1)
            cos_similarity_sum += max(cos_similarity.item(), 0)
            
            # Pearson correlation coefficient
            # pearson_corr = torch.corrcoef(patch_k1_flat, patch_k2_flat, rowvar=False)[0, 1]

        
        cos_similarity_avg = cos_similarity_sum / n_vars
        cos_similarity_list.append(1-cos_similarity_avg) 
    
    
    cos_similarity = torch.tensor(cos_similarity_list, dtype=torch.float32) 
    #min_max_scaling_list(cos_similarity_list)
    zero_tensor = torch.tensor([0.0])
    cos_similarity_list.insert(0, zero_tensor)
    
    cos_similarity = torch.tensor(cos_similarity_list)

    return cos_similarity


def series_decomposition(xb_patch):
    num_patch, n_vars, patch_len = xb_patch.shape
    flat_patches = xb_patch.reshape(num_patch, -1)
    residual_magnitudes = []
    # normalized_residuals = []
    for i in range(num_patch):

        current_patch = xb_patch[i, :, :]
        current_patch_1d = current_patch.flatten().numpy()

        # time series decomposition

        window_size = 3  
        ssa = SingularSpectrumAnalysis(window_size=window_size)
        components = ssa.fit_transform(current_patch_1d.reshape(1, -1))
        local_trend = components[0]
        residuals = current_patch_1d - local_trend

        residual_magnitude = torch.norm(torch.from_numpy(residuals), p=2)

        # residual_magnitudes.append(0)
        residual_magnitudes.append(residual_magnitude.item()) 
    
    if len(residual_magnitudes) > 1:      
        normalized_residuals = min_max_scaling_list(residual_magnitudes) #normalize
    # print(len(normalized_residuals))
        return normalized_residuals
    else:
        return residual_magnitudes


# Get the index of the anomalous patch
def get_anom_index(cos_similarity_list, residual_magnitudes_list, percentile,weight_similarity=0.8, weight_residual=0.2):

    weighted_cos_similarity = np.multiply(cos_similarity_list, weight_similarity)
    weighted_residual_magnitudes = np.multiply(residual_magnitudes_list, weight_residual)
    combined_score = np.add(weighted_cos_similarity, weighted_residual_magnitudes)
    # combined_score = weight_similarity * cos_similarity_list + weight_residual * residual_magnitudes_list

    # anomaly_scores_list = combined_score.squeeze().tolist()
    if combined_score.ndim > 0:
        anomaly_scores_list = combined_score.squeeze().tolist()
    else:
        anomaly_scores_list = [combined_score.tolist()]

    threshold = np.percentile(anomaly_scores_list, percentile)

    anomalous_patches_index = [i for i, score in enumerate(anomaly_scores_list) if score > threshold]

    return anomalous_patches_index
