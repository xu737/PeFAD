import torch
import torch.nn.functional as F
import numpy as np

# Returns the sequence xb as a patch
def create_patch(xb, yb, patch_len, stride): 
    """
    xb: [bs x seq_len x n_vars]
    """
    xb = torch.tensor(xb)
    yb = torch.tensor(yb)

    seq_len = xb.shape[0]
    num_patch = (max(seq_len, patch_len)-patch_len) // stride + 1  #Extract the number of sliding windows from the input sequence
    tgt_len = patch_len  + stride*(num_patch-1)
    s_begin = seq_len - tgt_len
        
    xb = xb[s_begin:, :]                                                    # xb: [bs x tgt_len x nvars]
    yb = yb[s_begin:]  # yb: [bs x tgt_len x n_labels]
    
    xb = xb.unfold(dimension=0, size=patch_len, step=stride)                 # xb: [bs x num_patch x n_vars x patch_len]
    yb = yb.unfold(dimension=0, size=patch_len, step=stride)  # yb: [bs x num_patch x n_labels x patch_len]

    return xb, yb, num_patch


# ADMS
def random_masking_with_anomalies(xb, mask_ratio, anomaly_indices, mask_factor=1.25):
    xb = xb.unsqueeze(0)
    bs, L, nvars, D = xb.shape
    x = xb.clone()

    len_keep = int(L * (1 - mask_ratio))

    # Generate noise with an increased range to potentially mask more patches
    noise = torch.rand(bs, L, nvars, device=xb.device)
    # print("noise:",noise.shape)
    # Modify mask based on anomaly indices
    for i in anomaly_indices:
        # Increase the mask factor for anomalies, making it more likely to be masked
        noise[:, i, :] *= mask_factor

    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)


    ids_keep = ids_shuffle[:, :len_keep, :]
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D))

    x_removed = torch.zeros(bs, L-len_keep, nvars, D, device=xb.device)
    x_ = torch.cat([x_kept, x_removed], dim=1)

    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,1,D))

    mask = torch.ones([bs, L, nvars], device=x.device)
    mask[:, :len_keep, :] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    x_masked = x_masked.squeeze(0)
    x_kept = x_kept.squeeze(0)
    mask = mask.squeeze(0)
    ids_restore = ids_restore.squeeze(0)

    return x_masked, x_kept, mask, ids_restore



def random_masking(xb, mask_ratio): 
    # xb: [bs x num_patch x n_vars x patch_len]
    xb = xb.unsqueeze(0)  # xb: [1 x num_patch x n_vars x patch_len]
    # xb = xb[np.newaxis, ...]  # or xb.reshape(1, num_patch, n_vars, patch_len)
    bs, L, nvars, D = xb.shape   
    x = xb.clone()
    
    len_keep = int(L * (1 - mask_ratio)) #Number of sliding windows that are not masked
        
    noise = torch.rand(bs, L, nvars,device=xb.device)  # noise in [0, 1], bs x L x nvars
        
    # sort noise for each sample 
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)                                     # ids_restore: [bs x L x nvars]

    # keep the first subset # 
    ids_keep = ids_shuffle[:, :len_keep, :]                                              # ids_keep: [bs x len_keep x nvars]         
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D))     # x_kept: [bs x len_keep x nvars  x patch_len]
    # y_kept = torch.gather(yb, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D))

    # removed x 
    x_removed = torch.zeros(bs, L-len_keep, nvars, D, device=xb.device)                 # x_removed: [bs x (L-len_keep) x nvars x patch_len]
    # y_removed = torch.zeros(bs, L - len_keep, nvars, D, device=xb.device)
    x_ = torch.cat([x_kept, x_removed], dim=1)                                          # x_: [bs x L x nvars x patch_len]
    # y_ = torch.cat([y_kept, y_removed], dim=1)

    # combine the kept part and the removed one 
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,1,D)) # x_masked: [bs x num_patch x nvars x patch_len]
    # y_masked = torch.gather(y_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, L, nvars], device=x.device)                                  # mask: [bs x num_patch x nvars]
    mask[:, :len_keep, :] = 0
    # unshuffle to get the binary mask 
    mask = torch.gather(mask, dim=1, index=ids_restore)                                  # [bs x num_patch x nvars]
    
    # Remove the batch dimension before returning
    x_masked = x_masked.squeeze(0)
    x_kept = x_kept.squeeze(0)
    mask = mask.squeeze(0)
    ids_restore = ids_restore.squeeze(0)
    
    return x_masked, x_kept, mask, ids_restore