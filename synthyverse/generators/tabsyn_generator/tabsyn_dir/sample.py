import warnings
import torch
from .latent_utils import recover_data, split_num_cat_target
from .diffusion_utils import sample
import numpy as np

warnings.filterwarnings("ignore")


def sample_tabsyn(
    n,
    num_inverse,
    cat_inverse,
    info,
    diffusion_model,
    pre_decoder,
    train_z_shape: tuple,
    train_z_mean,
    batch_size,
    device,
):

    in_dim = train_z_shape[1]
    mean = train_z_mean
    model = diffusion_model.to(device)
    num_samples = n
    sample_dim = in_dim

    # batch-wise inference
    syn_num_ = []
    syn_cat_ = []
    syn_target_ = []
    with torch.no_grad():
        for _ in range(0, num_samples, batch_size):
            # diffusion inference
            x_next = sample(model.denoise_fn_D, batch_size, sample_dim, device)
            x_next = x_next * 2 + mean.to(device)
            # x_next = x_next.float().cpu().numpy()
            # VAE inference
            syn_num, syn_cat, syn_target = split_num_cat_target(
                x_next, pre_decoder, info, num_inverse, cat_inverse, device
            )
            syn_num_.append(syn_num)
            syn_cat_.append(syn_cat)
            syn_target_.append(syn_target)

    syn_num = np.concatenate(syn_num_)
    syn_cat = np.concatenate(syn_cat_)
    syn_target = np.concatenate(syn_target_)

    syn_df = recover_data(syn_num, syn_cat, syn_target, info)

    idx_name_mapping = info["idx_name_mapping"]

    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}
    syn_df = syn_df.rename(columns=idx_name_mapping)

    return syn_df
