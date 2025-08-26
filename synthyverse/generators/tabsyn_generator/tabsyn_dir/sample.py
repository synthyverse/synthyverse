import warnings

from .latent_utils import recover_data, split_num_cat_target
from .diffusion_utils import sample


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
    device,
):

    in_dim = train_z_shape[1]
    mean = train_z_mean
    model = diffusion_model.to(device)
    num_samples = n
    sample_dim = in_dim

    x_next = sample(model.denoise_fn_D, num_samples, sample_dim, device)
    x_next = x_next * 2 + mean.to(device)

    syn_data = x_next.float().cpu().numpy()
    syn_num, syn_cat, syn_target = split_num_cat_target(
        syn_data, pre_decoder, info, num_inverse, cat_inverse, device
    )

    syn_df = recover_data(syn_num, syn_cat, syn_target, info)

    idx_name_mapping = info["idx_name_mapping"]

    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}
    syn_df = syn_df.rename(columns=idx_name_mapping)

    return syn_df
