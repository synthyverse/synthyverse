import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
from tqdm import tqdm


from .model import Model_VAE, Encoder_model, Decoder_model
from ..utils_train import preprocess, TabularDataset

warnings.filterwarnings("ignore")


def compute_loss(X_num, X_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z):
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss = (X_num - Recon_X_num).pow(2).mean()
    ce_loss = 0
    acc = 0
    total_num = 0

    for idx, x_cat in enumerate(Recon_X_cat):
        if x_cat is not None:
            ce_loss += ce_loss_fn(x_cat, X_cat[:, idx])
            x_hat = x_cat.argmax(dim=-1)
        acc += (x_hat == X_cat[:, idx]).float().sum()
        total_num += x_hat.shape[0]

    ce_loss /= idx + 1
    acc /= total_num
    # loss = mse_loss + ce_loss

    temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()

    loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
    return mse_loss, ce_loss, loss_kld, acc


def train_vae(
    X_num_train,
    X_cat_train,
    y_train,
    X_num_test,
    X_cat_test,
    y_test,
    info,
    device,
    vae_params,
):

    X_num, X_cat, categories, d_numerical = preprocess(
        X_num_train,
        X_cat_train,
        y_train,
        X_num_test,
        X_cat_test,
        y_test,
        info,
        task_type=info["task_type"],
    )

    X_train_num, _ = X_num
    X_train_cat, _ = X_cat

    X_train_num, X_test_num = X_num
    X_train_cat, X_test_cat = X_cat

    X_train_num, X_test_num = (
        torch.from_numpy(X_train_num).float(),
        torch.from_numpy(X_test_num).float(),
    )
    X_train_cat, X_test_cat = torch.from_numpy(X_train_cat), torch.from_numpy(
        X_test_cat
    )

    train_data = TabularDataset(X_train_num.float(), X_train_cat)

    X_test_num = X_test_num.float().to(device)
    X_test_cat = X_test_cat.to(device)

    batch_size = vae_params["BATCH_SIZE"]
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 4
    )

    model = Model_VAE(
        vae_params["NUM_LAYERS"],
        d_numerical,
        categories,
        vae_params["D_TOKEN"],
        n_head=vae_params["N_HEAD"],
        factor=vae_params["FACTOR"],
        bias=True,
    )
    model = model.to(device)

    pre_encoder = Encoder_model(
        vae_params["NUM_LAYERS"],
        d_numerical,
        categories,
        vae_params["D_TOKEN"],
        n_head=vae_params["N_HEAD"],
        factor=vae_params["FACTOR"],
    ).to(device)
    pre_decoder = Decoder_model(
        vae_params["NUM_LAYERS"],
        d_numerical,
        categories,
        vae_params["D_TOKEN"],
        n_head=vae_params["N_HEAD"],
        factor=vae_params["FACTOR"],
    ).to(device)

    pre_encoder.eval()
    pre_decoder.eval()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=vae_params["LR"], weight_decay=vae_params["WD"]
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.95,
        patience=10,
    )

    best_train_loss = float("inf")

    current_lr = optimizer.param_groups[0]["lr"]
    patience = 0

    beta = vae_params["MAX_BETA"]
    for epoch in range(vae_params["NUM_EPOCHS"]):
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{vae_params['NUM_EPOCHS']}")

        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0
        curr_loss_kl = 0.0

        curr_count = 0

        for batch_num, batch_cat in pbar:
            model.train()
            optimizer.zero_grad()

            batch_num = batch_num.to(device)
            batch_cat = batch_cat.to(device)

            Recon_X_num, Recon_X_cat, mu_z, std_z = model(batch_num, batch_cat)

            loss_mse, loss_ce, loss_kld, train_acc = compute_loss(
                batch_num, batch_cat, Recon_X_num, Recon_X_cat, mu_z, std_z
            )

            loss = loss_mse + loss_ce + beta * loss_kld
            loss.backward()
            optimizer.step()

            batch_length = batch_num.shape[0]
            curr_count += batch_length
            curr_loss_multi += loss_ce.item() * batch_length
            curr_loss_gauss += loss_mse.item() * batch_length
            curr_loss_kl += loss_kld.item() * batch_length

        num_loss = curr_loss_gauss / curr_count
        cat_loss = curr_loss_multi / curr_count
        kl_loss = curr_loss_kl / curr_count

        """
            Evaluation
        """
        model.eval()
        with torch.no_grad():
            Recon_X_num, Recon_X_cat, mu_z, std_z = model(X_test_num, X_test_cat)

            val_mse_loss, val_ce_loss, val_kl_loss, val_acc = compute_loss(
                X_test_num, X_test_cat, Recon_X_num, Recon_X_cat, mu_z, std_z
            )
            val_loss = val_mse_loss.item() * 0 + val_ce_loss.item()

            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]["lr"]

            if new_lr != current_lr:
                current_lr = new_lr
                print(f"Learning rate updated: {current_lr}")

            train_loss = val_loss
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                patience = 0
                best_model = model.state_dict()
            else:
                patience += 1
                if patience == 10:
                    if beta > vae_params["MIN_BETA"]:
                        beta = beta * vae_params["LAMBDA"]
        # print('epoch: {}, beta = {:.6f}, Train MSE: {:.6f}, Train CE:{:.6f}, Train KL:{:.6f}, Train ACC:{:6f}'.format(epoch, beta, num_loss, cat_loss, kl_loss, train_acc.item()))
        print(
            "epoch: {}, beta = {:.6f}, Train MSE: {:.6f}, Train CE:{:.6f}, Train KL:{:.6f}, Val MSE:{:.6f}, Val CE:{:.6f}, Train ACC:{:6f}, Val ACC:{:6f}".format(
                epoch,
                beta,
                num_loss,
                cat_loss,
                kl_loss,
                val_mse_loss.item(),
                val_ce_loss.item(),
                train_acc.item(),
                val_acc.item(),
            )
        )
    model.load_state_dict(best_model)
    # Saving latent embeddings
    with torch.no_grad():
        pre_encoder.load_weights(model)
        pre_decoder.load_weights(model)

        X_train_num = X_train_num.to(device)
        X_train_cat = X_train_cat.to(device)

        train_z = pre_encoder(X_train_num, X_train_cat).detach().cpu().numpy()

    return train_z, pre_decoder
