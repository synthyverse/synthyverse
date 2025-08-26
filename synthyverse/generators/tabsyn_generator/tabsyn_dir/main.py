import torch

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import warnings
import time

from tqdm import tqdm
from .model import MLPDiffusion, Model

warnings.filterwarnings("ignore")


def train_tabsyn(train_z, diffusion_params, device=None):

    train_z = torch.tensor(train_z).float()
    train_z = train_z[:, 1:, :]
    B, num_tokens, token_dim = train_z.size()
    in_dim = num_tokens * token_dim
    train_z = train_z.view(B, in_dim)

    in_dim = train_z.shape[1]

    mean, std = train_z.mean(0), train_z.std(0)

    train_z = (train_z - mean) / 2
    train_data = train_z

    batch_size = diffusion_params["BATCH_SIZE"]
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 4
    )

    denoise_fn = MLPDiffusion(in_dim, diffusion_params["DIM_T"]).to(device)

    model = Model(denoise_fn=denoise_fn, hid_dim=train_z.shape[1]).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=diffusion_params["LR"],
        weight_decay=diffusion_params["WD"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.9,
        patience=20,
    )

    model.train()

    best_loss = float("inf")
    patience = 0
    best_model = None
    for epoch in range(diffusion_params["NUM_EPOCHS"]):

        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{diffusion_params['NUM_EPOCHS']}")

        batch_loss = 0.0
        len_input = 0
        for batch in pbar:
            inputs = batch.float().to(device)
            loss = model(inputs)

            loss = loss.mean()

            batch_loss += loss.item() * len(inputs)
            len_input += len(inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"Loss": loss.item()})

        curr_loss = batch_loss / len_input
        scheduler.step(curr_loss)

        if curr_loss < best_loss:
            best_loss = curr_loss
            patience = 0
            best_model = model.state_dict()
        else:
            patience += 1
            if patience == diffusion_params["PATIENCE"]:
                print("Early stopping")
                break

    model.load_state_dict(best_model)

    return model, train_z.shape, train_z.mean(0), token_dim
