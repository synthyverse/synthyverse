# Copyright (c) 2021, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""

-----------------------------------------
One-class representations
-----------------------------------------

"""

from __future__ import absolute_import, division, print_function

import numpy as np
import sys
from tqdm import tqdm
import logging
import torch
import torch.nn as nn

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


from torch.autograd import Variable


ACTIVATION_DICT = {
    "ReLU": torch.nn.ReLU(),
    "Hardtanh": torch.nn.Hardtanh(),
    "ReLU6": torch.nn.ReLU6(),
    "Sigmoid": torch.nn.Sigmoid(),
    "Tanh": torch.nn.Tanh(),
    "ELU": torch.nn.ELU(),
    "CELU": torch.nn.CELU(),
    "SELU": torch.nn.SELU(),
    "GLU": torch.nn.GLU(),
    "LeakyReLU": torch.nn.LeakyReLU(),
    "LogSigmoid": torch.nn.LogSigmoid(),
    "Softplus": torch.nn.Softplus(),
}


def build_network(network_name, params):

    if network_name == "feedforward":

        net = feedforward_network(params)

    return net


def feedforward_network(params):
    """Architecture for a Feedforward Neural Network

    Args:

        ::params::

        ::params["input_dim"]::
        ::params[""rep_dim""]::
        ::params["num_hidden"]::
        ::params["activation"]::
        ::params["num_layers"]::
        ::params["dropout_prob"]::
        ::params["dropout_active"]::
        ::params["LossFn"]::

    Returns:

        ::_architecture::

    """

    modules = []

    if params["dropout_active"]:

        modules.append(torch.nn.Dropout(p=params["dropout_prob"]))

    # Input layer

    modules.append(
        torch.nn.Linear(params["input_dim"], params["num_hidden"], bias=False)
    )
    modules.append(ACTIVATION_DICT[params["activation"]])

    # Intermediate layers

    for u in range(params["num_layers"] - 1):

        if params["dropout_active"]:

            modules.append(torch.nn.Dropout(p=params["dropout_prob"]))

        modules.append(
            torch.nn.Linear(params["num_hidden"], params["num_hidden"], bias=False)
        )
        modules.append(ACTIVATION_DICT[params["activation"]])

    # Output layer

    modules.append(torch.nn.Linear(params["num_hidden"], params["rep_dim"], bias=False))

    _architecture = nn.Sequential(*modules)

    return _architecture


def OneClassLoss(outputs, c):

    dist = torch.sum((outputs - c) ** 2, dim=1)
    loss = torch.mean(dist)

    return loss


def SoftBoundaryLoss(outputs, R, c, nu):

    dist = torch.sum((outputs - c) ** 2, dim=1)
    scores = dist - R**2
    loss = R**2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))

    scores = dist
    loss = (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))

    return loss


LossFns = dict({"OneClass": OneClassLoss, "SoftBoundary": SoftBoundaryLoss})

# Base network
# ---------------------


class BaseNet(nn.Module):
    """Base class for all neural networks."""

    def __init__(self):

        super().__init__()

        self.logger = logging.getLogger(self.__class__.__name__)
        self.rep_dim = None  # representation dimensionality, i.e. dim of the last layer

    def forward(self, *input):
        """Forward pass logic

        :return: Network output
        """
        raise NotImplementedError

    def summary(self):
        """Network summary."""

        net_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in net_parameters])

        self.logger.info("Trainable parameters: {}".format(params))
        self.logger.info(self)


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""

    return np.quantile(np.sqrt(dist.clone().data.float().numpy()), 1 - nu)


class OneClassLayer(BaseNet):

    def __init__(self, params=None, hyperparams=None):

        super().__init__()

        # set all representation parameters - remove these lines

        self.rep_dim = params["rep_dim"]
        self.input_dim = params["input_dim"]
        self.num_layers = params["num_layers"]
        self.num_hidden = params["num_hidden"]
        self.activation = params["activation"]
        self.dropout_prob = params["dropout_prob"]
        self.dropout_active = params["dropout_active"]
        self.loss_type = params["LossFn"]
        self.train_prop = params["train_prop"]
        self.learningRate = params["lr"]
        self.epochs = params["epochs"]
        self.warm_up_epochs = params["warm_up_epochs"]
        self.weight_decay = params["weight_decay"]
        if torch.cuda.is_available():
            self.device = torch.device("cuda")  # Make this an option
        else:
            self.device = torch.device("cpu")
        # set up the network

        self.model = build_network(network_name="feedforward", params=params).to(
            self.device
        )

        # create the loss function

        self.c = hyperparams["center"].to(self.device)
        self.R = hyperparams["Radius"]
        self.nu = hyperparams["nu"]

        self.loss_fn = LossFns[self.loss_type]

    def forward(self, x):

        x = self.model(x)

        return x

    def predict(self, x: np.ndarray):
        with torch.no_grad():
            return (
                self.model(torch.from_numpy(x).float().to(self.device))
                .cpu()
                .detach()
                .numpy()
            )

    def fit(self, x_train, verbosity=True):

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learningRate,
            weight_decay=self.weight_decay,
        )
        self.X = torch.tensor(x_train.reshape((-1, self.input_dim))).float()

        if self.train_prop != 1:
            x_train, x_val = (
                x_train[: int(self.train_prop * len(x_train))],
                x_train[int(self.train_prop * len(x_train)) :],
            )
            inputs_val = Variable(torch.from_numpy(x_val).to(self.device)).float()

        self.losses = []
        self.loss_vals = []
        pbar = tqdm(range(self.epochs))

        for epoch in pbar:

            # Converting inputs and labels to Variable

            inputs = Variable(torch.from_numpy(x_train)).to(self.device).float()

            self.model.zero_grad()

            self.optimizer.zero_grad()

            # get output from the model, given the inputs
            outputs = self.model(inputs)

            # get loss for the predicted output

            if self.loss_type == "SoftBoundary":

                self.loss = self.loss_fn(
                    outputs=outputs, R=self.R, c=self.c, nu=self.nu
                )

            elif self.loss_type == "OneClass":

                self.loss = self.loss_fn(outputs=outputs, c=self.c)

            # self.c    = torch.mean(torch.tensor(outputs).float(), dim=0)

            # get gradients w.r.t to parameters
            self.loss.backward(retain_graph=True)
            self.losses.append(self.loss.detach().cpu().numpy())

            # update parameters
            self.optimizer.step()

            if (epoch >= self.warm_up_epochs) and (self.loss_type == "SoftBoundary"):

                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                # self.R = torch.tensor(get_radius(dist, self.nu))

            if self.train_prop != 1.0:
                with torch.no_grad():

                    # get output from the model, given the inputs
                    outputs = self.model(inputs_val)

                    # get loss for the predicted output

                    if self.loss_type == "SoftBoundary":

                        loss_val = self.loss_fn(
                            outputs=outputs, R=self.R, c=self.c, nu=self.nu
                        )

                    elif self.loss_type == "OneClass":

                        loss_val = (
                            self.loss_fn(outputs=outputs, c=self.c).detach.cpu().numpy()
                        )

                    self.loss_vals.append(loss_val)

            if verbosity:
                if self.train_prop == 1:
                    pbar.set_description(
                        "epoch {}, loss {}".format(epoch, self.loss.item())
                    )
                else:
                    pbar.set_description(
                        "epoch {:4}, train loss {:.4e}, val loss {:.4e}".format(
                            epoch, self.loss.item(), loss_val
                        )
                    )
