import numpy as np

from . import src
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    def __init__(self, X_num, X_cat):
        self.X_num = X_num
        self.X_cat = X_cat

    def __getitem__(self, index):
        this_num = self.X_num[index]
        this_cat = self.X_cat[index]

        sample = (this_num, this_cat)

        return sample

    def __len__(self):
        return self.X_num.shape[0]


def preprocess(
    X_num_train,
    X_cat_train,
    y_train,
    X_num_test,
    X_cat_test,
    y_test,
    info,
    task_type="binclass",
    inverse=False,
    cat_encoding=None,
):

    T_dict = {}

    T_dict["normalization"] = "quantile"
    T_dict["num_nan_policy"] = "mean"
    T_dict["cat_nan_policy"] = None
    T_dict["cat_min_frequency"] = None
    T_dict["cat_encoding"] = cat_encoding
    T_dict["y_policy"] = "default"

    T = src.Transformations(**T_dict)

    dataset = make_dataset(
        X_cat_train=X_cat_train,
        X_num_train=X_num_train,
        X_cat_test=X_cat_test,
        X_num_test=X_num_test,
        y_train=y_train,
        y_test=y_test,
        info=info,
        T=T,
        task_type=task_type,
        change_val=False,
    )

    if cat_encoding is None:
        X_num = dataset.X_num
        X_cat = dataset.X_cat

        X_train_num, X_test_num = X_num["train"], X_num["test"]
        X_train_cat, X_test_cat = X_cat["train"], X_cat["test"]

        categories = src.get_categories(X_train_cat)
        d_numerical = X_train_num.shape[1]

        X_num = (X_train_num, X_test_num)
        X_cat = (X_train_cat, X_test_cat)

        if inverse:
            num_inverse = dataset.num_transform.inverse_transform
            cat_inverse = dataset.cat_transform.inverse_transform

            return X_num, X_cat, categories, d_numerical, num_inverse, cat_inverse
        else:
            return X_num, X_cat, categories, d_numerical
    else:
        return dataset


def update_ema(target_params, source_params, rate=0.999):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for target, source in zip(target_params, source_params):
        target.detach().mul_(rate).add_(source.detach(), alpha=1 - rate)


def concat_y_to_X(X, y):
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), X], axis=1)


def make_dataset(
    X_cat_train,
    X_num_train,
    X_cat_test,
    X_num_test,
    y_train,
    y_test,
    info,
    T: src.Transformations,
    task_type,
    change_val: bool,
):

    if task_type == "binclass" or task_type == "multiclass":
        X_cat = {
            "train": concat_y_to_X(X_cat_train, y_train),
            "test": concat_y_to_X(X_cat_test, y_test),
        }
        X_num = {"train": X_num_train, "test": X_num_test}
        y = {"train": y_train, "test": y_test}
    else:
        X_cat = {"train": X_cat_train, "test": X_cat_test}
        X_num = {
            "train": concat_y_to_X(X_num_train, y_train),
            "test": concat_y_to_X(X_num_test, y_test),
        }
        y = {"train": y_train, "test": y_test}

    D = src.Dataset(
        X_num,
        X_cat,
        y,
        y_info={},
        task_type=src.TaskType(info["task_type"]),
        n_classes=info.get("n_classes"),
    )

    if change_val:
        D = src.change_val(D)

    return src.transform_dataset(D, T, None)
