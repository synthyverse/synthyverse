import numpy as np


TYPE_TRANSFORM = {"float", np.float32, "str", str, "int", int}

INFO_PATH = "data/Info"


def get_column_name_mapping(
    data_df, num_col_idx, cat_col_idx, target_col_idx, column_names=None
):

    if not column_names:
        column_names = np.array(data_df.columns.tolist())

    idx_mapping = {}

    curr_num_idx = 0
    curr_cat_idx = len(num_col_idx)
    curr_target_idx = curr_cat_idx + len(cat_col_idx)

    for idx in range(len(column_names)):

        if idx in num_col_idx:
            idx_mapping[int(idx)] = curr_num_idx
            curr_num_idx += 1
        elif idx in cat_col_idx:
            idx_mapping[int(idx)] = curr_cat_idx
            curr_cat_idx += 1
        else:
            idx_mapping[int(idx)] = curr_target_idx
            curr_target_idx += 1

    inverse_idx_mapping = {}
    for k, v in idx_mapping.items():
        inverse_idx_mapping[int(v)] = k

    idx_name_mapping = {}

    for i in range(len(column_names)):
        idx_name_mapping[int(i)] = column_names[i]

    return idx_mapping, inverse_idx_mapping, idx_name_mapping


def train_val_test_split(data_df, cat_columns, num_train=0, num_test=0):
    total_num = data_df.shape[0]
    idx = np.arange(total_num)

    seed = 1234

    while True:
        np.random.seed(seed)
        np.random.shuffle(idx)

        train_idx = idx[:num_train]
        test_idx = idx[-num_test:]

        train_df = data_df.loc[train_idx]
        test_df = data_df.loc[test_idx]

        flag = 0
        for i in cat_columns:
            if len(set(train_df[i])) != len(set(data_df[i])):
                flag = 1
                break

        if flag == 0:
            break
        else:
            seed += 1

    return train_df, test_df, seed


def process_data(train_df, test_df, info):

    column_names = (
        info["column_names"] if info["column_names"] else train_df.columns.tolist()
    )

    num_col_idx = info["num_col_idx"]
    cat_col_idx = info["cat_col_idx"]
    target_col_idx = info["target_col_idx"]

    idx_mapping, inverse_idx_mapping, idx_name_mapping = get_column_name_mapping(
        train_df, num_col_idx, cat_col_idx, target_col_idx, column_names
    )

    num_columns = [column_names[i] for i in num_col_idx]
    cat_columns = [column_names[i] for i in cat_col_idx]
    target_columns = [column_names[i] for i in target_col_idx]

    train_df.columns = range(len(train_df.columns))
    test_df.columns = range(len(test_df.columns))

    col_info = {}

    for col_idx in num_col_idx:
        col_info[col_idx] = {}
        col_info["type"] = "numerical"
        col_info["max"] = float(train_df[col_idx].max())
        col_info["min"] = float(train_df[col_idx].min())

    for col_idx in cat_col_idx:
        col_info[col_idx] = {}
        col_info["type"] = "categorical"
        col_info["categorizes"] = list(set(train_df[col_idx]))

    for col_idx in target_col_idx:
        if info["task_type"] == "regression":
            col_info[col_idx] = {}
            col_info["type"] = "numerical"
            col_info["max"] = float(train_df[col_idx].max())
            col_info["min"] = float(train_df[col_idx].min())
        else:
            col_info[col_idx] = {}
            col_info["type"] = "categorical"
            col_info["categorizes"] = list(set(train_df[col_idx]))

    info["column_info"] = col_info

    train_df.rename(columns=idx_name_mapping, inplace=True)
    test_df.rename(columns=idx_name_mapping, inplace=True)

    for col in num_columns:
        train_df.loc[train_df[col] == "?", col] = np.nan
    for col in cat_columns:
        train_df.loc[train_df[col] == "?", col] = "nan"
    for col in num_columns:
        test_df.loc[test_df[col] == "?", col] = np.nan
    for col in cat_columns:
        test_df.loc[test_df[col] == "?", col] = "nan"

    X_num_train = train_df[num_columns].to_numpy().astype(np.float32)
    X_cat_train = train_df[cat_columns].to_numpy()
    y_train = train_df[target_columns].to_numpy()

    X_num_test = test_df[num_columns].to_numpy().astype(np.float32)
    X_cat_test = test_df[cat_columns].to_numpy()
    y_test = test_df[target_columns].to_numpy()

    train_df[num_columns] = train_df[num_columns].astype(np.float32)
    test_df[num_columns] = test_df[num_columns].astype(np.float32)

    info["column_names"] = column_names
    info["train_num"] = train_df.shape[0]
    info["test_num"] = test_df.shape[0]

    info["idx_mapping"] = idx_mapping
    info["inverse_idx_mapping"] = inverse_idx_mapping
    info["idx_name_mapping"] = idx_name_mapping

    metadata = {"columns": {}}
    task_type = info["task_type"]
    num_col_idx = info["num_col_idx"]
    cat_col_idx = info["cat_col_idx"]
    target_col_idx = info["target_col_idx"]

    for i in num_col_idx:
        metadata["columns"][i] = {}
        metadata["columns"][i]["sdtype"] = "numerical"
        metadata["columns"][i]["computer_representation"] = "Float"

    for i in cat_col_idx:
        metadata["columns"][i] = {}
        metadata["columns"][i]["sdtype"] = "categorical"

    if task_type == "regression":

        for i in target_col_idx:
            metadata["columns"][i] = {}
            metadata["columns"][i]["sdtype"] = "numerical"
            metadata["columns"][i]["computer_representation"] = "Float"

    else:
        for i in target_col_idx:
            metadata["columns"][i] = {}
            metadata["columns"][i]["sdtype"] = "categorical"

    info["metadata"] = metadata

    return info, X_num_train, X_cat_train, y_train, X_num_test, X_cat_test, y_test
