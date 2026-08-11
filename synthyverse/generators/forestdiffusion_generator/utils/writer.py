# Third-party notice: based on MIT-licensed upstream code.
# See THIRD_PARTY_NOTICES.md for attribution and modification details.
import os
import json
import pickle

import numpy as np


class Writer:
    """Simple file I/O utility for saving and loading model artifacts and data."""

    def __init__(self, logdir):
        """Initialize Writer with a log directory.

        Args:
            logdir: Directory path for saving outputs. Will be created if it doesn't exist.
        """
        os.makedirs(logdir, exist_ok=True)
        self.logdir = logdir

    def write_json(self, tag, data):
        """Write data to a JSON file.

        Args:
            tag: Filename prefix (file saved as {tag}.json)
            data: Dictionary or serializable object to save
        """
        text = json.dumps(data, indent=4)
        json_path = os.path.join(self.logdir, f"{tag}.json")
        with open(json_path, "w") as f:
            f.write(text)

    def load_json(self, tag, load_dir=None):
        """Load data from a JSON file.

        Args:
            tag: Filename prefix (reads {tag}.json)
            load_dir: Optional directory to load from. Defaults to self.logdir

        Returns:
            Loaded JSON data (dict)
        """
        if load_dir:
            json_path = os.path.join(load_dir, f"{tag}.json")
        else:
            json_path = os.path.join(self.logdir, f"{tag}.json")
        with open(json_path, "r") as f:
            data = json.load(f)
        return data

    def write_pickle(self, tag, obj):
        """Pickle and save an object.

        Args:
            tag: Filename prefix (file saved as {tag}.pkl)
            obj: Any picklable Python object (e.g., ForestModel instance)
        """
        path = os.path.join(self.logdir, f"{tag}.pkl")
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    def load_pickle(self, tag, load_dir=None):
        """Load a pickled object.

        Args:
            tag: Filename prefix (reads {tag}.pkl)
            load_dir: Optional directory to load from. Defaults to self.logdir

        Returns:
            Unpickled Python object
        """
        if load_dir:
            path = os.path.join(load_dir, f"{tag}.pkl")
        else:
            path = os.path.join(self.logdir, f"{tag}.pkl")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj

    def write_numpy(self, tag, arr):
        """Save a numpy array as .npy file.

        Args:
            tag: Filename prefix (file saved as {tag}.npy)
            arr: Numpy array to save
        """
        path = os.path.join(self.logdir, f"{tag}.npy")
        np.save(path, arr)

    def write_pandas(self, tag, df):
        """Save a pandas DataFrame as CSV.

        Args:
            tag: Filename prefix (file saved as {tag}.csv)
            df: Pandas DataFrame to save
        """
        path = os.path.join(self.logdir, f"{tag}.csv")
        df.to_csv(path)
        print(f"Saved dataframe to {path}")

    def write_csv(self, tag, arr, header=""):
        """Save a numpy array as CSV file.

        Args:
            tag: Filename prefix (file saved as {tag}.csv)
            arr: Numpy array to save
            header: Optional header string for the CSV
        """
        path = os.path.join(self.logdir, f"{tag}.csv")
        np.savetxt(path, arr, delimiter=",", header=header)
        print(f"Saved array to {path}")
