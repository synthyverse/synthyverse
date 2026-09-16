import ast
from copy import deepcopy
from math import ceil, sqrt
from operator import add, mul, sub, truediv
from pathlib import Path
from typing import Optional

GENERATOR_CLASSES = {
    "arf": ("arf_generator/arf.py", "ARFGenerator"),
    "tabsyn": ("tabsyn_generator/tabsyn.py", "TabSynGenerator"),
    "cdtd": ("cdtd_generator/cdtd.py", "CDTDGenerator"),
    "tabddpm": ("tabddpm_generator/tabddpm.py", "TabDDPMGenerator"),
    "tabdiff": ("tabdiff_generator/tabdiff.py", "TabDiffGenerator"),
    "tabcascade": ("tabcascade_generator/tabcascade.py", "TabCascadeGenerator"),
    "univariate": ("univariate_generator/univariate.py", "UnivariateGenerator"),
    "smote": ("smote_generator/smote.py", "SMOTEGenerator"),
    "synthpop": ("synthpop_generator/synthpop.py", "SynthpopGenerator"),
    "xgbddpm": ("xgbddpm_generator/xgbddpm.py", "XGBDDPMGenerator"),
    "xgbdiffusion": ("xgbdiffusion_generator/xgbdiffusion.py", "XGBDiffusionGenerator"),
    "forestdiffusion": (
        "forestdiffusion_generator/fd.py",
        "ForestDiffusionGenerator",
    ),
    "ctgan": ("ctgan_generator/ct_gan.py", "CTGANGenerator"),
    "tvae": ("tvae_generator/tvae.py", "TVAEGenerator"),
}

DEFAULT_CONFIG = {
    "batch_size": 4096,
    "training_steps": 50_000,
    "val_size": 0.2,
    "val_steps": 5_000,
    "patience": 3,
    "max_validation_rows": 30_000,
}

GENERATOR_DEFAULT_CONFIGS = {
    "tabsyn": {
        "vae_training_steps": round(50_000 / 3.5),
        "training_steps": round(2.5 * 50_000 / 3.5),
    },
}

NETWORK_SIZE_CONFIGS = {
    "tabsyn": {
        "small": {
            "embedding_dim": 256,
            "mlp_dim": 256,
            "mlp_layers": 3,
        },
        "medium": {
            "embedding_dim": 512,
            "mlp_dim": 512,
            "mlp_layers": 4,
        },
        "large": {
            "embedding_dim": 1024,
            "mlp_dim": 1024,
            "mlp_layers": 4,
        },
    },
    "cdtd": {
        "small": {
            "embedding_dim": 256,
            "mlp_n_layers": 3,
            "mlp_n_units": 256,
        },
        "medium": {
            "embedding_dim": 512,
            "mlp_n_layers": 4,
            "mlp_n_units": 512,
        },
        "large": {
            "embedding_dim": 1024,
            "mlp_n_layers": 4,
            "mlp_n_units": 1024,
        },
    },
    "tabddpm": {
        "small": {
            "embedding_dim": 256,
            "model_params": {
                "n_layers_hidden": 3,
                "n_units_hidden": 256,
                "dropout": 0.0,
            },
        },
        "medium": {
            "embedding_dim": 512,
            "model_params": {
                "n_layers_hidden": 4,
                "n_units_hidden": 512,
                "dropout": 0.0,
            },
        },
        "large": {
            "embedding_dim": 1024,
            "model_params": {
                "n_layers_hidden": 4,
                "n_units_hidden": 1024,
                "dropout": 0.0,
            },
        },
    },
    "tabdiff": {
        "small": {
            "num_layers": 0,
            "embedding_dim": 256,
            "mlp_dim": 256,
            "mlp_layers": 3,
        },
        "medium": {
            "num_layers": 0,
            "embedding_dim": 512,
            "mlp_dim": 512,
            "mlp_layers": 4,
        },
        "large": {
            "num_layers": 0,
            "embedding_dim": 1024,
            "mlp_dim": 1024,
            "mlp_layers": 4,
        },
    },
    "tabcascade": {
        "small": {
            "embedding_dim": 256,
            "lowres_mlp_n_layers": 3,
            "lowres_mlp_n_units": 256,
            "highres_mlp_n_layers": 3,
            "highres_mlp_n_units": 256,
        },
        "medium": {
            "embedding_dim": 512,
            "lowres_mlp_n_layers": 4,
            "lowres_mlp_n_units": 512,
            "highres_mlp_n_layers": 4,
            "highres_mlp_n_units": 512,
        },
        "large": {
            "embedding_dim": 1024,
            "lowres_mlp_n_layers": 4,
            "lowres_mlp_n_units": 1024,
            "highres_mlp_n_layers": 4,
            "highres_mlp_n_units": 1024,
        },
    },
    "ctgan": {
        "small": {
            "embedding_dim": 256,
            "generator_dim": [256] * 3,
            "discriminator_dim": [256] * 3,
        },
        "medium": {
            "embedding_dim": 512,
            "generator_dim": [512] * 4,
            "discriminator_dim": [512] * 4,
        },
        "large": {
            "embedding_dim": 1024,
            "generator_dim": [1024] * 4,
            "discriminator_dim": [1024] * 4,
        },
    },
    "tvae": {
        "small": {
            "embedding_dim": 256,
            "compress_dims": [256] * 3,
            "decompress_dims": [256] * 3,
        },
        "medium": {
            "embedding_dim": 512,
            "compress_dims": [512] * 4,
            "decompress_dims": [512] * 4,
        },
        "large": {
            "embedding_dim": 1024,
            "compress_dims": [1024] * 4,
            "decompress_dims": [1024] * 4,
        },
    },
}


def _update_config(config, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            _update_config(config[key], value)
        else:
            config[key] = deepcopy(value)


def _update_batch_size(config, n):
    training_rows = n
    if config.get("val_size", 0) > 0 and config.get("val_steps", 0) > 0 and n > 1:
        val_rows = min(ceil(n * config["val_size"]), n - 1)
        if config.get("max_validation_rows") is not None:
            val_rows = min(val_rows, config["max_validation_rows"])
        training_rows -= val_rows

    batch_size = config["batch_size"]
    max_batch_size = max(1, training_rows // 5)
    if batch_size <= max_batch_size:
        return

    config["batch_size"] = max_batch_size
    lr_scale = sqrt(max_batch_size / batch_size)
    for key, value in config.items():
        if (key == "lr" or key.endswith("_lr")) and isinstance(value, (int, float)):
            config[key] = value * lr_scale


def _literal_default(node):
    try:
        return ast.literal_eval(node)
    except ValueError:
        pass

    if isinstance(node, ast.BinOp):
        operators = {
            ast.Add: add,
            ast.Sub: sub,
            ast.Mult: mul,
            ast.Div: truediv,
        }
        operator = operators.get(type(node.op))
        if operator is not None:
            return operator(
                _literal_default(node.left),
                _literal_default(node.right),
            )
    raise ValueError("Unsupported default value in generator constructor")


def _generator_params(name):
    if name not in GENERATOR_CLASSES:
        raise ValueError(f"Generator {name} not found")

    source_path, class_name = GENERATOR_CLASSES[name]
    source = Path(__file__).resolve().parent / source_path
    tree = ast.parse(source.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            init_node = next(
                item
                for item in node.body
                if isinstance(item, ast.FunctionDef) and item.name == "__init__"
            )
            args = init_node.args.args
            defaults = [None] * (len(args) - len(init_node.args.defaults))
            defaults += init_node.args.defaults
            params = [arg.arg for arg in args if arg.arg != "self"]
            config = {
                arg.arg: deepcopy(_literal_default(default))
                for arg, default in zip(args, defaults)
                if arg.arg != "self" and default is not None
            }
            return params, config

    raise ValueError(f"Generator {name} not found")


def get_config(generator: str, n: int, size: Optional[str] = "medium"):
    name = generator.lower()
    params, config = _generator_params(name)

    if size is not None:
        size = size.lower()
        if size not in {"small", "medium", "large"}:
            raise ValueError(f"Config size {size} not found")
        _update_config(config, NETWORK_SIZE_CONFIGS.get(name, {}).get(size, {}))

    _update_config(
        config,
        {key: value for key, value in DEFAULT_CONFIG.items() if key in params},
    )
    _update_config(
        config,
        {
            key: value
            for key, value in GENERATOR_DEFAULT_CONFIGS.get(name, {}).items()
            if key in params
        },
    )

    if "batch_size" in config:
        _update_batch_size(config, n)
    return config
