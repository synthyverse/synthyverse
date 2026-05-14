from pathlib import Path
import pickle


def save_generator_state(path, state, filename="generator.pkl"):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with (path / filename).open("wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_generator_state(path, filename="generator.pkl"):
    path = Path(path)
    with (path / filename).open("rb") as f:
        return pickle.load(f)


def load_generator_state_or_default(path, default=None, filename="generator.pkl"):
    try:
        return load_generator_state(path, filename)
    except FileNotFoundError:
        return {} if default is None else default


def restore_generator(cls, state):
    generator = cls.__new__(cls)
    generator.__dict__.update(state)
    post_load = getattr(generator, "__post_load__", None)
    if callable(post_load):
        post_load()
    return generator
