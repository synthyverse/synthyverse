def __getattr__(name: str):
    if name == "TabularSynthesisBenchmark":
        from .synthesis import TabularSynthesisBenchmark

        return TabularSynthesisBenchmark
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["TabularSynthesisBenchmark"]
