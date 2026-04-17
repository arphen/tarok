def __getattr__(name):
    if name == "GameLoop":
        from tarok.use_cases.game_loop import RustGameLoop
        return RustGameLoop
    if name == "NullObserver":
        from tarok.use_cases.game_loop import NullObserver
        return NullObserver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["GameLoop", "NullObserver"]
