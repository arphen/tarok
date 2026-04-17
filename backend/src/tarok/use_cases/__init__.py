def __getattr__(name):
    if name == "GameLoop":
        from tarok.adapters.ai.rust_game_loop import RustGameLoop
        return RustGameLoop
    if name == "NullObserver":
        from tarok.adapters.ai.rust_game_loop import NullObserver
        return NullObserver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["GameLoop", "NullObserver"]
