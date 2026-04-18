"""PPO contract validation helpers."""

from __future__ import annotations

from tarok_model.network import TarokNetV4


def validate_v4_contract_indices_with_rust() -> None:
    """Fail fast if Python and Rust contract indices diverge."""
    try:
        import tarok_engine as te
    except Exception:
        return

    expected = {
        "CONTRACT_KLOP": TarokNetV4._KLOP_IDX,
        "CONTRACT_THREE": TarokNetV4._THREE_IDX,
        "CONTRACT_TWO": TarokNetV4._TWO_IDX,
        "CONTRACT_ONE": TarokNetV4._ONE_IDX,
        "CONTRACT_SOLO_THREE": TarokNetV4._SOLO_THREE_IDX,
        "CONTRACT_SOLO_TWO": TarokNetV4._SOLO_TWO_IDX,
        "CONTRACT_SOLO_ONE": TarokNetV4._SOLO_ONE_IDX,
        "CONTRACT_SOLO": TarokNetV4._SOLO_IDX,
        "CONTRACT_BERAC": TarokNetV4._BERAC_IDX,
        "CONTRACT_BARVNI_VALAT": TarokNetV4._BARVNI_VALAT_IDX,
    }

    mismatches: list[str] = []
    for name, py_value in expected.items():
        rust_value = getattr(te, name, None)
        if rust_value is None or int(rust_value) != int(py_value):
            mismatches.append(f"{name}: rust={rust_value} python={py_value}")

    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise RuntimeError(
            "Rust/Python contract index mismatch detected. "
            f"Refusing to train with ambiguous v4 mode routing: {mismatch_text}"
        )