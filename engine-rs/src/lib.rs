pub mod card;
pub mod encoding;
pub mod expert_games_v5;
pub mod game_state;
pub mod legal_moves;
pub mod scoring;
pub mod stockskis_v5;
pub mod trick_eval;
pub mod warmup;

mod py_bindings;

use pyo3::prelude::*;

/// The Python module — `import tarok_engine`.
#[pymodule]
fn tarok_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    py_bindings::register(m)?;
    Ok(())
}
