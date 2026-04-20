pub mod arena;
pub mod card;
pub mod double_dummy;
pub mod encoding;
pub mod expert_games_v5;
pub mod game_state;
pub mod legal_moves;
pub mod bots;
pub mod pimc;
pub mod player;
pub mod player_bot;
pub mod player_centaur;
pub mod player_nn;
pub mod scoring;
pub mod self_play;
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
