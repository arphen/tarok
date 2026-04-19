pub mod arena;
pub mod card;
// pub mod double_dummy;
pub mod encoding;
pub mod expert_games_v5;
pub mod game_state;
pub mod legal_moves;
pub mod lustrek;
// pub mod pimc;
pub mod player;
pub mod player_bot;
pub mod player_nn;
pub mod scoring;
pub mod self_play;
pub mod stockskis_m6;
pub mod stockskis_pozrl;
pub mod stockskis_v1;
pub mod stockskis_v3;
pub mod stockskis_v5;
pub mod stockskis_v6;
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
