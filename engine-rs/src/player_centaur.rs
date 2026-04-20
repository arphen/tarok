//! Centaur (hybrid NN + PIMC) player for Tarok.
//!
//! Uses the neural network for early/mid-game decisions (bidding, king
//! calling, talon exchange, and tricks before the handoff point), then
//! switches to PIMC for the endgame where the information set is nearly
//! collapsed and search is practically exact.
//!
//! During training the PIMC-decided experiences are tagged with
//! `log_prob = NaN` so the PPO trainer can skip them while still
//! benefiting from the PIMC-optimal terminal rewards flowing back
//! through the NN's earlier decisions.

use crate::game_state::Contract;
use crate::pimc;
use crate::player::*;
use crate::player_nn::NeuralNetPlayer;
use tch::Device;

/// Default trick number at which PIMC takes over card play (0-indexed,
/// so 8 means PIMC plays tricks 9–12, i.e. the last 4 tricks).
pub const DEFAULT_HANDOFF_TRICK: usize = 8;

/// Default number of PIMC worlds to sample per decision.
pub const DEFAULT_PIMC_WORLDS: u32 = 100;

// -----------------------------------------------------------------------
// CentaurBot
// -----------------------------------------------------------------------

pub struct CentaurBot {
    nn: NeuralNetPlayer,
    handoff_trick: usize,
    pimc_worlds: u32,
}

impl CentaurBot {
    pub fn new(
        model_path: &str,
        device: Device,
        explore_rate: f64,
        handoff_trick: usize,
        pimc_worlds: u32,
    ) -> Self {
        CentaurBot {
            nn: NeuralNetPlayer::new(model_path, device, explore_rate),
            handoff_trick,
            pimc_worlds,
        }
    }

    /// Should this card-play decision be handled by PIMC?
    fn use_pimc(&self, ctx: &DecisionContext<'_>) -> bool {
        if ctx.decision_type != DecisionType::CardPlay {
            return false;
        }
        // Klop is individual scoring — PIMC's declarer-vs-opponent
        // maximisation doesn't apply.  Keep the NN for Klop.
        if ctx.gs.contract == Some(Contract::Klop) {
            return false;
        }
        ctx.gs.tricks_played() >= self.handoff_trick
    }
}

impl BatchPlayer for CentaurBot {
    fn batch_decide(&self, contexts: &[DecisionContext<'_>]) -> Vec<DecisionResult> {
        if contexts.is_empty() {
            return Vec::new();
        }

        // Let the NN decide everything first (gives us value estimates
        // for all positions and correct actions for non-PIMC decisions).
        let mut results = self.nn.batch_decide(contexts);

        // Override late-game card plays with PIMC.
        for (i, ctx) in contexts.iter().enumerate() {
            if self.use_pimc(ctx) {
                let card = pimc::pimc_choose_card(ctx.gs, ctx.player, self.pimc_worlds);
                results[i] = DecisionResult {
                    action: card.0 as usize,
                    // NaN sentinel: PPO should skip this experience's policy
                    // loss while still using the game's terminal reward.
                    log_prob: f32::NAN,
                    // Keep the NN's value estimate — useful for GAE on the
                    // preceding (NN-decided) time steps.
                    value: results[i].value,
                };
            }
        }

        results
    }

    fn name(&self) -> &str {
        "centaur"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game_state::*;

    #[test]
    fn pimc_threshold_not_reached() {
        let gs = GameState::new(0);
        // tricks_played() == 0, handoff_trick == 8 → NN should handle this
        let ctx = DecisionContext {
            gs: &gs,
            player: 0,
            decision_type: DecisionType::CardPlay,
            legal_mask: vec![],
            state_encoding: vec![],
        };
        // Manually test the predicate (can't construct a full CentaurBot
        // without a real TorchScript model in unit tests).
        let handoff = 8usize;
        let should_pimc = ctx.decision_type == DecisionType::CardPlay
            && ctx.gs.contract != Some(Contract::Klop)
            && ctx.gs.tricks_played() >= handoff;
        assert!(!should_pimc);
    }

    #[test]
    fn pimc_skips_klop() {
        let mut gs = GameState::new(0);
        gs.contract = Some(Contract::Klop);
        let ctx = DecisionContext {
            gs: &gs,
            player: 0,
            decision_type: DecisionType::CardPlay,
            legal_mask: vec![],
            state_encoding: vec![],
        };
        let handoff = 0usize; // even with handoff at 0
        let should_pimc = ctx.decision_type == DecisionType::CardPlay
            && ctx.gs.contract != Some(Contract::Klop)
            && ctx.gs.tricks_played() >= handoff;
        assert!(!should_pimc);
    }

    #[test]
    fn pimc_skips_non_card_decisions() {
        let mut gs = GameState::new(0);
        gs.contract = Some(Contract::Three);
        let ctx = DecisionContext {
            gs: &gs,
            player: 0,
            decision_type: DecisionType::Bid,
            legal_mask: vec![],
            state_encoding: vec![],
        };
        let handoff = 0usize;
        let should_pimc = ctx.decision_type == DecisionType::CardPlay
            && ctx.gs.contract != Some(Contract::Klop)
            && ctx.gs.tricks_played() >= handoff;
        assert!(!should_pimc);
    }
}
