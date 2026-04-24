//! Centaur (hybrid NN + endgame solver) player for Tarok.
//!
//! Uses the neural network for early/mid-game decisions (bidding, king
//! calling, talon exchange, and tricks before the handoff point), then
//! switches to a search-based endgame solver where the information set
//! is nearly collapsed and search is practically exact.
//!
//! The endgame solver is configurable:
//! - **PIMC** (default): Perfect-Information Monte Carlo — samples worlds
//!   and DD-solves each independently.  Fast, but susceptible to
//!   non-locality (strategy fusion is not enforced).
//! - **AlphaMu**: The αμ algorithm — enforces strategy fusion via Pareto
//!   fronts of per-world outcome vectors.  Stronger but slower.
//!
//! During training the solver-decided experiences are tagged with
//! `log_prob = NaN` so the PPO trainer can skip them while still
//! benefiting from the solver-optimal terminal rewards flowing back
//! through the NN's earlier decisions.

use crate::alpha_mu;
use crate::game_state::{Contract, GameState};
use crate::pimc;
use crate::player::*;
use crate::player_nn::NeuralNetPlayer;
use rayon::prelude::*;
use tch::Device;

/// Default trick number at which the endgame solver takes over card play
/// (0-indexed, so 8 means the solver plays tricks 9–12, i.e. the last 4).
pub const DEFAULT_HANDOFF_TRICK: usize = 8;

/// Default number of worlds to sample per decision.
pub const DEFAULT_NUM_WORLDS: u32 = 100;

/// Default αμ search depth (number of Max moves to look ahead).
pub const DEFAULT_ALPHA_MU_DEPTH: usize = 2;

// -----------------------------------------------------------------------
// Endgame policy
// -----------------------------------------------------------------------

/// Which search algorithm to use for endgame card play.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndgamePolicy {
    /// PIMC: sample worlds, DD-solve each independently.
    Pimc,
    /// αμ: sample worlds, enforce strategy fusion via Pareto fronts.
    AlphaMu { max_depth: usize },
}

impl EndgamePolicy {
    pub fn from_name(name: &str, alpha_mu_depth: usize) -> Self {
        match name {
            "alpha_mu" | "alphamu" => EndgamePolicy::AlphaMu {
                max_depth: alpha_mu_depth,
            },
            _ => EndgamePolicy::Pimc,
        }
    }
}

// -----------------------------------------------------------------------
// CentaurBot
// -----------------------------------------------------------------------

pub struct CentaurBot {
    nn: NeuralNetPlayer,
    handoff_trick: usize,
    num_worlds: u32,
    endgame_policy: EndgamePolicy,
    /// When `Some(salt)`, the endgame solver is seeded deterministically
    /// from a canonical fingerprint of the visible game state XORed with
    /// `salt`. Used by duplicate-RL so that active and shadow tables
    /// reaching the same state sample the same worlds → PIMC noise
    /// cancels in the reward difference. When `None` (the default),
    /// the solver uses fresh randomness per decision.
    deterministic_seed_salt: Option<u64>,
}

impl CentaurBot {
    pub fn new(
        model_path: &str,
        device: Device,
        explore_rate: f64,
        handoff_trick: usize,
        num_worlds: u32,
        endgame_policy: EndgamePolicy,
    ) -> Self {
        CentaurBot {
            nn: NeuralNetPlayer::new(model_path, device, explore_rate),
            handoff_trick,
            num_worlds,
            endgame_policy,
            deterministic_seed_salt: None,
        }
    }

    /// Enable deterministic seeding of the endgame solver. The salt is
    /// XORed into the state fingerprint so different training runs can
    /// still produce independent streams while any single run remains
    /// reproducible and duplicate-invariant.
    pub fn with_deterministic_seed(mut self, salt: u64) -> Self {
        self.deterministic_seed_salt = Some(salt);
        self
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

        // Override late-game card plays with the endgame solver.
        //
        // Each PIMC/alpha-mu call is independent and, under deterministic
        // seeding, a pure function of (state, viewer, salt). Running the
        // per-context overrides in parallel preserves determinism and lets
        // endgame-heavy batches saturate all cores instead of serialising
        // behind a single dispatch thread.
        let overrides: Vec<Option<(usize, usize)>> = contexts
            .par_iter()
            .enumerate()
            .map(|(i, ctx)| {
                if !self.use_pimc(ctx) {
                    return None;
                }
                let card = match (self.endgame_policy, self.deterministic_seed_salt) {
                    (EndgamePolicy::Pimc, None) => {
                        pimc::pimc_choose_card(ctx.gs, ctx.player, self.num_worlds)
                    }
                    (EndgamePolicy::Pimc, Some(salt)) => {
                        let seed = state_fingerprint(ctx.gs, ctx.player) ^ salt;
                        pimc::pimc_choose_card_with_seed(
                            ctx.gs,
                            ctx.player,
                            self.num_worlds,
                            seed,
                        )
                    }
                    (EndgamePolicy::AlphaMu { max_depth }, None) => alpha_mu::alpha_mu_choose_card(
                        ctx.gs,
                        ctx.player,
                        self.num_worlds,
                        max_depth,
                    ),
                    (EndgamePolicy::AlphaMu { max_depth }, Some(salt)) => {
                        let seed = state_fingerprint(ctx.gs, ctx.player) ^ salt;
                        alpha_mu::alpha_mu_choose_card_with_seed(
                            ctx.gs,
                            ctx.player,
                            self.num_worlds,
                            max_depth,
                            seed,
                        )
                    }
                };
                Some((i, card.0 as usize))
            })
            .collect();

        for entry in overrides.into_iter().flatten() {
            let (i, action) = entry;
            results[i] = DecisionResult {
                action,
                // NaN sentinel: PPO should skip this experience's policy
                // loss while still using the game's terminal reward.
                log_prob: f32::NAN,
                // Keep the NN's value estimate — useful for GAE on the
                // preceding (NN-decided) time steps.
                value: results[i].value,
            };
        }

        results
    }

    fn name(&self) -> &str {
        "centaur"
    }
}

// -----------------------------------------------------------------------
// Deterministic state fingerprint (for duplicate-RL PIMC seeding)
// -----------------------------------------------------------------------

/// SplitMix64 finalizer — stable, deterministic, no external deps.
#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn mix(acc: u64, x: u64) -> u64 {
    splitmix64(acc ^ splitmix64(x))
}

/// Hash every decision-relevant bit of the visible game state into a u64.
///
/// Two game states that reach this function with identical values produce
/// the same fingerprint. Different states — even ones that differ only in
/// played-card order — produce different fingerprints with overwhelming
/// probability (SplitMix64 avalanche).
///
/// This is the seed source for deterministic PIMC under duplicate-RL: when
/// active and shadow tables reach identical visible states, they pick the
/// same worlds and thus the same card, so the PIMC sampling noise
/// disappears from `R_active − R_shadow`.
pub fn state_fingerprint(gs: &GameState, viewer: u8) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325; // FNV offset; arbitrary nonzero start

    h = mix(h, viewer as u64);

    // Contract, declarer, partner — identifies who is maximizing what.
    let contract_tag: u64 = match gs.contract {
        None => 0,
        Some(c) => 1 + c as u64,
    };
    h = mix(h, contract_tag);
    h = mix(h, gs.declarer.map(|p| p as u64 + 1).unwrap_or(0));
    h = mix(h, gs.partner.map(|p| p as u64 + 1).unwrap_or(0));

    // Roles — declarer/partner/opponent per seat.
    let mut roles_word: u64 = 0;
    for (i, &r) in gs.roles.iter().enumerate() {
        roles_word |= (r as u64 & 0xF) << (i * 4);
    }
    h = mix(h, roles_word);

    // All cards played so far (order-independent by construction).
    h = mix(h, gs.played_cards.0);

    // Viewer's own hand — narrows the world-sampling space.
    h = mix(h, gs.hands[viewer as usize].0);

    // Current trick: lead + played cards in play order.
    if let Some(ref t) = gs.current_trick {
        h = mix(h, t.lead_player as u64);
        h = mix(h, t.count as u64);
        for i in 0..t.count as usize {
            let (p, c) = t.cards[i];
            h = mix(h, ((p as u64) << 8) | c.0 as u64);
        }
    } else {
        h = mix(h, 0xFFFF_FFFF_FFFF_FFFF); // distinguish "no trick" from trick with count=0
    }

    // Tricks played: count + who won each (viewer knows this from public play).
    h = mix(h, gs.tricks_played() as u64);
    for (i, trick) in gs.tricks.iter().enumerate() {
        h = mix(h, (i as u64) << 32 | trick.lead_player as u64);
        for j in 0..trick.count as usize {
            let (p, c) = trick.cards[j];
            h = mix(h, ((p as u64) << 16) | ((j as u64) << 8) | c.0 as u64);
        }
    }

    // Ensure no zero output (zero ^ salt = salt, which would make salt
    // trivially recoverable; not a security issue, but mildly untidy).
    splitmix64(h)
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

    // -------- state_fingerprint --------

    #[test]
    fn fingerprint_is_stable_for_identical_state() {
        let mut gs = GameState::new(0);
        gs.contract = Some(Contract::Three);
        gs.declarer = Some(1);
        gs.partner = Some(3);
        gs.played_cards = crate::card::CardSet(0xDEAD_BEEF_1234_5678);
        gs.hands[0] = crate::card::CardSet(0xAAAA_5555_AAAA_5555);

        let a = state_fingerprint(&gs, 0);
        let b = state_fingerprint(&gs, 0);
        assert_eq!(a, b, "fingerprint must be pure");
    }

    #[test]
    fn fingerprint_differs_across_viewers() {
        let mut gs = GameState::new(0);
        gs.contract = Some(Contract::Three);
        gs.played_cards = crate::card::CardSet(0x1111_2222_3333_4444);
        gs.hands[0] = crate::card::CardSet(0xFF);
        gs.hands[1] = crate::card::CardSet(0xFF00);

        let a = state_fingerprint(&gs, 0);
        let b = state_fingerprint(&gs, 1);
        assert_ne!(a, b, "different viewer → different fingerprint");
    }

    #[test]
    fn fingerprint_differs_when_played_cards_differ() {
        let mut gs = GameState::new(0);
        gs.contract = Some(Contract::Three);
        gs.played_cards = crate::card::CardSet(0x1);
        let a = state_fingerprint(&gs, 0);

        gs.played_cards = crate::card::CardSet(0x2);
        let b = state_fingerprint(&gs, 0);

        assert_ne!(a, b, "different played cards → different fingerprint");
    }

    #[test]
    fn fingerprint_is_nonzero() {
        let gs = GameState::new(0);
        assert_ne!(state_fingerprint(&gs, 0), 0);
    }
}
