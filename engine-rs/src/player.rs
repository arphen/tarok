//! Stable player interface for Tarok self-play.
//!
//! Implement [`BatchPlayer`] to create new bot types or AI models.
//! The self-play runner groups pending decisions by player instance
//! and calls [`BatchPlayer::batch_decide`] once per group per game
//! step, enabling efficient batched inference for neural networks
//! while heuristic bots simply loop.
//!
//! # Example
//!
//! ```ignore
//! struct MyBot;
//!
//! impl BatchPlayer for MyBot {
//!     fn batch_decide(&self, contexts: &[DecisionContext]) -> Vec<DecisionResult> {
//!         contexts.iter().map(|ctx| {
//!             let action = my_heuristic(&ctx.gs, ctx.player, ctx.decision_type, &ctx.legal_mask);
//!             DecisionResult { action, log_prob: 0.0, value: 0.0 }
//!         }).collect()
//!     }
//!     fn name(&self) -> &str { "my_bot" }
//! }
//! ```

use crate::card::Card;
use crate::game_state::{Contract, GameState};

// -----------------------------------------------------------------------
// Decision types and action spaces
// -----------------------------------------------------------------------

/// Which phase of the game the decision is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DecisionType {
    Bid = 0,
    KingCall = 1,
    TalonPick = 2,
    CardPlay = 3,
}

pub const BID_ACTION_SIZE: usize = 9;
pub const KING_ACTION_SIZE: usize = 4;
pub const TALON_ACTION_SIZE: usize = 6;
pub const CARD_ACTION_SIZE: usize = 54;

impl DecisionType {
    /// Number of possible actions for this decision type.
    pub fn action_size(self) -> usize {
        match self {
            DecisionType::Bid => BID_ACTION_SIZE,
            DecisionType::KingCall => KING_ACTION_SIZE,
            DecisionType::TalonPick => TALON_ACTION_SIZE,
            DecisionType::CardPlay => CARD_ACTION_SIZE,
        }
    }
}

// -----------------------------------------------------------------------
// Bid action ↔ contract mapping
// -----------------------------------------------------------------------

/// Maps bid action index → Contract (None = pass).
pub const BID_IDX_TO_CONTRACT: [Option<Contract>; BID_ACTION_SIZE] = [
    None,                       // 0 = pass
    Some(Contract::Three),      // 1
    Some(Contract::Two),        // 2
    Some(Contract::One),        // 3
    Some(Contract::SoloThree),  // 4
    Some(Contract::SoloTwo),    // 5
    Some(Contract::SoloOne),    // 6
    Some(Contract::Solo),       // 7
    Some(Contract::Berac),      // 8
];

/// Convert an `Option<Contract>` to a bid action index.
pub fn contract_to_bid_action(contract: Option<Contract>) -> usize {
    match contract {
        None => 0,
        Some(c) => Contract::BIDDABLE
            .iter()
            .position(|&b| b == c)
            .map(|i| i + 1)
            .unwrap_or(0),
    }
}

// -----------------------------------------------------------------------
// Decision context and result
// -----------------------------------------------------------------------

/// Everything a player needs to make one decision.
///
/// The runner pre-computes both the raw `GameState` (for heuristic bots)
/// and the encoded state tensor (for neural-network players).
pub struct DecisionContext {
    /// Full game state snapshot (cloned from the in-flight game).
    pub gs: GameState,
    /// Which player seat is deciding (0–3).
    pub player: u8,
    /// Type of decision required.
    pub decision_type: DecisionType,
    /// Binary mask over the action space (1.0 = legal, 0.0 = illegal).
    /// Length equals `decision_type.action_size()`.
    pub legal_mask: Vec<f32>,
    /// Pre-computed state encoding (450 floats).
    /// Heuristic bots can ignore this field.
    pub state_encoding: Vec<f32>,
}

/// Result of one player decision.
#[derive(Clone, Copy)]
pub struct DecisionResult {
    /// Index into the action space (0-based).
    pub action: usize,
    /// Log-probability of the chosen action (0.0 for deterministic bots).
    pub log_prob: f32,
    /// Estimated state value (0.0 for heuristic bots).
    pub value: f32,
}

// -----------------------------------------------------------------------
// The trait
// -----------------------------------------------------------------------

/// A Tarok player that makes decisions in batch.
///
/// The self-play runner groups pending decisions by player instance
/// and calls `batch_decide` once per group per game step.  This allows
/// neural-network players to do a single batched forward pass while
/// heuristic bots simply loop over the contexts.
pub trait BatchPlayer: Send + Sync {
    /// Decide actions for a batch of game positions.
    ///
    /// Must return exactly one [`DecisionResult`] per input context,
    /// in the same order.
    fn batch_decide(&self, contexts: &[DecisionContext]) -> Vec<DecisionResult>;

    /// Choose which cards to discard after picking a talon group.
    ///
    /// Called once per talon exchange.  Return `Some(cards)` to override
    /// the default heuristic (which discards low non-king, non-tarok cards).
    fn choose_discards(
        &self,
        _gs: &GameState,
        _player: u8,
        _must_discard: usize,
    ) -> Option<Vec<Card>> {
        None
    }

    /// Human-readable name for logging and Python-side identification.
    fn name(&self) -> &str;
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

/// Map a suit-card index to its suit (0–3).  Returns `None` for taroks.
pub fn card_suit_idx(card_idx: u8) -> Option<usize> {
    if card_idx >= 22 && card_idx < 54 {
        Some(((card_idx - 22) / 8) as usize)
    } else {
        None
    }
}
