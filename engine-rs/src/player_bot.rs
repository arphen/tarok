//! StockŠkis heuristic bot implementing [`BatchPlayer`].
//!
//! Wraps the `stockskis_v5` / `stockskis_v6` heuristic functions behind
//! the stable [`BatchPlayer`] trait.  Each call to `batch_decide` loops
//! over the contexts and calls the appropriate v5/v6 function.

use crate::card::*;
use crate::game_state::*;
use crate::player::*;
use crate::stockskis_v3;
use crate::stockskis_v1;
use crate::stockskis_v5;
use crate::stockskis_v6;
use crate::stockskis_m6;

// -----------------------------------------------------------------------
// Bot version selector
// -----------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BotVersion {
    V1,
    V3,
    V5,
    V6,
    M6,
}

// -----------------------------------------------------------------------
// StockSkisPlayer
// -----------------------------------------------------------------------

pub struct StockSkisPlayer {
    version: BotVersion,
}

impl StockSkisPlayer {
    pub fn new(version: BotVersion) -> Self {
        StockSkisPlayer { version }
    }

    pub fn v3() -> Self {
        Self::new(BotVersion::V3)
    }

    pub fn v1() -> Self {
        Self::new(BotVersion::V1)
    }

    pub fn v5() -> Self {
        Self::new(BotVersion::V5)
    }

    pub fn v6() -> Self {
        Self::new(BotVersion::V6)
    }

    pub fn m6() -> Self {
        Self::new(BotVersion::M6)
    }
}

impl BatchPlayer for StockSkisPlayer {
    fn batch_decide(&self, contexts: &[DecisionContext<'_>]) -> Vec<DecisionResult> {
        contexts
            .iter()
            .map(|ctx| {
                let action = match ctx.decision_type {
                    DecisionType::Bid => self.decide_bid(ctx),
                    DecisionType::KingCall => self.decide_king(ctx),
                    DecisionType::TalonPick => self.decide_talon(ctx),
                    DecisionType::CardPlay => self.decide_card(ctx),
                };
                DecisionResult {
                    action,
                    log_prob: 0.0,
                    value: 0.0,
                }
            })
            .collect()
    }

    fn choose_discards(
        &self,
        gs: &GameState,
        player: u8,
        must_discard: usize,
    ) -> Option<Vec<Card>> {
        let hand = gs.hands[player as usize];
        let called_king = gs.called_king;
        let discards = match self.version {
            BotVersion::V1 => stockskis_v1::choose_discards_v1(hand, must_discard, called_king),
            BotVersion::V3 => stockskis_v3::choose_discards_v3(hand, must_discard, called_king),
            BotVersion::V5 => stockskis_v5::choose_discards_v5(hand, must_discard, called_king),
            BotVersion::V6 => stockskis_v6::choose_discards_v6(hand, must_discard, called_king),
            BotVersion::M6 => stockskis_m6::choose_discards_m6(hand, must_discard, called_king),
        };
        Some(discards)
    }

    fn name(&self) -> &str {
        match self.version {
            BotVersion::V1 => "stockskis_v1",
            BotVersion::V3 => "stockskis_v3",
            BotVersion::V5 => "stockskis_v5",
            BotVersion::V6 => "stockskis_v6",
            BotVersion::M6 => "stockskis_m6",
        }
    }
}

// -----------------------------------------------------------------------
// Per-decision-type dispatch
// -----------------------------------------------------------------------

impl StockSkisPlayer {
    fn decide_bid(&self, ctx: &DecisionContext<'_>) -> usize {
        let hand = ctx.gs.hands[ctx.player as usize];
        let highest = ctx
            .gs
            .bids
            .iter()
            .filter_map(|b| b.contract)
            .max_by_key(|c| c.strength());

        let chosen = match self.version {
            BotVersion::V1 => stockskis_v1::evaluate_bid_v1(hand, highest),
            BotVersion::V3 => stockskis_v3::evaluate_bid_v3(hand, highest),
            BotVersion::V5 => stockskis_v5::evaluate_bid_v5(hand, highest),
            BotVersion::V6 => stockskis_v6::evaluate_bid_v6(hand, highest),
            BotVersion::M6 => stockskis_m6::evaluate_bid_m6(hand, highest),
        };

        let action = contract_to_bid_action(chosen);
        // Verify legality; fall back to pass
        if ctx.legal_mask.get(action).map_or(false, |&v| v > 0.5) {
            action
        } else {
            0 // pass
        }
    }

    fn decide_king(&self, ctx: &DecisionContext<'_>) -> usize {
        let hand = ctx.gs.hands[ctx.player as usize];
        let chosen = match self.version {
            BotVersion::V1 => stockskis_v1::choose_king_v1(hand),
            BotVersion::V3 => stockskis_v3::choose_king_v3(hand),
            BotVersion::V5 => stockskis_v5::choose_king_v5(hand),
            BotVersion::V6 => stockskis_v6::choose_king_v6(hand),
            BotVersion::M6 => stockskis_m6::choose_king_m6(hand),
        };

        match chosen.and_then(|c| card_suit_idx(c.0)) {
            Some(idx) if ctx.legal_mask.get(idx).map_or(false, |&v| v > 0.5) => idx,
            _ => ctx.legal_mask.iter().position(|&v| v > 0.5).unwrap_or(0),
        }
    }

    fn decide_talon(&self, ctx: &DecisionContext<'_>) -> usize {
        let hand = ctx.gs.hands[ctx.player as usize];
        let called_king = ctx.gs.called_king;
        let groups = &ctx.gs.talon_revealed;

        let chosen = match self.version {
            BotVersion::V1 => stockskis_v1::choose_talon_group_v1(groups, hand, called_king),
            BotVersion::V3 => stockskis_v3::choose_talon_group_v3(groups, hand, called_king),
            BotVersion::V5 => stockskis_v5::choose_talon_group_v5(groups, hand, called_king),
            BotVersion::V6 => stockskis_v6::choose_talon_group_v6(groups, hand, called_king),
            BotVersion::M6 => stockskis_m6::choose_talon_group_m6(groups, hand, called_king),
        };

        if ctx.legal_mask.get(chosen).map_or(false, |&v| v > 0.5) {
            chosen
        } else {
            ctx.legal_mask.iter().position(|&v| v > 0.5).unwrap_or(0)
        }
    }

    fn decide_card(&self, ctx: &DecisionContext<'_>) -> usize {
        let hand = ctx.gs.hands[ctx.player as usize];
        let card = match self.version {
            BotVersion::V1 => stockskis_v1::choose_card_v1(hand, &ctx.gs, ctx.player),
            BotVersion::V3 => stockskis_v3::choose_card_v3(hand, &ctx.gs, ctx.player),
            BotVersion::V5 => stockskis_v5::choose_card_v5(hand, &ctx.gs, ctx.player),
            BotVersion::V6 => stockskis_v6::choose_card_v6(hand, &ctx.gs, ctx.player),
            BotVersion::M6 => stockskis_m6::choose_card_m6(hand, &ctx.gs, ctx.player),
        };

        let action = card.0 as usize;
        if ctx.legal_mask.get(action).map_or(false, |&v| v > 0.5) {
            action
        } else {
            ctx.legal_mask.iter().position(|&v| v > 0.5).unwrap_or(0)
        }
    }
}
