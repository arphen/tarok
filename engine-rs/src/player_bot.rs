//! Heuristic bot infrastructure for Tarok self-play.
//!
//! # Adding a new bot
//!
//! 1. Implement the [`HeuristicStrategy`] trait for a zero-size struct.
//! 2. Add a match arm in [`try_make_bot_by_seat_label`].
//! 3. Add the seat label string to [`SUPPORTED_BOT_SEAT_LABELS`].
//!
//! Nothing else needs to change — `StockSkisPlayer` and `py_bindings`
//! consume the open interface automatically.

use crate::card::*;
use crate::game_state::*;
use crate::bots::lustrek;
use crate::bots::lapajne;
use crate::bots::m8;
use crate::bots::stockskis_m6;
use crate::bots::stockskis_pozrl;
use crate::bots::stockskis_v1;
use crate::bots::stockskis_v3;
use crate::bots::stockskis_v5;
use crate::bots::stockskis_v6;
use rayon::prelude::*;
use crate::player::*;

// -----------------------------------------------------------------------
// HeuristicStrategy — the extension point
// -----------------------------------------------------------------------

/// Implement this trait to add a new heuristic bot.
///
/// All methods receive only the information they need, extracted from
/// `DecisionContext` or `GameState` before the call, so implementations
/// stay focused on decision logic.
pub trait HeuristicStrategy: Send + Sync {
    /// Seat-token string used in `run_self_play` seat_config (e.g. `"bot_v5"`).
    fn seat_label(&self) -> &'static str;
    /// Human-readable name used in logging.
    fn name(&self) -> &'static str;

    fn evaluate_bid(&self, hand: CardSet, highest: Option<Contract>) -> Option<Contract>;
    fn choose_king(&self, hand: CardSet) -> Option<Card>;
    fn choose_talon_group(
        &self,
        groups: &[Vec<Card>],
        hand: CardSet,
        called_king: Option<Card>,
    ) -> usize;
    fn choose_discards(
        &self,
        hand: CardSet,
        must_discard: usize,
        called_king: Option<Card>,
    ) -> Vec<Card>;
    fn choose_card(&self, hand: CardSet, gs: &GameState, player: u8) -> Card;
}

// -----------------------------------------------------------------------
// Per-bot strategy structs
// -----------------------------------------------------------------------

pub struct BotLustrek;
impl HeuristicStrategy for BotLustrek {
    fn seat_label(&self) -> &'static str { "bot_lustrek" }
    fn name(&self) -> &'static str { "bot_lustrek" }
    fn evaluate_bid(&self, hand: CardSet, highest: Option<Contract>) -> Option<Contract> {
        lustrek::evaluate_bid_lustrek(hand, highest)
    }
    fn choose_king(&self, hand: CardSet) -> Option<Card> {
        lustrek::choose_king_lustrek(hand)
    }
    fn choose_talon_group(&self, groups: &[Vec<Card>], hand: CardSet, called_king: Option<Card>) -> usize {
        lustrek::choose_talon_group_lustrek(groups, hand, called_king)
    }
    fn choose_discards(&self, hand: CardSet, must_discard: usize, called_king: Option<Card>) -> Vec<Card> {
        lustrek::choose_discards_lustrek(hand, must_discard, called_king)
    }
    fn choose_card(&self, hand: CardSet, gs: &GameState, player: u8) -> Card {
        lustrek::choose_card_lustrek(hand, gs, player)
    }
}

pub struct BotV1;
impl HeuristicStrategy for BotV1 {
    fn seat_label(&self) -> &'static str { "bot_v1" }
    fn name(&self) -> &'static str { "stockskis_v1" }
    fn evaluate_bid(&self, hand: CardSet, highest: Option<Contract>) -> Option<Contract> {
        stockskis_v1::evaluate_bid_v1(hand, highest)
    }
    fn choose_king(&self, hand: CardSet) -> Option<Card> {
        stockskis_v1::choose_king_v1(hand)
    }
    fn choose_talon_group(&self, groups: &[Vec<Card>], hand: CardSet, called_king: Option<Card>) -> usize {
        stockskis_v1::choose_talon_group_v1(groups, hand, called_king)
    }
    fn choose_discards(&self, hand: CardSet, must_discard: usize, called_king: Option<Card>) -> Vec<Card> {
        stockskis_v1::choose_discards_v1(hand, must_discard, called_king)
    }
    fn choose_card(&self, hand: CardSet, gs: &GameState, player: u8) -> Card {
        stockskis_v1::choose_card_v1(hand, gs, player)
    }
}

pub struct BotV3;
impl HeuristicStrategy for BotV3 {
    fn seat_label(&self) -> &'static str { "bot_v3" }
    fn name(&self) -> &'static str { "stockskis_v3" }
    fn evaluate_bid(&self, hand: CardSet, highest: Option<Contract>) -> Option<Contract> {
        stockskis_v3::evaluate_bid_v3(hand, highest)
    }
    fn choose_king(&self, hand: CardSet) -> Option<Card> {
        stockskis_v3::choose_king_v3(hand)
    }
    fn choose_talon_group(&self, groups: &[Vec<Card>], hand: CardSet, called_king: Option<Card>) -> usize {
        stockskis_v3::choose_talon_group_v3(groups, hand, called_king)
    }
    fn choose_discards(&self, hand: CardSet, must_discard: usize, called_king: Option<Card>) -> Vec<Card> {
        stockskis_v3::choose_discards_v3(hand, must_discard, called_king)
    }
    fn choose_card(&self, hand: CardSet, gs: &GameState, player: u8) -> Card {
        stockskis_v3::choose_card_v3(hand, gs, player)
    }
}

pub struct BotV5;
impl HeuristicStrategy for BotV5 {
    fn seat_label(&self) -> &'static str { "bot_v5" }
    fn name(&self) -> &'static str { "stockskis_v5" }
    fn evaluate_bid(&self, hand: CardSet, highest: Option<Contract>) -> Option<Contract> {
        stockskis_v5::evaluate_bid_v5(hand, highest)
    }
    fn choose_king(&self, hand: CardSet) -> Option<Card> {
        stockskis_v5::choose_king_v5(hand)
    }
    fn choose_talon_group(&self, groups: &[Vec<Card>], hand: CardSet, called_king: Option<Card>) -> usize {
        stockskis_v5::choose_talon_group_v5(groups, hand, called_king)
    }
    fn choose_discards(&self, hand: CardSet, must_discard: usize, called_king: Option<Card>) -> Vec<Card> {
        stockskis_v5::choose_discards_v5(hand, must_discard, called_king)
    }
    fn choose_card(&self, hand: CardSet, gs: &GameState, player: u8) -> Card {
        stockskis_v5::choose_card_v5(hand, gs, player)
    }
}

pub struct BotV6;
impl HeuristicStrategy for BotV6 {
    fn seat_label(&self) -> &'static str { "bot_v6" }
    fn name(&self) -> &'static str { "stockskis_v6" }
    fn evaluate_bid(&self, hand: CardSet, highest: Option<Contract>) -> Option<Contract> {
        stockskis_v6::evaluate_bid_v6(hand, highest)
    }
    fn choose_king(&self, hand: CardSet) -> Option<Card> {
        stockskis_v6::choose_king_v6(hand)
    }
    fn choose_talon_group(&self, groups: &[Vec<Card>], hand: CardSet, called_king: Option<Card>) -> usize {
        stockskis_v6::choose_talon_group_v6(groups, hand, called_king)
    }
    fn choose_discards(&self, hand: CardSet, must_discard: usize, called_king: Option<Card>) -> Vec<Card> {
        stockskis_v6::choose_discards_v6(hand, must_discard, called_king)
    }
    fn choose_card(&self, hand: CardSet, gs: &GameState, player: u8) -> Card {
        stockskis_v6::choose_card_v6(hand, gs, player)
    }
}

pub struct BotM6;
impl HeuristicStrategy for BotM6 {
    fn seat_label(&self) -> &'static str { "bot_m6" }
    fn name(&self) -> &'static str { "stockskis_m6" }
    fn evaluate_bid(&self, hand: CardSet, highest: Option<Contract>) -> Option<Contract> {
        stockskis_m6::evaluate_bid_m6(hand, highest)
    }
    fn choose_king(&self, hand: CardSet) -> Option<Card> {
        stockskis_m6::choose_king_m6(hand)
    }
    fn choose_talon_group(&self, groups: &[Vec<Card>], hand: CardSet, called_king: Option<Card>) -> usize {
        stockskis_m6::choose_talon_group_m6(groups, hand, called_king)
    }
    fn choose_discards(&self, hand: CardSet, must_discard: usize, called_king: Option<Card>) -> Vec<Card> {
        stockskis_m6::choose_discards_m6(hand, must_discard, called_king)
    }
    fn choose_card(&self, hand: CardSet, gs: &GameState, player: u8) -> Card {
        stockskis_m6::choose_card_m6(hand, gs, player)
    }
}

pub struct BotM8;
impl HeuristicStrategy for BotM8 {
    fn seat_label(&self) -> &'static str { "bot_m8" }
    fn name(&self) -> &'static str { "stockskis_m8" }
    fn evaluate_bid(&self, hand: CardSet, highest: Option<Contract>) -> Option<Contract> {
        m8::evaluate_bid_m8(hand, highest)
    }
    fn choose_king(&self, hand: CardSet) -> Option<Card> {
        m8::choose_king_m8(hand)
    }
    fn choose_talon_group(&self, groups: &[Vec<Card>], hand: CardSet, called_king: Option<Card>) -> usize {
        m8::choose_talon_group_m8(groups, hand, called_king)
    }
    fn choose_discards(&self, hand: CardSet, must_discard: usize, called_king: Option<Card>) -> Vec<Card> {
        m8::choose_discards_m8(hand, must_discard, called_king)
    }
    fn choose_card(&self, hand: CardSet, gs: &GameState, player: u8) -> Card {
        m8::choose_card_m8(hand, gs, player)
    }
}

pub struct BotPozrl;
impl HeuristicStrategy for BotPozrl {
    fn seat_label(&self) -> &'static str { "bot_pozrl" }
    fn name(&self) -> &'static str { "stockskis_pozrl" }
    fn evaluate_bid(&self, hand: CardSet, highest: Option<Contract>) -> Option<Contract> {
        stockskis_pozrl::evaluate_bid_pozrl(hand, highest)
    }
    fn choose_king(&self, hand: CardSet) -> Option<Card> {
        stockskis_pozrl::choose_king_pozrl(hand)
    }
    fn choose_talon_group(&self, groups: &[Vec<Card>], hand: CardSet, called_king: Option<Card>) -> usize {
        stockskis_pozrl::choose_talon_group_pozrl(groups, hand, called_king)
    }
    fn choose_discards(&self, hand: CardSet, must_discard: usize, called_king: Option<Card>) -> Vec<Card> {
        stockskis_pozrl::choose_discards_pozrl(hand, must_discard, called_king)
    }
    fn choose_card(&self, hand: CardSet, gs: &GameState, player: u8) -> Card {
        stockskis_pozrl::choose_card_pozrl(hand, gs, player)
    }
}

// -----------------------------------------------------------------------
// Registry — the only two places to edit when adding a new bot
pub struct BotLapajne;
impl HeuristicStrategy for BotLapajne {
    fn seat_label(&self) -> &'static str { "bot_lapajne" }
    fn name(&self) -> &'static str { "lapajne" }
    fn evaluate_bid(&self, hand: CardSet, highest: Option<Contract>) -> Option<Contract> {
        lapajne::evaluate_bid_lapajne(hand, highest)
    }
    fn choose_king(&self, hand: CardSet) -> Option<Card> {
        lapajne::choose_king_lapajne(hand)
    }
    fn choose_talon_group(&self, groups: &[Vec<Card>], hand: CardSet, called_king: Option<Card>) -> usize {
        lapajne::choose_talon_group_lapajne(groups, hand, called_king)
    }
    fn choose_discards(&self, hand: CardSet, must_discard: usize, called_king: Option<Card>) -> Vec<Card> {
        lapajne::choose_discards_lapajne(hand, must_discard, called_king)
    }
    fn choose_card(&self, hand: CardSet, gs: &GameState, player: u8) -> Card {
        lapajne::choose_card_lapajne(hand, gs, player)
    }
}

// -----------------------------------------------------------------------

/// All recognised heuristic seat-label strings, used for error messages.
pub const SUPPORTED_BOT_SEAT_LABELS: [&str; 9] = [
    "bot_lapajne",
    "bot_lustrek",
    "bot_v1",
    "bot_v3",
    "bot_v5",
    "bot_v6",
    "bot_m6",
    "bot_m8",
    "bot_pozrl",
];

/// Construct a `StockSkisPlayer` from a seat-config label.
/// Returns `None` for labels that are not heuristic bots (e.g. `"nn"` or paths).
pub fn try_make_bot_by_seat_label(label: &str) -> Option<StockSkisPlayer> {
    let strategy: Box<dyn HeuristicStrategy> = match label {
        "bot_lapajne" => Box::new(BotLapajne),
        "bot_lustrek" => Box::new(BotLustrek),
        "bot_v1"      => Box::new(BotV1),
        "bot_v3"      => Box::new(BotV3),
        "bot_v5"      => Box::new(BotV5),
        "bot_v6"      => Box::new(BotV6),
        "bot_m6"      => Box::new(BotM6),
        "bot_m8"      => Box::new(BotM8),
        "bot_pozrl"   => Box::new(BotPozrl),
        _             => return None,
    };
    Some(StockSkisPlayer { strategy })
}

// -----------------------------------------------------------------------
// StockSkisPlayer — generic heuristic player backed by any HeuristicStrategy
// -----------------------------------------------------------------------

pub struct StockSkisPlayer {
    strategy: Box<dyn HeuristicStrategy>,
}

impl BatchPlayer for StockSkisPlayer {
    fn batch_decide(&self, contexts: &[DecisionContext<'_>]) -> Vec<DecisionResult> {
        contexts
            .par_iter()
            .map(|ctx| {
                let action = match ctx.decision_type {
                    DecisionType::Bid => self.decide_bid(ctx),
                    DecisionType::KingCall => self.decide_king(ctx),
                    DecisionType::TalonPick => self.decide_talon(ctx),
                    DecisionType::CardPlay => self.decide_card(ctx),
                };
                DecisionResult { action, log_prob: 0.0, value: 0.0 }
            })
            .collect()
    }

    fn choose_discards(&self, gs: &GameState, player: u8, must_discard: usize) -> Option<Vec<Card>> {
        let hand = gs.hands[player as usize];
        Some(self.strategy.choose_discards(hand, must_discard, gs.called_king))
    }

    fn name(&self) -> &str {
        self.strategy.name()
    }
}

impl StockSkisPlayer {
    fn decide_bid(&self, ctx: &DecisionContext<'_>) -> usize {
        let hand = ctx.gs.hands[ctx.player as usize];
        let highest = ctx.gs.bids.iter().filter_map(|b| b.contract).max_by_key(|c| c.strength());
        let action = contract_to_bid_action(self.strategy.evaluate_bid(hand, highest));
        if ctx.legal_mask.get(action).map_or(false, |&v| v > 0.5) { action } else { 0 }
    }

    fn decide_king(&self, ctx: &DecisionContext<'_>) -> usize {
        let hand = ctx.gs.hands[ctx.player as usize];
        let chosen = self.strategy.choose_king(hand);
        match chosen.and_then(|c| card_suit_idx(c.0)) {
            Some(idx) if ctx.legal_mask.get(idx).map_or(false, |&v| v > 0.5) => idx,
            _ => ctx.legal_mask.iter().position(|&v| v > 0.5).unwrap_or(0),
        }
    }

    fn decide_talon(&self, ctx: &DecisionContext<'_>) -> usize {
        let hand = ctx.gs.hands[ctx.player as usize];
        let chosen = self.strategy.choose_talon_group(
            &ctx.gs.talon_revealed, hand, ctx.gs.called_king,
        );
        if ctx.legal_mask.get(chosen).map_or(false, |&v| v > 0.5) {
            chosen
        } else {
            ctx.legal_mask.iter().position(|&v| v > 0.5).unwrap_or(0)
        }
    }

    fn decide_card(&self, ctx: &DecisionContext<'_>) -> usize {
        let hand = ctx.gs.hands[ctx.player as usize];
        let card = self.strategy.choose_card(hand, ctx.gs, ctx.player);
        let action = card.0 as usize;
        if ctx.legal_mask.get(action).map_or(false, |&v| v > 0.5) {
            action
        } else {
            ctx.legal_mask.iter().position(|&v| v > 0.5).unwrap_or(0)
        }
    }
}
