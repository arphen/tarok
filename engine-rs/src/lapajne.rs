/// Lapajne heuristic bot interface.
///
/// This adapter exposes the same Rust heuristic function shape used by
/// `stockskis_v*` bots, while preserving attribution to the original
/// Tarok AI work by Rok Lapajne.
///
/// Current implementation delegates to `stockskis_v1`, which is the
/// direct Rust port of the original StockŠkis-style rule set.
use crate::card::*;
use crate::game_state::*;
use crate::stockskis_v1;

#[inline]
pub fn evaluate_bid_lapajne(
    hand: CardSet,
    highest_so_far: Option<Contract>,
) -> Option<Contract> {
    stockskis_v1::evaluate_bid_v1(hand, highest_so_far)
}

#[inline]
pub fn choose_king_lapajne(hand: CardSet) -> Option<Card> {
    stockskis_v1::choose_king_v1(hand)
}

#[inline]
pub fn choose_talon_group_lapajne(
    groups: &[Vec<Card>],
    hand: CardSet,
    called_king: Option<Card>,
) -> usize {
    stockskis_v1::choose_talon_group_v1(groups, hand, called_king)
}

#[inline]
pub fn choose_discards_lapajne(
    hand: CardSet,
    must_discard: usize,
    called_king: Option<Card>,
) -> Vec<Card> {
    stockskis_v1::choose_discards_v1(hand, must_discard, called_king)
}

#[inline]
pub fn choose_card_lapajne(hand: CardSet, state: &GameState, player: u8) -> Card {
    stockskis_v1::choose_card_v1(hand, state, player)
}
