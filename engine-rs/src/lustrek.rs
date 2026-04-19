/// Luštrek paper baseline heuristic bot.
///
/// This module exposes the same function interface as other Rust heuristic bots
/// (`evaluate_bid_*`, `choose_king_*`, `choose_talon_group_*`,
/// `choose_discards_*`, `choose_card_*`) so it can be plugged into arena and
/// training seat dispatch without special cases.
///
/// Credit: Matjaž Luštrek et al., "The analysis of Tarok using computer
/// programs" (ICGA Journal, 2003).  The concrete baseline behavior here uses
/// the existing v1 StockŠkis heuristic implementation.
use crate::card::*;
use crate::game_state::*;
use crate::stockskis_v1;

#[inline]
pub fn evaluate_bid_lustrek(hand: CardSet, highest_so_far: Option<Contract>) -> Option<Contract> {
    stockskis_v1::evaluate_bid_v1(hand, highest_so_far)
}

#[inline]
pub fn choose_king_lustrek(hand: CardSet) -> Option<Card> {
    stockskis_v1::choose_king_v1(hand)
}

#[inline]
pub fn choose_talon_group_lustrek(
    groups: &[Vec<Card>],
    hand: CardSet,
    called_king: Option<Card>,
) -> usize {
    stockskis_v1::choose_talon_group_v1(groups, hand, called_king)
}

#[inline]
pub fn choose_discards_lustrek(
    hand: CardSet,
    must_discard: usize,
    called_king: Option<Card>,
) -> Vec<Card> {
    stockskis_v1::choose_discards_v1(hand, must_discard, called_king)
}

#[inline]
pub fn choose_card_lustrek(hand: CardSet, state: &GameState, player: u8) -> Card {
    stockskis_v1::choose_card_v1(hand, state, player)
}
