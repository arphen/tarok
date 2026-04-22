/// M9 — Luštrek-style heuristic + constraint-aware PIMC endgame, now
/// extended to cover Klop and Berač.
///
/// Same bid / king / talon / discard / mid-game card policy as the m6
/// heuristic (inherited via Luštrek), but switches to PIMC for the last
/// few tricks in ALL contract types — including the must-overplay modes
/// Klop and Berač, where constraint inference (void + rank ceilings) makes
/// world sampling very tight and the search almost exact.
use crate::bots::stockskis_m6;
use crate::card::*;
use crate::game_state::*;
use crate::pimc;

/// Trick index (0-based) at which PIMC takes over card play.
///
/// 8 means PIMC handles tricks 9–12 (the last 4). Same handoff point as
/// Luštrek; the extra klop/berac coverage is enabled by the PIMC dispatch
/// picking the right DD objective from `gs.contract`.
const PIMC_HANDOFF_TRICK: usize = 8;

/// Number of PIMC worlds to sample per card decision.
const PIMC_WORLDS: u32 = 100;

#[inline]
pub fn evaluate_bid_m9(hand: CardSet, highest_so_far: Option<Contract>) -> Option<Contract> {
    stockskis_m6::evaluate_bid_m6(hand, highest_so_far)
}

#[inline]
pub fn choose_king_m9(hand: CardSet) -> Option<Card> {
    stockskis_m6::choose_king_m6(hand)
}

#[inline]
pub fn choose_talon_group_m9(
    groups: &[Vec<Card>],
    hand: CardSet,
    called_king: Option<Card>,
) -> usize {
    stockskis_m6::choose_talon_group_m6(groups, hand, called_king)
}

#[inline]
pub fn choose_discards_m9(
    hand: CardSet,
    must_discard: usize,
    called_king: Option<Card>,
) -> Vec<Card> {
    stockskis_m6::choose_discards_m6(hand, must_discard, called_king)
}

pub fn choose_card_m9(hand: CardSet, state: &GameState, player: u8) -> Card {
    // Skip PIMC for barvni valat (single-player absolute contract — the
    // existing DD objective framing doesn't fit cleanly).
    let pimc_eligible = state
        .contract
        .map_or(false, |c| !c.is_barvni_valat());
    if pimc_eligible && state.tricks_played() >= PIMC_HANDOFF_TRICK {
        return pimc::pimc_choose_card(state, player, PIMC_WORLDS);
    }
    stockskis_m6::choose_card_m6(hand, state, player)
}
