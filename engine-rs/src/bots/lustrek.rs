/// Luštrek paper baseline heuristic bot.
///
/// This module exposes the same function interface as other Rust heuristic bots
/// (`evaluate_bid_*`, `choose_king_*`, `choose_talon_group_*`,
/// `choose_discards_*`, `choose_card_*`) so it can be plugged into arena and
/// training seat dispatch without special cases.
///
/// Credit: Matjaž Luštrek et al., "The analysis of Tarok using computer
/// programs" (ICGA Journal, 2003).  We plug in the m6 StockŠkis heuristic
/// for bidding / king / talon / discards / mid-game card play, and switch
/// to PIMC (Perfect Information Monte Carlo + double-dummy) once only a
/// few tricks remain.  At that point the information set is nearly
/// collapsed, so sampling + exact search is both cheap and near-optimal.
///
/// Klop / Berac / Barvni Valat fall back to pure m6 because the DD
/// solver's declarer-vs-opponent framing doesn't apply to those modes.
use crate::bots::stockskis_m6;
use crate::card::*;
use crate::game_state::*;
use crate::pimc;

/// Trick index (0-based) at which PIMC takes over card play.
/// 8 means PIMC plays tricks 9–12 (the last 4 tricks).
const PIMC_HANDOFF_TRICK: usize = 8;

/// Number of PIMC worlds to sample per card decision.
const PIMC_WORLDS: u32 = 100;

#[inline]
pub fn evaluate_bid_lustrek(hand: CardSet, highest_so_far: Option<Contract>) -> Option<Contract> {
    stockskis_m6::evaluate_bid_m6(hand, highest_so_far)
}

#[inline]
pub fn choose_king_lustrek(hand: CardSet) -> Option<Card> {
    stockskis_m6::choose_king_m6(hand)
}

#[inline]
pub fn choose_talon_group_lustrek(
    groups: &[Vec<Card>],
    hand: CardSet,
    called_king: Option<Card>,
) -> usize {
    stockskis_m6::choose_talon_group_m6(groups, hand, called_king)
}

#[inline]
pub fn choose_discards_lustrek(
    hand: CardSet,
    must_discard: usize,
    called_king: Option<Card>,
) -> Vec<Card> {
    stockskis_m6::choose_discards_m6(hand, must_discard, called_king)
}

pub fn choose_card_lustrek(hand: CardSet, state: &GameState, player: u8) -> Card {
    // PIMC + double-dummy only makes sense for two-team contracts; for
    // Klop (individual scoring), Berac and Barvni Valat fall back to m6.
    let pimc_eligible = state
        .contract
        .map_or(false, |c| !c.is_klop() && !c.is_berac() && !c.is_barvni_valat());
    if pimc_eligible && state.tricks_played() >= PIMC_HANDOFF_TRICK {
        return pimc::pimc_choose_card(state, player, PIMC_WORLDS);
    }
    stockskis_m6::choose_card_m6(hand, state, player)
}
