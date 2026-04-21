/// M8 heuristic bot.
///
/// Uses the same mid-game policy family as Luštrek (m6 heuristics), then
/// switches to αμ search for the final 4 tricks.
use crate::alpha_mu;
use crate::bots::stockskis_m6;
use crate::card::*;
use crate::game_state::*;

/// Trick index (0-based) at which alpha-mu takes over card play.
/// 8 means alpha-mu plays tricks 9-12 (the last 4 tricks).
const ALPHA_MU_HANDOFF_TRICK: usize = 8;

/// Number of worlds sampled for alpha-mu determinization.
const ALPHA_MU_WORLDS: u32 = 100;

/// Max-depth for alpha-mu search in Max moves.
const ALPHA_MU_DEPTH: usize = 2;

#[inline]
pub fn evaluate_bid_m8(hand: CardSet, highest_so_far: Option<Contract>) -> Option<Contract> {
    stockskis_m6::evaluate_bid_m6(hand, highest_so_far)
}

#[inline]
pub fn choose_king_m8(hand: CardSet) -> Option<Card> {
    stockskis_m6::choose_king_m6(hand)
}

#[inline]
pub fn choose_talon_group_m8(
    groups: &[Vec<Card>],
    hand: CardSet,
    called_king: Option<Card>,
) -> usize {
    stockskis_m6::choose_talon_group_m6(groups, hand, called_king)
}

#[inline]
pub fn choose_discards_m8(
    hand: CardSet,
    must_discard: usize,
    called_king: Option<Card>,
) -> Vec<Card> {
    stockskis_m6::choose_discards_m6(hand, must_discard, called_king)
}

pub fn choose_card_m8(hand: CardSet, state: &GameState, player: u8) -> Card {
    // Alpha-mu endgame search is only meaningful for team contracts.
    // For Klop, Berac, and Barvni Valat, use plain m6 logic.
    let alpha_mu_eligible = state
        .contract
        .map_or(false, |c| !c.is_klop() && !c.is_berac() && !c.is_barvni_valat());
    if alpha_mu_eligible && state.tricks_played() >= ALPHA_MU_HANDOFF_TRICK {
        return alpha_mu::alpha_mu_choose_card(
            state,
            player,
            ALPHA_MU_WORLDS,
            ALPHA_MU_DEPTH,
        );
    }
    stockskis_m6::choose_card_m6(hand, state, player)
}