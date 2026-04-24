/// PIMC (Perfect Information Monte Carlo) card selection.
///
/// Samples `N` consistent worlds by redistributing unknown cards among
/// opponents (respecting known suit voids AND — in overplay contracts like
/// klop/berac — rank ceilings inferred from the must-overplay rule),
/// double-dummy solves each world, and picks the legal move with the best
/// average viewer utility.
use crate::card::*;
use crate::double_dummy::{self, DDObjective, DDState};
use crate::game_state::*;
use crate::legal_moves;
use rand::prelude::*;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;

const VOID_CHANNELS: usize = 5;
const TAROK_VOID_IDX: usize = 4;
const NO_CEILING: u8 = u8::MAX;

/// Default number of worlds to sample.
pub const DEFAULT_NUM_WORLDS: u32 = 100;

/// Per-player constraints inferred from past play.
///
/// * `void[s]`           — player cannot hold any card of suit `s`
///   (0..=3 = suits, 4 = tarok void).
/// * `suit_max[s]`       — player's highest possible rank in suit `s`
///   (SuitRank u8 value 1..=8), or `NO_CEILING` if no upper bound known.
/// * `tarok_max`         — player's highest possible tarok value (1..=22),
///   or `NO_CEILING`.
///
/// Ceilings come from the must-overplay rule in klop/berac: when a player
/// follows suit/tarok but does not play a card that beats the current best
/// card in the trick, we infer they hold no higher card of that suit/tarok.
#[derive(Clone, Copy, Debug)]
pub struct PlayerConstraints {
    pub void: [bool; VOID_CHANNELS],
    pub suit_max: [u8; 4],
    pub tarok_max: u8,
}

impl PlayerConstraints {
    pub const fn new() -> Self {
        PlayerConstraints {
            void: [false; VOID_CHANNELS],
            suit_max: [NO_CEILING; 4],
            tarok_max: NO_CEILING,
        }
    }

    #[inline]
    fn tighten_suit_max(&mut self, suit: Suit, rank_value: u8) {
        let s = suit as usize;
        if rank_value < self.suit_max[s] {
            self.suit_max[s] = rank_value;
        }
    }

    #[inline]
    fn tighten_tarok_max(&mut self, tarok_value: u8) {
        if tarok_value < self.tarok_max {
            self.tarok_max = tarok_value;
        }
    }

    /// Would assigning `card` to this player violate any known constraint?
    #[inline]
    pub fn card_allowed(&self, card: Card) -> bool {
        match card.card_type() {
            CardType::Tarok => {
                if self.void[TAROK_VOID_IDX] {
                    return false;
                }
                card.value() <= self.tarok_max
            }
            CardType::Suit => {
                let suit = match card.suit() {
                    Some(s) => s,
                    None => return true,
                };
                if self.void[suit as usize] {
                    return false;
                }
                card.value() <= self.suit_max[suit as usize]
            }
        }
    }
}

impl Default for PlayerConstraints {
    fn default() -> Self {
        Self::new()
    }
}

/// Choose the best card for `viewer` using PIMC.
///
/// Panics (debug) if no legal moves exist.
pub fn pimc_choose_card(gs: &GameState, viewer: u8, num_worlds: u32) -> Card {
    pimc_choose_card_with_seed(gs, viewer, num_worlds, rand::rng().random())
}

/// Deterministic variant of [`pimc_choose_card`].
///
/// Identical `base_seed` → identical world samples → identical chosen card
/// (given identical `gs` and `viewer`). This is the entry point used by the
/// Centaur player under duplicate-RL, where the seed is derived from a
/// canonical fingerprint of the visible game state so that two tables that
/// reach the same state also pick the same worlds — and thus the PIMC noise
/// cancels in `R_active − R_shadow`.
pub fn pimc_choose_card_with_seed(
    gs: &GameState,
    viewer: u8,
    num_worlds: u32,
    base_seed: u64,
) -> Card {
    // Legal moves for the viewer
    let legal = {
        let ctx = legal_moves::MoveCtx::from_state(gs, viewer);
        legal_moves::generate_legal_moves(&ctx)
    };
    let legal_vec: Vec<Card> = legal.iter().collect();
    debug_assert!(!legal_vec.is_empty());

    if legal_vec.len() == 1 {
        return legal_vec[0];
    }

    let constraints = detect_constraints(gs);
    // O(1) map from card id to legal index in `scores`.
    let mut legal_idx: [Option<usize>; DECK_SIZE] = [None; DECK_SIZE];
    for (i, card) in legal_vec.iter().enumerate() {
        legal_idx[card.0 as usize] = Some(i);
    }

    // Pick the DD objective from the contract.
    let objective = objective_for_contract(gs.contract);

    // Parallel over worlds, then reduce into per-card aggregates:
    // (total_viewer_utility, sample_count) per legal card.
    //
    // DD utilities are integers, so summing them as i64 is associative
    // and the parallel rayon reduce is bit-deterministic regardless of
    // work-stealing order. This preserves the deterministic-seed
    // guarantee without the per-world Vec allocation + serial fold.
    let n_legal = legal_vec.len();
    let (sum_i64, counts) = (0..num_worlds)
        .into_par_iter()
        .map(|world_i| {
            let mut local_sum = vec![0i64; n_legal];
            let mut local_cnt = vec![0u32; n_legal];
            let mut rng = SmallRng::seed_from_u64(
                base_seed ^ (world_i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            );

            let sampled_hands = match sample_world(gs, viewer, &constraints, &mut rng) {
                Some(h) => h,
                None => return (local_sum, local_cnt),
            };

            let dd_state = DDState::new(
                sampled_hands,
                gs.current_trick.as_ref(),
                gs.current_player,
                gs.tricks_played(),
                gs.roles,
                gs.contract,
            );

            let move_values =
                double_dummy::solve_all_moves_viewer(&dd_state, viewer, objective);
            for (card, val) in &move_values {
                if let Some(idx) = legal_idx[card.0 as usize] {
                    local_sum[idx] += *val as i64;
                    local_cnt[idx] += 1;
                }
            }
            (local_sum, local_cnt)
        })
        .reduce(
            || (vec![0i64; n_legal], vec![0u32; n_legal]),
            |mut a, b| {
                for i in 0..n_legal {
                    a.0[i] += b.0[i];
                    a.1[i] += b.1[i];
                }
                a
            },
        );

    // Pick card with best average viewer utility (higher is always better).
    let best_idx = (0..n_legal)
        .max_by(|&a, &b| {
            let avg_a = if counts[a] > 0 {
                sum_i64[a] as f64 / counts[a] as f64
            } else {
                f64::NEG_INFINITY
            };
            let avg_b = if counts[b] > 0 {
                sum_i64[b] as f64 / counts[b] as f64
            } else {
                f64::NEG_INFINITY
            };
            avg_a
                .partial_cmp(&avg_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();

    legal_vec[best_idx]
}

fn objective_for_contract(contract: Option<Contract>) -> DDObjective {
    match contract {
        Some(Contract::Klop) => DDObjective::ViewerCardPoints,
        Some(Contract::Berac) => DDObjective::DeclarerTricks,
        _ => DDObjective::DeclarerTeamPoints,
    }
}

// -----------------------------------------------------------------------
// Constraint detection
// -----------------------------------------------------------------------

/// Detect per-player constraints from completed tricks and the current trick:
/// suit/tarok voids everywhere, plus rank ceilings when must-overplay is in
/// force (klop/berac).
pub fn detect_constraints(gs: &GameState) -> [PlayerConstraints; NUM_PLAYERS] {
    let mut constraints = [PlayerConstraints::new(); NUM_PLAYERS];
    let overplay = gs.contract.map_or(false, |c| c.requires_overplay());
    for trick in &gs.tricks {
        update_from_trick(trick, overplay, &mut constraints);
    }
    if let Some(ref trick) = gs.current_trick {
        update_from_trick(trick, overplay, &mut constraints);
    }
    constraints
}

fn update_from_trick(
    trick: &Trick,
    overplay: bool,
    constraints: &mut [PlayerConstraints; NUM_PLAYERS],
) {
    if trick.count == 0 {
        return;
    }
    let lead_card = trick.cards[0].1;
    let lead_is_tarok = lead_card.card_type() == CardType::Tarok;
    let lead_suit = lead_card.suit();
    // For trick-winning comparisons: when a tarok leads, lead_suit=None
    // is what `Card::beats` expects. Otherwise use the lead suit.
    let beats_lead_suit: Option<Suit> = if lead_is_tarok { None } else { lead_suit };

    // Running best card in the trick (used for overplay ceiling inference).
    let mut best_card = lead_card;

    for i in 1..trick.count as usize {
        let (player, card) = trick.cards[i];
        let pi = player as usize;

        // -- Void / ceiling inference relative to this card's play --
        if lead_is_tarok {
            if card.card_type() != CardType::Tarok {
                // Not following tarok → void in tarok.
                constraints[pi].void[TAROK_VOID_IDX] = true;
            } else if overplay {
                // In klop/berac: must beat current best tarok if possible.
                // If the played tarok does not beat `best_card`, the
                // player has no higher tarok than the one played.
                if !card.beats(best_card, None) {
                    constraints[pi].tighten_tarok_max(card.value());
                }
            }
        } else if let Some(ls) = lead_suit {
            if card.card_type() == CardType::Tarok {
                // Played a tarok on a suit lead → void in that suit.
                constraints[pi].void[ls as usize] = true;
            } else if card.suit() != Some(ls) {
                // Discarded off-suit (no taroks either) → void in both.
                constraints[pi].void[ls as usize] = true;
                constraints[pi].void[TAROK_VOID_IDX] = true;
            } else if overplay {
                // Followed the suit. In klop/berac the player must play
                // a higher card of the lead suit if they can.
                if !card.beats(best_card, Some(ls)) {
                    constraints[pi].tighten_suit_max(ls, card.value());
                }
            }
        }

        // Update running best.
        if card.beats(best_card, beats_lead_suit) {
            best_card = card;
        }
    }
}

// Back-compat alias for existing callers / tests that want just voids.
pub fn detect_voids(gs: &GameState) -> [[bool; VOID_CHANNELS]; NUM_PLAYERS] {
    let c = detect_constraints(gs);
    let mut voids = [[false; VOID_CHANNELS]; NUM_PLAYERS];
    for p in 0..NUM_PLAYERS {
        voids[p] = c[p].void;
    }
    voids
}

// -----------------------------------------------------------------------
// World sampling
// -----------------------------------------------------------------------

/// Redistribute unknown cards among other players, respecting `constraints`.
///
/// Returns `None` only when the card counts don't add up (should not happen
/// in a well-formed game state).
pub fn sample_world(
    gs: &GameState,
    viewer: u8,
    constraints: &[PlayerConstraints; NUM_PLAYERS],
    rng: &mut impl Rng,
) -> Option<[CardSet; NUM_PLAYERS]> {
    // Cards known to the viewer
    let mut known = gs
        .played_cards
        .union(gs.hands[viewer as usize])
        .union(gs.put_down)
        .union(gs.talon);
    if let Some(ref trick) = gs.current_trick {
        known = known.union(trick.played_cards_set());
    }

    let unknown_set = FULL_DECK.difference(known);
    let mut unknown: Vec<Card> = unknown_set.iter().collect();

    // Expected hand sizes — taken directly from the game state.
    let mut expected: [u32; NUM_PLAYERS] = [0; NUM_PLAYERS];
    let mut total_needed: u32 = 0;
    for p in 0..NUM_PLAYERS {
        expected[p] = gs.hands[p].len();
        if p as u8 != viewer {
            total_needed += expected[p];
        }
    }

    if total_needed != unknown.len() as u32 {
        return None; // sanity check
    }

    // Rejection sampling (cheap for ≤3 tricks / ≤9 unknown cards)
    for _ in 0..500 {
        unknown.shuffle(rng);
        if let Some(mut hands) = try_deal(&unknown, viewer, &expected, constraints) {
            hands[viewer as usize] = gs.hands[viewer as usize];
            return Some(hands);
        }
    }

    // Fallback: deal ignoring constraints
    unknown.shuffle(rng);
    let mut hands = [CardSet::EMPTY; NUM_PLAYERS];
    hands[viewer as usize] = gs.hands[viewer as usize];
    let mut idx = 0;
    for p in 0..NUM_PLAYERS {
        if p as u8 == viewer {
            continue;
        }
        let count = expected[p] as usize;
        for &card in &unknown[idx..idx + count] {
            hands[p].insert(card);
        }
        idx += count;
    }
    Some(hands)
}

/// Try to assign `unknown` cards to other players respecting constraints.
fn try_deal(
    unknown: &[Card],
    viewer: u8,
    expected: &[u32; NUM_PLAYERS],
    constraints: &[PlayerConstraints; NUM_PLAYERS],
) -> Option<[CardSet; NUM_PLAYERS]> {
    let mut hands = [CardSet::EMPTY; NUM_PLAYERS];
    let mut idx = 0;
    for p in 0..NUM_PLAYERS {
        if p as u8 == viewer {
            continue;
        }
        let count = expected[p] as usize;
        for &card in &unknown[idx..idx + count] {
            if !constraints[p].card_allowed(card) {
                return None;
            }
            hands[p].insert(card);
        }
        idx += count;
    }
    Some(hands)
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_voids_basic() {
        // Player 1 played a tarok when hearts was led → void in hearts
        let mut gs = GameState::new(0);
        gs.contract = Some(Contract::Three);
        let mut trick = Trick::new(0);
        trick.play(0, Card::suit_card(Suit::Hearts, SuitRank::King));
        trick.play(1, Card::tarok(5));
        trick.play(2, Card::suit_card(Suit::Hearts, SuitRank::Queen));
        trick.play(3, Card::suit_card(Suit::Hearts, SuitRank::Jack));
        gs.tricks.push(trick);

        let voids = detect_voids(&gs);
        assert!(voids[1][Suit::Hearts as usize]);
        assert!(!voids[0][Suit::Hearts as usize]);
        assert!(!voids[2][Suit::Hearts as usize]);
    }

    #[test]
    fn tarok_ceiling_in_klop() {
        // Klop: lead V (5), then XIII (13). Player 2 plays IV (4).
        // In klop the third player MUST overplay (XIII is the current
        // best). Playing IV means they had no tarok > XIII.
        let mut gs = GameState::new(0);
        gs.contract = Some(Contract::Klop);
        let mut trick = Trick::new(0);
        trick.play(0, Card::tarok(5));
        trick.play(1, Card::tarok(13));
        trick.play(2, Card::tarok(4));
        gs.current_trick = Some(trick);

        let c = detect_constraints(&gs);
        assert_eq!(c[2].tarok_max, 4);
        assert!(!c[2].card_allowed(Card::tarok(14)));
        assert!(c[2].card_allowed(Card::tarok(3)));
    }

    #[test]
    fn suit_ceiling_in_berac() {
        // Berač: hearts lead King (8), next plays Jack (5).
        // Must overplay → no heart higher than Jack.
        let mut gs = GameState::new(0);
        gs.contract = Some(Contract::Berac);
        let mut trick = Trick::new(0);
        trick.play(0, Card::suit_card(Suit::Hearts, SuitRank::King));
        trick.play(1, Card::suit_card(Suit::Hearts, SuitRank::Jack));
        gs.current_trick = Some(trick);

        let c = detect_constraints(&gs);
        assert_eq!(c[1].suit_max[Suit::Hearts as usize], 5);
        assert!(!c[1].card_allowed(Card::suit_card(Suit::Hearts, SuitRank::Queen)));
        assert!(c[1].card_allowed(Card::suit_card(Suit::Hearts, SuitRank::Jack)));
    }

    #[test]
    fn no_ceiling_in_normal_contract() {
        // Three: tarok lead V, next plays IV. Normal contracts don't force
        // overplay when following tarok (only void rules apply).
        let mut gs = GameState::new(0);
        gs.contract = Some(Contract::Three);
        let mut trick = Trick::new(0);
        trick.play(0, Card::tarok(5));
        trick.play(1, Card::tarok(4));
        gs.current_trick = Some(trick);

        let c = detect_constraints(&gs);
        assert_eq!(c[1].tarok_max, NO_CEILING);
        assert!(c[1].card_allowed(Card::tarok(20)));
    }
}
