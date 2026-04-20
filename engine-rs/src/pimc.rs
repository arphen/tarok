/// PIMC (Perfect Information Monte Carlo) card selection.
///
/// Samples `N` consistent worlds by redistributing unknown cards among
/// opponents (respecting known suit voids), double-dummy solves each world,
/// and picks the legal move with the best average outcome.
use crate::card::*;
use crate::double_dummy::{self, DDState};
use crate::game_state::*;
use crate::legal_moves;
use rand::prelude::*;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;

/// Default number of worlds to sample.
pub const DEFAULT_NUM_WORLDS: u32 = 100;

/// Choose the best card for `viewer` using PIMC.
///
/// Panics (debug) if no legal moves exist.
pub fn pimc_choose_card(gs: &GameState, viewer: u8, num_worlds: u32) -> Card {
    let viewer_team = gs.get_team(viewer);
    let maximizing = viewer_team == Team::DeclarerTeam;

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

    let voids = detect_voids(gs);
    // O(1) map from card id to legal index in `scores`.
    let mut legal_idx: [Option<usize>; DECK_SIZE] = [None; DECK_SIZE];
    for (i, card) in legal_vec.iter().enumerate() {
        legal_idx[card.0 as usize] = Some(i);
    }

    let base_seed: u64 = rand::rng().random();

    // Parallel over worlds, then reduce into per-card aggregates:
    // (total_dd_value, sample_count) per legal card.
    let scores: Vec<(f64, u32)> = (0..num_worlds)
        .into_par_iter()
        .map(|world_i| {
            let mut local = vec![(0.0, 0u32); legal_vec.len()];
            let mut rng = SmallRng::seed_from_u64(base_seed ^ (world_i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));

            let sampled_hands = match sample_world(gs, viewer, &voids, &mut rng) {
                Some(h) => h,
                None => return local,
            };

            let dd_state = DDState::new(
                sampled_hands,
                gs.current_trick.as_ref(),
                gs.current_player,
                gs.tricks_played(),
                gs.roles,
                gs.contract,
            );

            let move_values = double_dummy::solve_all_moves(&dd_state);
            for (card, val) in &move_values {
                if let Some(idx) = legal_idx[card.0 as usize] {
                    local[idx].0 += *val as f64;
                    local[idx].1 += 1;
                }
            }
            local
        })
        .reduce(
            || vec![(0.0, 0u32); legal_vec.len()],
            |mut acc, local| {
                for i in 0..acc.len() {
                    acc[i].0 += local[i].0;
                    acc[i].1 += local[i].1;
                }
                acc
            },
        );

    // Pick card with best average value
    let best_idx = scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            let avg_a = if a.1 > 0 {
                a.0 / a.1 as f64
            } else if maximizing {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
            let avg_b = if b.1 > 0 {
                b.0 / b.1 as f64
            } else if maximizing {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
            if maximizing {
                avg_a
                    .partial_cmp(&avg_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            } else {
                avg_b
                    .partial_cmp(&avg_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        })
        .map(|(i, _)| i)
        .unwrap();

    legal_vec[best_idx]
}

// -----------------------------------------------------------------------
// Void detection
// -----------------------------------------------------------------------

/// Detect known suit voids from completed tricks and the current trick.
/// `voids[player][suit]` is true when `player` failed to follow `suit`.
fn detect_voids(gs: &GameState) -> [[bool; 4]; NUM_PLAYERS] {
    let mut voids = [[false; 4]; NUM_PLAYERS];
    record_voids_from_tricks(&gs.tricks, &mut voids);
    if let Some(ref trick) = gs.current_trick {
        record_voids_from_trick(trick, &mut voids);
    }
    voids
}

fn record_voids_from_tricks(tricks: &[Trick], voids: &mut [[bool; 4]; NUM_PLAYERS]) {
    for trick in tricks {
        record_voids_from_trick(trick, voids);
    }
}

fn record_voids_from_trick(trick: &Trick, voids: &mut [[bool; 4]; NUM_PLAYERS]) {
    if trick.count == 0 {
        return;
    }
    let lead_card = trick.cards[0].1;
    if lead_card.card_type() == CardType::Tarok {
        return; // tarok lead — no suit-void inference
    }
    let lead_suit = match lead_card.suit() {
        Some(s) => s,
        None => return,
    };
    for i in 1..trick.count as usize {
        let (player, card) = trick.cards[i];
        if card.card_type() == CardType::Tarok || card.suit() != Some(lead_suit) {
            voids[player as usize][lead_suit as usize] = true;
        }
    }
}

// -----------------------------------------------------------------------
// World sampling
// -----------------------------------------------------------------------

/// Redistribute unknown cards among other players, respecting `voids`.
///
/// Returns `None` only when the card counts don't add up (should not happen
/// in a well-formed game state).
fn sample_world(
    gs: &GameState,
    viewer: u8,
    voids: &[[bool; 4]; NUM_PLAYERS],
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
        if let Some(mut hands) = try_deal(&unknown, viewer, &expected, voids) {
            hands[viewer as usize] = gs.hands[viewer as usize];
            return Some(hands);
        }
    }

    // Fallback: deal ignoring void constraints
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

/// Try to assign `unknown` cards to other players respecting void constraints.
fn try_deal(
    unknown: &[Card],
    viewer: u8,
    expected: &[u32; NUM_PLAYERS],
    voids: &[[bool; 4]; NUM_PLAYERS],
) -> Option<[CardSet; NUM_PLAYERS]> {
    let mut hands = [CardSet::EMPTY; NUM_PLAYERS];
    let mut idx = 0;
    for p in 0..NUM_PLAYERS {
        if p as u8 == viewer {
            continue;
        }
        let count = expected[p] as usize;
        for &card in &unknown[idx..idx + count] {
            if let Some(suit) = card.suit() {
                if voids[p][suit as usize] {
                    return None;
                }
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
}
