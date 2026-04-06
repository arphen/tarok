/// Scoring rules for Slovenian Tarok.
///
/// Cards are counted in groups of 3: (sum of 3 cards) - 2.
/// Total game points = 70. Declarer wins with > 35 (≥ 36).

use crate::card::*;
use crate::game_state::*;
use crate::trick_eval::evaluate_trick;

pub const TOTAL_GAME_POINTS: i32 = 70;
pub const POINT_HALF: i32 = 35;

// Silent bonus values
const SILENT_TRULA: i32 = 10;
const SILENT_KINGS: i32 = 10;
const SILENT_PAGAT_ULTIMO: i32 = 25;
const SILENT_VALAT: i32 = 250;
// Announced bonus values
const ANNOUNCED_TRULA: i32 = 20;
const ANNOUNCED_KINGS: i32 = 20;
const ANNOUNCED_PAGAT_ULTIMO: i32 = 50;
const ANNOUNCED_VALAT: i32 = 500;

/// Count card points using the groups-of-3 method.
pub fn compute_card_points(cards: &[Card]) -> i32 {
    let raw: i32 = cards.iter().map(|c| c.points() as i32).sum();
    let n = cards.len() as i32;
    let deduction = (n / 3) * 2 + if n % 3 == 2 { 1 } else { 0 };
    raw - deduction
}

/// Compute card points for a CardSet.
pub fn compute_card_points_set(set: CardSet) -> i32 {
    let cards: Vec<Card> = set.iter().collect();
    compute_card_points(&cards)
}

/// Evaluate all tricks and return per-trick winners.
fn trick_winners(state: &GameState) -> Vec<u8> {
    let contract = state.contract;
    state
        .tricks
        .iter()
        .enumerate()
        .map(|(i, trick)| {
            let is_last = i == state.tricks.len() - 1;
            evaluate_trick(trick, is_last, contract).winner
        })
        .collect()
}

/// Collect cards won by each team.
fn collect_team_cards(state: &GameState, winners: &[u8]) -> (CardSet, CardSet) {
    let mut decl_cards = CardSet::EMPTY;
    let mut opp_cards = CardSet::EMPTY;
    for (i, trick) in state.tricks.iter().enumerate() {
        let winner_team = state.get_team(winners[i]);
        for j in 0..trick.count as usize {
            match winner_team {
                Team::DeclarerTeam => decl_cards.insert(trick.cards[j].1),
                Team::OpponentTeam => opp_cards.insert(trick.cards[j].1),
            }
        }
    }
    (decl_cards, opp_cards)
}

/// Check if team achieved pagat ultimo (won the last trick with Pagat).
fn pagat_ultimo(state: &GameState, team: Team, winners: &[u8]) -> bool {
    if state.tricks.is_empty() {
        return false;
    }
    let last_idx = state.tricks.len() - 1;
    let last_trick = &state.tricks[last_idx];
    let winner = winners[last_idx];
    if state.get_team(winner) != team {
        return false;
    }
    let pagat = Card::tarok(PAGAT);
    for i in 0..last_trick.count as usize {
        let (p, c) = last_trick.cards[i];
        if c == pagat {
            return state.get_team(p) == team;
        }
    }
    false
}

/// Get which team announced a given announcement, if any.
fn announced_by(state: &GameState, ann: Announcement) -> Option<Team> {
    let bit = 1u8 << (ann as u8);
    for p in 0..NUM_PLAYERS {
        if state.announcements[p] & bit != 0 {
            return Some(state.get_team(p as u8));
        }
    }
    None
}

// -----------------------------------------------------------------------
// Scoring entry point
// -----------------------------------------------------------------------

/// Compute final scores for all players. Returns [i32; 4].
pub fn score_game(state: &GameState) -> [i32; NUM_PLAYERS] {
    let contract = state.contract.expect("score_game called without contract");

    match contract {
        Contract::Klop => score_klop(state),
        Contract::Berac => score_berac(state),
        Contract::BarvniValat => score_barvni_valat(state),
        _ => score_normal(state, contract),
    }
}

fn score_klop(state: &GameState) -> [i32; NUM_PLAYERS] {
    let winners = trick_winners(state);
    let mut player_points = [0i32; NUM_PLAYERS];
    let mut player_tricks_won = [0u32; NUM_PLAYERS];

    for (i, trick) in state.tricks.iter().enumerate() {
        let w = winners[i] as usize;
        // Collect cards for this trick's winner
        let trick_cards: Vec<Card> = (0..trick.count as usize)
            .map(|j| trick.cards[j].1)
            .collect();
        player_points[w] += compute_card_points(&trick_cards);
        player_tricks_won[w] += 1;
    }

    let mut scores = [0i32; NUM_PLAYERS];
    for p in 0..NUM_PLAYERS {
        if player_points[p] > POINT_HALF {
            scores[p] = -TOTAL_GAME_POINTS;
        } else if player_tricks_won[p] == 0 {
            scores[p] = TOTAL_GAME_POINTS;
        } else {
            scores[p] = -player_points[p];
        }
    }
    scores
}

fn score_berac(state: &GameState) -> [i32; NUM_PLAYERS] {
    let declarer = state.declarer.expect("berac without declarer") as usize;
    let winners = trick_winners(state);
    let base = Contract::Berac.base_value(); // 70
    let declarer_trick_count = winners.iter().filter(|&&w| w as usize == declarer).count();

    let mut scores = [0i32; NUM_PLAYERS];
    if declarer_trick_count == 0 {
        scores[declarer] = base;
    } else {
        scores[declarer] = -base;
    }
    scores
}

fn score_barvni_valat(state: &GameState) -> [i32; NUM_PLAYERS] {
    let declarer = state.declarer.expect("barvni_valat without declarer") as usize;
    let winners = trick_winners(state);
    let mut base = Contract::BarvniValat.base_value(); // 125
    let all_won = winners.iter().all(|&w| w as usize == declarer);

    if !all_won {
        base = -base;
    }
    base *= state.kontra_multiplier(KontraTarget::Game);

    let mut scores = [0i32; NUM_PLAYERS];
    scores[declarer] = base;
    scores
}

fn score_normal(state: &GameState, contract: Contract) -> [i32; NUM_PLAYERS] {
    let declarer = state.declarer.expect("normal game without declarer");
    let winners = trick_winners(state);
    let (decl_card_set, opp_card_set) = collect_team_cards(state, &winners);

    // Add put-down cards to declarer pile
    let full_decl_set = decl_card_set.union(state.put_down);

    let decl_cards: Vec<Card> = full_decl_set.iter().collect();
    let _opp_cards: Vec<Card> = opp_card_set.iter().collect();

    let declarer_points = compute_card_points(&decl_cards);
    let declarer_won = declarer_points > POINT_HALF;

    let point_diff = (declarer_points - POINT_HALF).abs();
    let mut base_score = contract.base_value() + point_diff;
    if !declarer_won {
        base_score = -base_score;
    }
    base_score *= state.kontra_multiplier(KontraTarget::Game);

    // --- Bonuses ---
    let mut bonus = 0i32;

    // Trula
    let decl_has_trula = full_decl_set.has_trula();
    let opp_has_trula = opp_card_set.has_trula();
    let mut trula_bonus = 0i32;
    if let Some(ann_team) = announced_by(state, Announcement::Trula) {
        if ann_team == Team::DeclarerTeam {
            trula_bonus = if decl_has_trula {
                ANNOUNCED_TRULA
            } else {
                -ANNOUNCED_TRULA
            };
        } else {
            trula_bonus = if opp_has_trula {
                -ANNOUNCED_TRULA
            } else {
                ANNOUNCED_TRULA
            };
        }
        trula_bonus *= state.kontra_multiplier(KontraTarget::Trula);
    } else if decl_has_trula {
        trula_bonus = SILENT_TRULA;
    } else if opp_has_trula {
        trula_bonus = -SILENT_TRULA;
    }
    bonus += trula_bonus;

    // Kings
    let decl_has_kings = full_decl_set.has_all_kings();
    let opp_has_kings = opp_card_set.has_all_kings();
    let mut kings_bonus = 0i32;
    if let Some(ann_team) = announced_by(state, Announcement::Kings) {
        if ann_team == Team::DeclarerTeam {
            kings_bonus = if decl_has_kings {
                ANNOUNCED_KINGS
            } else {
                -ANNOUNCED_KINGS
            };
        } else {
            kings_bonus = if opp_has_kings {
                -ANNOUNCED_KINGS
            } else {
                ANNOUNCED_KINGS
            };
        }
        kings_bonus *= state.kontra_multiplier(KontraTarget::Kings);
    } else if decl_has_kings {
        kings_bonus = SILENT_KINGS;
    } else if opp_has_kings {
        kings_bonus = -SILENT_KINGS;
    }
    bonus += kings_bonus;

    // Pagat ultimo
    let decl_pagat = pagat_ultimo(state, Team::DeclarerTeam, &winners);
    let opp_pagat = pagat_ultimo(state, Team::OpponentTeam, &winners);
    let mut pagat_bonus = 0i32;
    if let Some(ann_team) = announced_by(state, Announcement::PagatUltimo) {
        if ann_team == Team::DeclarerTeam {
            pagat_bonus = if decl_pagat {
                ANNOUNCED_PAGAT_ULTIMO
            } else {
                -ANNOUNCED_PAGAT_ULTIMO
            };
        } else {
            pagat_bonus = if opp_pagat {
                -ANNOUNCED_PAGAT_ULTIMO
            } else {
                ANNOUNCED_PAGAT_ULTIMO
            };
        }
        pagat_bonus *= state.kontra_multiplier(KontraTarget::PagatUltimo);
    } else if decl_pagat {
        pagat_bonus = SILENT_PAGAT_ULTIMO;
    } else if opp_pagat {
        pagat_bonus = -SILENT_PAGAT_ULTIMO;
    }
    bonus += pagat_bonus;

    // Valat
    let mut valat_bonus = 0i32;
    if let Some(ann_team) = announced_by(state, Announcement::Valat) {
        let all_won = winners.iter().all(|&w| state.get_team(w) == ann_team);
        if ann_team == Team::DeclarerTeam {
            valat_bonus = if all_won {
                ANNOUNCED_VALAT
            } else {
                -ANNOUNCED_VALAT
            };
        } else {
            valat_bonus = if all_won {
                -ANNOUNCED_VALAT
            } else {
                ANNOUNCED_VALAT
            };
        }
        valat_bonus *= state.kontra_multiplier(KontraTarget::Valat);
    } else {
        let decl_all = winners.iter().all(|&w| state.get_team(w) == Team::DeclarerTeam);
        let opp_all = winners.iter().all(|&w| state.get_team(w) == Team::OpponentTeam);
        if decl_all {
            valat_bonus = SILENT_VALAT;
        } else if opp_all {
            valat_bonus = -SILENT_VALAT;
        }
    }
    bonus += valat_bonus;

    let total_declarer = base_score + bonus;

    // Distribute scores — only declarer team scores, opponents get 0
    let mut scores = [0i32; NUM_PLAYERS];
    for p in 0..NUM_PLAYERS {
        if state.get_team(p as u8) == Team::DeclarerTeam {
            scores[p] = total_declarer;
        }
    }
    scores
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn card_points_groups_of_3() {
        // 3 pip cards: 1+1+1 - 2 = 1
        let cards = vec![
            Card::suit_card(Suit::Hearts, SuitRank::Pip1),
            Card::suit_card(Suit::Hearts, SuitRank::Pip2),
            Card::suit_card(Suit::Hearts, SuitRank::Pip3),
        ];
        assert_eq!(compute_card_points(&cards), 1);
    }

    #[test]
    fn card_points_king_queen_knight() {
        // K(5) + Q(4) + C(3) - 2 = 10
        let cards = vec![
            Card::suit_card(Suit::Spades, SuitRank::King),
            Card::suit_card(Suit::Spades, SuitRank::Queen),
            Card::suit_card(Suit::Spades, SuitRank::Knight),
        ];
        assert_eq!(compute_card_points(&cards), 10);
    }

    #[test]
    fn total_deck_is_70() {
        let all: Vec<Card> = FULL_DECK.iter().collect();
        assert_eq!(compute_card_points(&all), 70);
    }
}
