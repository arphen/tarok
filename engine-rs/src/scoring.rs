/// Scoring rules for Slovenian Tarok.
///
/// Cards are counted in groups of 3: (sum of 3 cards) - 2.
/// Total game points = 70. Declarer wins with > 35 (≥ 36).
use crate::card::*;
use crate::game_state::*;
use crate::trick_eval::evaluate_trick;
use serde::Serialize;

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
        // Valat as a *contract* (3p only — in 4p it stays an announcement that
        // rides on top of another contract). All-or-nothing on declarer winning
        // every trick. Loses to the announced-Valat case which gives 500.
        Contract::Valat => score_valat_contract(state),
        _ => score_normal(state, contract),
    }
}

/// Compute per-player **training reward** signals (separate from leaderboard
/// scoring). The leaderboard scoring in [`score_game`] is locked at
/// "opponents=0"; this function produces a non-zero-sum reward signal so the
/// RL agent learns to defend actively instead of coasting to a guaranteed 0.
///
/// See [`score_normal_reward`] for the per-seat formula. For solo contracts
/// (Berač / Solo* / Barvni-valat) the declarer/partner split does not apply
/// (no partner on the declarer's team), so defenders receive ``−declarer_score``
/// directly. Klop is unchanged — every seat already has a meaningful per-player
/// score.
pub fn score_game_reward(state: &GameState) -> [i32; NUM_PLAYERS] {
    let contract = state
        .contract
        .expect("score_game_reward called without contract");

    match contract {
        Contract::Klop => score_klop(state),
        Contract::Berac => score_berac_reward(state),
        Contract::BarvniValat => score_barvni_valat_reward(state),
        Contract::Valat => score_valat_contract_reward(state),
        _ => score_normal_reward(state, contract),
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

    let n = state.num_players();
    let mut scores = [0i32; NUM_PLAYERS];
    for p in 0..n {
        if player_points[p] > POINT_HALF {
            scores[p] = -TOTAL_GAME_POINTS;
        } else if player_tricks_won[p] == 0 {
            scores[p] = TOTAL_GAME_POINTS;
        } else {
            scores[p] = -player_points[p];
        }
    }
    // Phantom seat 3 in 3p stays at 0.
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

/// Reward signal for Berač: defenders receive ``−declarer_score`` so losing to
/// a declarer-won Berač produces a punishing signal (and defeating a Berač
/// produces a positive signal).
fn score_berac_reward(state: &GameState) -> [i32; NUM_PLAYERS] {
    let scores = score_berac(state);
    let declarer = state.declarer.expect("berac without declarer") as usize;
    let defender_reward = -scores[declarer];
    let n = state.num_players();
    let mut out = [0i32; NUM_PLAYERS];
    for p in 0..n {
        out[p] = if p == declarer { scores[declarer] } else { defender_reward };
    }
    out
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

/// Reward signal for Barvni-valat: defenders receive ``−declarer_score``.
fn score_barvni_valat_reward(state: &GameState) -> [i32; NUM_PLAYERS] {
    let scores = score_barvni_valat(state);
    let declarer = state.declarer.expect("barvni_valat without declarer") as usize;
    let defender_reward = -scores[declarer];
    let n = state.num_players();
    let mut out = [0i32; NUM_PLAYERS];
    for p in 0..n {
        out[p] = if p == declarer { scores[declarer] } else { defender_reward };
    }
    out
}

/// Intermediate quantities produced by the normal-game scoring pipeline, before
/// seat-level distribution. Used by both [`score_normal`] (leaderboard scoring)
/// and [`score_normal_reward`] (training reward signal) so the core scoring
/// math is computed exactly once.
#[allow(dead_code)]
struct NormalTotals {
    /// Signed total for the declarer seat (contract + diff + bonuses, or valat).
    total_declarer: i32,
    /// Signed contract-base component (included in `total_declarer`). Zero when
    /// valat is achieved (valat replaces all scoring including the contract).
    /// Currently kept for documentation / future callers; the reward
    /// distribution uses `partner_total` which already accounts for the split.
    contract_base: i32,
    /// Signed total for the partner seat (``total_declarer - contract_base``
    /// outside valat; equals ``total_declarer`` when valat is achieved).
    partner_total: i32,
    declarer_won: bool,
    declarer_idx: usize,
    valat_achieved: bool,
}

fn compute_normal_totals(state: &GameState, contract: Contract) -> NormalTotals {
    let winners = trick_winners(state);
    let (decl_card_set, opp_card_set) = collect_team_cards(state, &winners);

    // Add put-down cards to declarer pile
    let full_decl_set = decl_card_set.union(state.put_down);

    // Unchosen talon cards go to opponent team
    let full_opp_set = opp_card_set.union(state.talon);

    let decl_cards: Vec<Card> = full_decl_set.iter().collect();
    let _opp_cards: Vec<Card> = full_opp_set.iter().collect();

    let declarer_points = compute_card_points(&decl_cards);
    let declarer_won = declarer_points > POINT_HALF;

    let point_diff = (declarer_points - POINT_HALF).abs();
    // Split the "game" portion into a contract-base component (awarded only
    // to the declarer) and a point-difference component (shared with the
    // partner). Bonuses (trula/kings/pagat/valat) are shared, always.
    // Rationale: the contract value is a reward for *bidding*, which only
    // the declarer did. Point-diff and bonuses reward *play*, which the
    // partner contributed to — sharing them encourages the partner seat to
    // play actively instead of coasting on the contract base.
    let sign = if declarer_won { 1 } else { -1 };
    let km_game = state.kontra_multiplier(KontraTarget::Game);
    // Variant-aware contract base value. 3p uses a compressed table
    // (see Contract::base_value_for in game_state.rs).
    let contract_base = sign * contract.base_value_for(state.variant) * km_game;
    let point_diff_score = sign * point_diff * km_game;
    let base_score = contract_base + point_diff_score;

    // --- Bonuses ---
    let mut bonus = 0i32;

    // Trula
    let decl_has_trula = full_decl_set.has_trula();
    let opp_has_trula = full_opp_set.has_trula();
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
    let opp_has_kings = full_opp_set.has_all_kings();
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

    // Valat — when achieved, replaces ALL other scoring (base game + bonuses).
    // Only the valat value applies (250 silent / 500 announced), with kontra.
    let decl_all = winners
        .iter()
        .all(|&w| state.get_team(w) == Team::DeclarerTeam);
    let opp_all = winners
        .iter()
        .all(|&w| state.get_team(w) == Team::OpponentTeam);
    let valat_achieved = decl_all || opp_all;

    let total_declarer;
    if valat_achieved {
        if let Some(ann_team) = announced_by(state, Announcement::Valat) {
            let valat_sign = if (ann_team == Team::DeclarerTeam) == decl_all {
                1
            } else {
                -1
            };
            total_declarer =
                valat_sign * ANNOUNCED_VALAT * state.kontra_multiplier(KontraTarget::Valat);
        } else {
            let valat_base = if decl_all {
                SILENT_VALAT
            } else {
                -SILENT_VALAT
            };
            total_declarer = valat_base * state.kontra_multiplier(KontraTarget::Valat);
        }
    } else {
        // Valat not achieved — check for failed announcement penalty
        let mut valat_bonus = 0i32;
        if let Some(ann_team) = announced_by(state, Announcement::Valat) {
            // Announced but failed
            if ann_team == Team::DeclarerTeam {
                valat_bonus = -ANNOUNCED_VALAT;
            } else {
                valat_bonus = ANNOUNCED_VALAT;
            }
            valat_bonus *= state.kontra_multiplier(KontraTarget::Valat);
        }
        bonus += valat_bonus;
        total_declarer = base_score + bonus;
    }

    // Distribute scores — only declarer team scores, opponents get 0.
    // The declarer gets the full total; the partner gets the total minus
    // the contract-base component (declarer-only). Valat replaces all
    // scoring, so both share the full valat value in that branch.
    let partner_total = if valat_achieved {
        total_declarer
    } else {
        total_declarer - contract_base
    };
    let declarer_idx = state.declarer.expect("normal game without declarer") as usize;
    NormalTotals {
        total_declarer,
        contract_base: if valat_achieved { 0 } else { contract_base },
        partner_total,
        declarer_won,
        declarer_idx,
        valat_achieved,
    }
}

fn score_normal(state: &GameState, contract: Contract) -> [i32; NUM_PLAYERS] {
    let t = compute_normal_totals(state, contract);
    let mut scores = [0i32; NUM_PLAYERS];
    for p in 0..NUM_PLAYERS {
        if state.get_team(p as u8) != Team::DeclarerTeam {
            continue;
        }
        scores[p] = if p == t.declarer_idx {
            t.total_declarer
        } else {
            t.partner_total
        };
    }
    scores
}

/// Reward-signal distribution for normal games.
///
/// Unlike [`score_normal`] (leaderboard scoring where opponents get 0), this
/// function produces a training reward signal that punishes the defender seats
/// when the declarer wins and rewards them when the declarer fails. See
/// `.github/copilot-instructions.md` — the *leaderboard* scoring is locked at
/// opponents=0; this is a separate signal fed only into the RL optimizer.
///
/// Per-seat reward (let `total = total_declarer`, `C = contract_base`):
/// - Declarer: `total`
/// - Partner:  `total − C`  (same as leaderboard)
/// - Each defender:
///     - declarer won  → `−(total − C)` = ``−partner_total``
///     - declarer lost → `−total`
///
/// Valat: `C` is 0 in that branch, so the two cases collapse to
/// `defender = −total_declarer`, which is the intended symmetry (the full valat
/// magnitude flows to both sides).
fn score_normal_reward(state: &GameState, contract: Contract) -> [i32; NUM_PLAYERS] {
    let t = compute_normal_totals(state, contract);
    let defender_reward = if t.declarer_won {
        -t.partner_total
    } else {
        -t.total_declarer
    };
    let n = state.num_players();
    let mut scores = [0i32; NUM_PLAYERS];
    for p in 0..n {
        if state.get_team(p as u8) == Team::DeclarerTeam {
            scores[p] = if p == t.declarer_idx {
                t.total_declarer
            } else {
                t.partner_total
            };
        } else {
            scores[p] = defender_reward;
        }
    }
    scores
}

/// Valat as a *bid contract* (3-player only). All-or-nothing: declarer must
/// win every trick. Distinct from the announced-Valat bonus that rides on top
/// of a normal contract (handled inside `compute_normal_totals`). Mirrors
/// `score_barvni_valat` structurally; uses `Contract::Valat.base_value_for(variant)`.
fn score_valat_contract(state: &GameState) -> [i32; NUM_PLAYERS] {
    let declarer = state.declarer.expect("valat without declarer") as usize;
    let winners = trick_winners(state);
    let mut base = Contract::Valat.base_value_for(state.variant);
    let all_won = winners.iter().all(|&w| w as usize == declarer);
    if !all_won {
        base = -base;
    }
    base *= state.kontra_multiplier(KontraTarget::Game);
    let mut scores = [0i32; NUM_PLAYERS];
    scores[declarer] = base;
    scores
}

fn score_valat_contract_reward(state: &GameState) -> [i32; NUM_PLAYERS] {
    let scores = score_valat_contract(state);
    let declarer = state.declarer.expect("valat without declarer") as usize;
    let defender_reward = -scores[declarer];
    let n = state.num_players();
    let mut out = [0i32; NUM_PLAYERS];
    for p in 0..n {
        out[p] = if p == declarer { scores[declarer] } else { defender_reward };
    }
    out
}

// -----------------------------------------------------------------------
// Scoring breakdown — structured data for the UI display layer
// -----------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct BreakdownLine {
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct TrickSummaryCard {
    pub player: u8,
    pub label: String,
    pub points: u8,
}

#[derive(Debug, Serialize)]
pub struct TrickSummaryEntry {
    pub trick_num: usize,
    pub lead_player: u8,
    pub cards: Vec<TrickSummaryCard>,
    pub winner: u8,
    pub card_points: i32,
}

#[derive(Debug, Serialize)]
pub struct ScoreBreakdown {
    pub contract: String,
    pub mode: String,
    pub declarer_won: Option<bool>,
    pub declarer_points: Option<i32>,
    pub opponent_points: Option<i32>,
    pub lines: Vec<BreakdownLine>,
    pub scores: [i32; NUM_PLAYERS],
    pub trick_summary: Vec<TrickSummaryEntry>,
}

fn contract_label(c: Contract) -> &'static str {
    match c {
        Contract::Klop => "Klop",
        Contract::Three => "Three",
        Contract::Two => "Two",
        Contract::One => "One",
        Contract::SoloThree => "Solo Three",
        Contract::SoloTwo => "Solo Two",
        Contract::SoloOne => "Solo One",
        Contract::Solo => "Solo",
        Contract::Berac => "Berač",
        Contract::BarvniValat => "Barvni Valat",
        Contract::Valat => "Valat",
    }
}

fn contract_mode(c: Contract) -> &'static str {
    if c.is_klop() {
        "klop"
    } else if c.is_solo() || c.is_berac() || c.is_barvni_valat() {
        "solo"
    } else {
        "2v2"
    }
}

fn build_trick_summary(state: &GameState) -> Vec<TrickSummaryEntry> {
    let contract = state.contract;
    state
        .tricks
        .iter()
        .enumerate()
        .map(|(i, trick)| {
            let is_last = i == state.tricks.len() - 1;
            let result = evaluate_trick(trick, is_last, contract);
            let cards: Vec<TrickSummaryCard> = (0..trick.count as usize)
                .map(|j| {
                    let (player, card) = trick.cards[j];
                    TrickSummaryCard {
                        player,
                        label: card.label(),
                        points: card.points(),
                    }
                })
                .collect();
            let trick_cards: Vec<Card> = (0..trick.count as usize)
                .map(|j| trick.cards[j].1)
                .collect();
            TrickSummaryEntry {
                trick_num: i + 1,
                lead_player: trick.lead_player,
                cards,
                winner: result.winner,
                card_points: compute_card_points(&trick_cards),
            }
        })
        .collect()
}

/// Full scoring breakdown for UI display. Returns all intermediate values
/// so Python never re-derives scoring decisions.
pub fn score_game_breakdown(state: &GameState) -> ScoreBreakdown {
    let contract = state
        .contract
        .expect("score_game_breakdown without contract");
    let scores = score_game(state);
    let trick_summary = build_trick_summary(state);

    match contract {
        Contract::Klop => breakdown_klop(state, scores, trick_summary),
        Contract::Berac => breakdown_berac(state, contract, scores, trick_summary),
        Contract::BarvniValat => breakdown_barvni_valat(state, contract, scores, trick_summary),
        _ => breakdown_normal(state, contract, scores, trick_summary),
    }
}

fn breakdown_klop(
    state: &GameState,
    scores: [i32; NUM_PLAYERS],
    trick_summary: Vec<TrickSummaryEntry>,
) -> ScoreBreakdown {
    let winners = trick_winners(state);
    let mut lines = Vec::new();

    for p in 0..NUM_PLAYERS {
        let trick_cards: Vec<Card> = state
            .tricks
            .iter()
            .enumerate()
            .filter(|(i, _)| winners[*i] as usize == p)
            .flat_map(|(_, t)| (0..t.count as usize).map(move |j| t.cards[j].1))
            .collect();
        let pts = compute_card_points(&trick_cards);
        let tricks_won = winners.iter().filter(|&&w| w as usize == p).count();
        let detail = if tricks_won == 0 {
            format!("won 0 tricks → +{}", TOTAL_GAME_POINTS)
        } else if pts > POINT_HALF {
            format!("{} pts (>{}) → −{}", pts, POINT_HALF, TOTAL_GAME_POINTS)
        } else {
            format!("{} pts in {} trick(s)", pts, tricks_won)
        };
        lines.push(BreakdownLine {
            label: format!("Player {}", p),
            value: Some(scores[p]),
            detail: Some(detail),
        });
    }

    ScoreBreakdown {
        contract: contract_label(Contract::Klop).to_string(),
        mode: "klop".to_string(),
        declarer_won: None,
        declarer_points: None,
        opponent_points: None,
        lines,
        scores,
        trick_summary,
    }
}

fn breakdown_berac(
    state: &GameState,
    contract: Contract,
    scores: [i32; NUM_PLAYERS],
    trick_summary: Vec<TrickSummaryEntry>,
) -> ScoreBreakdown {
    let declarer = state.declarer.expect("berac without declarer") as usize;
    let winners = trick_winners(state);
    let decl_tricks = winners.iter().filter(|&&w| w as usize == declarer).count();
    let won = decl_tricks == 0;

    let mut lines = vec![
        BreakdownLine {
            label: "Contract".to_string(),
            value: Some(contract.base_value()),
            detail: Some(contract_label(contract).to_string()),
        },
        BreakdownLine {
            label: if won {
                "Declarer won all tricks avoided"
            } else {
                "Declarer won a trick — loss"
            }
            .to_string(),
            value: Some(scores[declarer]),
            detail: Some(format!("Declarer tricks: {}", decl_tricks)),
        },
    ];

    // Kontra
    let km = state.kontra_multiplier(KontraTarget::Game);
    if km > 1 {
        lines.push(BreakdownLine {
            label: "Kontra multiplier".to_string(),
            value: Some(km),
            detail: None,
        });
    }

    ScoreBreakdown {
        contract: contract_label(contract).to_string(),
        mode: "solo".to_string(),
        declarer_won: Some(won),
        declarer_points: None,
        opponent_points: None,
        lines,
        scores,
        trick_summary,
    }
}

fn breakdown_barvni_valat(
    state: &GameState,
    contract: Contract,
    scores: [i32; NUM_PLAYERS],
    trick_summary: Vec<TrickSummaryEntry>,
) -> ScoreBreakdown {
    let declarer = state.declarer.expect("barvni_valat without declarer") as usize;
    let winners = trick_winners(state);
    let all_won = winners.iter().all(|&w| w as usize == declarer);

    let mut lines = vec![
        BreakdownLine {
            label: "Contract".to_string(),
            value: Some(contract.base_value()),
            detail: Some(contract_label(contract).to_string()),
        },
        BreakdownLine {
            label: if all_won {
                "All tricks won"
            } else {
                "Failed — didn't win all tricks"
            }
            .to_string(),
            value: Some(scores[declarer]),
            detail: None,
        },
    ];

    let km = state.kontra_multiplier(KontraTarget::Game);
    if km > 1 {
        lines.push(BreakdownLine {
            label: "Kontra multiplier".to_string(),
            value: Some(km),
            detail: None,
        });
    }

    ScoreBreakdown {
        contract: contract_label(contract).to_string(),
        mode: "solo".to_string(),
        declarer_won: Some(all_won),
        declarer_points: None,
        opponent_points: None,
        lines,
        scores,
        trick_summary,
    }
}

fn breakdown_normal(
    state: &GameState,
    contract: Contract,
    scores: [i32; NUM_PLAYERS],
    trick_summary: Vec<TrickSummaryEntry>,
) -> ScoreBreakdown {
    let _declarer = state.declarer.expect("normal game without declarer");
    let winners = trick_winners(state);
    let (decl_card_set, opp_card_set) = collect_team_cards(state, &winners);

    let full_decl_set = decl_card_set.union(state.put_down);
    let full_opp_set = opp_card_set.union(state.talon);

    let decl_cards: Vec<Card> = full_decl_set.iter().collect();
    let opp_cards: Vec<Card> = full_opp_set.iter().collect();

    let declarer_points = compute_card_points(&decl_cards);
    let opponent_points = compute_card_points(&opp_cards);
    let declarer_won = declarer_points > POINT_HALF;

    let point_diff = (declarer_points - POINT_HALF).abs();
    let base = contract.base_value();

    let mut lines = Vec::new();
    lines.push(BreakdownLine {
        label: "Contract".to_string(),
        value: Some(base),
        detail: Some(contract_label(contract).to_string()),
    });
    lines.push(BreakdownLine {
        label: "Card points (declarer)".to_string(),
        value: Some(declarer_points),
        detail: Some(format!(
            "{} by {}",
            if declarer_won { "Won" } else { "Lost" },
            point_diff
        )),
    });

    let sign = if declarer_won { 1 } else { -1 };
    let km_game = state.kontra_multiplier(KontraTarget::Game);
    let contract_component = sign * base * km_game;
    let point_diff_component = sign * point_diff * km_game;
    lines.push(BreakdownLine {
        label: "Contract score (declarer only)".to_string(),
        value: Some(contract_component),
        detail: if km_game > 1 {
            Some(format!("{} × kontra {}", sign * base, km_game))
        } else {
            None
        },
    });
    lines.push(BreakdownLine {
        label: "Point-diff score (shared)".to_string(),
        value: Some(point_diff_component),
        detail: if km_game > 1 {
            Some(format!("{} × kontra {}", sign * point_diff, km_game))
        } else {
            None
        },
    });

    // Trula
    let decl_has_trula = full_decl_set.has_trula();
    let opp_has_trula = full_opp_set.has_trula();
    if let Some(ann_team) = announced_by(state, Announcement::Trula) {
        let succeeded = if ann_team == Team::DeclarerTeam {
            decl_has_trula
        } else {
            opp_has_trula
        };
        let km = state.kontra_multiplier(KontraTarget::Trula);
        let raw = if (ann_team == Team::DeclarerTeam) == succeeded {
            ANNOUNCED_TRULA
        } else {
            -ANNOUNCED_TRULA
        };
        lines.push(BreakdownLine {
            label: "Trula (announced)".to_string(),
            value: Some(raw * km),
            detail: Some(format!("{}", if succeeded { "achieved" } else { "failed" })),
        });
    } else if decl_has_trula {
        lines.push(BreakdownLine {
            label: "Trula (silent)".to_string(),
            value: Some(SILENT_TRULA),
            detail: Some("declarer team".to_string()),
        });
    } else if opp_has_trula {
        lines.push(BreakdownLine {
            label: "Trula (silent)".to_string(),
            value: Some(-SILENT_TRULA),
            detail: Some("opponent team".to_string()),
        });
    }

    // Kings
    let decl_has_kings = full_decl_set.has_all_kings();
    let opp_has_kings = full_opp_set.has_all_kings();
    if let Some(ann_team) = announced_by(state, Announcement::Kings) {
        let succeeded = if ann_team == Team::DeclarerTeam {
            decl_has_kings
        } else {
            opp_has_kings
        };
        let km = state.kontra_multiplier(KontraTarget::Kings);
        let raw = if (ann_team == Team::DeclarerTeam) == succeeded {
            ANNOUNCED_KINGS
        } else {
            -ANNOUNCED_KINGS
        };
        lines.push(BreakdownLine {
            label: "Kings (announced)".to_string(),
            value: Some(raw * km),
            detail: Some(format!("{}", if succeeded { "achieved" } else { "failed" })),
        });
    } else if decl_has_kings {
        lines.push(BreakdownLine {
            label: "Kings (silent)".to_string(),
            value: Some(SILENT_KINGS),
            detail: Some("declarer team".to_string()),
        });
    } else if opp_has_kings {
        lines.push(BreakdownLine {
            label: "Kings (silent)".to_string(),
            value: Some(-SILENT_KINGS),
            detail: Some("opponent team".to_string()),
        });
    }

    // Pagat ultimo
    let decl_pagat = pagat_ultimo(state, Team::DeclarerTeam, &winners);
    let opp_pagat = pagat_ultimo(state, Team::OpponentTeam, &winners);
    if let Some(ann_team) = announced_by(state, Announcement::PagatUltimo) {
        let succeeded = if ann_team == Team::DeclarerTeam {
            decl_pagat
        } else {
            opp_pagat
        };
        let km = state.kontra_multiplier(KontraTarget::PagatUltimo);
        let raw = if (ann_team == Team::DeclarerTeam) == succeeded {
            ANNOUNCED_PAGAT_ULTIMO
        } else {
            -ANNOUNCED_PAGAT_ULTIMO
        };
        lines.push(BreakdownLine {
            label: "Pagat ultimo (announced)".to_string(),
            value: Some(raw * km),
            detail: Some(format!("{}", if succeeded { "achieved" } else { "failed" })),
        });
    } else if decl_pagat {
        lines.push(BreakdownLine {
            label: "Pagat ultimo (silent)".to_string(),
            value: Some(SILENT_PAGAT_ULTIMO),
            detail: Some("declarer team".to_string()),
        });
    } else if opp_pagat {
        lines.push(BreakdownLine {
            label: "Pagat ultimo (silent)".to_string(),
            value: Some(-SILENT_PAGAT_ULTIMO),
            detail: Some("opponent team".to_string()),
        });
    }

    // Valat
    let decl_all = winners
        .iter()
        .all(|&w| state.get_team(w) == Team::DeclarerTeam);
    let opp_all = winners
        .iter()
        .all(|&w| state.get_team(w) == Team::OpponentTeam);
    let valat_achieved = decl_all || opp_all;

    if valat_achieved {
        if let Some(ann_team) = announced_by(state, Announcement::Valat) {
            let succeeded = if ann_team == Team::DeclarerTeam {
                decl_all
            } else {
                opp_all
            };
            let km = state.kontra_multiplier(KontraTarget::Valat);
            let raw = if (ann_team == Team::DeclarerTeam) == succeeded {
                ANNOUNCED_VALAT
            } else {
                -ANNOUNCED_VALAT
            };
            lines.push(BreakdownLine {
                label: "VALAT (announced) — replaces all other scoring".to_string(),
                value: Some(raw * km),
                detail: None,
            });
        } else {
            let raw = if decl_all {
                SILENT_VALAT
            } else {
                -SILENT_VALAT
            };
            let km = state.kontra_multiplier(KontraTarget::Valat);
            lines.push(BreakdownLine {
                label: "VALAT (silent) — replaces all other scoring".to_string(),
                value: Some(raw * km),
                detail: Some(format!(
                    "{} team won all tricks",
                    if decl_all { "Declarer" } else { "Opponent" }
                )),
            });
        }
    } else if let Some(ann_team) = announced_by(state, Announcement::Valat) {
        let km = state.kontra_multiplier(KontraTarget::Valat);
        let raw = if ann_team == Team::DeclarerTeam {
            -ANNOUNCED_VALAT
        } else {
            ANNOUNCED_VALAT
        };
        lines.push(BreakdownLine {
            label: "Valat (announced, failed)".to_string(),
            value: Some(raw * km),
            detail: None,
        });
    }

    ScoreBreakdown {
        contract: contract_label(contract).to_string(),
        mode: contract_mode(contract).to_string(),
        declarer_won: Some(declarer_won),
        declarer_points: Some(declarer_points),
        opponent_points: Some(opponent_points),
        lines,
        scores,
        trick_summary,
    }
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
