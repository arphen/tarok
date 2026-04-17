/// StockŠkis v5 — strongest heuristic bot, combining all v1–v4 lessons
/// with major bidding overhaul, belief tracking, and smarter trick play.
///
/// Key improvements over v4:
/// - **Bidding overhaul**: Much less aggressive solo bidding — solos require
///   guaranteed points (kings + škis/mond) > 30.  No solo with fewer than
///   4 taroks.  Tighter thresholds across the board.
/// - **Berač gating**: No berač with any tarok ≥ 9, škis, void suits,
///   singletons, or more than 2 taroks.  Kings allowed only if 5+ cards
///   in that suit.
/// - **Belief tracking**: Infer opponent hand composition from observed
///   plays and voids — track tarok probability per-player to make smarter
///   following decisions.
/// - **Last-seat economy**: When last to play and winning, play the
///   *lowest* tarok that still wins the trick.
/// - **Partner preservation**: Don't overtrump partner when it doesn't
///   change who wins the trick.
/// - **Opposition coordination**: Defenders avoid wasting high taroks
///   and lead through declarer's weak suits.

use crate::card::*;
use crate::game_state::*;
use crate::legal_moves;

// -----------------------------------------------------------------------
// Card tracking — enhanced with belief estimation
// -----------------------------------------------------------------------

#[allow(dead_code)]
struct CardTracker {
    remaining: CardSet,
    taroks_in_hand: u8,
    taroks_remaining: CardSet,
    suit_counts: [u8; 4],
    player_voids: [[bool; 4]; NUM_PLAYERS],
    /// Estimated probability that each opponent still has taroks (0.0–1.0).
    player_tarok_likelihood: [f64; NUM_PLAYERS],
    tricks_left: usize,
    phase: GamePhase,
    /// Points already secured by declarer team.
    declarer_team_points: i32,
    /// Points already secured by opposition.
    opposition_points: i32,
}

#[derive(Clone, Copy, PartialEq)]
enum GamePhase {
    Early,
    Mid,
    Late,
}

impl CardTracker {
    fn from_state(state: &GameState, player: u8) -> Self {
        let hand = state.hands[player as usize];

        let mut played = state.played_cards;
        if let Some(ref trick) = state.current_trick {
            for i in 0..trick.count as usize {
                played.insert(trick.cards[i].1);
            }
        }
        played = played.union(state.put_down);

        let full_deck = CardSet((1u64 << DECK_SIZE) - 1);
        let remaining = full_deck.difference(played).difference(hand);
        let taroks_remaining = remaining.taroks();

        let mut suit_counts = [0u8; 4];
        for suit in Suit::ALL {
            suit_counts[suit as usize] = hand.suit_count(suit) as u8;
        }

        let mut player_voids = [[false; 4]; NUM_PLAYERS];
        // Track tarok-related plays per opponent
        let mut player_played_tarok = [false; NUM_PLAYERS];
        let mut player_followed_suit_when_could = [0u32; NUM_PLAYERS];
        let mut player_total_follows = [0u32; NUM_PLAYERS];

        for trick in &state.tricks {
            if trick.count == 0 {
                continue;
            }
            let lead_card = trick.cards[0].1;
            if let Some(lead_suit) = lead_card.suit() {
                for i in 1..trick.count as usize {
                    let (p, c) = trick.cards[i];
                    if p == player {
                        continue;
                    }
                    player_total_follows[p as usize] += 1;
                    if c.card_type() == CardType::Tarok {
                        player_voids[p as usize][lead_suit as usize] = true;
                        player_played_tarok[p as usize] = true;
                    } else if c.suit() == Some(lead_suit) {
                        player_followed_suit_when_could[p as usize] += 1;
                    }
                }
            } else {
                // Lead was tarok
                for i in 1..trick.count as usize {
                    let (p, c) = trick.cards[i];
                    if p == player {
                        continue;
                    }
                    if c.card_type() == CardType::Tarok {
                        player_played_tarok[p as usize] = true;
                    }
                }
            }
        }

        // Estimate tarok likelihood per opponent
        let mut player_tarok_likelihood = [0.8f64; NUM_PLAYERS];
        let total_taroks_remaining = taroks_remaining.len() as f64;
        let opponents_count = (NUM_PLAYERS - 1) as f64;
        let avg_taroks_per_opponent = if opponents_count > 0.0 {
            total_taroks_remaining / opponents_count
        } else {
            0.0
        };

        for p in 0..NUM_PLAYERS {
            if p as u8 == player {
                player_tarok_likelihood[p] = 0.0;
                continue;
            }
            // If they played suit card when tarok was led and they had to
            // follow with tarok, they are tarok-void
            let mut tarok_void = false;
            for trick in &state.tricks {
                if trick.count == 0 {
                    continue;
                }
                let lead_card = trick.cards[0].1;
                if lead_card.card_type() == CardType::Tarok {
                    for i in 1..trick.count as usize {
                        let (tp, tc) = trick.cards[i];
                        if tp as usize == p && tc.card_type() != CardType::Tarok {
                            tarok_void = true;
                        }
                    }
                }
            }

            if tarok_void {
                player_tarok_likelihood[p] = 0.0;
            } else if total_taroks_remaining == 0.0 {
                player_tarok_likelihood[p] = 0.0;
            } else {
                // Base likelihood from remaining distribution
                let base = (avg_taroks_per_opponent / 4.0).min(1.0);
                player_tarok_likelihood[p] = base;
            }
        }

        // Compute points accumulated
        let mut declarer_team_points = 0i32;
        let mut opposition_points = 0i32;
        for trick in &state.tricks {
            if trick.count < 4 {
                continue;
            }
            let lead_suit = trick.lead_suit();
            let mut winner = trick.cards[0].0;
            let mut best = trick.cards[0].1;
            let mut trick_pts = 0i32;
            for i in 0..trick.count as usize {
                trick_pts += trick.cards[i].1.points() as i32;
                if i > 0 && trick.cards[i].1.beats(best, lead_suit) {
                    best = trick.cards[i].1;
                    winner = trick.cards[i].0;
                }
            }
            let winner_is_declarer_team = state.declarer == Some(winner)
                || state.partner == Some(winner);
            if winner_is_declarer_team {
                declarer_team_points += trick_pts;
            } else {
                opposition_points += trick_pts;
            }
        }

        let tricks_left = (hand.len() as usize).saturating_sub(state.tricks.len()).max(1);
        let phase = match state.tricks.len() {
            0..=3 => GamePhase::Early,
            4..=8 => GamePhase::Mid,
            _ => GamePhase::Late,
        };

        CardTracker {
            remaining,
            taroks_in_hand: hand.tarok_count(),
            taroks_remaining,
            suit_counts,
            player_voids,
            player_tarok_likelihood,
            tricks_left,
            phase,
            declarer_team_points,
            opposition_points,
        }
    }

    fn higher_taroks_out(&self, value: u8) -> u8 {
        let mut count = 0u8;
        for card in self.taroks_remaining.iter() {
            if card.value() > value {
                count += 1;
            }
        }
        count
    }

    fn suit_is_master(&self, suit: Suit, value: u8) -> bool {
        for card in self.remaining.suit(suit).iter() {
            if card.value() > value {
                return false;
            }
        }
        true
    }

    fn remaining_in_suit(&self, suit: Suit) -> u8 {
        self.remaining.suit_count(suit) as u8
    }

    fn opponents_void_in(&self, suit: Suit, player: u8) -> u8 {
        let mut count = 0u8;
        for p in 0..NUM_PLAYERS {
            if p as u8 != player && self.player_voids[p][suit as usize] {
                count += 1;
            }
        }
        count
    }

    /// Lowest tarok in remaining that is higher than `value`.
    #[allow(dead_code)]
    fn lowest_winning_tarok_remaining(&self, value: u8) -> Option<u8> {
        let mut lowest: Option<u8> = None;
        for card in self.taroks_remaining.iter() {
            if card.value() > value {
                match lowest {
                    None => lowest = Some(card.value()),
                    Some(v) if card.value() < v => lowest = Some(card.value()),
                    _ => {}
                }
            }
        }
        lowest
    }
}

// -----------------------------------------------------------------------
// Bidding — completely overhauled
// -----------------------------------------------------------------------

/// Compute "guaranteed" points: kings (5 each) + škis (5) + mond (5 if safe).
fn guaranteed_points(hand: CardSet) -> f64 {
    let mut pts = 0.0f64;
    let has_skis = hand.contains(Card::tarok(SKIS));
    let has_mond = hand.contains(Card::tarok(MOND));

    // Kings: 5 points each
    for suit in Suit::ALL {
        if hand.contains(Card::suit_card(suit, SuitRank::King)) {
            pts += 5.0;
        }
    }

    // Škis: always guaranteed
    if has_skis {
        pts += 5.0;
    }

    // Mond: guaranteed if we also have škis, or if we have many taroks (safe)
    if has_mond {
        if has_skis || hand.tarok_count() >= 6 {
            pts += 5.0;
        } else {
            // Mond somewhat vulnerable
            pts += 2.0;
        }
    }

    pts
}

pub fn evaluate_bid_v5(hand: CardSet, highest_so_far: Option<Contract>) -> Option<Contract> {
    let tarok_count = hand.tarok_count();
    let high_taroks = hand.high_tarok_count();
    let king_count = hand.king_count();

    let has_skis = hand.contains(Card::tarok(SKIS));
    let has_mond = hand.contains(Card::tarok(MOND));
    let has_pagat = hand.contains(Card::tarok(PAGAT));

    // --- Compute hand rating ---
    let mut rating = 0.0f64;
    rating += tarok_count as f64 * 5.5;
    rating += high_taroks as f64 * 4.5;
    if has_skis {
        rating += 12.0;
    }
    if has_mond {
        rating += 9.0;
        if has_skis {
            rating += 4.0;
        }
    }
    if has_pagat {
        if tarok_count >= 7 {
            rating += 7.0;
        } else if tarok_count >= 5 {
            rating += 2.0;
        } else {
            rating -= 4.0; // unprotected pagat is dangerous
        }
    }

    rating += king_count as f64 * 7.0;

    let mut has_all_suits = true;
    let mut has_void = false;
    let mut has_singleton = false;
    let mut max_tarok_val = 0u8;

    // Track max tarok value in hand
    for card in hand.taroks().iter() {
        if card.value() > max_tarok_val {
            max_tarok_val = card.value();
        }
    }

    for suit in Suit::ALL {
        let count = hand.suit_count(suit);
        let has_king = hand.contains(Card::suit_card(suit, SuitRank::King));
        let has_queen = hand.contains(Card::suit_card(suit, SuitRank::Queen));

        if count == 0 {
            has_all_suits = false;
            has_void = true;
            rating += 7.0;
        } else if count == 1 {
            has_singleton = true;
            if has_king {
                rating += 3.0;
            } else {
                rating += 2.0;
            }
        } else if count == 2 && has_king {
            rating += 2.0;
        } else if count >= 3 && !has_king {
            rating -= count as f64 * 2.5; // slightly harsher penalty
        }

        if has_king && has_queen && count >= 2 {
            rating += 2.0;
        }
    }

    if tarok_count >= 8 {
        rating += (tarok_count - 7) as f64 * 3.0;
    }

    let ratio = (rating / 130.0).min(1.0);

    // --- Berač gating (much tighter) ---
    // No berač with: any void suit, any singleton, >2 taroks, any tarok ≥ 9,
    // kings unless 5+ in that suit
    let mut king_blocks_berac = false;
    for suit in Suit::ALL {
        let has_king = hand.contains(Card::suit_card(suit, SuitRank::King));
        if has_king && hand.suit_count(suit) < 5 {
            king_blocks_berac = true;
        }
    }

    let can_berac = ratio < 0.16
        && tarok_count <= 2
        && max_tarok_val < 9 // no tarok 9 or above (excludes škis etc.)
        && has_all_suits
        && !has_void
        && !has_singleton
        && !king_blocks_berac
        && high_taroks == 0;

    // --- Normal contract thresholds (raised to be less aggressive) ---
    let thresholds: [(Contract, f64); 7] = [
        (Contract::Three, 0.26),
        (Contract::Two, 0.33),
        (Contract::One, 0.42),
        (Contract::SoloThree, 0.55),
        (Contract::SoloTwo, 0.63),
        (Contract::SoloOne, 0.72),
        (Contract::Solo, 0.82),
    ];

    let mut best: Option<Contract> = None;
    for &(contract, threshold) in &thresholds {
        if ratio >= threshold {
            best = Some(contract);
        }
    }

    // --- Solo gating: guaranteed points must exceed 30 ---
    if let Some(contract) = best {
        if contract.is_solo() {
            let gp = guaranteed_points(hand);
            if gp < 30.0 {
                // Downgrade: find highest non-solo that still qualifies
                best = None;
                for &(c, t) in &thresholds {
                    if !c.is_solo() && ratio >= t {
                        best = Some(c);
                    }
                }
            }
            // Additional: no solo at all with fewer than 4 taroks
            if tarok_count < 4 {
                best = None;
                for &(c, t) in &thresholds {
                    if !c.is_solo() && ratio >= t {
                        best = Some(c);
                    }
                }
            }
        }
    }

    // Berač override
    if can_berac && best.map_or(true, |current| Contract::Berac.strength() > current.strength()) {
        best = Some(Contract::Berac);
    }

    // Filter against highest bid so far
    if let Some(contract) = best {
        if let Some(highest) = highest_so_far {
            if contract.strength() <= highest.strength() {
                return None;
            }
        }
        Some(contract)
    } else {
        None
    }
}

// -----------------------------------------------------------------------
// King calling — same as v4 (already solid)
// -----------------------------------------------------------------------

pub fn choose_king_v5(hand: CardSet) -> Option<Card> {
    let mut best_king: Option<Card> = None;
    let mut best_score = -1i32;

    for suit in Suit::ALL {
        let king = Card::suit_card(suit, SuitRank::King);
        if hand.contains(king) {
            continue;
        }
        let count = hand.suit_count(suit) as i32;
        let low_cards = hand
            .suit(suit)
            .iter()
            .filter(|card| card.value() <= 4)
            .count() as i32;
        let has_queen = hand.contains(Card::suit_card(suit, SuitRank::Queen));
        let score = count * 3 + low_cards * 2 + if has_queen { 1 } else { 0 };
        if score > best_score {
            best_score = score;
            best_king = Some(king);
        }
    }

    if best_king.is_none() {
        for suit in Suit::ALL {
            let queen = Card::suit_card(suit, SuitRank::Queen);
            if !hand.contains(queen) {
                best_king = Some(queen);
                break;
            }
        }
    }

    best_king
}

// -----------------------------------------------------------------------
// Talon — slightly improved void-building + called-suit bonus
// -----------------------------------------------------------------------

fn evaluate_talon_group_v5(group: &[Card], hand: CardSet, called_king: Option<Card>) -> f64 {
    let called_suit = called_king.and_then(|king| king.suit());
    let mut total = 0.0f64;

    for &card in group {
        total += card.points() as f64 * 2.0 + worth_over(card) as f64 * 0.3;
        if card.card_type() == CardType::Tarok {
            total += 10.0;
        }
        if let (Some(king_suit), Some(suit)) = (called_suit, card.suit()) {
            if suit == king_suit {
                total += 6.0;
            }
        }
    }

    for suit in Suit::ALL {
        if Some(suit) == called_suit {
            continue;
        }
        let hand_count = hand.suit_count(suit);
        let group_adds = group
            .iter()
            .filter(|card| card.suit() == Some(suit))
            .count() as u32;
        if hand_count <= 1 && hand_count + group_adds <= 1 {
            total += 6.0;
        }
        if hand_count == 0 && group_adds == 0 {
            total += 4.0;
        }
    }

    total
}

pub fn choose_talon_group_v5(
    groups: &[Vec<Card>],
    hand: CardSet,
    called_king: Option<Card>,
) -> usize {
    let mut best_idx = 0usize;
    let mut best_score = f64::NEG_INFINITY;
    for (idx, group) in groups.iter().enumerate() {
        let score = evaluate_talon_group_v5(group, hand, called_king);
        if score > best_score {
            best_score = score;
            best_idx = idx;
        }
    }
    best_idx
}

// -----------------------------------------------------------------------
// Discard — same as v4 (already solid)
// -----------------------------------------------------------------------

pub fn choose_discards_v5(
    hand: CardSet,
    must_discard: usize,
    called_king: Option<Card>,
) -> Vec<Card> {
    let called_suit = called_king.and_then(|king| king.suit());

    let mut discardable: Vec<Card> = hand
        .iter()
        .filter(|card| card.card_type() != CardType::Tarok && !card.is_king())
        .collect();

    if discardable.len() < must_discard {
        let mut extra: Vec<Card> = hand
            .iter()
            .filter(|card| {
                card.card_type() == CardType::Tarok
                    && !card.is_trula()
                    && !discardable.contains(card)
            })
            .collect();
        extra.sort_by_key(|card| card.value());
        discardable.extend(extra);
    }

    let mut by_suit: [Vec<Card>; 4] = Default::default();
    for &card in &discardable {
        if let Some(suit) = card.suit() {
            by_suit[suit as usize].push(card);
        }
    }

    let mut result: Vec<Card> = Vec::with_capacity(must_discard);
    let mut suit_order: Vec<(Suit, u32)> = Suit::ALL
        .iter()
        .filter(|&&suit| Some(suit) != called_suit && !by_suit[suit as usize].is_empty())
        .map(|&suit| (suit, hand.suit_count(suit)))
        .collect();
    suit_order.sort_by_key(|&(_, count)| count);

    for (suit, _) in suit_order {
        if result.len() >= must_discard {
            break;
        }
        let suit_discardable: Vec<Card> = hand
            .suit(suit)
            .iter()
            .filter(|card| !card.is_king() && discardable.contains(card) && !result.contains(card))
            .collect();
        if suit_discardable.len() + result.len() <= must_discard {
            result.extend(suit_discardable);
        }
    }

    if result.len() < must_discard {
        let mut remaining: Vec<Card> = discardable
            .iter()
            .filter(|card| !result.contains(card))
            .copied()
            .collect();
        remaining.sort_by_key(|card| (card.points(), worth_over(*card)));
        for card in remaining {
            if result.len() >= must_discard {
                break;
            }
            result.push(card);
        }
    }

    result.truncate(must_discard);
    result
}

// -----------------------------------------------------------------------
// Card play — v5 with belief tracking, partner preservation, last-seat
//             economy, and role-aware evaluation
// -----------------------------------------------------------------------

#[allow(private_interfaces)]
pub fn evaluate_card_play_v5(
    card: Card,
    hand: CardSet,
    state: &GameState,
    player: u8,
    is_leading: bool,
    tracker: &CardTracker,
) -> f64 {
    let is_declarer = state.declarer == Some(player);
    let is_partner = state.partner == Some(player);
    let _is_playing = is_declarer || is_partner;
    let is_klop = state
        .contract
        .map_or(false, |contract| contract.is_klop());
    let is_berac = state
        .contract
        .map_or(false, |contract| contract.is_berac());

    if is_klop || is_berac {
        return eval_klop_berac_v5(card, hand, state, player, is_leading, tracker);
    }

    if is_leading {
        eval_leading_v5(
            card,
            hand,
            state,
            player,
            is_declarer,
            is_partner,
            tracker,
        )
    } else {
        eval_following_v5(card, hand, state, player, is_declarer, is_partner, tracker)
    }
}

fn eval_klop_berac_v5(
    card: Card,
    _hand: CardSet,
    state: &GameState,
    player: u8,
    is_leading: bool,
    tracker: &CardTracker,
) -> f64 {
    let wo = worth_over(card) as f64;
    let pts = card.points() as f64;

    if is_leading {
        if card.card_type() == CardType::Tarok {
            let mut score = -wo * 2.5;
            if card.value() == PAGAT {
                if tracker.taroks_in_hand == 1 && tracker.phase == GamePhase::Late {
                    return -5.0;
                }
                score -= 500.0;
            }
            if tracker.phase == GamePhase::Late
                && card.value() <= 5
                && tracker.higher_taroks_out(card.value()) == 0
            {
                return 10.0;
            }
            score
        } else {
            let suit = card.suit().unwrap();
            let count = tracker.suit_counts[suit as usize] as f64;
            let remaining = tracker.remaining_in_suit(suit) as f64;
            let opponents_void = tracker.opponents_void_in(suit, player) as f64;

            if opponents_void > 0.0 {
                -wo * 4.0 - opponents_void * 25.0
            } else if remaining == 0.0 {
                -wo * 3.0
            } else {
                let mut score = -wo + remaining * 3.0 - count * 2.0;
                if count == 1.0 {
                    score += 5.0;
                }
                score
            }
        }
    } else if let Some(ref trick) = state.current_trick {
        if trick.count == 0 {
            return -wo;
        }
        let lead_suit = trick.lead_suit();
        if let Some(best) = trick.best_card() {
            let is_last = trick.count as usize == NUM_PLAYERS - 1;
            let trick_pts = (0..trick.count as usize)
                .map(|idx| trick.cards[idx].1.points() as f64)
                .sum::<f64>()
                + pts;

            if card.beats(best, lead_suit) {
                if is_last && trick_pts <= 3.0 {
                    -wo * 0.5
                } else {
                    -wo * 3.0 - trick_pts * 10.0
                }
            } else {
                let mut score = pts * 5.0 + wo * 0.3;
                if let Some(suit) = card.suit() {
                    let danger = tracker.opponents_void_in(suit, player);
                    if danger > 0 {
                        score += pts * 3.0;
                    }
                }
                score
            }
        } else {
            -wo
        }
    } else {
        -wo
    }
}

fn eval_leading_v5(
    card: Card,
    hand: CardSet,
    state: &GameState,
    _player: u8,
    is_declarer: bool,
    is_partner: bool,
    tracker: &CardTracker,
) -> f64 {
    let wo = worth_over(card) as f64;
    let pts = card.points() as f64;
    let is_opposition = !is_declarer && !is_partner;

    if card.card_type() == CardType::Tarok {
        let higher_out = tracker.higher_taroks_out(card.value());

        // --- Pagat ---
        if card.value() == PAGAT {
            if tracker.tricks_left <= 2 && tracker.taroks_in_hand <= 2 {
                return 300.0;
            }
            if tracker.tricks_left == 1 {
                return 400.0;
            }
            return -250.0;
        }

        // --- Partner leading ---
        if is_partner {
            if card.value() == SKIS {
                return match tracker.phase {
                    GamePhase::Early => 130.0,
                    GamePhase::Mid => 105.0,
                    GamePhase::Late => {
                        if tracker.tricks_left <= 1 {
                            -300.0
                        } else {
                            70.0
                        }
                    }
                };
            }
            if card.value() == MOND {
                let skis_out = tracker.taroks_remaining.contains(Card::tarok(SKIS));
                let has_skis = hand.contains(Card::tarok(SKIS));
                if skis_out && !has_skis {
                    return 72.0;
                }
                return 95.0 + tracker.taroks_remaining.len() as f64 * 2.0;
            }
            // Partner should lead high taroks to pull opponents' taroks
            return 90.0 + wo * 1.4 - higher_out as f64 * 4.0;
        }

        // --- Opposition leading ---
        if is_opposition {
            // Preserve taroks — only lead them if master or late game
            if higher_out == 0 {
                if tracker.phase == GamePhase::Late {
                    return 25.0 + wo * 0.3;
                }
                return 20.0 + wo * 0.2;
            }
            if higher_out <= 1 && tracker.tricks_left <= 3 {
                return 8.0 + wo * 0.1;
            }
            // Don't waste taroks on leads as opposition
            return -50.0 - wo * 0.5 - higher_out as f64 * 7.0;
        }

        // --- Declarer leading ---
        if card.value() == SKIS {
            return match tracker.phase {
                GamePhase::Early => 95.0,
                GamePhase::Mid => 75.0,
                GamePhase::Late => {
                    if tracker.tricks_left <= 1 {
                        -300.0
                    } else {
                        55.0
                    }
                }
            };
        }

        if card.value() == MOND {
            let skis_out = tracker.taroks_remaining.contains(Card::tarok(SKIS));
            let has_skis = hand.contains(Card::tarok(SKIS));
            if skis_out && !has_skis {
                return -150.0;
            }
            if has_skis {
                return 70.0;
            }
            return 75.0 + tracker.taroks_remaining.len() as f64 * 3.0;
        }

        if higher_out == 0 {
            return 65.0 + wo;
        }

        let base = ((wo - 11.0).max(0.0) / 3.0).powf(1.5);
        if tracker.phase == GamePhase::Early {
            base + 10.0
        } else {
            base
        }
    } else {
        let suit = card.suit().unwrap();
        let count = tracker.suit_counts[suit as usize] as f64;
        let remaining = tracker.remaining_in_suit(suit) as f64;

        if is_partner {
            if card.is_king() && tracker.suit_is_master(suit, card.value()) {
                18.0 + pts
            } else if count == 1.0 {
                6.0 - pts
            } else if count == 2.0 && !card.is_king() {
                4.0 - pts
            } else {
                -pts - count
            }
        } else if is_declarer {
            if card.is_king() {
                if tracker.suit_is_master(suit, card.value()) {
                    30.0 + pts * 2.0
                } else if count >= 3.0 {
                    15.0 + pts
                } else {
                    pts - 8.0
                }
            } else if count == 1.0 {
                18.0 - pts
            } else if count == 2.0 && !card.is_king() {
                12.0 - pts
            } else {
                -pts * 1.5 - count * 2.0
            }
        } else {
            // Opposition suit leads
            let mut declarer_team_voids = 0.0;
            if let Some(declarer) = state.declarer {
                if tracker.player_voids[declarer as usize][suit as usize] {
                    declarer_team_voids += 1.0;
                }
            }
            if let Some(partner) = state.partner {
                if tracker.player_voids[partner as usize][suit as usize] {
                    declarer_team_voids += 1.0;
                }
            }

            let mut score = 18.0 - pts * 4.0 - count * 1.5;
            if card.is_king() {
                score -= 25.0;
            }
            if card.value() <= SuitRank::Pip2 as u8 {
                score += 8.0;
            }
            if declarer_team_voids > 0.0 {
                // Lead into suits where declarer team is void — they'll
                // be forced to play taroks they'd rather keep
                score += declarer_team_voids * 14.0 - pts * 2.0;
            } else if remaining > 0.0 {
                score += remaining.min(4.0);
            }

            // Bonus: lead into suits where we believe declarer has low
            // tarok likelihood (so our partner might win)
            if let Some(declarer) = state.declarer {
                let dl = tracker.player_tarok_likelihood[declarer as usize];
                if dl < 0.3 {
                    score += 5.0; // declarer may be low on taroks
                }
            }

            score
        }
    }
}

fn eval_following_v5(
    card: Card,
    _hand: CardSet,
    state: &GameState,
    player: u8,
    is_declarer: bool,
    is_partner: bool,
    tracker: &CardTracker,
) -> f64 {
    let wo = worth_over(card) as f64;
    let pts = card.points() as f64;
    let is_playing = is_declarer || is_partner;

    let trick = match &state.current_trick {
        Some(trick) if trick.count > 0 => trick,
        _ => return 0.0,
    };

    let lead_suit = trick.lead_suit();
    let num_played = trick.count as usize;
    let is_last = num_played == NUM_PLAYERS - 1;

    let mut best_card = trick.cards[0].1;
    let mut best_player = trick.cards[0].0;
    for idx in 1..num_played {
        if trick.cards[idx].1.beats(best_card, lead_suit) {
            best_card = trick.cards[idx].1;
            best_player = trick.cards[idx].0;
        }
    }

    let best_is_ally = if is_playing {
        best_player == state.declarer.unwrap_or(255) || state.partner == Some(best_player)
    } else {
        best_player != state.declarer.unwrap_or(255) && state.partner != Some(best_player)
    };

    let trick_pts = (0..num_played)
        .map(|idx| trick.cards[idx].1.points() as f64)
        .sum::<f64>()
        + pts;
    let would_win = card.beats(best_card, lead_suit);

    // === Special card handling ===

    // Pagat
    if card.card_type() == CardType::Tarok && card.value() == PAGAT {
        if tracker.tricks_left <= 2 && tracker.taroks_in_hand <= 2 {
            return if would_win { 500.0 } else { -500.0 };
        }
        if tracker.tricks_left == 1 {
            return if would_win { 600.0 } else { -600.0 };
        }
        return -250.0;
    }

    // Mond
    if card.card_type() == CardType::Tarok && card.value() == MOND {
        if !would_win {
            return -600.0;
        }
        let skis_out = tracker.taroks_remaining.contains(Card::tarok(SKIS));
        if skis_out && !is_last {
            return -100.0;
        }
        return trick_pts * 2.0 + 40.0;
    }

    // Škis
    if card.card_type() == CardType::Tarok && card.value() == SKIS {
        if tracker.tricks_left <= 1 {
            return -400.0;
        }
        return trick_pts * 2.5 + 35.0;
    }

    // === Partner preservation: don't overtrump ally if outcome unchanged ===
    if would_win && best_is_ally && card.card_type() == CardType::Tarok {
        // If our ally is already winning, don't play a *higher* tarok
        // unless there are opponents yet to play who could beat ally
        if !is_last {
            // Check if any remaining opponent could potentially beat ally's card
            let opponents_remaining = NUM_PLAYERS - 1 - num_played;
            if opponents_remaining > 0 {
                // If our tarok is higher than partner's winning card but there
                // are opponents left, still might be useful to overrule.
                // But if ally's card is already very strong, don't waste ours.
                let ally_val = if best_card.card_type() == CardType::Tarok {
                    best_card.value()
                } else {
                    0
                };
                let higher_than_ally_out = tracker.higher_taroks_out(ally_val);
                if higher_than_ally_out == 0 {
                    // Nobody can beat ally — don't waste our tarok
                    return -wo * 2.0 - 50.0;
                }
            }
        } else {
            // We're last and ally is winning — don't overtrump
            return -wo * 2.0 - 50.0;
        }
    }

    // === Last-seat economy: play lowest winning tarok ===
    if would_win && is_last && !best_is_ally && card.card_type() == CardType::Tarok {
        // Reward low taroks that still win (economy bonus)
        let min_winning_val = if best_card.card_type() == CardType::Tarok {
            best_card.value() + 1
        } else {
            1 // any tarok beats suit cards
        };
        if card.value() == min_winning_val || card.value() <= min_winning_val + 1 {
            // This is the cheapest (or nearly cheapest) winning tarok — bonus
            return trick_pts * 4.0 + 50.0 - wo * 0.1;
        }
        // Higher tarok than needed in last seat — penalty proportional to waste
        let waste = (card.value() - min_winning_val) as f64;
        return trick_pts * 3.0 - waste * 8.0;
    }

    // === Normal following ===
    if would_win {
        if best_is_ally {
            if is_last {
                pts * 4.5 // šmir
            } else {
                -wo * 0.6 - (wo / 3.0).powf(1.5)
            }
        } else if is_last {
            trick_pts * 3.5 - wo
        } else if num_played == 1 {
            // 2nd seat: be careful with high taroks if more players to come
            if card.card_type() == CardType::Tarok
                && tracker.higher_taroks_out(card.value()) > 0
            {
                trick_pts * 0.6 - wo * 0.4
            } else {
                trick_pts * 2.0 + wo * 0.2
            }
        } else {
            trick_pts * 2.0 + wo * 0.3
        }
    } else if best_is_ally {
        // Šmir — dump high-value cards to ally
        if is_last {
            pts * 6.0
        } else if tracker.phase == GamePhase::Late {
            pts * 4.5
        } else {
            pts * 3.0
        }
    } else {
        // Losing to enemy — dump lowest value
        let mut score = -(pts * 3.0) - wo * 0.4;
        if let Some(suit) = card.suit() {
            for p in 0..NUM_PLAYERS {
                if p as u8 != player && tracker.player_voids[p][suit as usize] {
                    score += 2.0;
                }
            }
        }
        // Use beliefs: if opponents likely have no taroks,
        // suit cards in hand are more valuable to keep
        if card.card_type() != CardType::Tarok {
            let mut opp_tarok_pressure = 0.0;
            for p in 0..NUM_PLAYERS {
                if p as u8 != player && !is_ally(p as u8, state, is_playing) {
                    opp_tarok_pressure += tracker.player_tarok_likelihood[p];
                }
            }
            if opp_tarok_pressure < 0.3 {
                // Opponents low on taroks, suit cards cheaper to shed
                score += 2.0;
            }
        }
        score
    }
}

/// Check if a player is an ally given the current playing status.
fn is_ally(p: u8, state: &GameState, is_playing: bool) -> bool {
    if is_playing {
        p == state.declarer.unwrap_or(255) || state.partner == Some(p)
    } else {
        p != state.declarer.unwrap_or(255) && state.partner != Some(p)
    }
}

// -----------------------------------------------------------------------
// Main entry point
// -----------------------------------------------------------------------

pub fn choose_card_v5(hand: CardSet, state: &GameState, player: u8) -> Card {
    let ctx = legal_moves::MoveCtx::from_state(state, player);
    let legal = legal_moves::generate_legal_moves(&ctx);
    let legal_vec: Vec<Card> = legal.iter().collect();

    if legal_vec.len() == 1 {
        return legal_vec[0];
    }

    let is_leading = state
        .current_trick
        .as_ref()
        .map_or(true, |trick| trick.count == 0);
    let tracker = CardTracker::from_state(state, player);

    let mut best_card = legal_vec[0];
    let mut best_score = f64::NEG_INFINITY;
    for &card in &legal_vec {
        let score = evaluate_card_play_v5(card, hand, state, player, is_leading, &tracker);
        if score > best_score {
            best_score = score;
            best_card = card;
        }
    }

    best_card
}

#[inline]
fn worth_over(card: Card) -> i32 {
    if card.card_type() == CardType::Tarok {
        10 + card.value() as i32
    } else {
        let suit_base = match card.suit().unwrap() {
            Suit::Hearts => 33,
            Suit::Diamonds => 41,
            Suit::Clubs => 49,
            Suit::Spades => 57,
        };
        suit_base + card.value() as i32
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn hand_from(cards: &[Card]) -> CardSet {
        let mut hand = CardSet::EMPTY;
        for &card in cards {
            hand.insert(card);
        }
        hand
    }

    fn make_state(hand: CardSet, player: u8, declarer: u8, partner: Option<u8>) -> GameState {
        let mut state = GameState::new(0);
        state.phase = Phase::TrickPlay;
        state.contract = Some(Contract::Three);
        state.declarer = Some(declarer);
        state.partner = partner;
        state.current_player = player;
        state.hands[player as usize] = hand;
        state.roles[declarer as usize] = PlayerRole::Declarer;
        for idx in 0..NUM_PLAYERS {
            if idx as u8 != declarer {
                state.roles[idx] = PlayerRole::Opponent;
            }
        }
        if let Some(partner_player) = partner {
            state.roles[partner_player as usize] = PlayerRole::Partner;
        }
        state
    }

    // --- Bidding tests ---

    #[test]
    fn berac_rejects_more_than_two_taroks() {
        let hand = hand_from(&[
            Card::tarok(2),
            Card::tarok(3),
            Card::tarok(4),
            Card::suit_card(Suit::Hearts, SuitRank::Pip1),
            Card::suit_card(Suit::Hearts, SuitRank::Pip2),
            Card::suit_card(Suit::Hearts, SuitRank::Pip3),
            Card::suit_card(Suit::Diamonds, SuitRank::Pip1),
            Card::suit_card(Suit::Diamonds, SuitRank::Pip2),
            Card::suit_card(Suit::Diamonds, SuitRank::Pip3),
            Card::suit_card(Suit::Clubs, SuitRank::Pip1),
            Card::suit_card(Suit::Clubs, SuitRank::Pip2),
            Card::suit_card(Suit::Spades, SuitRank::Pip1),
        ]);
        assert_ne!(evaluate_bid_v5(hand, None), Some(Contract::Berac));
    }

    #[test]
    fn berac_rejects_singleton_suit() {
        let hand = hand_from(&[
            Card::tarok(2),
            Card::suit_card(Suit::Hearts, SuitRank::Pip1),
            Card::suit_card(Suit::Hearts, SuitRank::Pip2),
            Card::suit_card(Suit::Hearts, SuitRank::Pip3),
            Card::suit_card(Suit::Diamonds, SuitRank::Pip1),
            Card::suit_card(Suit::Diamonds, SuitRank::Pip2),
            Card::suit_card(Suit::Diamonds, SuitRank::Pip3),
            Card::suit_card(Suit::Clubs, SuitRank::Pip1),
            Card::suit_card(Suit::Clubs, SuitRank::Pip2),
            Card::suit_card(Suit::Clubs, SuitRank::Pip3),
            Card::suit_card(Suit::Spades, SuitRank::Pip1),
            Card::suit_card(Suit::Spades, SuitRank::Pip2),
        ]);
        // Spades has only 2 cards, but Hearts has 3, Diamonds 3, Clubs 3
        // Actually this hand has no singleton. Let's make one:
        let hand2 = hand_from(&[
            Card::tarok(2),
            Card::suit_card(Suit::Hearts, SuitRank::Pip1),
            Card::suit_card(Suit::Hearts, SuitRank::Pip2),
            Card::suit_card(Suit::Hearts, SuitRank::Pip3),
            Card::suit_card(Suit::Hearts, SuitRank::Pip4),
            Card::suit_card(Suit::Diamonds, SuitRank::Pip1),
            Card::suit_card(Suit::Diamonds, SuitRank::Pip2),
            Card::suit_card(Suit::Diamonds, SuitRank::Pip3),
            Card::suit_card(Suit::Clubs, SuitRank::Pip1),
            Card::suit_card(Suit::Clubs, SuitRank::Pip2),
            Card::suit_card(Suit::Clubs, SuitRank::Pip3),
            Card::suit_card(Suit::Spades, SuitRank::Pip1), // singleton
        ]);
        assert_ne!(evaluate_bid_v5(hand2, None), Some(Contract::Berac));
    }

    #[test]
    fn berac_rejects_void_suit() {
        let hand = hand_from(&[
            Card::tarok(2),
            Card::suit_card(Suit::Hearts, SuitRank::Pip1),
            Card::suit_card(Suit::Hearts, SuitRank::Pip2),
            Card::suit_card(Suit::Hearts, SuitRank::Pip3),
            Card::suit_card(Suit::Hearts, SuitRank::Pip4),
            Card::suit_card(Suit::Diamonds, SuitRank::Pip1),
            Card::suit_card(Suit::Diamonds, SuitRank::Pip2),
            Card::suit_card(Suit::Diamonds, SuitRank::Pip3),
            Card::suit_card(Suit::Clubs, SuitRank::Pip1),
            Card::suit_card(Suit::Clubs, SuitRank::Pip2),
            Card::suit_card(Suit::Clubs, SuitRank::Pip3),
            Card::suit_card(Suit::Clubs, SuitRank::Pip4),
            // No spades = void
        ]);
        assert_ne!(evaluate_bid_v5(hand, None), Some(Contract::Berac));
    }

    #[test]
    fn berac_rejects_tarok_9_or_above() {
        let hand = hand_from(&[
            Card::tarok(9), // too high
            Card::suit_card(Suit::Hearts, SuitRank::Pip1),
            Card::suit_card(Suit::Hearts, SuitRank::Pip2),
            Card::suit_card(Suit::Hearts, SuitRank::Pip3),
            Card::suit_card(Suit::Diamonds, SuitRank::Pip1),
            Card::suit_card(Suit::Diamonds, SuitRank::Pip2),
            Card::suit_card(Suit::Diamonds, SuitRank::Pip3),
            Card::suit_card(Suit::Clubs, SuitRank::Pip1),
            Card::suit_card(Suit::Clubs, SuitRank::Pip2),
            Card::suit_card(Suit::Clubs, SuitRank::Pip3),
            Card::suit_card(Suit::Spades, SuitRank::Pip1),
            Card::suit_card(Suit::Spades, SuitRank::Pip2),
        ]);
        assert_ne!(evaluate_bid_v5(hand, None), Some(Contract::Berac));
    }

    #[test]
    fn berac_rejects_skis() {
        let hand = hand_from(&[
            Card::tarok(SKIS),
            Card::suit_card(Suit::Hearts, SuitRank::Pip1),
            Card::suit_card(Suit::Hearts, SuitRank::Pip2),
            Card::suit_card(Suit::Hearts, SuitRank::Pip3),
            Card::suit_card(Suit::Diamonds, SuitRank::Pip1),
            Card::suit_card(Suit::Diamonds, SuitRank::Pip2),
            Card::suit_card(Suit::Diamonds, SuitRank::Pip3),
            Card::suit_card(Suit::Clubs, SuitRank::Pip1),
            Card::suit_card(Suit::Clubs, SuitRank::Pip2),
            Card::suit_card(Suit::Clubs, SuitRank::Pip3),
            Card::suit_card(Suit::Spades, SuitRank::Pip1),
            Card::suit_card(Suit::Spades, SuitRank::Pip2),
        ]);
        assert_ne!(evaluate_bid_v5(hand, None), Some(Contract::Berac));
    }

    #[test]
    fn berac_rejects_king_without_5_in_suit() {
        let hand = hand_from(&[
            Card::tarok(2),
            Card::suit_card(Suit::Hearts, SuitRank::King), // only 3 hearts
            Card::suit_card(Suit::Hearts, SuitRank::Pip1),
            Card::suit_card(Suit::Hearts, SuitRank::Pip2),
            Card::suit_card(Suit::Diamonds, SuitRank::Pip1),
            Card::suit_card(Suit::Diamonds, SuitRank::Pip2),
            Card::suit_card(Suit::Diamonds, SuitRank::Pip3),
            Card::suit_card(Suit::Clubs, SuitRank::Pip1),
            Card::suit_card(Suit::Clubs, SuitRank::Pip2),
            Card::suit_card(Suit::Clubs, SuitRank::Pip3),
            Card::suit_card(Suit::Spades, SuitRank::Pip1),
            Card::suit_card(Suit::Spades, SuitRank::Pip2),
        ]);
        assert_ne!(evaluate_bid_v5(hand, None), Some(Contract::Berac));
    }

    #[test]
    fn solo_requires_guaranteed_points_above_30() {
        // Hand with only 2 kings (10 pts) and no škis/mond — shouldn't solo
        let hand = hand_from(&[
            Card::tarok(15),
            Card::tarok(16),
            Card::tarok(17),
            Card::tarok(18),
            Card::tarok(19),
            Card::tarok(20),
            Card::suit_card(Suit::Hearts, SuitRank::King),
            Card::suit_card(Suit::Diamonds, SuitRank::King),
            Card::suit_card(Suit::Clubs, SuitRank::Pip1),
            Card::suit_card(Suit::Clubs, SuitRank::Pip2),
            Card::suit_card(Suit::Spades, SuitRank::Pip1),
            Card::suit_card(Suit::Spades, SuitRank::Pip2),
        ]);
        let bid = evaluate_bid_v5(hand, None);
        match bid {
            Some(c) => assert!(!c.is_solo(), "Should not bid solo with only 10 guaranteed pts"),
            None => {} // passing is also fine
        }
    }

    #[test]
    fn solo_rejected_with_fewer_than_4_taroks() {
        // Very strong hand but only 3 taroks
        let hand = hand_from(&[
            Card::tarok(SKIS),
            Card::tarok(MOND),
            Card::tarok(20),
            Card::suit_card(Suit::Hearts, SuitRank::King),
            Card::suit_card(Suit::Diamonds, SuitRank::King),
            Card::suit_card(Suit::Clubs, SuitRank::King),
            Card::suit_card(Suit::Spades, SuitRank::King),
            Card::suit_card(Suit::Hearts, SuitRank::Queen),
            Card::suit_card(Suit::Diamonds, SuitRank::Queen),
            Card::suit_card(Suit::Clubs, SuitRank::Queen),
            Card::suit_card(Suit::Spades, SuitRank::Queen),
            Card::suit_card(Suit::Hearts, SuitRank::Knight),
        ]);
        let bid = evaluate_bid_v5(hand, None);
        match bid {
            Some(c) => assert!(!c.is_solo(), "Should not bid solo with only 3 taroks"),
            None => {}
        }
    }

    #[test]
    fn no_aggressive_solo_with_weak_hand() {
        // Only 1 tarok — absolutely should not solo
        let hand = hand_from(&[
            Card::tarok(5),
            Card::suit_card(Suit::Hearts, SuitRank::King),
            Card::suit_card(Suit::Hearts, SuitRank::Pip1),
            Card::suit_card(Suit::Hearts, SuitRank::Pip2),
            Card::suit_card(Suit::Diamonds, SuitRank::Pip1),
            Card::suit_card(Suit::Diamonds, SuitRank::Pip2),
            Card::suit_card(Suit::Diamonds, SuitRank::Pip3),
            Card::suit_card(Suit::Clubs, SuitRank::Pip1),
            Card::suit_card(Suit::Clubs, SuitRank::Pip2),
            Card::suit_card(Suit::Spades, SuitRank::Pip1),
            Card::suit_card(Suit::Spades, SuitRank::Pip2),
            Card::suit_card(Suit::Spades, SuitRank::Pip3),
        ]);
        let bid = evaluate_bid_v5(hand, None);
        match bid {
            Some(c) => assert!(!c.is_solo(), "Should not bid solo with 1 tarok"),
            None => {}
        }
    }

    // --- Card play tests ---

    #[test]
    fn partner_prefers_highest_tarok_lead() {
        let hand = hand_from(&[
            Card::tarok(20),
            Card::tarok(15),
            Card::suit_card(Suit::Hearts, SuitRank::Pip1),
            Card::suit_card(Suit::Clubs, SuitRank::Pip2),
        ]);
        let state = make_state(hand, 1, 0, Some(1));
        assert_eq!(choose_card_v5(hand, &state, 1), Card::tarok(20));
    }

    #[test]
    fn last_seat_plays_lowest_winning_tarok() {
        // Player is last to play, opponent is currently winning with tarok 10
        let hand = hand_from(&[
            Card::tarok(11),
            Card::tarok(15),
            Card::tarok(19),
            Card::suit_card(Suit::Hearts, SuitRank::Pip1),
        ]);
        let mut state = make_state(hand, 3, 3, None);
        state.contract = Some(Contract::SoloThree);

        // Set up a trick where player 3 is last (3 cards already played)
        let mut trick = Trick::new(0);
        trick.play(0, Card::suit_card(Suit::Diamonds, SuitRank::Pip1)); // opponent leads
        trick.play(1, Card::suit_card(Suit::Diamonds, SuitRank::Pip2)); // opponent
        trick.play(2, Card::tarok(10)); // opponent trumps
        state.current_trick = Some(trick);

        let chosen = choose_card_v5(hand, &state, 3);
        // Should pick tarok 11 (lowest that still wins) over 15 or 19
        assert_eq!(chosen, Card::tarok(11));
    }

    #[test]
    fn do_not_overtrump_winning_ally() {
        // Declarer (player 0) is winning with tarok 18.
        // Player 2 is partner — should NOT waste tarok 20 on top of ally's winner.
        let hand = hand_from(&[
            Card::tarok(20),
            Card::tarok(10),
            Card::suit_card(Suit::Hearts, SuitRank::Pip1),
            Card::suit_card(Suit::Clubs, SuitRank::Pip2),
        ]);
        let mut state = make_state(hand, 2, 0, Some(2));

        // Trick: opponent (3) leads diamond, declarer (0) trumps with tarok 18
        // Opponent 1 follows suit.  Now partner (2) plays.
        let mut trick = Trick::new(3);
        trick.play(3, Card::suit_card(Suit::Diamonds, SuitRank::King));
        trick.play(0, Card::tarok(18)); // ally winning
        trick.play(1, Card::suit_card(Suit::Diamonds, SuitRank::Pip1));
        state.current_trick = Some(trick);

        // Player 2 has no diamonds — free to play any tarok (no overtrump
        // obligation in normal contracts).
        let chosen = choose_card_v5(hand, &state, 2);
        // Should NOT play tarok 20 — ally is already winning and nobody left
        assert_ne!(chosen, Card::tarok(20));
    }

    #[test]
    fn opposition_preserves_taroks_on_lead() {
        let hand = hand_from(&[
            Card::tarok(20),
            Card::tarok(10),
            Card::suit_card(Suit::Hearts, SuitRank::Pip1),
            Card::suit_card(Suit::Spades, SuitRank::Pip2),
        ]);
        let state = make_state(hand, 2, 0, Some(1));

        let chosen = choose_card_v5(hand, &state, 2);
        // Opposition should prefer suit lead over taroks
        assert_ne!(
            chosen.card_type(),
            CardType::Tarok,
            "Opposition should lead suits, not taroks"
        );
    }
}
