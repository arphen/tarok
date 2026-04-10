/// StockŠkis v2 — improved heuristic bot with card counting.
///
/// Enhancements over v1:
/// - Card counting (tracks played cards, remaining taroks)
/// - Positional awareness (2nd seat vs 4th seat)
/// - Void-building in talon/discard
/// - Better partner play (šmiranje)
/// - Mond/Škis safety
/// - Improved bidding with suit distribution

use crate::card::*;
use crate::game_state::*;
use crate::legal_moves;

// -----------------------------------------------------------------------
// Card tracking — derived from game state
// -----------------------------------------------------------------------

struct CardTracker {
    remaining: CardSet,        // Cards not played and not in our hand
    taroks_in_hand: u8,
    taroks_remaining: CardSet, // Taroks still held by opponents
    suit_counts: [u8; 4],      // Our suit card counts
    player_voids: [[bool; 4]; NUM_PLAYERS], // player_voids[p][suit] = void
    tricks_left: usize,
}

impl CardTracker {
    fn from_state(state: &GameState, player: u8) -> Self {
        let hand = state.hands[player as usize];

        // All played cards
        let mut played = state.played_cards;
        if let Some(ref trick) = state.current_trick {
            for i in 0..trick.count as usize {
                played.insert(trick.cards[i].1);
            }
        }
        played = played.union(state.put_down);

        // Remaining = full deck - played - hand
        let full_deck = CardSet((1u64 << DECK_SIZE) - 1);
        let remaining = full_deck.difference(played).difference(hand);

        let taroks_remaining = remaining.taroks();

        let mut suit_counts = [0u8; 4];
        for s in Suit::ALL {
            suit_counts[s as usize] = hand.suit_count(s) as u8;
        }

        // Track player voids from previous tricks
        let mut player_voids = [[false; 4]; NUM_PLAYERS];
        for trick in &state.tricks {
            if trick.count == 0 {
                continue;
            }
            let lead_card = trick.cards[0].1;
            if let Some(lead_suit) = lead_card.suit() {
                for i in 1..trick.count as usize {
                    let (p, c) = trick.cards[i];
                    if c.card_type() == CardType::Tarok && p != player {
                        player_voids[p as usize][lead_suit as usize] = true;
                    }
                }
            }
        }

        let tricks_left = (hand.len() as usize).saturating_sub(state.tricks.len()).max(1);

        CardTracker {
            remaining,
            taroks_in_hand: hand.tarok_count(),
            taroks_remaining,
            suit_counts,
            player_voids,
            tricks_left,
        }
    }

    fn higher_taroks_out(&self, value: u8) -> u8 {
        let mut count = 0u8;
        for c in self.taroks_remaining.iter() {
            if c.value() > value {
                count += 1;
            }
        }
        count
    }

    fn suit_is_master(&self, suit: Suit, value: u8) -> bool {
        for c in self.remaining.suit(suit).iter() {
            if c.value() > value {
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
}

// -----------------------------------------------------------------------
// Bidding — improved with suit distribution
// -----------------------------------------------------------------------

pub fn evaluate_bid_v2(hand: CardSet, highest_so_far: Option<Contract>) -> Option<Contract> {
    let tarok_count = hand.tarok_count();
    let high_taroks = hand.high_tarok_count();
    let king_count = hand.king_count();
    let void_count = hand.void_count();

    let has_skis = hand.contains(Card::tarok(SKIS));
    let has_mond = hand.contains(Card::tarok(MOND));
    let has_pagat = hand.contains(Card::tarok(PAGAT));

    let mut singletons = 0u8;
    for s in Suit::ALL {
        if hand.suit_count(s) == 1 {
            singletons += 1;
        }
    }

    let mut rating = 0.0f64;

    rating += tarok_count as f64 * 6.0;
    rating += high_taroks as f64 * 4.0;
    if has_skis { rating += 12.0; }
    if has_mond { rating += 10.0; }
    if has_pagat && tarok_count >= 6 { rating += 5.0; }

    rating += king_count as f64 * 8.0;
    rating += void_count as f64 * 7.0;
    rating += singletons as f64 * 2.0;

    // Suit length penalty
    for s in Suit::ALL {
        let count = hand.suit_count(s);
        if count >= 3 {
            let has_king = hand.contains(Card::suit_card(s, SuitRank::King));
            if !has_king {
                rating -= count as f64 * 2.0;
            }
        }
    }

    let max_rating = 120.0f64;
    let ratio = (rating / max_rating).min(1.0);

    let thresholds: [(Contract, f64); 7] = [
        (Contract::Three, 0.25),
        (Contract::Two, 0.32),
        (Contract::One, 0.40),
        (Contract::SoloThree, 0.52),
        (Contract::SoloTwo, 0.60),
        (Contract::SoloOne, 0.68),
        (Contract::Solo, 0.78),
    ];

    // Berač check
    let has_all_suits = Suit::ALL.iter().all(|&s| hand.has_suit(s));
    let can_berac = ratio < 0.18 && tarok_count <= 2 && has_all_suits && high_taroks == 0;

    let mut best: Option<Contract> = None;
    for &(contract, threshold) in &thresholds {
        if ratio >= threshold {
            best = Some(contract);
        }
    }
    if can_berac {
        if best.map_or(true, |b| Contract::Berac.strength() > b.strength()) {
            best = Some(Contract::Berac);
        }
    }

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
// King calling — same as v1 (with slight improvement)
// -----------------------------------------------------------------------

pub fn choose_king_v2(hand: CardSet) -> Option<Card> {
    let mut best_king: Option<Card> = None;
    let mut best_score = -1i32;

    for s in Suit::ALL {
        let king = Card::suit_card(s, SuitRank::King);
        if hand.contains(king) {
            continue;
        }
        let count = hand.suit_count(s) as i32;
        let low_cards = hand.suit(s).iter()
            .filter(|c| c.value() <= 4)
            .count() as i32;
        let score = count * 3 + low_cards;
        if score > best_score {
            best_score = score;
            best_king = Some(king);
        }
    }

    if best_king.is_none() {
        for s in Suit::ALL {
            let queen = Card::suit_card(s, SuitRank::Queen);
            if !hand.contains(queen) {
                best_king = Some(queen);
                break;
            }
        }
    }

    best_king
}

// -----------------------------------------------------------------------
// Talon — void-building aware
// -----------------------------------------------------------------------

fn evaluate_talon_group_v2(
    group: &[Card],
    hand: CardSet,
    called_king: Option<Card>,
) -> f64 {
    let called_suit = called_king.and_then(|k| k.suit());

    let mut total = 0.0f64;
    for &card in group {
        total += card.points() as f64 * 2.0 + worth_over(card) as f64 * 0.3;
        if card.card_type() == CardType::Tarok {
            total += 8.0;
        }
        if let (Some(cs), Some(s)) = (called_suit, card.suit()) {
            if s == cs {
                total += 4.0;
            }
        }
    }

    // Void potential
    for s in Suit::ALL {
        if Some(s) == called_suit { continue; }
        let hand_count = hand.suit_count(s);
        let group_adds = group.iter().filter(|c| c.suit() == Some(s)).count() as u32;
        if hand_count <= 1 && hand_count + group_adds <= 1 {
            total += 5.0;
        }
        if hand_count == 0 && group_adds == 0 {
            total += 3.0;
        }
    }

    total
}

pub fn choose_talon_group_v2(
    groups: &[Vec<Card>],
    hand: CardSet,
    called_king: Option<Card>,
) -> usize {
    let mut best_idx = 0;
    let mut best_score = f64::NEG_INFINITY;
    for (i, group) in groups.iter().enumerate() {
        let score = evaluate_talon_group_v2(group, hand, called_king);
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }
    best_idx
}

// -----------------------------------------------------------------------
// Discard — void-building focused
// -----------------------------------------------------------------------

pub fn choose_discards_v2(
    hand: CardSet,
    must_discard: usize,
    called_king: Option<Card>,
) -> Vec<Card> {
    let called_suit = called_king.and_then(|k| k.suit());

    let mut discardable: Vec<Card> = hand
        .iter()
        .filter(|c| c.card_type() != CardType::Tarok && !c.is_king())
        .collect();

    if discardable.len() < must_discard {
        let mut extra: Vec<Card> = hand
            .iter()
            .filter(|c| c.card_type() == CardType::Tarok && !c.is_trula() && !discardable.contains(c))
            .collect();
        extra.sort_by_key(|c| c.value());
        discardable.extend(extra);
    }

    // Group by suit for void-building
    let mut by_suit: [Vec<Card>; 4] = Default::default();
    for &c in &discardable {
        if let Some(s) = c.suit() {
            by_suit[s as usize].push(c);
        }
    }

    let mut result: Vec<Card> = Vec::with_capacity(must_discard);

    // Void shortest non-called suits first
    let mut suit_order: Vec<(Suit, u32)> = Suit::ALL.iter()
        .filter(|&&s| Some(s) != called_suit && !by_suit[s as usize].is_empty())
        .map(|&s| (s, hand.suit_count(s)))
        .collect();
    suit_order.sort_by_key(|&(_, count)| count);

    for (suit, _) in suit_order {
        if result.len() >= must_discard {
            break;
        }
        let suit_discardable: Vec<Card> = hand.suit(suit).iter()
            .filter(|c| !c.is_king() && discardable.contains(c) && !result.contains(c))
            .collect();
        if suit_discardable.len() + result.len() <= must_discard {
            result.extend(suit_discardable);
        }
    }

    // Fill remaining with lowest value
    if result.len() < must_discard {
        let mut remaining: Vec<Card> = discardable.iter()
            .filter(|c| !result.contains(c))
            .copied()
            .collect();
        remaining.sort_by_key(|c| (c.points(), worth_over(*c)));
        for c in remaining {
            if result.len() >= must_discard { break; }
            result.push(c);
        }
    }

    result.truncate(must_discard);
    result
}

// -----------------------------------------------------------------------
// Card play — v2 with card counting
// -----------------------------------------------------------------------

pub fn evaluate_card_play_v2(
    card: Card,
    hand: CardSet,
    state: &GameState,
    player: u8,
    is_leading: bool,
    tracker: &CardTracker,
) -> f64 {
    let wo = worth_over(card) as f64;
    let pts = card.points() as f64;

    let is_declarer = state.declarer == Some(player);
    let is_partner = state.partner == Some(player);
    let is_playing = is_declarer || is_partner;
    let is_klop = state.contract.map_or(false, |c| c.is_klop());
    let is_berac = state.contract.map_or(false, |c| c.is_berac());

    if is_klop || is_berac {
        return eval_klop_berac_v2(card, hand, state, player, is_leading, tracker);
    }

    if is_leading {
        eval_leading_v2(card, hand, state, player, is_playing, tracker)
    } else {
        eval_following_v2(card, hand, state, player, is_playing, tracker)
    }
}

fn eval_klop_berac_v2(
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
            let mut score = -wo * 2.0;
            if card.value() == PAGAT && tracker.taroks_in_hand > 1 {
                score -= 500.0;
            }
            score
        } else {
            let suit = card.suit().unwrap();
            let count = tracker.suit_counts[suit as usize] as f64;
            let remaining = tracker.remaining_in_suit(suit) as f64;
            let opponents_void = tracker.opponents_void_in(suit, player) as f64;

            if opponents_void > 0.0 {
                -wo * 3.0 - opponents_void * 20.0
            } else {
                -wo + remaining * 2.0 - count * 3.0
            }
        }
    } else if let Some(ref trick) = state.current_trick {
        if trick.count > 0 {
            let lead_suit = trick.lead_suit();
            if let Some(best) = trick.best_card() {
                let is_last = trick.count as usize == NUM_PLAYERS - 1;
                let trick_pts: f64 = (0..trick.count as usize)
                    .map(|i| trick.cards[i].1.points() as f64)
                    .sum::<f64>() + pts;

                if card.beats(best, lead_suit) {
                    if is_last && trick_pts <= 4.0 {
                        -wo
                    } else {
                        -wo * 3.0 - trick_pts * 10.0
                    }
                } else {
                    pts * 5.0 + wo * 0.2
                }
            } else {
                -wo
            }
        } else {
            -wo
        }
    } else {
        -wo
    }
}

#[allow(unused_variables)]
fn eval_leading_v2(
    card: Card,
    _hand: CardSet,
    _state: &GameState,
    _player: u8,
    is_playing: bool,
    tracker: &CardTracker,
) -> f64 {
    let wo = worth_over(card) as f64;
    let pts = card.points() as f64;

    if card.card_type() == CardType::Tarok {
        let higher_out = tracker.higher_taroks_out(card.value());

        if card.value() == SKIS {
            if tracker.taroks_remaining.len() > 2 { 80.0 } else { 40.0 }
        } else if card.value() == MOND {
            let skis_out = tracker.taroks_remaining.contains(Card::tarok(SKIS));
            if skis_out { -100.0 } else { 60.0 }
        } else if card.value() == PAGAT {
            if tracker.tricks_left <= 2 && tracker.taroks_in_hand <= 2 {
                200.0
            } else {
                -200.0
            }
        } else {
            // Regular tarok
            if higher_out == 0 {
                50.0 + wo
            } else if is_playing {
                (((wo - 11.0).max(0.0)) / 3.0).powf(1.5) + 5.0
            } else {
                if higher_out <= 1 { 30.0 + wo * 0.5 } else { wo * 0.3 }
            }
        }
    } else {
        let suit = card.suit().unwrap();
        let count = tracker.suit_counts[suit as usize] as f64;

        if is_playing {
            if card.is_king() {
                if tracker.suit_is_master(suit, card.value()) {
                    25.0 + pts * 2.0
                } else {
                    pts - 5.0
                }
            } else if count == 1.0 {
                15.0 - pts
            } else {
                -pts * 1.5 - count * 2.0
            }
        } else {
            if count >= 3.0 {
                if card.is_king() { pts * 2.0 } else { count * 3.0 - pts * 1.5 }
            } else if count == 1.0 {
                10.0 - pts
            } else {
                -count * 2.0 - pts
            }
        }
    }
}

fn eval_following_v2(
    card: Card,
    _hand: CardSet,
    state: &GameState,
    _player: u8,
    is_playing: bool,
    tracker: &CardTracker,
) -> f64 {
    let wo = worth_over(card) as f64;
    let pts = card.points() as f64;

    let trick = match &state.current_trick {
        Some(t) if t.count > 0 => t,
        _ => return 0.0,
    };

    let lead_suit = trick.lead_suit();
    let is_last = trick.count as usize == NUM_PLAYERS - 1;

    let mut best_card = trick.cards[0].1;
    let mut best_player = trick.cards[0].0;
    for i in 1..trick.count as usize {
        if trick.cards[i].1.beats(best_card, lead_suit) {
            best_card = trick.cards[i].1;
            best_player = trick.cards[i].0;
        }
    }

    let best_is_ally = if is_playing {
        best_player == state.declarer.unwrap_or(255)
            || state.partner == Some(best_player)
    } else {
        best_player != state.declarer.unwrap_or(255)
            && state.partner != Some(best_player)
    };

    let trick_pts: f64 = (0..trick.count as usize)
        .map(|i| trick.cards[i].1.points() as f64)
        .sum::<f64>() + pts;

    let would_win = card.beats(best_card, lead_suit);

    // Pagat
    if card.card_type() == CardType::Tarok && card.value() == PAGAT {
        if tracker.tricks_left <= 2 && tracker.taroks_in_hand <= 2 {
            return if would_win { 500.0 } else { -500.0 };
        }
        return -200.0;
    }

    // Mond
    if card.card_type() == CardType::Tarok && card.value() == MOND {
        if !would_win { return -500.0; }
    }

    // Škis
    if card.card_type() == CardType::Tarok && card.value() == SKIS {
        if tracker.tricks_left <= 1 { return -300.0; }
        return trick_pts * 2.0 + 30.0;
    }

    if would_win {
        if best_is_ally {
            if is_last {
                pts * 4.0 // šmir
            } else {
                -wo * 0.5 - (wo / 3.0).powf(1.5)
            }
        } else {
            if is_last {
                trick_pts * 3.0 - wo * 0.8
            } else {
                trick_pts * 1.5 + wo * 0.2
            }
        }
    } else {
        if best_is_ally {
            if is_last { pts * 5.0 } else { pts * 2.5 }
        } else {
            -(pts * 3.0) - wo * 0.3
        }
    }
}

/// Choose the best legal card (v2).
pub fn choose_card_v2(hand: CardSet, state: &GameState, player: u8) -> Card {
    let ctx = legal_moves::MoveCtx::from_state(state, player);
    let legal = legal_moves::generate_legal_moves(&ctx);
    let legal_vec: Vec<Card> = legal.iter().collect();

    if legal_vec.len() == 1 {
        return legal_vec[0];
    }

    let is_leading = state.current_trick.as_ref().map_or(true, |t| t.count == 0);
    let tracker = CardTracker::from_state(state, player);

    let mut best_card = legal_vec[0];
    let mut best_score = f64::NEG_INFINITY;

    for &card in &legal_vec {
        let score = evaluate_card_play_v2(card, hand, state, player, is_leading, &tracker);
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
