/// StockŠkis m6 — refined heuristic bot based on v5/v6 with targeted tweaks.
///
/// Changes from v5/v6:
/// - **King calling**: Prefer suits with ~2 cards and high-value cards (queen,
///   jack) rather than maximum length.  Queen + 1 other card is the sweet spot.
/// - **Discarding**: Prioritize putting down point cards (queens, knights) to
///   secure points, rather than void-building.
/// - **Pagat caution**: Never lead pagat unless all remaining taroks are
///   accounted for.  Only attempt pagat ultimo with 10+ taroks.
/// - **Less aggressive play**: Toned-down multipliers on trick-point scoring
///   and more conservative tarok usage.
use crate::card::*;
use crate::game_state::*;
use crate::legal_moves;

// -----------------------------------------------------------------------
// Card tracking — same as v5/v6
// -----------------------------------------------------------------------

#[allow(dead_code)]
struct CardTracker {
    remaining: CardSet,
    taroks_in_hand: u8,
    taroks_remaining: CardSet,
    suit_counts: [u8; 4],
    player_voids: [[bool; 4]; NUM_PLAYERS],
    player_tarok_likelihood: [f64; NUM_PLAYERS],
    tricks_left: usize,
    phase: GamePhase,
    /// Estimated number of taroks this player started with.
    starting_taroks_estimate: u8,
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
        let mut player_played_tarok = [false; NUM_PLAYERS];
        let mut player_followed_suit_when_could = [0u32; NUM_PLAYERS];
        let mut player_total_follows = [0u32; NUM_PLAYERS];

        // Count taroks this player has played in completed tricks
        let mut own_taroks_played = 0u8;
        for trick in &state.tricks {
            if trick.count == 0 {
                continue;
            }
            let lead_card = trick.cards[0].1;
            for i in 0..trick.count as usize {
                let (p, c) = trick.cards[i];
                if p == player && c.card_type() == CardType::Tarok {
                    own_taroks_played += 1;
                }
            }
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
                let base = (avg_taroks_per_opponent / 4.0).min(1.0);
                player_tarok_likelihood[p] = base;
            }
        }

        let taroks_in_hand = hand.tarok_count();
        let tricks_left = (hand.len() as usize)
            .saturating_sub(state.tricks.len())
            .max(1);
        let phase = match state.tricks.len() {
            0..=3 => GamePhase::Early,
            4..=8 => GamePhase::Mid,
            _ => GamePhase::Late,
        };

        CardTracker {
            remaining,
            taroks_in_hand,
            taroks_remaining,
            suit_counts,
            player_voids,
            player_tarok_likelihood,
            tricks_left,
            phase,
            starting_taroks_estimate: taroks_in_hand + own_taroks_played,
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
}

// -----------------------------------------------------------------------
// Bidding — tighter gating for One and harder
// -----------------------------------------------------------------------

fn guaranteed_points(hand: CardSet) -> f64 {
    let mut pts = 0.0f64;
    let has_skis = hand.contains(Card::tarok(SKIS));
    let has_mond = hand.contains(Card::tarok(MOND));

    for suit in Suit::ALL {
        if hand.contains(Card::suit_card(suit, SuitRank::King)) {
            pts += 5.0;
        }
    }
    if has_skis {
        pts += 5.0;
    }
    if has_mond {
        if has_skis || hand.tarok_count() >= 6 {
            pts += 5.0;
        } else {
            pts += 2.0;
        }
    }
    pts
}

/// Count cards worth 5 points: kings, škis, mond, pagat.
fn five_point_card_count(hand: CardSet) -> u8 {
    let mut count = hand.king_count();
    if hand.contains(Card::tarok(SKIS)) {
        count += 1;
    }
    if hand.contains(Card::tarok(MOND)) {
        count += 1;
    }
    if hand.contains(Card::tarok(PAGAT)) {
        count += 1;
    }
    count
}

pub fn evaluate_bid_m6(hand: CardSet, highest_so_far: Option<Contract>) -> Option<Contract> {
    let tarok_count = hand.tarok_count();
    let high_taroks = hand.high_tarok_count();
    let king_count = hand.king_count();

    let has_skis = hand.contains(Card::tarok(SKIS));
    let has_mond = hand.contains(Card::tarok(MOND));
    let has_pagat = hand.contains(Card::tarok(PAGAT));

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
            rating -= 4.0;
        }
    }

    rating += king_count as f64 * 7.0;

    let mut has_all_suits = true;
    let mut has_void = false;
    let mut has_singleton = false;
    let mut max_tarok_val = 0u8;

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
            rating -= count as f64 * 2.5;
        }

        if has_king && has_queen && count >= 2 {
            rating += 2.0;
        }
    }

    if tarok_count >= 8 {
        rating += (tarok_count - 7) as f64 * 3.0;
    }

    let ratio = (rating / 130.0).min(1.0);

    let mut king_blocks_berac = false;
    for suit in Suit::ALL {
        let has_king = hand.contains(Card::suit_card(suit, SuitRank::King));
        if has_king && hand.suit_count(suit) < 5 {
            king_blocks_berac = true;
        }
    }

    let can_berac = ratio < 0.16
        && tarok_count <= 2
        && max_tarok_val < 9
        && has_all_suits
        && !has_void
        && !has_singleton
        && !king_blocks_berac
        && high_taroks == 0;

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

    // Gate: One or harder requires at least 2 five-point cards and 6+ taroks.
    // Five-point cards are kings, škis, mond, pagat.
    if let Some(contract) = best {
        if contract.strength() >= Contract::One.strength() {
            let fives = five_point_card_count(hand);
            if fives < 2 || tarok_count < 6 {
                // Downgrade to best contract below One
                best = None;
                for &(c, t) in &thresholds {
                    if c.strength() < Contract::One.strength() && ratio >= t {
                        best = Some(c);
                    }
                }
            }
        }
    }

    // Solo gating: guaranteed points must exceed 30, and 4+ taroks
    if let Some(contract) = best {
        if contract.is_solo() {
            let gp = guaranteed_points(hand);
            if gp < 30.0 || tarok_count < 4 {
                best = None;
                for &(c, t) in &thresholds {
                    if !c.is_solo() && ratio >= t {
                        best = Some(c);
                    }
                }
            }
        }
    }

    if can_berac
        && best.map_or(true, |current| {
            Contract::Berac.strength() > current.strength()
        })
    {
        best = Some(Contract::Berac);
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
// King calling — prefer ~2 cards with queen/jack, not longest suit
// -----------------------------------------------------------------------

pub fn choose_king_m6(hand: CardSet) -> Option<Card> {
    let mut best_king: Option<Card> = None;
    let mut best_score = -1i32;

    for suit in Suit::ALL {
        let king = Card::suit_card(suit, SuitRank::King);
        if hand.contains(king) {
            continue;
        }
        let count = hand.suit_count(suit) as i32;
        let has_queen = hand.contains(Card::suit_card(suit, SuitRank::Queen));
        let has_knight = hand.contains(Card::suit_card(suit, SuitRank::Knight));
        let has_jack = hand.contains(Card::suit_card(suit, SuitRank::Jack));

        let count_score = match count {
            0 => 0,
            1 => 4,
            2 => 10,
            3 => 6,
            _ => 2,
        };

        let high_bonus = if has_queen { 8 } else { 0 }
            + if has_knight { 5 } else { 0 }
            + if has_jack { 3 } else { 0 };

        let synergy = if has_queen && count == 2 { 4 } else { 0 };

        let score = count_score + high_bonus + synergy;
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
// Talon — same as v6
// -----------------------------------------------------------------------

fn evaluate_talon_group_m6(group: &[Card], hand: CardSet, _called_king: Option<Card>) -> f64 {
    let mut score = 0.0;
    if group.contains(&Card::tarok(MOND)) {
        score += 100000.0;
    }
    if group.contains(&Card::tarok(SKIS)) {
        score += 10000.0;
    }
    if group.contains(&Card::tarok(PAGAT)) {
        score += 1000.0;
    }

    for &c in group {
        if c.card_type() == CardType::Tarok {
            score += 5.0;
            if c.value() >= 15 {
                score += 10.0;
            }
        } else if c.is_king() {
            let mut king_score = 25.0;
            let suit_count = hand.suit_count(c.suit().unwrap())
                + group.iter().filter(|&&g| g.suit() == c.suit()).count() as u32;
            if suit_count >= 3 {
                king_score -= 15.0;
            }
            score += king_score;
        } else if c.value() == SuitRank::Queen as u8 {
            score += 2.0;
        } else {
            score += 1.0;
        }
    }
    score
}

pub fn choose_talon_group_m6(
    groups: &[Vec<Card>],
    hand: CardSet,
    called_king: Option<Card>,
) -> usize {
    let mut best_idx = 0usize;
    let mut best_score = f64::NEG_INFINITY;
    for (idx, group) in groups.iter().enumerate() {
        let score = evaluate_talon_group_m6(group, hand, called_king);
        if score > best_score {
            best_score = score;
            best_idx = idx;
        }
    }
    best_idx
}

// -----------------------------------------------------------------------
// Discard — prioritize putting down point cards to secure points
// -----------------------------------------------------------------------

pub fn choose_discards_m6(
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

    discardable.sort_by(|a, b| {
        let a_called = a.suit() == called_suit && called_suit.is_some();
        let b_called = b.suit() == called_suit && called_suit.is_some();
        if a_called != b_called {
            return a_called.cmp(&b_called);
        }
        // Queens (4pts) > Knights (3pts) > Jacks (2pts) > Pips (1pt)
        b.points().cmp(&a.points())
    });

    discardable.into_iter().take(must_discard).collect()
}

// -----------------------------------------------------------------------
// Card play
// -----------------------------------------------------------------------

#[allow(private_interfaces)]
pub fn evaluate_card_play_m6(
    card: Card,
    hand: CardSet,
    state: &GameState,
    player: u8,
    is_leading: bool,
    tracker: &CardTracker,
) -> f64 {
    let is_declarer = state.declarer == Some(player);
    let is_partner = state.partner == Some(player);
    let is_klop = state.contract.map_or(false, |contract| contract.is_klop());
    let is_berac = state.contract.map_or(false, |contract| contract.is_berac());

    if is_klop || is_berac {
        return eval_klop_berac_m6(card, hand, state, player, is_leading, tracker);
    }

    if is_leading {
        eval_leading_m6(card, hand, state, player, is_declarer, is_partner, tracker)
    } else {
        eval_following_m6(card, hand, state, player, is_declarer, is_partner, tracker)
    }
}

fn eval_klop_berac_m6(
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

fn eval_leading_m6(
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

        // --- Pagat: only lead if guaranteed safe ---
        if card.value() == PAGAT {
            // Last trick and we have it — play it
            if tracker.tricks_left == 1 {
                return 400.0;
            }
            // Only lead pagat if no higher taroks remain anywhere (guaranteed win)
            if higher_out == 0 && tracker.taroks_remaining.len() == 0 {
                return 300.0;
            }
            // With 10+ starting taroks and late game, allow ultimo attempt
            if tracker.starting_taroks_estimate >= 10
                && tracker.tricks_left <= 2
                && tracker.taroks_in_hand <= 2
            {
                return 250.0;
            }
            // Otherwise never lead pagat
            return -400.0;
        }

        // --- Partner leading ---
        if is_partner {
            if card.value() == SKIS {
                return match tracker.phase {
                    GamePhase::Early => 120.0,
                    GamePhase::Mid => 95.0,
                    GamePhase::Late => {
                        if tracker.tricks_left <= 1 {
                            -300.0
                        } else {
                            60.0
                        }
                    }
                };
            }
            if card.value() == MOND {
                let skis_out = tracker.taroks_remaining.contains(Card::tarok(SKIS));
                let has_skis = hand.contains(Card::tarok(SKIS));
                if skis_out && !has_skis {
                    return 65.0;
                }
                return 85.0 + tracker.taroks_remaining.len() as f64 * 1.5;
            }
            return 80.0 + wo * 1.2 - higher_out as f64 * 5.0;
        }

        // --- Opposition leading ---
        if is_opposition {
            if higher_out == 0 {
                if tracker.phase == GamePhase::Late {
                    return 22.0 + wo * 0.2;
                }
                return 16.0 + wo * 0.15;
            }
            if higher_out <= 1 && tracker.tricks_left <= 3 {
                return 6.0 + wo * 0.1;
            }
            return -55.0 - wo * 0.6 - higher_out as f64 * 8.0;
        }

        // --- Declarer leading ---
        if card.value() == SKIS {
            return match tracker.phase {
                GamePhase::Early => 85.0,
                GamePhase::Mid => 65.0,
                GamePhase::Late => {
                    if tracker.tricks_left <= 1 {
                        -300.0
                    } else {
                        50.0
                    }
                }
            };
        }

        if card.value() == MOND {
            let skis_out = tracker.taroks_remaining.contains(Card::tarok(SKIS));
            let has_skis = hand.contains(Card::tarok(SKIS));
            if skis_out && !has_skis {
                return -180.0;
            }
            if has_skis {
                return 60.0;
            }
            return 65.0 + tracker.taroks_remaining.len() as f64 * 2.5;
        }

        if higher_out == 0 {
            return 55.0 + wo * 0.8;
        }

        let base = ((wo - 11.0).max(0.0) / 3.5).powf(1.5);
        if tracker.phase == GamePhase::Early {
            base + 8.0
        } else {
            base
        }
    } else {
        let suit = card.suit().unwrap();
        let count = tracker.suit_counts[suit as usize] as f64;
        let remaining = tracker.remaining_in_suit(suit) as f64;

        if is_partner {
            if card.is_king() && tracker.suit_is_master(suit, card.value()) {
                16.0 + pts
            } else if count == 1.0 {
                5.0 - pts
            } else if count == 2.0 && !card.is_king() {
                3.0 - pts
            } else {
                -pts - count
            }
        } else if is_declarer {
            if card.is_king() {
                if tracker.suit_is_master(suit, card.value()) {
                    26.0 + pts * 1.5
                } else if count >= 3.0 {
                    12.0 + pts
                } else {
                    pts - 10.0
                }
            } else if count == 1.0 {
                15.0 - pts
            } else if count == 2.0 && !card.is_king() {
                10.0 - pts
            } else {
                -pts * 1.5 - count * 2.0
            }
        } else {
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

            let mut score = 15.0 - pts * 4.0 - count * 1.5;
            if card.is_king() {
                score -= 25.0;
            }
            if card.value() <= SuitRank::Pip2 as u8 {
                score += 7.0;
            }
            if declarer_team_voids > 0.0 {
                score += declarer_team_voids * 12.0 - pts * 2.0;
            } else if remaining > 0.0 {
                score += remaining.min(4.0);
            }

            if let Some(declarer) = state.declarer {
                let dl = tracker.player_tarok_likelihood[declarer as usize];
                if dl < 0.3 {
                    score += 4.0;
                }
            }

            score
        }
    }
}

fn eval_following_m6(
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

    // === Pagat: very careful handling ===
    if card.card_type() == CardType::Tarok && card.value() == PAGAT {
        // Last trick: play it if it wins
        if tracker.tricks_left == 1 {
            return if would_win { 600.0 } else { -600.0 };
        }
        // Ultimo attempt: only with 10+ starting taroks and near the end
        if tracker.starting_taroks_estimate >= 10
            && tracker.tricks_left <= 2
            && tracker.taroks_in_hand <= 2
        {
            return if would_win { 500.0 } else { -500.0 };
        }
        // Can safely play if ally is winning (šmir the pagat to partner's trick)
        if best_is_ally && !would_win {
            // Only šmir pagat if no taroks remain (safe from capture next trick)
            if tracker.taroks_remaining.len() == 0 {
                return 200.0;
            }
        }
        // Otherwise: do not play pagat unless absolutely forced
        return -400.0;
    }

    // Mond
    if card.card_type() == CardType::Tarok && card.value() == MOND {
        if !would_win {
            return -600.0;
        }
        let skis_out = tracker.taroks_remaining.contains(Card::tarok(SKIS));
        if skis_out && !is_last {
            return -120.0;
        }
        return trick_pts * 1.8 + 35.0;
    }

    // Škis
    if card.card_type() == CardType::Tarok && card.value() == SKIS {
        if tracker.tricks_left <= 1 {
            return -400.0;
        }
        return trick_pts * 2.2 + 30.0;
    }

    // === Partner preservation: don't overtrump ally if outcome unchanged ===
    if would_win && best_is_ally && card.card_type() == CardType::Tarok {
        if !is_last {
            let opponents_remaining = NUM_PLAYERS - 1 - num_played;
            if opponents_remaining > 0 {
                let ally_val = if best_card.card_type() == CardType::Tarok {
                    best_card.value()
                } else {
                    0
                };
                let higher_than_ally_out = tracker.higher_taroks_out(ally_val);
                if higher_than_ally_out == 0 {
                    return -wo * 2.0 - 50.0;
                }
            }
        } else {
            return -wo * 2.0 - 50.0;
        }
    }

    // === Last-seat economy: play lowest winning tarok ===
    if would_win && is_last && !best_is_ally && card.card_type() == CardType::Tarok {
        let min_winning_val = if best_card.card_type() == CardType::Tarok {
            best_card.value() + 1
        } else {
            1
        };
        if card.value() == min_winning_val || card.value() <= min_winning_val + 1 {
            return trick_pts * 3.5 + 45.0 - wo * 0.1;
        }
        let waste = (card.value() - min_winning_val) as f64;
        return trick_pts * 2.5 - waste * 9.0;
    }

    // === Normal following (slightly less aggressive multipliers) ===
    if would_win {
        if best_is_ally {
            if is_last {
                pts * 4.0
            } else {
                -wo * 0.7 - (wo / 3.0).powf(1.5)
            }
        } else if is_last {
            trick_pts * 3.0 - wo
        } else if num_played == 1 {
            if card.card_type() == CardType::Tarok && tracker.higher_taroks_out(card.value()) > 0 {
                trick_pts * 0.5 - wo * 0.5
            } else {
                trick_pts * 1.7 + wo * 0.15
            }
        } else {
            trick_pts * 1.7 + wo * 0.2
        }
    } else if best_is_ally {
        // Šmir
        if is_last {
            pts * 5.5
        } else if tracker.phase == GamePhase::Late {
            pts * 4.0
        } else {
            pts * 2.8
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
        if card.card_type() != CardType::Tarok {
            let mut opp_tarok_pressure = 0.0;
            for p in 0..NUM_PLAYERS {
                if p as u8 != player && !is_ally(p as u8, state, is_playing) {
                    opp_tarok_pressure += tracker.player_tarok_likelihood[p];
                }
            }
            if opp_tarok_pressure < 0.3 {
                score += 2.0;
            }
        }
        score
    }
}

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

pub fn choose_card_m6(hand: CardSet, state: &GameState, player: u8) -> Card {
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
        let score = evaluate_card_play_m6(card, hand, state, player, is_leading, &tracker);
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
// Announcements — pagat ultimo only with 10+ taroks
// -----------------------------------------------------------------------

pub fn choose_announcements_m6(hand: CardSet, state: &GameState, player: u8) -> Vec<u8> {
    let mut anns = Vec::new();

    // King ultimo for partner holding the called king
    if let Some(partner) = state.partner {
        if player == partner {
            if let Some(king) = state.called_king {
                if hand.contains(king) {
                    if let Some(suit) = king.suit() {
                        if hand.suit_count(suit) >= 3 {
                            anns.push(4); // King Ultimo
                        }
                    }
                }
            }
        }
    }

    // Pagat ultimo: only if we hold pagat AND started with 10+ taroks
    let has_pagat = hand.contains(Card::tarok(PAGAT));
    if has_pagat && hand.tarok_count() >= 10 {
        anns.push(3); // Pagat Ultimo
    }

    anns
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

    // --- King calling: prefer queen/jack with ~2 cards ---

    #[test]
    fn king_call_prefers_queen_with_two_cards() {
        let hand = hand_from(&[
            Card::tarok(15),
            Card::tarok(10),
            Card::tarok(5),
            Card::tarok(3),
            Card::tarok(8),
            Card::tarok(12),
            // Hearts: queen + 1 pip = 2 cards, no king
            Card::suit_card(Suit::Hearts, SuitRank::Queen),
            Card::suit_card(Suit::Hearts, SuitRank::Pip1),
            // Diamonds: 4 pips, no king
            Card::suit_card(Suit::Diamonds, SuitRank::Pip1),
            Card::suit_card(Suit::Diamonds, SuitRank::Pip2),
            Card::suit_card(Suit::Diamonds, SuitRank::Pip3),
            Card::suit_card(Suit::Diamonds, SuitRank::Pip4),
        ]);
        let chosen = choose_king_m6(hand);
        assert_eq!(
            chosen,
            Some(Card::suit_card(Suit::Hearts, SuitRank::King)),
            "Should call king of hearts (queen + 2 cards)"
        );
    }

    #[test]
    fn king_call_prefers_jack_over_empty_long_suit() {
        let hand = hand_from(&[
            Card::tarok(15),
            Card::tarok(10),
            Card::tarok(5),
            Card::tarok(3),
            Card::tarok(8),
            Card::tarok(12),
            Card::tarok(18),
            // Spades: jack + 1 pip = 2 cards
            Card::suit_card(Suit::Spades, SuitRank::Jack),
            Card::suit_card(Suit::Spades, SuitRank::Pip1),
            // Clubs: 3 low pips
            Card::suit_card(Suit::Clubs, SuitRank::Pip1),
            Card::suit_card(Suit::Clubs, SuitRank::Pip2),
            Card::suit_card(Suit::Clubs, SuitRank::Pip3),
        ]);
        let chosen = choose_king_m6(hand);
        assert_eq!(
            chosen,
            Some(Card::suit_card(Suit::Spades, SuitRank::King)),
            "Should call king of spades (jack + 2 cards = high card bonus)"
        );
    }

    // --- Discarding: queens first for points ---

    #[test]
    fn discard_puts_down_queens_for_points() {
        let hand = hand_from(&[
            Card::tarok(15),
            Card::tarok(10),
            Card::tarok(5),
            Card::tarok(3),
            Card::tarok(8),
            Card::tarok(12),
            Card::tarok(18),
            Card::suit_card(Suit::Hearts, SuitRank::Queen),
            Card::suit_card(Suit::Hearts, SuitRank::Pip1),
            Card::suit_card(Suit::Diamonds, SuitRank::Pip1),
            Card::suit_card(Suit::Clubs, SuitRank::Pip1),
            Card::suit_card(Suit::Spades, SuitRank::Pip1),
        ]);
        let discards = choose_discards_m6(hand, 3, None);
        assert!(
            discards.contains(&Card::suit_card(Suit::Hearts, SuitRank::Queen)),
            "Should discard queen to secure 4 points"
        );
    }

    // --- Pagat: never lead unless guaranteed ---

    #[test]
    fn never_lead_pagat_with_taroks_remaining() {
        let hand = hand_from(&[
            Card::tarok(PAGAT),
            Card::tarok(5),
            Card::tarok(10),
            Card::suit_card(Suit::Hearts, SuitRank::Pip1),
        ]);
        let state = make_state(hand, 0, 0, None);
        let chosen = choose_card_m6(hand, &state, 0);
        assert_ne!(
            chosen,
            Card::tarok(PAGAT),
            "Should never lead pagat when opponents likely have higher taroks"
        );
    }

    #[test]
    fn pagat_ok_on_last_trick() {
        let hand = hand_from(&[Card::tarok(PAGAT)]);
        let mut state = make_state(hand, 0, 0, None);
        state.contract = Some(Contract::SoloThree);
        // Simulate that 11 tricks have been played (last trick)
        for _ in 0..11 {
            let mut trick = Trick::new(0);
            trick.play(0, Card::tarok(2)); // placeholder
            trick.play(1, Card::tarok(3));
            trick.play(2, Card::tarok(4));
            trick.play(3, Card::tarok(5));
            state.tricks.push(trick);
        }
        let chosen = choose_card_m6(hand, &state, 0);
        assert_eq!(
            chosen,
            Card::tarok(PAGAT),
            "Pagat is fine on the very last trick"
        );
    }

    // --- Less aggressive: opposition preserves taroks ---

    #[test]
    fn opposition_preserves_taroks_on_lead() {
        let hand = hand_from(&[
            Card::tarok(20),
            Card::tarok(10),
            Card::suit_card(Suit::Hearts, SuitRank::Pip1),
            Card::suit_card(Suit::Spades, SuitRank::Pip2),
        ]);
        let state = make_state(hand, 2, 0, Some(1));
        let chosen = choose_card_m6(hand, &state, 2);
        assert_ne!(
            chosen.card_type(),
            CardType::Tarok,
            "Opposition should lead suits, not taroks"
        );
    }

    #[test]
    fn do_not_overtrump_winning_ally() {
        let hand = hand_from(&[
            Card::tarok(20),
            Card::tarok(10),
            Card::suit_card(Suit::Hearts, SuitRank::Pip1),
            Card::suit_card(Suit::Clubs, SuitRank::Pip2),
        ]);
        let mut state = make_state(hand, 2, 0, Some(2));
        let mut trick = Trick::new(3);
        trick.play(3, Card::suit_card(Suit::Diamonds, SuitRank::King));
        trick.play(0, Card::tarok(18));
        trick.play(1, Card::suit_card(Suit::Diamonds, SuitRank::Pip1));
        state.current_trick = Some(trick);

        let chosen = choose_card_m6(hand, &state, 2);
        assert_ne!(chosen, Card::tarok(20));
    }

    #[test]
    fn last_seat_plays_lowest_winning_tarok() {
        let hand = hand_from(&[
            Card::tarok(11),
            Card::tarok(15),
            Card::tarok(19),
            Card::suit_card(Suit::Hearts, SuitRank::Pip1),
        ]);
        let mut state = make_state(hand, 3, 3, None);
        state.contract = Some(Contract::SoloThree);
        let mut trick = Trick::new(0);
        trick.play(0, Card::suit_card(Suit::Diamonds, SuitRank::Pip1));
        trick.play(1, Card::suit_card(Suit::Diamonds, SuitRank::Pip2));
        trick.play(2, Card::tarok(10));
        state.current_trick = Some(trick);

        let chosen = choose_card_m6(hand, &state, 3);
        assert_eq!(chosen, Card::tarok(11));
    }
}
