/// Centaur bidding heuristics — cloned from `stockskis_m6` so they can
/// evolve independently without affecting m6 or any other bot.
///
/// All functions in this module are intentional copies of their m6 counterparts
/// at the time of the fork.  Improve centaur bidding here; leave m6 alone.
use crate::card::*;
use crate::game_state::*;

// -----------------------------------------------------------------------
// Internal helpers
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

// -----------------------------------------------------------------------
// Bidding
// -----------------------------------------------------------------------

pub fn evaluate_bid_centaur(hand: CardSet, highest_so_far: Option<Contract>) -> Option<Contract> {
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

    let mut king_blocks_berac = true;
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
    if let Some(contract) = best {
        if contract.strength() >= Contract::One.strength() {
            let fives = five_point_card_count(hand);
            if fives < 2 || tarok_count < 6 {
                best = None;
                for &(c, t) in &thresholds {
                    if c.strength() < Contract::One.strength() && ratio >= t {
                        best = Some(c);
                    }
                }
            }
        }
    }

    // Solo gating: guaranteed points must exceed 30, and 4+ taroks.
    if let Some(contract) = best {
        if contract.is_solo() {
            let gp = guaranteed_points(hand);
            if gp < 30.0 || tarok_count < 6 {
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
// King calling
// -----------------------------------------------------------------------

pub fn choose_king_centaur(hand: CardSet) -> Option<Card> {
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
// Talon group selection
// -----------------------------------------------------------------------

fn evaluate_talon_group(group: &[Card], hand: CardSet, _called_king: Option<Card>) -> f64 {
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

pub fn choose_talon_group_centaur(
    groups: &[Vec<Card>],
    hand: CardSet,
    called_king: Option<Card>,
) -> usize {
    let mut best_idx = 0usize;
    let mut best_score = f64::NEG_INFINITY;
    for (idx, group) in groups.iter().enumerate() {
        let score = evaluate_talon_group(group, hand, called_king);
        if score > best_score {
            best_score = score;
            best_idx = idx;
        }
    }
    best_idx
}

// -----------------------------------------------------------------------
// Discarding
// -----------------------------------------------------------------------

pub fn choose_discards_centaur(
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
