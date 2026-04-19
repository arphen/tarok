/// StockŠkis heuristic bot — pure Rust port of the Python StockSkisPlayer.
///
/// This implements the same hand-evaluation and card-play heuristics from
/// the original StockŠkis engine (mytja/Tarok), ported from the Python
/// version in `backend/src/tarok/adapters/ai/stockskis_player.py`.
///
/// Used by the expert-game generator to play millions of competent games
/// at full Rust speed (~350K games/sec) for imitation learning pre-training.

use crate::card::*;
use crate::game_state::*;
use crate::legal_moves;

// -----------------------------------------------------------------------
// Card ordering (StockŠkis "worthOver")
// -----------------------------------------------------------------------

/// Global ordering value matching StockŠkis' worthOver field.
/// Taroks: 11 (Pagat) .. 32 (Škis)
/// Suits:  Hearts 33..40, Diamonds 41..48, Clubs 49..56, Spades 57..64
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
// Hand evaluation for bidding
// -----------------------------------------------------------------------

/// Evaluate hand and return the strongest biddable contract, or None to pass.
pub fn evaluate_bid_v1(hand: CardSet, highest_so_far: Option<Contract>) -> Option<Contract> {
    let mut taroks = 0u32;
    let mut my_rating = 0i64;

    for card in hand.iter() {
        let wo = worth_over(card) as i64;
        let pts = card.points() as i64;
        my_rating += wo + pts * pts;
        if card.card_type() == CardType::Tarok {
            taroks += 1;
        }
    }

    // Max rating: best 12 cards from a 54-card deck (pre-computed approximation).
    // The top 12 cards are: Škis(32+25), Mond(31+25), plus 4 kings(~62+25 each),
    // plus next highest suit cards. We use a fixed constant.
    let max_rating = compute_max_rating(hand.len() as usize);
    if max_rating == 0 {
        return None;
    }

    let ratio = my_rating as f64 / max_rating as f64;
    let t = taroks as f64;

    // Thresholds from StockŠkis (4-player NORMALNI path)
    let thresholds: [(Contract, f64); 7] = [
        (Contract::Three, 0.29 - t * 0.003),
        (Contract::Two, 0.33 - t * 0.0035),
        (Contract::One, 0.42 - t * 0.004),
        (Contract::SoloThree, 0.55 - t * 0.007),
        (Contract::SoloTwo, 0.62 - t * 0.007),
        (Contract::SoloOne, 0.68 - t * 0.007),
        (Contract::Solo, 0.80),
    ];

    // Berač check: weak hand with all suits present
    let mut suit_counts = [0u32; 4];
    for card in hand.iter() {
        if let Some(s) = card.suit() {
            suit_counts[s as usize] += 1;
        }
    }
    let has_all_suits = suit_counts.iter().all(|&c| c > 0);
    let can_berac = ratio < 0.20 && taroks <= 3 && has_all_suits;

    // Find the highest qualifying contract
    let mut best: Option<Contract> = None;
    for &(contract, threshold) in &thresholds {
        if ratio >= threshold {
            best = Some(contract);
        }
    }
    if can_berac {
        // Berac beats everything up to Solo in strength
        if best.map_or(true, |b| Contract::Berac.strength() > b.strength()) {
            best = Some(Contract::Berac);
        }
    }

    // Must be stronger than the current highest bid
    if let Some(contract) = best {
        if let Some(highest) = highest_so_far {
            if contract.strength() <= highest.strength() {
                return None; // Can't outbid
            }
        }
        Some(contract)
    } else {
        None
    }
}

/// Compute max possible rating for `n` cards from the best cards in the deck.
fn compute_max_rating(n: usize) -> i64 {
    let mut ratings: Vec<i64> = Vec::with_capacity(DECK_SIZE);
    for i in 0..DECK_SIZE as u8 {
        let card = Card(i);
        let wo = worth_over(card) as i64;
        let pts = card.points() as i64;
        ratings.push(wo + pts * pts);
    }
    ratings.sort_unstable_by(|a, b| b.cmp(a));
    ratings.iter().take(n).sum()
}

// -----------------------------------------------------------------------
// King calling heuristic
// -----------------------------------------------------------------------

/// Call king of the suit where we have the most cards.
pub fn choose_king_v1(hand: CardSet) -> Option<Card> {
    let mut best_king: Option<Card> = None;
    let mut best_count = -1i32;

    for s in Suit::ALL {
        let king = Card::suit_card(s, SuitRank::King);
        if hand.contains(king) {
            continue; // Can't call a king we already have
        }
        let count = hand.suit_count(s) as i32;
        if count > best_count {
            best_count = count;
            best_king = Some(king);
        }
    }

    // Fallback to queens if we have all kings
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
// Talon selection heuristic
// -----------------------------------------------------------------------

/// Score a talon group. Higher = better pick for declarer.
fn evaluate_talon_group(group: &[Card], called_king: Option<Card>) -> f64 {
    let mut total = 0.0f64;
    for &card in group {
        total += card.points() as f64 * 2.0 + worth_over(card) as f64 * 0.3;
        if card.card_type() == CardType::Tarok {
            total += 5.0;
        }
        if let (Some(king), Some(card_suit)) = (called_king, card.suit()) {
            if king.suit() == Some(card_suit) {
                total += 3.0;
            }
        }
    }
    total
}

/// Choose best talon group index.
pub fn choose_talon_group_v1(groups: &[Vec<Card>], _hand: CardSet, called_king: Option<Card>) -> usize {
    let mut best_idx = 0;
    let mut best_score = f64::NEG_INFINITY;
    for (i, group) in groups.iter().enumerate() {
        let score = evaluate_talon_group(group, called_king);
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }
    best_idx
}

// -----------------------------------------------------------------------
// Discard heuristic
// -----------------------------------------------------------------------

/// Choose cards to discard. Prefers low-value suit cards, void-building.
pub fn choose_discards_v1(hand: CardSet, must_discard: usize, called_king: Option<Card>) -> Vec<Card> {
    let called_suit = called_king.and_then(|k| k.suit());

    let mut discardable: Vec<Card> = hand
        .iter()
        .filter(|c| c.card_type() != CardType::Tarok && !c.is_king())
        .collect();

    // If not enough non-tarok non-king, add lowest taroks (not trula)
    if discardable.len() < must_discard {
        let mut extra: Vec<Card> = hand
            .iter()
            .filter(|c| {
                c.card_type() == CardType::Tarok
                    && !c.is_trula()
                    && !discardable.contains(c)
            })
            .collect();
        extra.sort_by_key(|c| c.value());
        discardable.extend(extra);
    }

    // Sort by discard priority: prefer voiding suits (except called suit)
    discardable.sort_by(|a, b| {
        let a_penalty = if a.suit() == called_suit { 50i32 } else { 0 };
        let b_penalty = if b.suit() == called_suit { 50i32 } else { 0 };
        let a_count = a.suit().map_or(99, |s| hand.suit_count(s) as i32);
        let b_count = b.suit().map_or(99, |s| hand.suit_count(s) as i32);
        let a_key = (a_penalty + a_count, a.points());
        let b_key = (b_penalty + b_count, b.points());
        a_key.cmp(&b_key)
    });

    discardable.into_iter().take(must_discard).collect()
}

// -----------------------------------------------------------------------
// Card play heuristic
// -----------------------------------------------------------------------

/// Evaluate a card for playing. Higher score = better choice.
pub fn evaluate_card_play(
    card: Card,
    hand: CardSet,
    state: &GameState,
    player: u8,
    is_leading: bool,
) -> f64 {
    let wo = worth_over(card) as f64;
    let pts = card.points() as f64;

    let is_declarer = state.declarer == Some(player);
    let is_partner = state.partner == Some(player);
    let is_playing = is_declarer || is_partner;
    let is_klop = state.contract.map_or(false, |c| c.is_klop());
    let is_berac = state.contract.map_or(false, |c| c.is_berac());

    let taroks_in_hand = hand.tarok_count() as i32;

    let mut score = 0.0f64;

    // === KLOP / BERAC: avoid taking tricks ===
    if is_klop || is_berac {
        if is_leading {
            if card.card_type() == CardType::Tarok {
                score = -wo;
                // Protect pagat
                if card.value() == PAGAT && taroks_in_hand > 1 {
                    score -= 1000.0;
                }
            } else {
                let count = card.suit().map_or(1, |s| hand.suit_count(s)) as f64;
                score = -wo - count * 5.0;
            }
        } else if let Some(ref trick) = state.current_trick {
            if trick.count > 0 {
                let lead_suit = trick.lead_suit();
                if let Some(best) = trick.best_card() {
                    if card.beats(best, lead_suit) {
                        score = -wo * 3.0 - pts * 10.0;
                    } else {
                        score = pts + wo * 0.1;
                    }
                } else {
                    score = -wo;
                }
            } else {
                score = -wo;
            }
        }
        return score;
    }

    // === NORMAL GAMES ===
    if is_leading {
        if card.card_type() == CardType::Tarok {
            let adjusted = (wo - 11.0).max(0.0);
            score = (adjusted / 3.0_f64).powf(1.5);
            if card.value() == PAGAT && taroks_in_hand > 1 {
                score -= 50.0;
            }
            if card.value() == MOND && !is_playing {
                score -= 20.0;
            }
        } else {
            let count = card.suit().map_or(1, |s| hand.suit_count(s)) as f64;
            if is_playing {
                score = pts * 2.0 - count * 3.0;
                if card.is_king() {
                    score += 10.0;
                }
            } else {
                let suit_worth: f64 = hand
                    .iter()
                    .filter(|c| c.suit() == card.suit())
                    .map(|c| c.points() as f64)
                    .sum();
                score = suit_worth - count.powf(1.5) - (pts / 2.0).powi(2);
            }
        }
    } else if let Some(ref trick) = state.current_trick {
        if trick.count == 0 {
            return 0.0;
        }

        let lead_suit = trick.lead_suit();
        let best_card = trick.best_card().unwrap();
        let best_player = {
            let mut bp = trick.cards[0].0;
            let mut bc = trick.cards[0].1;
            for i in 1..trick.count as usize {
                if trick.cards[i].1.beats(bc, lead_suit) {
                    bc = trick.cards[i].1;
                    bp = trick.cards[i].0;
                }
            }
            bp
        };

        let best_is_ally = if is_playing {
            best_player == state.declarer.unwrap_or(255)
                || state.partner == Some(best_player)
        } else {
            best_player != state.declarer.unwrap_or(255)
                && state.partner != Some(best_player)
        };

        let trick_pts: f64 = (0..trick.count as usize)
            .map(|i| trick.cards[i].1.points() as f64)
            .sum::<f64>()
            + pts;

        let would_win = card.beats(best_card, lead_suit);

        let mut penalty = 0.0f64;

        if would_win {
            if is_playing {
                score = trick_pts * 2.0 + wo * 0.5;
            } else if !best_is_ally {
                score = trick_pts * 2.0 + wo * 0.3;
            } else {
                score = -wo * 0.5;
                penalty += (wo / 3.0).powf(1.5);
            }
        } else if best_is_ally {
            score = pts * 3.0 + wo * 0.2;
        } else {
            score = -(pts * 2.0) - wo * 0.3;
        }

        // Pagat ultimo awareness
        if card.card_type() == CardType::Tarok && card.value() == PAGAT {
            if state.tricks.len() < TRICKS_PER_GAME - 1 {
                penalty += 200.0;
            }
        }

        // Mond protection
        if card.card_type() == CardType::Tarok && card.value() == MOND && !would_win {
            penalty += 500.0;
        }

        // Škis: good to play early, bad in last trick
        if card.card_type() == CardType::Tarok && card.value() == SKIS {
            if state.tricks.len() >= TRICKS_PER_GAME - 1 {
                penalty += 300.0;
            } else {
                score += trick_pts;
            }
        }

        score -= penalty;
    }

    score
}

/// Choose the best legal card to play.
pub fn choose_card_v1(hand: CardSet, state: &GameState, player: u8) -> Card {
    let ctx = legal_moves::MoveCtx::from_state(state, player);
    let legal = legal_moves::generate_legal_moves(&ctx);
    let legal_vec: Vec<Card> = legal.iter().collect();

    if legal_vec.len() == 1 {
        return legal_vec[0];
    }

    let is_leading = state
        .current_trick
        .as_ref()
        .map_or(true, |t| t.count == 0);

    let mut best_card = legal_vec[0];
    let mut best_score = f64::NEG_INFINITY;

    for &card in &legal_vec {
        let score = evaluate_card_play(card, hand, state, player, is_leading);
        if score > best_score {
            best_score = score;
            best_card = card;
        }
    }

    best_card
}
