/// StockŠkis-v3 ported to **3-player Tarok** (`tarok v treh`).
///
/// Differences vs the 4-player [`super::stockskis_v3`] bot:
///
///  - **No king-calling** — there is no partner. Every non-klop/non-berac
///    contract is 1-vs-2 (declarer alone vs the other two seats).
///  - **No 2v2 contracts** — Three/Two/One/Solo are removed; the bid menu is
///    `{Berac, SoloThree, SoloTwo, SoloOne}` plus an implicit Klop on
///    all-pass and BarvniValat as a post-deal announcement (see
///    [`Contract::BIDDABLE_THREE_PLAYER`]).
///  - **16-card hands** vs 12 — phase boundaries and tarok-density bonuses
///    recalibrated.
///  - **All references to `state.partner` removed** — in 3p the partner field
///    is always `None`, so legacy partner-aware bonuses become dead weight.
///
/// Calibration choices marked `// 3P-CALIB` are the spots most worth tuning
/// once we have arena data; defaults are scaled from the 4p numbers by a
/// 16/12 ≈ 1.33 hand-size ratio (additive bonuses scaled, thresholds raised
/// since contract base values are 20/30/40 in 3p vs 10/20/30 in 4p).
use crate::card::*;
use crate::game_state::*;
use crate::legal_moves;

// -----------------------------------------------------------------------
// Card tracking
// -----------------------------------------------------------------------

#[allow(dead_code)]
struct CardTracker {
    remaining: CardSet,
    taroks_in_hand: u8,
    taroks_remaining: CardSet,
    suit_counts: [u8; 4],
    /// Indexed by absolute seat (0..=2 in 3p; slot 3 is the phantom and
    /// always stays all-false).
    player_voids: [[bool; 4]; NUM_PLAYERS],
    tricks_left: usize,
    phase: GamePhase,
}

#[derive(Clone, Copy, PartialEq)]
enum GamePhase {
    Early,
    Mid,
    Late,
}

impl CardTracker {
    fn from_state(state: &GameState, player: u8) -> Self {
        debug_assert_eq!(
            state.variant,
            Variant::ThreePlayer,
            "stockskis_v3_3p::CardTracker called on {:?} state",
            state.variant
        );

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
        for s in Suit::ALL {
            suit_counts[s as usize] = hand.suit_count(s) as u8;
        }

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

        let tricks_left = (hand.len() as usize)
            .saturating_sub(state.tricks.len())
            .max(1);

        // 3P-CALIB: 16-trick game splits roughly evenly. Early covers
        // talon-pickup / first lead establishment; late = "endgame triggers
        // pagat / mond protection now matters".
        let phase = match state.tricks.len() {
            0..=4 => GamePhase::Early,
            5..=10 => GamePhase::Mid,
            _ => GamePhase::Late,
        };

        CardTracker {
            remaining,
            taroks_in_hand: hand.tarok_count(),
            taroks_remaining,
            suit_counts,
            player_voids,
            tricks_left,
            phase,
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
        // Loop over the 3 active seats only; the phantom seat is never an opponent.
        for p in 0..3 {
            if p as u8 != player && self.player_voids[p][suit as usize] {
                count += 1;
            }
        }
        count
    }
}

// -----------------------------------------------------------------------
// Bidding
// -----------------------------------------------------------------------

/// Evaluate a bid for the 3p game. `highest_so_far` is the strongest contract
/// already named in this auction (or `None`); the returned bid (if any) must
/// strictly outrank it.
pub fn evaluate_bid_v3_3p(hand: CardSet, highest_so_far: Option<Contract>) -> Option<Contract> {
    debug_assert_eq!(hand.len(), 16, "3p hand should be 16 cards, got {}", hand.len());

    let tarok_count = hand.tarok_count();
    let high_taroks = hand.high_tarok_count();
    let king_count = hand.king_count();

    let has_skis = hand.contains(Card::tarok(SKIS));
    let has_mond = hand.contains(Card::tarok(MOND));
    let has_pagat = hand.contains(Card::tarok(PAGAT));

    let mut rating = 0.0f64;

    // 3P-CALIB: same per-card weights as 4p; the larger hand naturally
    // shifts ratings upward, which is offset by raised thresholds below.
    rating += tarok_count as f64 * 6.0;
    rating += high_taroks as f64 * 5.0;
    if has_skis {
        rating += 14.0;
    }
    if has_mond {
        rating += 11.0;
        if has_skis {
            rating += 3.0;
        }
    }
    if has_pagat {
        if tarok_count >= 9 {
            rating += 8.0; // 3P-CALIB: protection threshold lifted (was 7 in 4p)
        } else if tarok_count >= 7 {
            rating += 3.0;
        } else {
            rating += 1.0;
        }
    }

    rating += king_count as f64 * 8.0;

    for s in Suit::ALL {
        let count = hand.suit_count(s);
        let has_king = hand.contains(Card::suit_card(s, SuitRank::King));
        let has_queen = hand.contains(Card::suit_card(s, SuitRank::Queen));

        if count == 0 {
            rating += 8.0;
        } else if count == 1 {
            rating += if has_king { 4.0 } else { 3.0 };
        } else if count == 2 && has_king {
            rating += 2.0;
        } else if count >= 3 && !has_king {
            rating -= count as f64 * 2.0;
        }

        if has_king && has_queen && count >= 2 {
            rating += 2.0;
        }
    }

    // Tarok density bonus — 3P-CALIB: kicks in later because hands hold 16 cards.
    if tarok_count >= 10 {
        rating += (tarok_count - 9) as f64 * 3.0;
    }

    // Max possible rating: 16 taroks × 6 + 5 high × 5 + skis 14 + mond 14 +
    //   pagat 8 + 4 kings × 8 + suit bonuses ≈ 200. Use 200 as the cap so
    //   the same ratio scale (≤ 1) plays nicely with the thresholds below.
    let max_rating = 200.0f64;
    let ratio = (rating / max_rating).min(1.0);

    // 3P-CALIB: thresholds scaled from 4p. SoloThree replaces the partnership
    // contracts as the *bottom* of the solo ladder, so its threshold sits
    // around where SoloThree was in the 4p table.
    let thresholds: [(Contract, f64); 3] = [
        (Contract::SoloThree, 0.34),
        (Contract::SoloTwo, 0.42),
        (Contract::SoloOne, 0.52),
    ];

    let has_all_suits = Suit::ALL.iter().all(|&s| hand.has_suit(s));
    // Berac in 3p: same intuition as 4p — paltry hand, no high taroks, no kings.
    let can_berac =
        ratio < 0.18 && tarok_count <= 3 && has_all_suits && high_taroks == 0 && king_count == 0;

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
// Talon (no called-king bias in 3p)
// -----------------------------------------------------------------------

fn evaluate_talon_group_v3_3p(group: &[Card], hand: CardSet) -> f64 {
    let mut total = 0.0f64;
    for &card in group {
        total += card.points() as f64 * 2.0 + worth_over(card) as f64 * 0.3;
        if card.card_type() == CardType::Tarok {
            total += 10.0;
        }
    }

    // Reward groups that complete or near-complete a void in our hand.
    for s in Suit::ALL {
        let hand_count = hand.suit_count(s);
        let group_adds = group.iter().filter(|c| c.suit() == Some(s)).count() as u32;
        if hand_count <= 1 && hand_count + group_adds <= 1 {
            total += 6.0;
        }
        if hand_count == 0 && group_adds == 0 {
            total += 4.0;
        }
    }

    total
}

pub fn choose_talon_group_v3_3p(groups: &[Vec<Card>], hand: CardSet) -> usize {
    let mut best_idx = 0;
    let mut best_score = f64::NEG_INFINITY;
    for (i, group) in groups.iter().enumerate() {
        let score = evaluate_talon_group_v3_3p(group, hand);
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }
    best_idx
}

// -----------------------------------------------------------------------
// Discard
// -----------------------------------------------------------------------

pub fn choose_discards_v3_3p(hand: CardSet, must_discard: usize) -> Vec<Card> {
    let mut discardable: Vec<Card> = hand
        .iter()
        .filter(|c| c.card_type() != CardType::Tarok && !c.is_king())
        .collect();

    // Fall back to non-trula taroks if we don't have enough plain cards.
    if discardable.len() < must_discard {
        let mut extra: Vec<Card> = hand
            .iter()
            .filter(|c| {
                c.card_type() == CardType::Tarok && !c.is_trula() && !discardable.contains(c)
            })
            .collect();
        extra.sort_by_key(|c| c.value());
        discardable.extend(extra);
    }

    let mut by_suit: [Vec<Card>; 4] = Default::default();
    for &c in &discardable {
        if let Some(s) = c.suit() {
            by_suit[s as usize].push(c);
        }
    }

    let mut result: Vec<Card> = Vec::with_capacity(must_discard);

    // Prefer dumping shortest suits first to create voids for trumping.
    let mut suit_order: Vec<(Suit, u32)> = Suit::ALL
        .iter()
        .filter(|&&s| !by_suit[s as usize].is_empty())
        .map(|&s| (s, hand.suit_count(s)))
        .collect();
    suit_order.sort_by_key(|&(_, count)| count);

    for (suit, _) in suit_order {
        if result.len() >= must_discard {
            break;
        }
        let suit_discardable: Vec<Card> = hand
            .suit(suit)
            .iter()
            .filter(|c| !c.is_king() && discardable.contains(c) && !result.contains(c))
            .collect();
        if suit_discardable.len() + result.len() <= must_discard {
            result.extend(suit_discardable);
        }
    }

    if result.len() < must_discard {
        let mut remaining: Vec<Card> = discardable
            .iter()
            .filter(|c| !result.contains(c))
            .copied()
            .collect();
        remaining.sort_by_key(|c| (c.points(), worth_over(*c)));
        for c in remaining {
            if result.len() >= must_discard {
                break;
            }
            result.push(c);
        }
    }

    result.truncate(must_discard);
    result
}

// -----------------------------------------------------------------------
// Card play
// -----------------------------------------------------------------------

#[allow(private_interfaces)]
pub fn evaluate_card_play_v3_3p(
    card: Card,
    hand: CardSet,
    state: &GameState,
    player: u8,
    is_leading: bool,
    tracker: &CardTracker,
) -> f64 {
    // In 3p there is no partner — declarer plays alone.
    let is_declarer = state.declarer == Some(player);
    let is_klop = state.contract.map_or(false, |c| c.is_klop());
    let is_berac = state.contract.map_or(false, |c| c.is_berac());

    if is_klop || is_berac {
        return eval_klop_berac_v3_3p(card, hand, state, player, is_leading, tracker);
    }

    if is_leading {
        eval_leading_v3_3p(card, hand, state, player, is_declarer, tracker)
    } else {
        eval_following_v3_3p(card, hand, state, player, is_declarer, tracker)
    }
}

fn eval_klop_berac_v3_3p(
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
            // 3P: trick is "last seat" when 2 cards already played (3 total).
            let is_last = trick.count as usize == 2;
            let trick_pts: f64 = (0..trick.count as usize)
                .map(|i| trick.cards[i].1.points() as f64)
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

#[allow(unused_variables)]
fn eval_leading_v3_3p(
    card: Card,
    hand: CardSet,
    state: &GameState,
    player: u8,
    is_declarer: bool,
    tracker: &CardTracker,
) -> f64 {
    let wo = worth_over(card) as f64;
    let pts = card.points() as f64;

    if card.card_type() == CardType::Tarok {
        let higher_out = tracker.higher_taroks_out(card.value());

        if card.value() == SKIS {
            match tracker.phase {
                GamePhase::Early => 90.0,
                GamePhase::Mid => 70.0,
                GamePhase::Late => {
                    if tracker.tricks_left <= 1 {
                        -300.0
                    } else {
                        50.0
                    }
                }
            }
        } else if card.value() == MOND {
            let skis_out = tracker.taroks_remaining.contains(Card::tarok(SKIS));
            let has_skis = hand.contains(Card::tarok(SKIS));
            if skis_out && !has_skis {
                -150.0
            } else if has_skis {
                65.0
            } else {
                70.0 + tracker.taroks_remaining.len() as f64 * 3.0
            }
        } else if card.value() == PAGAT {
            if tracker.tricks_left <= 2 && tracker.taroks_in_hand <= 2 {
                300.0
            } else if tracker.tricks_left == 1 {
                400.0
            } else {
                -250.0
            }
        } else if higher_out == 0 {
            let score = 55.0 + wo;
            if is_declarer { score + 10.0 } else { score }
        } else if is_declarer {
            let base = (((wo - 11.0).max(0.0)) / 3.0).powf(1.5);
            if tracker.phase == GamePhase::Early {
                base + 10.0
            } else {
                base
            }
        } else if higher_out <= 2 {
            25.0 + wo * 0.5
        } else {
            wo * 0.2
        }
    } else {
        let suit = card.suit().unwrap();
        let count = tracker.suit_counts[suit as usize] as f64;
        let _remaining = tracker.remaining_in_suit(suit) as f64;

        if is_declarer {
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
            } else if count == 2.0 {
                12.0 - pts
            } else {
                -pts * 1.5 - count * 2.0
            }
        } else {
            // Defender: avoid leading suits where declarer is void.
            let mut score = if count >= 3.0 {
                if card.is_king() {
                    pts * 2.0 + 5.0
                } else {
                    count * 3.0 - pts * 1.5
                }
            } else if count == 1.0 {
                8.0 - pts
            } else {
                -count * 2.0 - pts
            };

            if let Some(decl) = state.declarer {
                if tracker.player_voids[decl as usize][suit as usize] {
                    score -= 15.0;
                }
            }
            score
        }
    }
}

fn eval_following_v3_3p(
    card: Card,
    _hand: CardSet,
    state: &GameState,
    player: u8,
    is_declarer: bool,
    tracker: &CardTracker,
) -> f64 {
    let wo = worth_over(card) as f64;
    let pts = card.points() as f64;

    let trick = match &state.current_trick {
        Some(t) if t.count > 0 => t,
        _ => return 0.0,
    };

    let lead_suit = trick.lead_suit();
    let num_played = trick.count as usize;
    // 3P: tricks have 3 cards, so "last seat" = 2 already played.
    let is_last = num_played == 2;

    let mut best_card = trick.cards[0].1;
    let mut best_player = trick.cards[0].0;
    for i in 1..num_played {
        if trick.cards[i].1.beats(best_card, lead_suit) {
            best_card = trick.cards[i].1;
            best_player = trick.cards[i].0;
        }
    }

    // 3P: the only ally for the declarer is themself; defenders' "ally" is
    // the *other* defender (the seat that is neither declarer nor self).
    let best_is_ally = if is_declarer {
        best_player == player
    } else {
        best_player != state.declarer.unwrap_or(255) && best_player != player
    };

    let trick_pts: f64 = (0..num_played)
        .map(|i| trick.cards[i].1.points() as f64)
        .sum::<f64>()
        + pts;

    let would_win = card.beats(best_card, lead_suit);

    // Pagat ultimo
    if card.card_type() == CardType::Tarok && card.value() == PAGAT {
        if tracker.tricks_left <= 2 && tracker.taroks_in_hand <= 2 {
            return if would_win { 500.0 } else { -500.0 };
        } else if tracker.tricks_left == 1 {
            return if would_win { 600.0 } else { -600.0 };
        }
        return -250.0;
    }

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

    if card.card_type() == CardType::Tarok && card.value() == SKIS {
        if tracker.tricks_left <= 1 {
            return -400.0;
        }
        return trick_pts * 2.5 + 35.0;
    }

    if would_win {
        if best_is_ally {
            if is_last {
                pts * 4.5
            } else {
                -wo * 0.6 - (wo / 3.0).powf(1.5)
            }
        } else if is_last {
            trick_pts * 3.5 - wo * 1.0
        } else if num_played == 1 {
            if card.card_type() == CardType::Tarok && tracker.higher_taroks_out(card.value()) > 0 {
                trick_pts * 0.8 - wo * 0.3
            } else {
                trick_pts * 2.0 + wo * 0.2
            }
        } else {
            trick_pts * 2.0 + wo * 0.3
        }
    } else if best_is_ally {
        // Šmir.
        if is_last {
            pts * 6.0
        } else if tracker.phase == GamePhase::Late {
            pts * 4.0
        } else {
            pts * 3.0
        }
    } else {
        let mut score = -(pts * 3.0) - wo * 0.4;
        if let Some(suit) = card.suit() {
            for p in 0..3 {
                if p as u8 != player && tracker.player_voids[p][suit as usize] {
                    score += 2.0;
                }
            }
        }
        score
    }
}

/// Choose the best legal card in a 3-player game.
pub fn choose_card_v3_3p(hand: CardSet, state: &GameState, player: u8) -> Card {
    debug_assert_eq!(state.variant, Variant::ThreePlayer);

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
        let score = evaluate_card_play_v3_3p(card, hand, state, player, is_leading, &tracker);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn build_3p_hand(cards: &[Card]) -> CardSet {
        let mut h = CardSet::EMPTY;
        for &c in cards {
            h.insert(c);
        }
        h
    }

    #[test]
    fn weak_hand_passes() {
        // 16 plain low cards (no taroks, no kings). All four suits represented.
        let mut hand = CardSet::EMPTY;
        for s in Suit::ALL {
            for r in [SuitRank::Pip1, SuitRank::Pip2, SuitRank::Pip3, SuitRank::Pip4] {
                hand.insert(Card::suit_card(s, r));
            }
        }
        assert_eq!(hand.len(), 16);
        let bid = evaluate_bid_v3_3p(hand, None);
        // Expected: pass or berac (no high taroks, no kings, all suits present).
        assert!(
            bid.is_none() || bid == Some(Contract::Berac),
            "weak hand produced unexpected bid {:?}",
            bid
        );
    }

    #[test]
    fn strong_hand_bids_solo() {
        // Construct a hand with skis, mond, several high taroks and 3 kings.
        let mut hand = CardSet::EMPTY;
        for v in [SKIS, MOND, PAGAT, 19, 18, 17, 16, 15, 14, 13] {
            hand.insert(Card::tarok(v));
        }
        for s in [Suit::Hearts, Suit::Diamonds, Suit::Clubs] {
            hand.insert(Card::suit_card(s, SuitRank::King));
        }
        // Pad to 16 with low cards.
        let mut filler: Vec<Card> = Vec::new();
        for s in [Suit::Spades, Suit::Hearts, Suit::Diamonds] {
            for r in [SuitRank::Jack, SuitRank::Knight] {
                let c = Card::suit_card(s, r);
                if !hand.contains(c) {
                    filler.push(c);
                }
            }
        }
        for c in filler.into_iter().take(16 - hand.len() as usize) {
            hand.insert(c);
        }
        let bid = evaluate_bid_v3_3p(hand, None);
        // Strong hand should at minimum bid SoloThree.
        assert!(matches!(
            bid,
            Some(Contract::SoloThree | Contract::SoloTwo | Contract::SoloOne)
        ), "bid was {:?}", bid);
    }

    #[test]
    fn does_not_bid_4p_only_contracts() {
        // Even a moderate hand must never produce Three / Two / One / Solo.
        let mut hand = CardSet::EMPTY;
        for v in [SKIS, MOND, PAGAT, 19, 18, 17] {
            hand.insert(Card::tarok(v));
        }
        for s in [Suit::Hearts, Suit::Diamonds] {
            hand.insert(Card::suit_card(s, SuitRank::King));
            hand.insert(Card::suit_card(s, SuitRank::Queen));
        }
        for s in [Suit::Clubs, Suit::Spades] {
            hand.insert(Card::suit_card(s, SuitRank::Jack));
            hand.insert(Card::suit_card(s, SuitRank::Knight));
            hand.insert(Card::suit_card(s, SuitRank::Queen));
        }
        // Pad up to 16.
        for v in [12, 11, 10, 9, 8] {
            if hand.len() < 16 {
                hand.insert(Card::tarok(v));
            }
        }
        let bid = evaluate_bid_v3_3p(hand, None);
        assert!(
            !matches!(
                bid,
                Some(Contract::Three) | Some(Contract::Two) | Some(Contract::One) | Some(Contract::Solo)
            ),
            "3p bid produced 4p-only contract: {:?}",
            bid
        );
    }
}
