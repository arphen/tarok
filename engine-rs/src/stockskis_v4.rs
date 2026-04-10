/// StockŠkis v4 — v3 plus tighter berač gating and clearer opening roles.
///
/// Enhancements over v3:
/// - Do not bid berač with more than two taroks.
/// - Do not bid berač with any singleton suit.
/// - Declarer partner prefers opening with the highest tarok.
/// - Opposition preserves taroks and leads cheap suit cards that are more
///   likely to pull taroks from the declarer team.

use crate::card::*;
use crate::game_state::*;
use crate::legal_moves;

struct CardTracker {
    remaining: CardSet,
    taroks_in_hand: u8,
    taroks_remaining: CardSet,
    suit_counts: [u8; 4],
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
            tricks_left,
            phase,
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

pub fn evaluate_bid_v4(hand: CardSet, highest_so_far: Option<Contract>) -> Option<Contract> {
    let tarok_count = hand.tarok_count();
    let high_taroks = hand.high_tarok_count();
    let king_count = hand.king_count();

    let has_skis = hand.contains(Card::tarok(SKIS));
    let has_mond = hand.contains(Card::tarok(MOND));
    let has_pagat = hand.contains(Card::tarok(PAGAT));

    let mut rating = 0.0f64;
    rating += tarok_count as f64 * 6.0;
    rating += high_taroks as f64 * 5.0;
    if has_skis { rating += 14.0; }
    if has_mond {
        rating += 11.0;
        if has_skis { rating += 3.0; }
    }
    if has_pagat {
        if tarok_count >= 7 { rating += 8.0; }
        else if tarok_count >= 5 { rating += 3.0; }
        else { rating -= 2.0; }
    }

    rating += king_count as f64 * 8.0;

    let mut has_all_suits = true;
    let mut has_singleton = false;
    for suit in Suit::ALL {
        let count = hand.suit_count(suit);
        let has_king = hand.contains(Card::suit_card(suit, SuitRank::King));
        let has_queen = hand.contains(Card::suit_card(suit, SuitRank::Queen));

        if count == 0 {
            has_all_suits = false;
            rating += 8.0;
        } else if count == 1 {
            has_singleton = true;
            if has_king { rating += 4.0; } else { rating += 3.0; }
        } else if count == 2 && has_king {
            rating += 2.0;
        } else if count >= 3 && !has_king {
            rating -= count as f64 * 2.0;
        }

        if has_king && has_queen && count >= 2 {
            rating += 2.0;
        }
    }

    if tarok_count >= 8 {
        rating += (tarok_count - 7) as f64 * 3.0;
    }

    let ratio = (rating / 130.0).min(1.0);
    let thresholds: [(Contract, f64); 7] = [
        (Contract::Three, 0.24),
        (Contract::Two, 0.30),
        (Contract::One, 0.38),
        (Contract::SoloThree, 0.50),
        (Contract::SoloTwo, 0.58),
        (Contract::SoloOne, 0.66),
        (Contract::Solo, 0.76),
    ];

    let can_berac = ratio < 0.16
        && tarok_count <= 2
        && has_all_suits
        && !has_singleton
        && high_taroks == 0
        && king_count == 0;

    let mut best: Option<Contract> = None;
    for &(contract, threshold) in &thresholds {
        if ratio >= threshold {
            best = Some(contract);
        }
    }
    if can_berac && best.map_or(true, |current| Contract::Berac.strength() > current.strength()) {
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

pub fn choose_king_v4(hand: CardSet) -> Option<Card> {
    let mut best_king: Option<Card> = None;
    let mut best_score = -1i32;

    for suit in Suit::ALL {
        let king = Card::suit_card(suit, SuitRank::King);
        if hand.contains(king) {
            continue;
        }
        let count = hand.suit_count(suit) as i32;
        let low_cards = hand.suit(suit).iter().filter(|card| card.value() <= 4).count() as i32;
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

fn evaluate_talon_group_v4(group: &[Card], hand: CardSet, called_king: Option<Card>) -> f64 {
    let called_suit = called_king.and_then(|king| king.suit());
    let mut total = 0.0f64;

    for &card in group {
        total += card.points() as f64 * 2.0 + worth_over(card) as f64 * 0.3;
        if card.card_type() == CardType::Tarok {
            total += 10.0;
        }
        if let (Some(king_suit), Some(suit)) = (called_suit, card.suit()) {
            if suit == king_suit {
                total += 5.0;
            }
        }
    }

    for suit in Suit::ALL {
        if Some(suit) == called_suit {
            continue;
        }
        let hand_count = hand.suit_count(suit);
        let group_adds = group.iter().filter(|card| card.suit() == Some(suit)).count() as u32;
        if hand_count <= 1 && hand_count + group_adds <= 1 {
            total += 6.0;
        }
        if hand_count == 0 && group_adds == 0 {
            total += 4.0;
        }
    }

    total
}

pub fn choose_talon_group_v4(groups: &[Vec<Card>], hand: CardSet, called_king: Option<Card>) -> usize {
    let mut best_idx = 0usize;
    let mut best_score = f64::NEG_INFINITY;
    for (idx, group) in groups.iter().enumerate() {
        let score = evaluate_talon_group_v4(group, hand, called_king);
        if score > best_score {
            best_score = score;
            best_idx = idx;
        }
    }
    best_idx
}

pub fn choose_discards_v4(hand: CardSet, must_discard: usize, called_king: Option<Card>) -> Vec<Card> {
    let called_suit = called_king.and_then(|king| king.suit());

    let mut discardable: Vec<Card> = hand.iter()
        .filter(|card| card.card_type() != CardType::Tarok && !card.is_king())
        .collect();

    if discardable.len() < must_discard {
        let mut extra: Vec<Card> = hand.iter()
            .filter(|card| card.card_type() == CardType::Tarok && !card.is_trula() && !discardable.contains(card))
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
    let mut suit_order: Vec<(Suit, u32)> = Suit::ALL.iter()
        .filter(|&&suit| Some(suit) != called_suit && !by_suit[suit as usize].is_empty())
        .map(|&suit| (suit, hand.suit_count(suit)))
        .collect();
    suit_order.sort_by_key(|&(_, count)| count);

    for (suit, _) in suit_order {
        if result.len() >= must_discard {
            break;
        }
        let suit_discardable: Vec<Card> = hand.suit(suit).iter()
            .filter(|card| !card.is_king() && discardable.contains(card) && !result.contains(card))
            .collect();
        if suit_discardable.len() + result.len() <= must_discard {
            result.extend(suit_discardable);
        }
    }

    if result.len() < must_discard {
        let mut remaining: Vec<Card> = discardable.iter()
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

pub fn evaluate_card_play_v4(
    card: Card,
    hand: CardSet,
    state: &GameState,
    player: u8,
    is_leading: bool,
    tracker: &CardTracker,
) -> f64 {
    let is_declarer = state.declarer == Some(player);
    let is_partner = state.partner == Some(player);
    let is_playing = is_declarer || is_partner;
    let is_klop = state.contract.map_or(false, |contract| contract.is_klop());
    let is_berac = state.contract.map_or(false, |contract| contract.is_berac());

    if is_klop || is_berac {
        return eval_klop_berac_v4(card, hand, state, player, is_leading, tracker);
    }

    if is_leading {
        eval_leading_v4(card, hand, state, player, is_declarer, is_partner, tracker)
    } else {
        eval_following_v4(card, state, player, is_playing, tracker)
    }
}

fn eval_klop_berac_v4(
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
            if tracker.phase == GamePhase::Late && card.value() <= 5 && tracker.higher_taroks_out(card.value()) == 0 {
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
                .sum::<f64>() + pts;

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

fn eval_leading_v4(
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

        if card.value() == PAGAT {
            if tracker.tricks_left <= 2 && tracker.taroks_in_hand <= 2 {
                return 300.0;
            }
            if tracker.tricks_left == 1 {
                return 400.0;
            }
            return -250.0;
        }

        if is_partner {
            if card.value() == SKIS {
                return match tracker.phase {
                    GamePhase::Early => 125.0,
                    GamePhase::Mid => 105.0,
                    GamePhase::Late => {
                        if tracker.tricks_left <= 1 { -300.0 } else { 70.0 }
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
            return 90.0 + wo * 1.4 - higher_out as f64 * 4.0;
        }

        if is_opposition {
            if higher_out == 0 && tracker.phase == GamePhase::Late {
                return 20.0 + wo * 0.2;
            }
            if higher_out <= 1 && tracker.tricks_left <= 3 {
                return 8.0 + wo * 0.1;
            }
            return -45.0 - wo * 0.4 - higher_out as f64 * 6.0;
        }

        if card.value() == SKIS {
            return match tracker.phase {
                GamePhase::Early => 90.0,
                GamePhase::Mid => 70.0,
                GamePhase::Late => {
                    if tracker.tricks_left <= 1 { -300.0 } else { 50.0 }
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
                return 65.0;
            }
            return 70.0 + tracker.taroks_remaining.len() as f64 * 3.0;
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
                score += declarer_team_voids * 14.0 - pts * 2.0;
            } else if remaining > 0.0 {
                score += remaining.min(4.0);
            }
            score
        }
    }
}

fn eval_following_v4(
    card: Card,
    state: &GameState,
    player: u8,
    is_playing: bool,
    tracker: &CardTracker,
) -> f64 {
    let wo = worth_over(card) as f64;
    let pts = card.points() as f64;

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
        .sum::<f64>() + pts;
    let would_win = card.beats(best_card, lead_suit);

    if card.card_type() == CardType::Tarok && card.value() == PAGAT {
        if tracker.tricks_left <= 2 && tracker.taroks_in_hand <= 2 {
            return if would_win { 500.0 } else { -500.0 };
        }
        if tracker.tricks_left == 1 {
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
            trick_pts * 3.5 - wo
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
            for p in 0..NUM_PLAYERS {
                if p as u8 != player && tracker.player_voids[p][suit as usize] {
                    score += 2.0;
                }
            }
        }
        score
    }
}

pub fn choose_card_v4(hand: CardSet, state: &GameState, player: u8) -> Card {
    let ctx = legal_moves::MoveCtx::from_state(state, player);
    let legal = legal_moves::generate_legal_moves(&ctx);
    let legal_vec: Vec<Card> = legal.iter().collect();

    if legal_vec.len() == 1 {
        return legal_vec[0];
    }

    let is_leading = state.current_trick.as_ref().map_or(true, |trick| trick.count == 0);
    let tracker = CardTracker::from_state(state, player);

    let mut best_card = legal_vec[0];
    let mut best_score = f64::NEG_INFINITY;
    for &card in &legal_vec {
        let score = evaluate_card_play_v4(card, hand, state, player, is_leading, &tracker);
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

        assert_ne!(evaluate_bid_v4(hand, None), Some(Contract::Berac));
    }

    #[test]
    fn berac_rejects_singleton_suit() {
        let hand = hand_from(&[
            Card::tarok(2),
            Card::tarok(3),
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
        ]);

        assert_ne!(evaluate_bid_v4(hand, None), Some(Contract::Berac));
    }

    #[test]
    fn partner_prefers_highest_tarok_lead() {
        let hand = hand_from(&[
            Card::tarok(20),
            Card::tarok(15),
            Card::suit_card(Suit::Hearts, SuitRank::Pip1),
            Card::suit_card(Suit::Clubs, SuitRank::Pip2),
        ]);
        let state = make_state(hand, 1, 0, Some(1));

        assert_eq!(choose_card_v4(hand, &state, 1), Card::tarok(20));
    }

    #[test]
    fn opposition_prefers_low_suit_lead() {
        let hand = hand_from(&[
            Card::tarok(20),
            Card::tarok(10),
            Card::suit_card(Suit::Hearts, SuitRank::Pip1),
            Card::suit_card(Suit::Spades, SuitRank::King),
        ]);
        let mut state = make_state(hand, 2, 0, Some(1));
        let mut prior = Trick::new(3);
        prior.play(3, Card::suit_card(Suit::Hearts, SuitRank::Pip2));
        prior.play(0, Card::suit_card(Suit::Hearts, SuitRank::Pip3));
        prior.play(1, Card::tarok(5));
        prior.play(2, Card::suit_card(Suit::Hearts, SuitRank::Pip1));
        state.tricks.push(prior);

        assert_eq!(choose_card_v4(hand, &state, 2), Card::suit_card(Suit::Hearts, SuitRank::Pip1));
    }
}