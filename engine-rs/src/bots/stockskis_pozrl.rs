/// POŽRL heuristic bot based on:
/// Domen Požrl, "Reinforcement learning in Tarock", University of Ljubljana, 2021.
///
/// This implementation ports the paper's described heuristic meta-game ideas
/// (bidding, talon choice, discard priorities) and a lightweight deterministic
/// trick-play policy into the Rust heuristic-bot interface used by arena/training.
use crate::card::*;
use crate::game_state::*;
use crate::legal_moves;

pub fn evaluate_bid_pozrl(hand: CardSet, highest_so_far: Option<Contract>) -> Option<Contract> {
    let mut score: u8 = 0;

    score += hand.king_count();

    for suit in Suit::ALL {
        let queen = Card::suit_card(suit, SuitRank::Queen);
        if hand.contains(queen) && hand.suit_count(suit) >= 2 {
            score += 1;
        }
    }

    for trula in [Card::tarok(PAGAT), Card::tarok(MOND), Card::tarok(SKIS)] {
        if hand.contains(trula) {
            score += 1;
        }
    }

    if hand.tarok_count() >= 7 {
        score += 1;
    }

    score += hand.void_count();

    let chosen = match score {
        0..=4 => None,
        5 => Some(Contract::Three),
        6 => Some(Contract::Two),
        7 => Some(Contract::One),
        _ => Some(Contract::Solo),
    };

    match (chosen, highest_so_far) {
        (Some(c), Some(h)) if c.strength() <= h.strength() => None,
        (other, _) => other,
    }
}

pub fn choose_king_pozrl(hand: CardSet) -> Option<Card> {
    let mut best: Option<(i32, Card)> = None;

    for suit in Suit::ALL {
        let king = Card::suit_card(suit, SuitRank::King);
        if hand.contains(king) {
            continue;
        }
        let suit_count = hand.suit_count(suit) as i32;
        let suit_points: i32 = hand.suit(suit).iter().map(|c| c.points() as i32).sum();
        let score = suit_count * 10 + suit_points;
        if best.map_or(true, |(b, _)| score > b) {
            best = Some((score, king));
        }
    }

    if let Some((_, king)) = best {
        return Some(king);
    }

    for suit in Suit::ALL {
        let queen = Card::suit_card(suit, SuitRank::Queen);
        if !hand.contains(queen) {
            return Some(queen);
        }
    }

    None
}

pub fn choose_talon_group_pozrl(
    groups: &[Vec<Card>],
    _hand: CardSet,
    _called_king: Option<Card>,
) -> usize {
    let mut best_idx = 0usize;
    let mut best_score = f64::NEG_INFINITY;

    for (idx, group) in groups.iter().enumerate() {
        let score: f64 = group.iter().map(|&c| talon_card_score(c)).sum();
        if score > best_score {
            best_score = score;
            best_idx = idx;
        }
    }

    best_idx
}

pub fn choose_discards_pozrl(
    hand: CardSet,
    must_discard: usize,
    called_king: Option<Card>,
) -> Vec<Card> {
    if must_discard == 0 {
        return Vec::new();
    }

    let mut selected: Vec<Card> = Vec::with_capacity(must_discard);
    let mut working = hand;

    // 1) Remove an entire suit if possible (fewest cards first).
    loop {
        if selected.len() >= must_discard {
            break;
        }
        let remaining = must_discard - selected.len();
        let mut best_suit: Option<(Suit, usize)> = None;
        for suit in Suit::ALL {
            let suit_cards: Vec<Card> = working
                .suit(suit)
                .iter()
                .filter(|&c| can_discard_primary(c, called_king))
                .collect();
            let count = suit_cards.len();
            if count == 0 || count > remaining {
                continue;
            }
            if best_suit.map_or(true, |(_, best_count)| count < best_count) {
                best_suit = Some((suit, count));
            }
        }
        let Some((suit, _)) = best_suit else {
            break;
        };

        let mut suit_cards: Vec<Card> = working
            .suit(suit)
            .iter()
            .filter(|&c| can_discard_primary(c, called_king))
            .collect();
        suit_cards.sort_by_key(|&c| discard_priority_key(c));
        for c in suit_cards {
            if selected.len() >= must_discard {
                break;
            }
            selected.push(c);
            working.remove(c);
        }
    }

    // 2) Discard cards that strengthen a suit (e.g. leave a high winner).
    if selected.len() < must_discard {
        let mut strengthen_candidates: Vec<Card> = Vec::new();
        for suit in Suit::ALL {
            let suit_cards: Vec<Card> = working.suit(suit).iter().collect();
            if suit_cards.len() != 2 {
                continue;
            }
            let has_king = suit_cards.iter().any(|c| c.is_king());
            let has_queen = suit_cards
                .iter()
                .any(|c| c.card_type() == CardType::Suit && c.value() == SuitRank::Queen as u8);
            if !(has_king || has_queen) {
                continue;
            }
            for &c in &suit_cards {
                if c.is_king() || c.value() == SuitRank::Queen as u8 {
                    continue;
                }
                if can_discard_primary(c, called_king) {
                    strengthen_candidates.push(c);
                }
            }
        }
        strengthen_candidates.sort_by_key(|&c| std::cmp::Reverse((c.points(), c.value(), c.0)));
        for c in strengthen_candidates {
            if selected.len() >= must_discard {
                break;
            }
            if working.contains(c) {
                selected.push(c);
                working.remove(c);
            }
        }
    }

    // 3) Fill remaining discards: preserve points where possible.
    fill_discards(&mut selected, &mut working, must_discard, called_king);
    selected
}

pub fn choose_card_pozrl(hand: CardSet, state: &GameState, player: u8) -> Card {
    let ctx = legal_moves::MoveCtx::from_state(state, player);
    let legal_set = legal_moves::generate_legal_moves(&ctx);
    let legal_cards: Vec<Card> = legal_set.iter().collect();
    if legal_cards.is_empty() {
        return hand.iter().next().unwrap_or(Card::tarok(PAGAT));
    }

    let trick = state.current_trick.as_ref();
    if trick.is_none() || trick.map_or(true, |t| t.count == 0) {
        return lead_card_choice(&legal_cards);
    }

    let trick = trick.unwrap();
    let lead_suit = trick.lead_suit();
    let best_now = trick.best_card().unwrap();
    let can_win: Vec<Card> = legal_cards
        .iter()
        .copied()
        .filter(|&c| c.beats(best_now, lead_suit))
        .collect();

    let is_last = trick.count == 3;
    let partner = partner_of(state, player);
    let partner_winning = partner.map_or(false, |p| current_winner(trick) == p);
    let trick_points = trick.points() as i32;
    let playing_solo = state.is_effectively_solo();

    if is_last {
        if partner_winning {
            return pick_weakest(&legal_cards);
        }
        if can_win.is_empty() {
            return pick_weakest(&legal_cards);
        }
        if can_win.len() == 1
            && can_win[0].card_type() == CardType::Tarok
            && can_win[0].value() == SKIS
        {
            return can_win[0];
        }
        if can_win.iter().all(|c| c.points() <= 1) {
            return pick_weakest(&can_win);
        }
        return pick_max_points_then_strength(&can_win);
    }

    if can_win.is_empty() {
        return pick_weakest(&legal_cards);
    }

    if !playing_solo && partner_winning && trick_points <= 3 {
        return pick_weakest(&legal_cards);
    }

    if trick_points >= 4 {
        return pick_max_points_then_strength(&can_win);
    }

    pick_weakest(&can_win)
}

fn talon_card_score(card: Card) -> f64 {
    if card.card_type() == CardType::Tarok {
        return match card.value() {
            PAGAT | MOND | SKIS => 1_000_000.0,
            v => (v as f64) * 0.1,
        };
    }

    match card.value() {
        v if v == SuitRank::Jack as u8 => 0.2,
        v if v == SuitRank::Knight as u8 => 0.5,
        v if v == SuitRank::Queen as u8 => 1.0,
        v if v == SuitRank::King as u8 => 5.0,
        pip => {
            let is_red = matches!(card.suit().unwrap(), Suit::Hearts | Suit::Diamonds);
            if is_red {
                match pip {
                    4 => 0.01, // 4
                    3 => 0.02, // 3
                    2 => 0.03, // 2
                    _ => 0.05, // 1
                }
            } else {
                match pip {
                    1 => 0.01, // 7
                    2 => 0.02, // 8
                    3 => 0.03, // 9
                    _ => 0.05, // 10
                }
            }
        }
    }
}

fn can_discard_primary(card: Card, called_king: Option<Card>) -> bool {
    card.card_type() == CardType::Suit && !card.is_king() && Some(card) != called_king
}

fn can_discard_secondary(card: Card, called_king: Option<Card>) -> bool {
    card.card_type() == CardType::Suit && Some(card) != called_king
}

fn discard_priority_key(card: Card) -> (u8, u8, u8) {
    let suit_bias = if card.card_type() == CardType::Tarok {
        1
    } else {
        0
    };
    (card.points(), suit_bias, card.value())
}

fn fill_discards(
    selected: &mut Vec<Card>,
    working: &mut CardSet,
    must_discard: usize,
    called_king: Option<Card>,
) {
    if selected.len() >= must_discard {
        return;
    }
    let tiers: [fn(Card, Option<Card>) -> bool; 3] =
        [can_discard_primary, can_discard_secondary, |c, k| {
            Some(c) != k
        }];

    for predicate in tiers {
        if selected.len() >= must_discard {
            break;
        }
        let mut candidates: Vec<Card> = working
            .iter()
            .filter(|&c| predicate(c, called_king))
            .collect();
        candidates.sort_by_key(|&c| discard_priority_key(c));
        for c in candidates {
            if selected.len() >= must_discard {
                break;
            }
            if working.contains(c) {
                selected.push(c);
                working.remove(c);
            }
        }
    }
}

fn lead_card_choice(legal_cards: &[Card]) -> Card {
    let non_tarok_faces: Vec<Card> = legal_cards
        .iter()
        .copied()
        .filter(|c| c.card_type() == CardType::Suit && c.points() >= 3)
        .collect();
    if !non_tarok_faces.is_empty() {
        return pick_max_points_then_strength(&non_tarok_faces);
    }

    let non_tarok: Vec<Card> = legal_cards
        .iter()
        .copied()
        .filter(|c| c.card_type() == CardType::Suit)
        .collect();
    if !non_tarok.is_empty() {
        return pick_weakest(&non_tarok);
    }

    pick_weakest(legal_cards)
}

fn pick_weakest(cards: &[Card]) -> Card {
    cards
        .iter()
        .copied()
        .min_by_key(|c| (c.points(), c.card_type() == CardType::Tarok, c.value(), c.0))
        .unwrap_or(cards[0])
}

fn pick_max_points_then_strength(cards: &[Card]) -> Card {
    cards
        .iter()
        .copied()
        .max_by_key(|c| (c.points(), c.card_type() == CardType::Tarok, c.value(), c.0))
        .unwrap_or(cards[0])
}

fn current_winner(trick: &Trick) -> u8 {
    let lead_suit = trick.lead_suit();
    let mut winner = trick.cards[0].0;
    let mut best = trick.cards[0].1;
    for i in 1..trick.count as usize {
        let (p, c) = trick.cards[i];
        if c.beats(best, lead_suit) {
            best = c;
            winner = p;
        }
    }
    winner
}

fn partner_of(state: &GameState, player: u8) -> Option<u8> {
    let my_team = state.get_team(player);
    for p in 0..NUM_PLAYERS {
        let pu8 = p as u8;
        if pu8 != player && state.get_team(pu8) == my_team {
            return Some(pu8);
        }
    }
    None
}
