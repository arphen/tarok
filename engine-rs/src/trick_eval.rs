/// Trick evaluation — determine the winner and points for a completed trick.
///
/// Compiles the YAML rule priorities directly into Rust match logic.
use crate::card::*;
use crate::game_state::*;

/// Result of evaluating a trick.
#[derive(Debug, Clone, Copy)]
pub struct TrickResult {
    pub winner: u8,
    pub points: u8,
    pub rule: &'static str,
}

/// Evaluate a completed trick, returning the winner and points.
pub fn evaluate_trick(
    trick: &Trick,
    is_last_trick: bool,
    contract: Option<Contract>,
) -> TrickResult {
    debug_assert!(trick.is_complete());
    let cards_set = trick.played_cards_set();
    let lead_suit = trick.lead_suit();

    // P200: PagatWinsTrulaTrick — all 3 trula in same trick
    if cards_set.0 & TRULA_MASK == TRULA_MASK {
        let pagat = Card::tarok(PAGAT);
        let winner = find_player_of(trick, pagat);
        return TrickResult {
            winner,
            points: trick.points(),
            rule: "PagatWinsTrulaTrick",
        };
    }

    // P150: BarvniValatWinner — suit cards beat taroks
    if contract == Some(Contract::BarvniValat) {
        return barvni_valat_winner(trick, lead_suit);
    }

    // P100: SkisCapturedLastTrick — same as standard, Škis captured
    if is_last_trick && cards_set.contains(Card::tarok(SKIS)) {
        // Standard winner — Škis stays, captured
        return TrickResult {
            winner: standard_winner(trick, lead_suit),
            points: trick.points(),
            rule: "SkisCapturedLastTrick",
        };
    }

    // P0: StandardTrickWinner
    TrickResult {
        winner: standard_winner(trick, lead_suit),
        points: trick.points(),
        rule: "StandardTrickWinner",
    }
}

fn standard_winner(trick: &Trick, lead_suit: Option<Suit>) -> u8 {
    let (mut best_player, mut best_card) = trick.cards[0];
    for i in 1..4 {
        let (player, card) = trick.cards[i];
        if card.beats(best_card, lead_suit) {
            best_player = player;
            best_card = card;
        }
    }
    best_player
}

fn barvni_valat_winner(trick: &Trick, lead_suit: Option<Suit>) -> TrickResult {
    // Separate suit cards and taroks
    let mut has_suit_card = false;
    for i in 0..4 {
        if trick.cards[i].1.card_type() == CardType::Suit {
            has_suit_card = true;
            break;
        }
    }

    if has_suit_card {
        // Suit cards beat all taroks; lead suit wins among suits
        let mut best_player = 0u8;
        let mut best_card: Option<Card> = None;

        for i in 0..4 {
            let (player, card) = trick.cards[i];
            if card.card_type() != CardType::Suit {
                continue;
            }
            match best_card {
                None => {
                    best_player = player;
                    best_card = Some(card);
                }
                Some(bc) => {
                    if card.suit() == bc.suit() {
                        if card.value() > bc.value() {
                            best_player = player;
                            best_card = Some(card);
                        }
                    } else if card.suit() == lead_suit {
                        best_player = player;
                        best_card = Some(card);
                    }
                }
            }
        }

        TrickResult {
            winner: best_player,
            points: trick.points(),
            rule: "BarvniValatWinner",
        }
    } else {
        // All taroks — standard
        TrickResult {
            winner: standard_winner(trick, lead_suit),
            points: trick.points(),
            rule: "BarvniValatWinner",
        }
    }
}

fn find_player_of(trick: &Trick, target: Card) -> u8 {
    for i in 0..4 {
        if trick.cards[i].1 == target {
            return trick.cards[i].0;
        }
    }
    unreachable!("Card not found in trick")
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trick(cards: [(u8, Card); 4]) -> Trick {
        Trick {
            lead_player: cards[0].0,
            cards,
            count: 4,
        }
    }

    #[test]
    fn higher_tarok_wins() {
        let trick = make_trick([
            (0, Card::tarok(5)),
            (1, Card::tarok(10)),
            (2, Card::tarok(3)),
            (3, Card::tarok(7)),
        ]);
        let r = evaluate_trick(&trick, false, None);
        assert_eq!(r.winner, 1); // tarok 10
    }

    #[test]
    fn tarok_beats_suit() {
        let trick = make_trick([
            (0, Card::suit_card(Suit::Hearts, SuitRank::King)),
            (1, Card::tarok(1)),
            (2, Card::suit_card(Suit::Hearts, SuitRank::Queen)),
            (3, Card::suit_card(Suit::Hearts, SuitRank::Pip1)),
        ]);
        let r = evaluate_trick(&trick, false, None);
        assert_eq!(r.winner, 1); // even Pagat beats suit
    }

    #[test]
    fn lead_suit_wins() {
        let trick = make_trick([
            (0, Card::suit_card(Suit::Hearts, SuitRank::Pip1)),
            (1, Card::suit_card(Suit::Clubs, SuitRank::King)),
            (2, Card::suit_card(Suit::Hearts, SuitRank::King)),
            (3, Card::suit_card(Suit::Diamonds, SuitRank::King)),
        ]);
        let r = evaluate_trick(&trick, false, None);
        assert_eq!(r.winner, 2); // Hearts king (lead suit)
    }

    #[test]
    fn pagat_wins_trula_trick() {
        let trick = make_trick([
            (0, Card::tarok(MOND)),
            (1, Card::tarok(PAGAT)),
            (2, Card::tarok(SKIS)),
            (3, Card::tarok(10)),
        ]);
        let r = evaluate_trick(&trick, false, None);
        assert_eq!(r.winner, 1); // Pagat wins
        assert_eq!(r.rule, "PagatWinsTrulaTrick");
    }

    #[test]
    fn barvni_valat_suit_beats_tarok() {
        let trick = make_trick([
            (0, Card::suit_card(Suit::Hearts, SuitRank::Pip1)),
            (1, Card::tarok(MOND)),
            (2, Card::suit_card(Suit::Hearts, SuitRank::King)),
            (3, Card::tarok(15)),
        ]);
        let r = evaluate_trick(&trick, false, Some(Contract::BarvniValat));
        assert_eq!(r.winner, 2); // Hearts King beats Mond in barvni valat
    }
}
