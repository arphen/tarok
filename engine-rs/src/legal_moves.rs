/// Legal move generation — all rules hardcoded (no YAML at runtime).
///
/// The YAML-driven rules engine was great for Python prototyping. In Rust
/// we compile the exact same cascading-filter + ban-list logic directly,
/// gaining zero-overhead dispatch and bitmask operations.
use crate::card::*;
use crate::game_state::*;

/// Context for legal move generation (mirrors Python MoveContext).
#[derive(Debug)]
pub struct MoveCtx {
    pub hand: CardSet,
    pub lead_card: Option<Card>,
    pub lead_suit: Option<Suit>,
    pub best_card: Option<Card>,
    pub contract_name: Option<&'static str>,
    pub is_last_trick: bool,
    pub trick_cards: CardSet,
}

impl MoveCtx {
    pub fn from_state(state: &GameState, player: u8) -> MoveCtx {
        let hand = state.hands[player as usize];
        let (lead_card, lead_suit, best_card, trick_cards) =
            if let Some(ref trick) = state.current_trick {
                if trick.count > 0 {
                    (
                        trick.lead_card(),
                        trick.lead_suit(),
                        trick.best_card(),
                        trick.played_cards_set(),
                    )
                } else {
                    (None, None, None, CardSet::EMPTY)
                }
            } else {
                (None, None, None, CardSet::EMPTY)
            };
        MoveCtx {
            hand,
            lead_card,
            lead_suit,
            best_card,
            contract_name: state.contract_name(),
            is_last_trick: state.is_last_trick(),
            trick_cards,
        }
    }

    fn is_overplay(&self) -> bool {
        matches!(self.contract_name, Some("klop") | Some("berac"))
    }

    fn is_leading(&self) -> bool {
        self.lead_card.is_none()
    }

    fn hand_suit(&self) -> CardSet {
        match self.lead_suit {
            Some(s) => self.hand.suit(s),
            None => CardSet::EMPTY,
        }
    }

    fn hand_taroks(&self) -> CardSet {
        self.hand.taroks()
    }

    fn has_lead_suit(&self) -> bool {
        match self.lead_suit {
            Some(s) => self.hand.has_suit(s),
            None => false,
        }
    }

    fn has_taroks(&self) -> bool {
        self.hand.has_taroks()
    }

    fn lead_is_tarok(&self) -> bool {
        self.lead_card
            .map(|c| c.card_type() == CardType::Tarok)
            .unwrap_or(false)
    }
}

// -----------------------------------------------------------------------
// Main entry point
// -----------------------------------------------------------------------

/// Generate legal moves for the current player. Returns a CardSet of playable cards.
pub fn generate_legal_moves(ctx: &MoveCtx) -> CardSet {
    if ctx.hand.is_empty() {
        return CardSet::EMPTY;
    }

    // --- Cascading filter pipeline (highest priority first) ---
    let legal = cascading_pipeline(ctx);

    // --- Ban list pass ---
    apply_bans(ctx, legal)
}

fn cascading_pipeline(ctx: &MoveCtx) -> CardSet {
    // P200: LeadAnything
    if ctx.is_leading() {
        return ctx.hand;
    }

    let overplay = ctx.is_overplay();

    // P150: FollowSuitOverplay
    if overplay && ctx.has_lead_suit() {
        let same_suit = ctx.hand_suit();
        if let Some(best) = ctx.best_card {
            let higher = same_suit.cards_beating(best, ctx.lead_suit);
            if !higher.is_empty() {
                return higher;
            }
        }
        return same_suit;
    }

    // P140: FollowTarokOverplay
    if overplay && ctx.lead_is_tarok() && ctx.has_taroks() {
        let taroks = ctx.hand_taroks();
        if let Some(best) = ctx.best_card {
            let higher = taroks.cards_beating(best, None);
            if !higher.is_empty() {
                return higher;
            }
        }
        return taroks;
    }

    // P130: TrumpInOverplay — can't follow suit, has taroks, overplay
    if overplay
        && !ctx.is_leading()
        && !ctx.lead_is_tarok()
        && !ctx.has_lead_suit()
        && ctx.has_taroks()
    {
        let taroks = ctx.hand_taroks();
        if let Some(best) = ctx.best_card {
            let higher = taroks.cards_beating(best, None);
            if !higher.is_empty() {
                return higher;
            }
        }
        return taroks;
    }

    // P90: MustFollowSuit
    if ctx.has_lead_suit() {
        return ctx.hand_suit();
    }

    // P80: MustFollowTarok
    if ctx.lead_is_tarok() && ctx.has_taroks() {
        return ctx.hand_taroks();
    }

    // P70: MustTrumpIn — can't follow suit, has taroks (non-overplay)
    if !ctx.is_leading() && !ctx.lead_is_tarok() && !ctx.has_lead_suit() && ctx.has_taroks() {
        return ctx.hand_taroks();
    }

    // P0: PlayAnything (fallback)
    ctx.hand
}

fn apply_bans(ctx: &MoveCtx, legal: CardSet) -> CardSet {
    if !ctx.is_overplay() {
        return legal;
    }

    let pagat = Card::tarok(PAGAT);
    let has_pagat = legal.contains(pagat);
    if !has_pagat {
        return legal;
    }

    // P200: ForcePagatWhenMondSkisInTrick
    if mond_and_skis_in_trick(ctx.trick_cards) {
        // Pagat MUST be played (it wins via trula rule)
        return CardSet::single(pagat);
    }

    // P100: BanPagatInOverplay — ban pagat if other taroks exist
    let legal_taroks = legal.taroks();
    if legal_taroks.len() > 1 {
        // Remove pagat
        let without_pagat = legal.difference(CardSet::single(pagat));
        if !without_pagat.is_empty() {
            return without_pagat;
        }
    }

    legal
}

fn mond_and_skis_in_trick(trick_cards: CardSet) -> bool {
    trick_cards.0 & MOND_MASK != 0 && trick_cards.0 & SKIS_MASK != 0
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(
        hand: CardSet,
        lead: Option<Card>,
        best: Option<Card>,
        contract: Option<&'static str>,
    ) -> MoveCtx {
        let lead_suit = lead.and_then(|c| {
            if c.card_type() == CardType::Tarok {
                None
            } else {
                c.suit()
            }
        });
        MoveCtx {
            hand,
            lead_card: lead,
            lead_suit,
            best_card: best,
            contract_name: contract,
            is_last_trick: false,
            trick_cards: CardSet::EMPTY,
        }
    }

    #[test]
    fn leading_can_play_anything() {
        let mut hand = CardSet::EMPTY;
        hand.insert(Card::tarok(5));
        hand.insert(Card::suit_card(Suit::Hearts, SuitRank::King));
        let c = ctx(hand, None, None, None);
        assert_eq!(generate_legal_moves(&c), hand);
    }

    #[test]
    fn must_follow_suit() {
        let mut hand = CardSet::EMPTY;
        hand.insert(Card::suit_card(Suit::Hearts, SuitRank::King));
        hand.insert(Card::suit_card(Suit::Hearts, SuitRank::Pip1));
        hand.insert(Card::suit_card(Suit::Clubs, SuitRank::Jack));
        hand.insert(Card::tarok(5));

        let lead = Card::suit_card(Suit::Hearts, SuitRank::Queen);
        let c = ctx(hand, Some(lead), Some(lead), None);
        let legal = generate_legal_moves(&c);
        assert_eq!(legal.len(), 2); // Only hearts
        assert!(legal.contains(Card::suit_card(Suit::Hearts, SuitRank::King)));
        assert!(legal.contains(Card::suit_card(Suit::Hearts, SuitRank::Pip1)));
    }

    #[test]
    fn must_trump_in() {
        let mut hand = CardSet::EMPTY;
        hand.insert(Card::suit_card(Suit::Clubs, SuitRank::Jack));
        hand.insert(Card::tarok(5));
        hand.insert(Card::tarok(10));

        let lead = Card::suit_card(Suit::Hearts, SuitRank::Queen);
        let c = ctx(hand, Some(lead), Some(lead), None);
        let legal = generate_legal_moves(&c);
        assert_eq!(legal.len(), 2); // Only taroks
    }

    #[test]
    fn klop_overplay_follow_suit() {
        let mut hand = CardSet::EMPTY;
        hand.insert(Card::suit_card(Suit::Hearts, SuitRank::King));
        hand.insert(Card::suit_card(Suit::Hearts, SuitRank::Pip1));
        hand.insert(Card::tarok(5));

        let lead = Card::suit_card(Suit::Hearts, SuitRank::Queen);
        let c = ctx(hand, Some(lead), Some(lead), Some("klop"));
        let legal = generate_legal_moves(&c);
        // Only ♥K beats ♥Q
        assert_eq!(legal.len(), 1);
        assert!(legal.contains(Card::suit_card(Suit::Hearts, SuitRank::King)));
    }

    #[test]
    fn klop_pagat_banned_when_other_taroks() {
        let mut hand = CardSet::EMPTY;
        hand.insert(Card::tarok(PAGAT));
        hand.insert(Card::tarok(5));
        hand.insert(Card::tarok(10));

        let lead = Card::tarok(3);
        let c = ctx(hand, Some(lead), Some(lead), Some("klop"));
        let legal = generate_legal_moves(&c);
        // Pagat should be banned, only 5 and 10
        assert!(!legal.contains(Card::tarok(PAGAT)));
        assert_eq!(legal.len(), 2);
    }

    #[test]
    fn pagat_forced_when_mond_skis_in_trick() {
        let mut hand = CardSet::EMPTY;
        hand.insert(Card::tarok(PAGAT));
        hand.insert(Card::tarok(5));
        hand.insert(Card::tarok(10));

        let lead = Card::tarok(MOND);
        let mut tc = CardSet::EMPTY;
        tc.insert(Card::tarok(MOND));
        tc.insert(Card::tarok(SKIS));

        let c = MoveCtx {
            hand,
            lead_card: Some(lead),
            lead_suit: None,
            best_card: Some(Card::tarok(SKIS)),
            contract_name: Some("berac"),
            is_last_trick: false,
            trick_cards: tc,
        };
        let legal = generate_legal_moves(&c);
        // Must play Pagat (it wins)
        assert_eq!(legal.len(), 1);
        assert!(legal.contains(Card::tarok(PAGAT)));
    }
}
