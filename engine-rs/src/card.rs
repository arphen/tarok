/// Card types, suits, and the 54-card Tarok deck.
///
/// Cards are represented compactly: a `Card` is a `u8` index (0..54) into
/// the canonical deck order.  Hands and sets of cards are `CardSet` (a `u64`
/// bitmask), giving O(1) membership tests, intersections, and suit filtering.

// -----------------------------------------------------------------------
// Card type / suit enums
// -----------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CardType {
    Tarok = 0,
    Suit = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Suit {
    Hearts = 0,
    Diamonds = 1,
    Clubs = 2,
    Spades = 3,
}

impl Suit {
    pub const ALL: [Suit; 4] = [Suit::Hearts, Suit::Diamonds, Suit::Clubs, Suit::Spades];

    pub fn from_u8(v: u8) -> Option<Suit> {
        match v {
            0 => Some(Suit::Hearts),
            1 => Some(Suit::Diamonds),
            2 => Some(Suit::Clubs),
            3 => Some(Suit::Spades),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SuitRank {
    Pip1 = 1,
    Pip2 = 2,
    Pip3 = 3,
    Pip4 = 4,
    Jack = 5,
    Knight = 6,
    Queen = 7,
    King = 8,
}

impl SuitRank {
    pub fn from_u8(v: u8) -> Option<SuitRank> {
        match v {
            1 => Some(SuitRank::Pip1),
            2 => Some(SuitRank::Pip2),
            3 => Some(SuitRank::Pip3),
            4 => Some(SuitRank::Pip4),
            5 => Some(SuitRank::Jack),
            6 => Some(SuitRank::Knight),
            7 => Some(SuitRank::Queen),
            8 => Some(SuitRank::King),
            _ => None,
        }
    }
}

// -----------------------------------------------------------------------
// Card — a single card identified by index 0..54
// -----------------------------------------------------------------------

pub const DECK_SIZE: usize = 54;
pub const NUM_TAROKS: usize = 22;
pub const PAGAT: u8 = 1;
pub const MOND: u8 = 21;
pub const SKIS: u8 = 22;

/// Compact card representation: a u8 index into the canonical deck.
///
/// Layout (matches Python DECK order):
///   0..22  → Taroks (value 1..22, index = value - 1)
///   22..54 → Suit cards (4 suits × 8 ranks)
///            suit_idx * 8 + (rank - 1) + 22
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Card(pub u8);

impl std::fmt::Debug for Card {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Card({})", self.label())
    }
}

impl Card {
    #[inline]
    pub fn tarok(value: u8) -> Card {
        debug_assert!((1..=22).contains(&value));
        Card(value - 1)
    }

    #[inline]
    pub fn suit_card(suit: Suit, rank: SuitRank) -> Card {
        Card(NUM_TAROKS as u8 + suit as u8 * 8 + rank as u8 - 1)
    }

    #[inline]
    pub fn card_type(self) -> CardType {
        if self.0 < NUM_TAROKS as u8 {
            CardType::Tarok
        } else {
            CardType::Suit
        }
    }

    /// Tarok value (1..22) or suit rank value (1..8).
    #[inline]
    pub fn value(self) -> u8 {
        if self.card_type() == CardType::Tarok {
            self.0 + 1
        } else {
            (self.0 - NUM_TAROKS as u8) % 8 + 1
        }
    }

    #[inline]
    pub fn suit(self) -> Option<Suit> {
        if self.card_type() == CardType::Tarok {
            None
        } else {
            Suit::from_u8((self.0 - NUM_TAROKS as u8) / 8)
        }
    }

    #[inline]
    pub fn points(self) -> u8 {
        if self.card_type() == CardType::Tarok {
            let v = self.value();
            if v == PAGAT || v == MOND || v == SKIS {
                5
            } else {
                1
            }
        } else {
            match SuitRank::from_u8(self.value()) {
                Some(SuitRank::King) => 5,
                Some(SuitRank::Queen) => 4,
                Some(SuitRank::Knight) => 3,
                Some(SuitRank::Jack) => 2,
                _ => 1,
            }
        }
    }

    #[inline]
    pub fn is_trula(self) -> bool {
        self.card_type() == CardType::Tarok && matches!(self.value(), 1 | 21 | 22)
    }

    #[inline]
    pub fn is_king(self) -> bool {
        self.card_type() == CardType::Suit && self.value() == SuitRank::King as u8
    }

    /// Does `self` beat `other` given lead suit?
    #[inline]
    pub fn beats(self, other: Card, lead_suit: Option<Suit>) -> bool {
        match (self.card_type(), other.card_type()) {
            (CardType::Tarok, CardType::Tarok) => {
                if self.value() == SKIS {
                    return true;
                }
                if other.value() == SKIS {
                    return false;
                }
                self.value() > other.value()
            }
            (CardType::Tarok, CardType::Suit) => true,
            (CardType::Suit, CardType::Tarok) => false,
            (CardType::Suit, CardType::Suit) => {
                if self.suit() == other.suit() {
                    self.value() > other.value()
                } else {
                    self.suit() == lead_suit
                }
            }
        }
    }

    pub fn label(self) -> String {
        if self.card_type() == CardType::Tarok {
            let v = self.value();
            match v {
                22 => "Škis".to_string(),
                _ => {
                    const ROMAN: [&str; 21] = [
                        "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII",
                        "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX", "XXI",
                    ];
                    ROMAN[(v - 1) as usize].to_string()
                }
            }
        } else {
            let suit = self.suit().unwrap();
            let rank = SuitRank::from_u8(self.value()).unwrap();
            let sym = match suit {
                Suit::Hearts => "♥",
                Suit::Diamonds => "♦",
                Suit::Clubs => "♣",
                Suit::Spades => "♠",
            };
            let r = match rank {
                SuitRank::King => "K",
                SuitRank::Queen => "Q",
                SuitRank::Knight => "C",
                SuitRank::Jack => "J",
                _ => {
                    let is_red = matches!(suit, Suit::Hearts | Suit::Diamonds);
                    match (rank as u8, is_red) {
                        (1, true) => "1",
                        (2, true) => "2",
                        (3, true) => "3",
                        (4, true) => "4",
                        (1, false) => "7",
                        (2, false) => "8",
                        (3, false) => "9",
                        (4, false) => "10",
                        _ => "?",
                    }
                }
            };
            format!("{r}{sym}")
        }
    }
}

// -----------------------------------------------------------------------
// CardSet — bitmask over 54 cards (fits in u64)
// -----------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct CardSet(pub u64);

impl std::fmt::Debug for CardSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cards: Vec<_> = self.iter().map(|c| c.label()).collect();
        write!(f, "CardSet({cards:?})")
    }
}

/// Precomputed masks for suit filtering.
const fn suit_mask(suit_idx: u8) -> u64 {
    let base = NUM_TAROKS as u8 + suit_idx * 8;
    let mut mask = 0u64;
    let mut i = 0u8;
    while i < 8 {
        mask |= 1u64 << (base + i);
        i += 1;
    }
    mask
}

const fn tarok_mask() -> u64 {
    (1u64 << NUM_TAROKS) - 1
}

pub const TAROK_MASK: u64 = tarok_mask();
pub const SUIT_MASKS: [u64; 4] = [
    suit_mask(0), // Hearts
    suit_mask(1), // Diamonds
    suit_mask(2), // Clubs
    suit_mask(3), // Spades
];
pub const ALL_KINGS_MASK: u64 = {
    let mut mask = 0u64;
    let mut s = 0u8;
    while s < 4 {
        // King = rank 8, index = 22 + s*8 + 7
        mask |= 1u64 << (NUM_TAROKS as u8 + s * 8 + 7);
        s += 1;
    }
    mask
};
pub const PAGAT_MASK: u64 = 1u64; // Card(0) = tarok 1
pub const MOND_MASK: u64 = 1u64 << 20; // Card(20) = tarok 21
pub const SKIS_MASK: u64 = 1u64 << 21; // Card(21) = tarok 22
pub const TRULA_MASK: u64 = PAGAT_MASK | MOND_MASK | SKIS_MASK;

impl CardSet {
    pub const EMPTY: CardSet = CardSet(0);

    #[inline]
    pub fn single(c: Card) -> CardSet {
        CardSet(1u64 << c.0)
    }

    #[inline]
    pub fn contains(self, c: Card) -> bool {
        self.0 & (1u64 << c.0) != 0
    }

    #[inline]
    pub fn insert(&mut self, c: Card) {
        self.0 |= 1u64 << c.0;
    }

    #[inline]
    pub fn remove(&mut self, c: Card) {
        self.0 &= !(1u64 << c.0);
    }

    #[inline]
    pub fn len(self) -> u32 {
        self.0.count_ones()
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub fn intersect(self, other: CardSet) -> CardSet {
        CardSet(self.0 & other.0)
    }

    #[inline]
    pub fn union(self, other: CardSet) -> CardSet {
        CardSet(self.0 | other.0)
    }

    #[inline]
    pub fn difference(self, other: CardSet) -> CardSet {
        CardSet(self.0 & !other.0)
    }

    #[inline]
    pub fn taroks(self) -> CardSet {
        CardSet(self.0 & TAROK_MASK)
    }

    #[inline]
    pub fn suit(self, suit: Suit) -> CardSet {
        CardSet(self.0 & SUIT_MASKS[suit as usize])
    }

    #[inline]
    pub fn has_suit(self, suit: Suit) -> bool {
        self.0 & SUIT_MASKS[suit as usize] != 0
    }

    /// Number of cards of a specific suit.
    #[inline]
    pub fn suit_count(self, suit: Suit) -> u32 {
        self.suit(suit).len()
    }

    #[inline]
    pub fn has_taroks(self) -> bool {
        self.0 & TAROK_MASK != 0
    }

    /// Iterate over all cards in the set (lowest index first).
    pub fn iter(self) -> CardSetIter {
        CardSetIter(self.0)
    }

    /// Collect into a Vec<Card>.
    pub fn to_vec(self) -> Vec<Card> {
        self.iter().collect()
    }

    /// Build from a slice of Card.
    pub fn from_slice(cards: &[Card]) -> CardSet {
        let mut set = CardSet::EMPTY;
        for &c in cards {
            set.insert(c);
        }
        set
    }

    /// Returns cards in `self` that beat `target` with the given lead suit.
    pub fn cards_beating(self, target: Card, lead_suit: Option<Suit>) -> CardSet {
        let mut result = CardSet::EMPTY;
        for c in self.iter() {
            if c.beats(target, lead_suit) {
                result.insert(c);
            }
        }
        result
    }

    /// Count of suit voids (suits with 0 cards).
    pub fn void_count(self) -> u8 {
        let mut count = 0u8;
        for s in Suit::ALL {
            if !self.has_suit(s) {
                count += 1;
            }
        }
        count
    }

    /// Number of kings in the set.
    pub fn king_count(self) -> u8 {
        (self.0 & ALL_KINGS_MASK).count_ones() as u8
    }

    /// Number of taroks.
    pub fn tarok_count(self) -> u8 {
        self.taroks().len() as u8
    }

    /// Number of taroks with value >= 15.
    pub fn high_tarok_count(self) -> u8 {
        let mut count = 0u8;
        // Taroks 15..=22 → indices 14..=21
        for idx in 14..=21u8 {
            if self.0 & (1u64 << idx) != 0 {
                count += 1;
            }
        }
        count
    }

    /// Has all 3 trula cards.
    pub fn has_trula(self) -> bool {
        self.0 & TRULA_MASK == TRULA_MASK
    }

    /// Has all 4 kings.
    pub fn has_all_kings(self) -> bool {
        self.0 & ALL_KINGS_MASK == ALL_KINGS_MASK
    }
}

pub struct CardSetIter(u64);

impl Iterator for CardSetIter {
    type Item = Card;

    #[inline]
    fn next(&mut self) -> Option<Card> {
        if self.0 == 0 {
            None
        } else {
            let idx = self.0.trailing_zeros() as u8;
            self.0 &= self.0 - 1; // clear lowest set bit
            Some(Card(idx))
        }
    }
}

// -----------------------------------------------------------------------
// Deck — the canonical 54 cards
// -----------------------------------------------------------------------

pub fn build_deck() -> [Card; DECK_SIZE] {
    let mut deck = [Card(0); DECK_SIZE];
    for i in 0..DECK_SIZE {
        deck[i] = Card(i as u8);
    }
    deck
}

pub const FULL_DECK: CardSet = CardSet((1u64 << DECK_SIZE) - 1);

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deck_has_54_cards() {
        assert_eq!(build_deck().len(), 54);
        assert_eq!(FULL_DECK.len(), 54);
    }

    #[test]
    fn tarok_beats_suit() {
        let t5 = Card::tarok(5);
        let kh = Card::suit_card(Suit::Hearts, SuitRank::King);
        assert!(t5.beats(kh, Some(Suit::Hearts)));
        assert!(!kh.beats(t5, Some(Suit::Hearts)));
    }

    #[test]
    fn skis_beats_everything() {
        let skis = Card::tarok(SKIS);
        let mond = Card::tarok(MOND);
        assert!(skis.beats(mond, None));
    }

    #[test]
    fn suit_card_points() {
        assert_eq!(Card::suit_card(Suit::Hearts, SuitRank::King).points(), 5);
        assert_eq!(Card::suit_card(Suit::Clubs, SuitRank::Pip1).points(), 1);
        assert_eq!(Card::suit_card(Suit::Spades, SuitRank::Jack).points(), 2);
    }

    #[test]
    fn card_set_suit_filter() {
        let mut hand = CardSet::EMPTY;
        hand.insert(Card::suit_card(Suit::Hearts, SuitRank::King));
        hand.insert(Card::suit_card(Suit::Hearts, SuitRank::Queen));
        hand.insert(Card::suit_card(Suit::Clubs, SuitRank::Jack));
        hand.insert(Card::tarok(5));

        assert_eq!(hand.suit(Suit::Hearts).len(), 2);
        assert_eq!(hand.suit(Suit::Clubs).len(), 1);
        assert_eq!(hand.taroks().len(), 1);
    }

    #[test]
    fn trula_detection() {
        let mut set = CardSet::EMPTY;
        set.insert(Card::tarok(PAGAT));
        set.insert(Card::tarok(MOND));
        assert!(!set.has_trula());
        set.insert(Card::tarok(SKIS));
        assert!(set.has_trula());
    }
}
