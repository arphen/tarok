/// Game state, enums, and the Trick data structure.
///
/// This is a mutable game state designed for fast simulation — no Python
/// objects, no heap allocations per move where possible.
use crate::card::*;
use crate::trick_eval;
use rand::prelude::*;

// -----------------------------------------------------------------------
// Enums
// -----------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Phase {
    Dealing = 0,
    Bidding = 1,
    KingCalling = 2,
    TalonExchange = 3,
    Announcements = 4,
    TrickPlay = 5,
    Scoring = 6,
    Finished = 7,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Contract {
    Klop = 0,
    Three = 1,
    Two = 2,
    One = 3,
    SoloThree = 4,
    SoloTwo = 5,
    SoloOne = 6,
    Solo = 7,
    Berac = 8,
    BarvniValat = 9,
}

impl Contract {
    pub const NUM: usize = 10;

    pub fn strength(self) -> u8 {
        // Bidding order intentionally differs from enum discriminants.
        // Barvni Valat is not an active bid here; it is handled separately.
        match self {
            Contract::Klop => 0,
            Contract::Three => 1,
            Contract::Two => 2,
            Contract::One => 3,
            Contract::SoloThree => 4,
            Contract::SoloTwo => 5,
            Contract::SoloOne => 6,
            Contract::Berac => 7,
            Contract::Solo => 8,
            Contract::BarvniValat => 9,
        }
    }

    pub fn is_solo(self) -> bool {
        matches!(
            self,
            Contract::SoloThree | Contract::SoloTwo | Contract::SoloOne | Contract::Solo
        )
    }

    pub fn is_klop(self) -> bool {
        self == Contract::Klop
    }

    pub fn is_berac(self) -> bool {
        self == Contract::Berac
    }

    pub fn is_barvni_valat(self) -> bool {
        self == Contract::BarvniValat
    }

    pub fn requires_overplay(self) -> bool {
        matches!(self, Contract::Klop | Contract::Berac)
    }

    pub fn talon_cards(self) -> u8 {
        match self {
            Contract::Three | Contract::SoloThree => 3,
            Contract::Two | Contract::SoloTwo => 2,
            Contract::One | Contract::SoloOne => 1,
            _ => 0,
        }
    }

    pub fn base_value(self) -> i32 {
        match self {
            Contract::Klop => 0,
            Contract::Three => 10,
            Contract::Two => 20,
            Contract::One => 30,
            Contract::SoloThree => 40,
            Contract::SoloTwo => 50,
            Contract::SoloOne => 60,
            Contract::Solo => 80,
            Contract::Berac => 70,
            Contract::BarvniValat => 125,
        }
    }

    pub fn is_biddable(self) -> bool {
        !matches!(self, Contract::Klop | Contract::BarvniValat)
    }

    pub fn from_u8(v: u8) -> Option<Contract> {
        match v {
            0 => Some(Contract::Klop),
            1 => Some(Contract::Three),
            2 => Some(Contract::Two),
            3 => Some(Contract::One),
            4 => Some(Contract::SoloThree),
            5 => Some(Contract::SoloTwo),
            6 => Some(Contract::SoloOne),
            7 => Some(Contract::Solo),
            8 => Some(Contract::Berac),
            9 => Some(Contract::BarvniValat),
            _ => None,
        }
    }

    /// Contracts that can be actively bid (excludes Klop and BarvniValat).
    pub const BIDDABLE: [Contract; 8] = [
        Contract::Three,
        Contract::Two,
        Contract::One,
        Contract::SoloThree,
        Contract::SoloTwo,
        Contract::SoloOne,
        Contract::Solo,
        Contract::Berac,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PlayerRole {
    Declarer = 0,
    Partner = 1,
    Opponent = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Team {
    DeclarerTeam = 0,
    OpponentTeam = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Announcement {
    Trula = 0,
    Kings = 1,
    PagatUltimo = 2,
    Valat = 3,
}

impl Announcement {
    pub const ALL: [Announcement; 4] = [
        Announcement::Trula,
        Announcement::Kings,
        Announcement::PagatUltimo,
        Announcement::Valat,
    ];

    pub fn from_u8(v: u8) -> Option<Announcement> {
        match v {
            0 => Some(Announcement::Trula),
            1 => Some(Announcement::Kings),
            2 => Some(Announcement::PagatUltimo),
            3 => Some(Announcement::Valat),
            _ => None,
        }
    }
}

/// Kontra escalation levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum KontraLevel {
    None = 0,
    Kontra = 1,
    Re = 2,
    Sub = 3,
}

impl KontraLevel {
    pub fn multiplier(self) -> i32 {
        match self {
            KontraLevel::None => 1,
            KontraLevel::Kontra => 2,
            KontraLevel::Re => 4,
            KontraLevel::Sub => 8,
        }
    }

    pub fn next_level(self) -> Option<KontraLevel> {
        match self {
            KontraLevel::None => Some(KontraLevel::Kontra),
            KontraLevel::Kontra => Some(KontraLevel::Re),
            KontraLevel::Re => Some(KontraLevel::Sub),
            KontraLevel::Sub => None,
        }
    }

    pub fn is_opponent_turn(self) -> bool {
        matches!(self, KontraLevel::None | KontraLevel::Re)
    }
}

/// Kontra targets: base game + 4 bonus announcements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum KontraTarget {
    Game = 0,
    Trula = 1,
    Kings = 2,
    PagatUltimo = 3,
    Valat = 4,
}

impl KontraTarget {
    pub const NUM: usize = 5;

    pub fn from_announcement(a: Announcement) -> KontraTarget {
        match a {
            Announcement::Trula => KontraTarget::Trula,
            Announcement::Kings => KontraTarget::Kings,
            Announcement::PagatUltimo => KontraTarget::PagatUltimo,
            Announcement::Valat => KontraTarget::Valat,
        }
    }
}

// -----------------------------------------------------------------------
// Bid
// -----------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct Bid {
    pub player: u8,
    pub contract: Option<Contract>, // None = pass
}

// -----------------------------------------------------------------------
// Trick
// -----------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Trick {
    pub lead_player: u8,
    pub cards: [(u8, Card); 4], // (player_idx, card) — fixed-size
    pub count: u8,              // how many cards played so far (0..=4)
}

impl Trick {
    pub fn new(lead_player: u8) -> Trick {
        Trick {
            lead_player,
            cards: [(0, Card(0)); 4],
            count: 0,
        }
    }

    pub fn play(&mut self, player: u8, card: Card) {
        debug_assert!((self.count as usize) < 4);
        self.cards[self.count as usize] = (player, card);
        self.count += 1;
    }

    pub fn is_complete(&self) -> bool {
        self.count == 4
    }

    pub fn lead_card(&self) -> Option<Card> {
        if self.count == 0 {
            None
        } else {
            Some(self.cards[0].1)
        }
    }

    pub fn lead_suit(&self) -> Option<Suit> {
        self.lead_card().and_then(|c| {
            if c.card_type() == CardType::Tarok {
                None
            } else {
                c.suit()
            }
        })
    }

    pub fn best_card(&self) -> Option<Card> {
        if self.count == 0 {
            return None;
        }
        let ls = self.lead_suit();
        let mut best = self.cards[0].1;
        for i in 1..self.count as usize {
            if self.cards[i].1.beats(best, ls) {
                best = self.cards[i].1;
            }
        }
        Some(best)
    }

    pub fn played_cards_set(&self) -> CardSet {
        let mut s = CardSet::EMPTY;
        for i in 0..self.count as usize {
            s.insert(self.cards[i].1);
        }
        s
    }

    pub fn points(&self) -> u8 {
        let mut total = 0u8;
        for i in 0..self.count as usize {
            total += self.cards[i].1.points();
        }
        total
    }
}

// -----------------------------------------------------------------------
// GameState
// -----------------------------------------------------------------------

pub const NUM_PLAYERS: usize = 4;
pub const TRICKS_PER_GAME: usize = 12;
pub const HAND_SIZE: usize = 12;
pub const TALON_SIZE: usize = 6;

#[derive(Debug, Clone)]
pub struct GameState {
    pub phase: Phase,
    pub hands: [CardSet; NUM_PLAYERS],
    pub talon: CardSet,

    // Bidding
    pub bids: Vec<Bid>,
    pub current_bidder: u8,
    pub declarer: Option<u8>,
    pub contract: Option<Contract>,

    // King calling
    pub called_king: Option<Card>,
    pub partner: Option<u8>,

    // Talon
    pub talon_revealed: Vec<Vec<Card>>,
    pub put_down: CardSet,

    // Announcements: per-player bitflags (announcement index → bit)
    pub announcements: [u8; NUM_PLAYERS], // bits: Trula=0, Kings=1, Pagat=2, Valat=3
    pub kontra_levels: [KontraLevel; KontraTarget::NUM],

    // Trick play
    pub tricks: Vec<Trick>,
    pub current_trick: Option<Trick>,
    pub current_player: u8,

    // All cards played so far (fast lookup for encoding)
    pub played_cards: CardSet,

    // Roles
    pub roles: [PlayerRole; NUM_PLAYERS],

    // Tracking
    pub dealer: u8,

    // Scores
    pub scores: [i32; NUM_PLAYERS],
}

impl GameState {
    pub fn new(dealer: u8) -> GameState {
        GameState {
            phase: Phase::Dealing,
            hands: [CardSet::EMPTY; NUM_PLAYERS],
            talon: CardSet::EMPTY,
            bids: Vec::new(),
            // Forehand (obvezen) goes last; start bidding at dealer+2
            current_bidder: (dealer + 2) % NUM_PLAYERS as u8,
            declarer: None,
            contract: None,
            called_king: None,
            partner: None,
            talon_revealed: Vec::new(),
            put_down: CardSet::EMPTY,
            announcements: [0; NUM_PLAYERS],
            kontra_levels: [KontraLevel::None; KontraTarget::NUM],
            tricks: Vec::with_capacity(TRICKS_PER_GAME),
            current_trick: None,
            current_player: (dealer + 1) % NUM_PLAYERS as u8,
            played_cards: CardSet::EMPTY,
            roles: [PlayerRole::Opponent; NUM_PLAYERS],
            dealer,
            scores: [0; NUM_PLAYERS],
        }
    }

    /// The forehand / obvezen player (first after dealer).
    pub fn forehand(&self) -> u8 {
        (self.dealer + 1) % NUM_PLAYERS as u8
    }

    pub fn get_team(&self, player: u8) -> Team {
        match self.roles[player as usize] {
            PlayerRole::Declarer | PlayerRole::Partner => Team::DeclarerTeam,
            PlayerRole::Opponent => Team::OpponentTeam,
        }
    }

    pub fn tricks_played(&self) -> usize {
        self.tricks.len()
    }

    pub fn is_last_trick(&self) -> bool {
        self.tricks_played() == 11 && self.current_trick.is_some()
    }

    pub fn is_partner_revealed(&self) -> bool {
        if let Some(king) = self.called_king {
            if self.played_cards.contains(king) {
                return true;
            }
            // Also check current trick
            if let Some(ref trick) = self.current_trick {
                if trick.played_cards_set().contains(king) {
                    return true;
                }
            }
        }
        false
    }

    /// Has a given announcement been made by any player?
    pub fn is_announced(&self, ann: Announcement) -> bool {
        let bit = 1u8 << (ann as u8);
        self.announcements.iter().any(|a| a & bit != 0)
    }

    /// Get all announcements made by all players as a set.
    pub fn all_announcements(&self) -> u8 {
        self.announcements.iter().fold(0u8, |acc, a| acc | a)
    }

    /// Get kontra level for a target.
    pub fn kontra(&self, target: KontraTarget) -> KontraLevel {
        self.kontra_levels[target as usize]
    }

    /// Get kontra multiplier for a target.
    pub fn kontra_multiplier(&self, target: KontraTarget) -> i32 {
        self.kontra_levels[target as usize].multiplier()
    }

    /// Contract name string for rules matching.
    pub fn contract_name(&self) -> Option<&'static str> {
        self.contract.map(|c| match c {
            Contract::Klop => "klop",
            Contract::Three => "three",
            Contract::Two => "two",
            Contract::One => "one",
            Contract::SoloThree => "solo_three",
            Contract::SoloTwo => "solo_two",
            Contract::SoloOne => "solo_one",
            Contract::Solo => "solo",
            Contract::Berac => "berac",
            Contract::BarvniValat => "barvni_valat",
        })
    }

    /// Whether the state is effectively solo (solo contract or partner is None).
    pub fn is_effectively_solo(&self) -> bool {
        self.contract.map_or(false, |c| c.is_solo()) || self.partner.is_none()
    }

    /// Generate a legal-bid mask for a given player.
    ///
    /// Index 0 = pass, indices 1..=8 map to Contract::BIDDABLE.
    /// Enforces forehand (obvezen) rules:
    ///  - Only forehand can bid THREE.
    ///  - Forehand can *match* the current highest (>=), others must outbid (>).
    pub fn legal_bid_mask(&self, player: u8) -> [u8; 9] {
        let mut mask = [0u8; 9];
        mask[0] = 1; // pass always legal

        let is_forehand = player == self.forehand();
        let highest = self
            .bids
            .iter()
            .filter_map(|b| b.contract)
            .max_by_key(|c| c.strength());

        for (idx, &contract) in Contract::BIDDABLE.iter().enumerate() {
            if contract == Contract::Three && !is_forehand {
                continue;
            }
            let legal = match highest {
                Some(h) if is_forehand => contract.strength() >= h.strength(),
                Some(h) => contract.strength() > h.strength(),
                None => true,
            };
            if legal {
                mask[idx + 1] = 1;
            }
        }
        mask
    }

    // ------------------------------------------------------------------
    // Game lifecycle methods (moved from PyGameState for self-play use)
    // ------------------------------------------------------------------

    /// Shuffle a full deck and deal 12 cards per player + 6 to talon.
    pub fn deal(&mut self, rng: &mut impl Rng) {
        let mut deck = build_deck();
        deck.shuffle(rng);
        for (i, &card) in deck.iter().enumerate() {
            if i < 48 {
                self.hands[i / 12].insert(card);
            } else {
                self.talon.insert(card);
            }
        }
        self.phase = Phase::Bidding;
    }

    /// Record a bid (pass = None).
    pub fn add_bid(&mut self, player: u8, contract: Option<Contract>) {
        self.bids.push(Bid { player, contract });
    }

    /// Return the list of contracts this player may legally bid.
    pub fn legal_bids(&self, player: u8) -> Vec<Contract> {
        let is_forehand = player == self.forehand();
        let highest = self
            .bids
            .iter()
            .filter_map(|b| b.contract)
            .max_by_key(|c| c.strength());

        let mut result = Vec::new();
        for c in Contract::BIDDABLE {
            if c == Contract::Three && !is_forehand {
                continue;
            }
            let legal = match highest {
                Some(h) if is_forehand => c.strength() >= h.strength(),
                Some(h) => c.strength() > h.strength(),
                None => true,
            };
            if legal {
                result.push(c);
            }
        }
        result
    }

    /// Kings (or queens) that the declarer may call.
    pub fn callable_kings(&self) -> Vec<Card> {
        let declarer = match self.declarer {
            Some(d) => d as usize,
            None => return Vec::new(),
        };
        let hand = self.hands[declarer];
        let mut kings = Vec::new();
        for s in Suit::ALL {
            let king = Card::suit_card(s, SuitRank::King);
            if !hand.contains(king) {
                kings.push(king);
            }
        }
        if kings.is_empty() {
            for s in Suit::ALL {
                let queen = Card::suit_card(s, SuitRank::Queen);
                if !hand.contains(queen) {
                    kings.push(queen);
                }
            }
        }
        kings
    }

    /// Begin a new trick with the given lead player.
    pub fn start_trick(&mut self, lead_player: u8) {
        self.current_trick = Some(Trick::new(lead_player));
    }

    /// Play a card: remove from hand, add to trick, mark as played.
    pub fn play_card(&mut self, player: u8, card: Card) {
        self.hands[player as usize].remove(card);
        if let Some(ref mut trick) = self.current_trick {
            trick.play(player, card);
        }
        self.played_cards.insert(card);
    }

    /// Evaluate the current trick, archive it, return (winner, points).
    pub fn finish_trick(&mut self) -> (u8, u8) {
        let trick = self.current_trick.take().expect("No current trick");
        let is_last = self.tricks.len() == 11;
        let result = trick_eval::evaluate_trick(&trick, is_last, self.contract);
        self.tricks.push(trick);
        (result.winner, result.points)
    }
}
