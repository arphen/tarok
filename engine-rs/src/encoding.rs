/// State encoding — writes features directly into a numpy buffer.
///
/// v2 encoding matches Python STATE_SIZE = 450 layout:
///  - v1 features (270 dims): hand, played, trick, talon, position, contract, etc.
///  - Belief probabilities: 3×54 opponent card likelihoods with void inference
///  - Opponent card-play stats: 3×4 features (taroks, suits, kings, total)
///  - Trick context: 6 features (position + lead suit)

use crate::card::*;
use crate::game_state::*;
use crate::trick_eval::evaluate_trick;

/// v1 base features size (for backward compat detection)
pub const V1_STATE_SIZE: usize = 270;
/// v2 belief features: 3 opponents × 54 cards
const BELIEF_SIZE: usize = 3 * DECK_SIZE;
/// v2 opponent play stats: 3 opponents × 4 features
const OPP_STATS_SIZE: usize = 3 * 4;
/// v2 trick context: position(1) + tarok_lead(1) + suit_lead(4)
const TRICK_CTX_SIZE: usize = 6;
/// Full v2 state size
pub const STATE_SIZE: usize = V1_STATE_SIZE + BELIEF_SIZE + OPP_STATS_SIZE + TRICK_CTX_SIZE; // 450
pub const ORACLE_EXTRA: usize = 3 * DECK_SIZE; // 162
pub const ORACLE_STATE_SIZE: usize = STATE_SIZE + ORACLE_EXTRA; // 612

/// Decision type codes matching Python DecisionType enum.
pub const DT_BID: u8 = 0;
pub const DT_KING_CALL: u8 = 1;
pub const DT_TALON_PICK: u8 = 2;
pub const DT_CARD_PLAY: u8 = 3;
pub const DT_ANNOUNCE: u8 = 4;

/// Encode a state into a pre-existing float buffer.
/// Buffer must be at least STATE_SIZE long and pre-zeroed.
pub fn encode_state(buf: &mut [f32], state: &GameState, player: u8, decision_type: u8) {
    debug_assert!(buf.len() >= STATE_SIZE);
    let mut o = 0usize;

    // Hand (54 binary)
    let hand = state.hands[player as usize];
    for c in hand.iter() {
        buf[o + c.0 as usize] = 1.0;
    }
    o += DECK_SIZE;

    // Played cards (54 binary)
    for c in state.played_cards.iter() {
        buf[o + c.0 as usize] = 1.0;
    }
    o += DECK_SIZE;

    // Current trick cards (54 binary)
    if let Some(ref trick) = state.current_trick {
        for i in 0..trick.count as usize {
            buf[o + trick.cards[i].1 .0 as usize] = 1.0;
        }
    }
    o += DECK_SIZE;

    // Talon visible (54 binary)
    if !state.talon_revealed.is_empty() && Some(player) == state.declarer {
        for group in &state.talon_revealed {
            for &card in group {
                buf[o + card.0 as usize] = 1.0;
            }
        }
    }
    o += DECK_SIZE;

    // Player position relative to dealer (4 one-hot)
    let rel_pos = ((player as usize + NUM_PLAYERS - state.dealer as usize) % NUM_PLAYERS) as usize;
    buf[o + rel_pos] = 1.0;
    o += 4;

    // Contract (10 one-hot)
    if let Some(c) = state.contract {
        let idx = c as usize;
        if idx < 10 {
            buf[o + idx] = 1.0;
        }
    }
    o += 10;

    // Phase (3 one-hot: bidding, trick_play, other)
    match state.phase {
        Phase::Bidding => buf[o] = 1.0,
        Phase::TrickPlay => buf[o + 1] = 1.0,
        _ => buf[o + 2] = 1.0,
    }
    o += 3;

    // Partner known
    if state.is_partner_revealed() {
        buf[o] = 1.0;
    }
    o += 1;

    // Tricks won by my team (normalized 0-1)
    let my_team = state.get_team(player);
    let contract_opt = state.contract;
    let my_tricks: f32 = state
        .tricks
        .iter()
        .enumerate()
        .filter(|(i, trick)| {
            let is_last = *i == state.tricks.len() - 1;
            let w = evaluate_trick(trick, is_last, contract_opt).winner;
            state.get_team(w) == my_team
        })
        .count() as f32;
    buf[o] = my_tricks / 12.0;
    o += 1;

    // Tricks played (normalized 0-1)
    buf[o] = state.tricks_played() as f32 / 12.0;
    o += 1;

    // Decision type (5 one-hot)
    if (decision_type as usize) < 5 {
        buf[o + decision_type as usize] = 1.0;
    }
    o += 5;

    // Highest bid so far (9 one-hot: no_bid + 8 contracts)
    let highest_bid = state
        .bids
        .iter()
        .filter_map(|b| b.contract)
        .max_by_key(|c| c.strength());
    match highest_bid {
        Some(c) => {
            // BID_ACTIONS: [pass, THREE, TWO, ONE, S3, S2, S1, SOLO, BERAC]
            // Map contract to bid index
            let bid_idx = match c {
                Contract::Three => 1,
                Contract::Two => 2,
                Contract::One => 3,
                Contract::SoloThree => 4,
                Contract::SoloTwo => 5,
                Contract::SoloOne => 6,
                Contract::Solo => 7,
                Contract::Berac => 8,
                _ => 0,
            };
            buf[o + bid_idx] = 1.0;
        }
        None => buf[o] = 1.0, // No bid yet
    }
    o += 9;

    // Passed players (4 binary, relative to dealer)
    let passed: u8 = state
        .bids
        .iter()
        .filter(|b| b.contract.is_none())
        .fold(0u8, |acc, b| acc | (1 << b.player));
    for i in 0..NUM_PLAYERS {
        let p = ((state.dealer as usize + 1 + i) % NUM_PLAYERS) as u8;
        if passed & (1 << p) != 0 {
            buf[o + i] = 1.0;
        }
    }
    o += 4;

    // Hand strength features (normalized)
    buf[o] = hand.tarok_count() as f32 / 12.0;
    buf[o + 1] = hand.high_tarok_count() as f32 / 7.0;
    buf[o + 2] = hand.king_count() as f32 / 4.0;
    buf[o + 3] = hand.void_count() as f32 / 4.0;
    o += 4;

    // Announcements made (4 binary)
    let all_ann = state.all_announcements();
    for i in 0..4u8 {
        if all_ann & (1 << i) != 0 {
            buf[o + i as usize] = 1.0;
        }
    }
    o += 4;

    // Kontra levels (5 features, normalized)
    for i in 0..KontraTarget::NUM {
        let level = state.kontra_levels[i];
        buf[o + i] = (level.multiplier() as f32 - 1.0) / 7.0;
    }
    o += 5;

    // Role one-hot (3 features: is_declarer, is_partner, is_opposition)
    if let Some(decl) = state.declarer {
        if player == decl {
            buf[o] = 1.0;     // is_declarer
        } else if state.partner == Some(player) {
            buf[o + 1] = 1.0; // is_partner (revealed)
        } else if state.get_team(player) == Team::DeclarerTeam {
            buf[o + 1] = 1.0; // is_partner (hidden — we know our own role)
        } else {
            buf[o + 2] = 1.0; // is_opposition
        }
    }
    // During bidding (no declarer yet), all three stay 0 — role unknown
    o += 3;

    debug_assert_eq!(o, V1_STATE_SIZE);

    // ===================================================================
    // v2 FEATURES: Belief tracking + card-play statistics
    // ===================================================================

    // --- 3×54 opponent belief probabilities ---
    // Compute cards known to this player
    let mut known = hand;
    known = known.union(state.played_cards);
    if let Some(ref trick) = state.current_trick {
        known = known.union(trick.played_cards_set());
    }
    if !state.talon_revealed.is_empty() && Some(player) == state.declarer {
        for group in &state.talon_revealed {
            for &card in group {
                known.insert(card);
            }
        }
    }

    // Detect opponent void suits from trick history
    let mut opp_void: [u8; NUM_PLAYERS] = [0; NUM_PLAYERS]; // bitmask per player: bit=suit_idx → void
    for trick in &state.tricks {
        if trick.count == 0 {
            continue;
        }
        let lead_card = trick.cards[0].1;
        let lead_suit = if lead_card.card_type() == CardType::Suit {
            lead_card.suit()
        } else {
            None
        };
        if let Some(ls) = lead_suit {
            for i in 1..trick.count as usize {
                let (p, card) = trick.cards[i];
                if p == player {
                    continue;
                }
                if card.suit() != Some(ls) {
                    // Player didn't follow suit → void in that suit
                    opp_void[p as usize] |= 1 << (ls as u8);
                }
            }
        }
    }

    // Write belief probabilities for each opponent
    for opp_offset in 1..NUM_PLAYERS {
        let opp_idx = ((player as usize + opp_offset) % NUM_PLAYERS) as u8;
        let void_mask = opp_void[opp_idx as usize];
        for cidx in 0..DECK_SIZE {
            let c = Card(cidx as u8);
            if known.contains(c) {
                // Known card — not in any opponent's hand
                // buf stays 0.0
            } else {
                // Check void constraint
                let is_void = if let Some(s) = c.suit() {
                    void_mask & (1 << (s as u8)) != 0
                } else {
                    false
                };
                if !is_void {
                    buf[o + cidx] = 1.0 / 3.0; // uniform prior across 3 opponents
                }
            }
        }
        o += DECK_SIZE;
    }

    // --- Per-opponent card-play counts (3×4 features) ---
    for opp_offset in 1..NUM_PLAYERS {
        let opp_idx = ((player as usize + opp_offset) % NUM_PLAYERS) as u8;
        let mut taroks_played: f32 = 0.0;
        let mut suit_played: f32 = 0.0;
        let mut kings_played: f32 = 0.0;
        let mut total_played: f32 = 0.0;

        for trick in &state.tricks {
            for i in 0..trick.count as usize {
                let (p, card) = trick.cards[i];
                if p == opp_idx {
                    total_played += 1.0;
                    if card.card_type() == CardType::Tarok {
                        taroks_played += 1.0;
                    } else {
                        suit_played += 1.0;
                    }
                    if card.is_king() {
                        kings_played += 1.0;
                    }
                }
            }
        }
        if let Some(ref trick) = state.current_trick {
            for i in 0..trick.count as usize {
                let (p, card) = trick.cards[i];
                if p == opp_idx {
                    total_played += 1.0;
                    if card.card_type() == CardType::Tarok {
                        taroks_played += 1.0;
                    } else {
                        suit_played += 1.0;
                    }
                    if card.is_king() {
                        kings_played += 1.0;
                    }
                }
            }
        }

        buf[o] = taroks_played / 12.0;
        buf[o + 1] = suit_played / 12.0;
        buf[o + 2] = kings_played / 4.0;
        buf[o + 3] = total_played / 12.0;
        o += 4;
    }

    // --- Trick context features (6 features) ---
    if let Some(ref trick) = state.current_trick {
        if trick.count > 0 {
            buf[o] = trick.count as f32 / 4.0; // trick position
            let lead_card = trick.cards[0].1;
            if lead_card.card_type() == CardType::Tarok {
                buf[o + 1] = 1.0; // tarok lead
            } else if let Some(s) = lead_card.suit() {
                buf[o + 2 + s as usize] = 1.0; // suit lead (4 positions)
            }
        }
    }
    o += TRICK_CTX_SIZE;

    debug_assert_eq!(o, STATE_SIZE);
}

/// Encode oracle (perfect info) state. Buffer must be ORACLE_STATE_SIZE long.
pub fn encode_oracle_state(buf: &mut [f32], state: &GameState, player: u8, decision_type: u8) {
    debug_assert!(buf.len() >= ORACLE_STATE_SIZE);
    encode_state(buf, state, player, decision_type);

    let mut o = STATE_SIZE;
    for offset in 1..NUM_PLAYERS {
        let opp = ((player as usize + offset) % NUM_PLAYERS) as u8;
        for c in state.hands[opp as usize].iter() {
            buf[o + c.0 as usize] = 1.0;
        }
        o += DECK_SIZE;
    }
}

/// Write a legal‐move binary mask into a 54‐element buffer.
pub fn encode_legal_mask(buf: &mut [f32], legal: CardSet) {
    for c in legal.iter() {
        buf[c.0 as usize] = 1.0;
    }
}
