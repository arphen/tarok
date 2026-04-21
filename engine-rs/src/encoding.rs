/// State encoding — writes features directly into a numpy buffer.
///
/// v5 encoding (626 dims):
///  - v1 features (270 dims): hand, played+unpicked_talon, current_trick,
///    declarer forced-retention plane, position, contract, phase,
///    partner_known, tricks, decision_type, bid history, passed players,
///    hand strength, announcements, kontra, role.
///  - Belief block 270..431 (162 dims): 3×54 opponent card likelihoods
///    with suit-void, tarok-void, unpicked-talon-retirement, and
///    forced-retention (declarer must-hold) constraints applied.
///  - Trick context 432..437 (6 dims): position + lead suit/type.
///  - Per-opponent played planes 438..599 (162 dims): 3×54 cards played.
///  - Per-opponent tarok-void flags 600..602 (3 dims).
///  - Per-opponent suit-void flags 603..614 (12 dims).
///  - Live kings one-hot 615..618 (4 dims).
///  - Live trula one-hot 619..621 (3 dims): pagat / mond / skis.
///  - Called-king suit one-hot 622..625 (4 dims).
///
/// Rule semantics:
///  - `talon_revealed` is public (Slovenian Tarok).  Unpicked talon cards
///    appear in the `played_cards` plane and in the belief `known` set.
///  - The "declarer forced retention" plane contains cards that declarer
///    is publicly known to still hold: taroks and kings from the picked
///    talon group (these cards cannot legally be discarded).  The belief
///    block pins these cards to the declarer's column.
use crate::card::*;
use crate::game_state::*;
use crate::trick_eval::evaluate_trick;

/// v1 base features size (frozen: belief block always starts at 270).
pub const V1_STATE_SIZE: usize = 270;
/// Contract one-hot feature offset in the flat state vector.
pub const CONTRACT_OFFSET: usize = 220;
/// Contract one-hot feature length.
pub const CONTRACT_SIZE: usize = 10;
/// Belief feature block offset in the flat state vector.
pub const BELIEF_OFFSET: usize = V1_STATE_SIZE;
/// Belief features: 3 opponents × 54 cards.
const BELIEF_SIZE: usize = 3 * DECK_SIZE;
/// Trick context: position(1) + tarok_lead(1) + suit_lead(4).
const TRICK_CTX_SIZE: usize = 6;
/// Per-opponent played planes: 3 × 54.
const OPP_PLAYED_SIZE: usize = 3 * DECK_SIZE;
/// Per-opponent tarok-void flags.
const TAROK_VOID_SIZE: usize = 3;
/// Per-opponent suit-void flags: 3 opponents × 4 suits.
const SUIT_VOID_SIZE: usize = 3 * 4;
/// Live kings one-hot (hearts/diamonds/clubs/spades).
const LIVE_KINGS_SIZE: usize = 4;
/// Live trula one-hot (pagat/mond/skis).
const LIVE_TRULA_SIZE: usize = 3;
/// Called-king suit one-hot.
const CALLED_KING_SIZE: usize = 4;
/// v5 full state size.
pub const STATE_SIZE: usize = V1_STATE_SIZE
    + BELIEF_SIZE
    + TRICK_CTX_SIZE
    + OPP_PLAYED_SIZE
    + TAROK_VOID_SIZE
    + SUIT_VOID_SIZE
    + LIVE_KINGS_SIZE
    + LIVE_TRULA_SIZE
    + CALLED_KING_SIZE; // 626
pub const ORACLE_EXTRA: usize = 3 * DECK_SIZE; // 162
pub const ORACLE_STATE_SIZE: usize = STATE_SIZE + ORACLE_EXTRA; // 788

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

    // --- Compute v5 public-talon sets up front (used by several blocks) ---
    // Unpicked talon cards: revealed groups whose cards still live in
    // state.talon.  A group is "picked" once none of its cards remain.
    let (unpicked_talon_set, picked_group) = compute_talon_visibility(state);
    // Declarer forced-retention: taroks + kings in the picked group (they
    // cannot legally be discarded, so declarer must still hold them).
    let mut forced_retention = CardSet::EMPTY;
    if let Some(ref picked) = picked_group {
        for &c in picked {
            if c.card_type() == CardType::Tarok || c.is_king() {
                forced_retention.insert(c);
            }
        }
    }

    // Hand (54 binary)
    let hand = state.hands[player as usize];
    for c in hand.iter() {
        buf[o + c.0 as usize] = 1.0;
    }
    o += DECK_SIZE;

    // Played cards plane (54 binary): all tricks + unpicked talon cards.
    // Unpicked talon cards are publicly retired under Slovenian rules and
    // so they belong in the same "cards permanently out of play" plane.
    for c in state.played_cards.iter() {
        buf[o + c.0 as usize] = 1.0;
    }
    for c in unpicked_talon_set.iter() {
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

    // Declarer forced-retention plane (54 binary, public).
    // Contains cards from the picked talon group that declarer MUST still
    // hold (taroks and kings — not legally discardable).  From every
    // player's perspective this is public knowledge; non-declarers use it
    // to pin these cards onto declarer in the belief block below.
    for c in forced_retention.iter() {
        buf[o + c.0 as usize] = 1.0;
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

    // Partner known: true if the pairing is public (called king played)
    // OR this player already knows the pairing privately (they are the
    // declarer, or they are the partner themselves — whether publicly
    // revealed or still the hidden partner).
    let self_knows_pairing = match state.declarer {
        Some(decl) => player == decl || state.get_team(player) == Team::DeclarerTeam,
        None => false,
    };
    if state.is_partner_revealed() || self_knows_pairing {
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
            buf[o] = 1.0; // is_declarer
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
    // v5 BELIEF BLOCK (162 dims, offsets 270..431)
    // 3 opponents × 54 cards, with:
    //   - known cards (own hand / tricks / current trick / unpicked talon) → 0
    //   - suit-void / tarok-void constraints → 0 for that opp
    //   - declarer forced-retention (taroks+kings picked from talon) → 1.0 on
    //     declarer column, 0 on other opp columns
    //   - otherwise 1/3 uniform prior
    // ===================================================================

    // Cards known to be out of every opponent's hand.
    let mut known = hand;
    known = known.union(state.played_cards);
    known = known.union(unpicked_talon_set);
    if let Some(ref trick) = state.current_trick {
        known = known.union(trick.played_cards_set());
    }

    // Detect opponent void suits and tarok-void from trick history.
    let mut opp_void: [u8; NUM_PLAYERS] = [0; NUM_PLAYERS];
    let mut opp_tarok_void: [bool; NUM_PLAYERS] = [false; NUM_PLAYERS];
    for trick in &state.tricks {
        if trick.count == 0 {
            continue;
        }
        let lead_card = trick.cards[0].1;
        match lead_card.card_type() {
            CardType::Suit => {
                let ls = match lead_card.suit() {
                    Some(s) => s,
                    None => continue,
                };
                for i in 1..trick.count as usize {
                    let (p, card) = trick.cards[i];
                    if p == player {
                        continue;
                    }
                    match card.card_type() {
                        CardType::Suit => {
                            if card.suit() != Some(ls) {
                                opp_void[p as usize] |= 1 << (ls as u8);
                            }
                        }
                        CardType::Tarok => {
                            opp_void[p as usize] |= 1 << (ls as u8);
                        }
                    }
                }
            }
            CardType::Tarok => {
                for i in 1..trick.count as usize {
                    let (p, card) = trick.cards[i];
                    if p == player {
                        continue;
                    }
                    if card.card_type() != CardType::Tarok {
                        opp_tarok_void[p as usize] = true;
                    }
                }
            }
        }
    }

    let declarer_opt = state.declarer;
    for opp_offset in 1..NUM_PLAYERS {
        let opp_idx = ((player as usize + opp_offset) % NUM_PLAYERS) as u8;
        let is_declarer_col = declarer_opt == Some(opp_idx);
        let void_mask = opp_void[opp_idx as usize];
        let tarok_void = opp_tarok_void[opp_idx as usize];
        for cidx in 0..DECK_SIZE {
            let c = Card(cidx as u8);
            if known.contains(c) {
                continue;
            }
            // Forced retention pins the card onto the declarer column.
            if forced_retention.contains(c) {
                if is_declarer_col {
                    buf[o + cidx] = 1.0;
                }
                continue;
            }
            let impossible = match c.card_type() {
                CardType::Tarok => tarok_void,
                CardType::Suit => match c.suit() {
                    Some(s) => void_mask & (1 << (s as u8)) != 0,
                    None => false,
                },
            };
            if !impossible {
                buf[o + cidx] = 1.0 / 3.0;
            }
        }
        o += DECK_SIZE;
    }

    // --- Trick context features (6 dims) ---
    if let Some(ref trick) = state.current_trick {
        if trick.count > 0 {
            buf[o] = trick.count as f32 / 4.0;
            let lead_card = trick.cards[0].1;
            if lead_card.card_type() == CardType::Tarok {
                buf[o + 1] = 1.0;
            } else if let Some(s) = lead_card.suit() {
                buf[o + 2 + s as usize] = 1.0;
            }
        }
    }
    o += TRICK_CTX_SIZE;

    // ===================================================================
    // v5 PER-OPPONENT PLAYED PLANES (3 × 54 = 162 dims)
    // Preserves "who played what" identity that the global played plane
    // loses.  Enables the network to track per-opponent tarok depletion
    // and spot the partner after the called king falls.
    // ===================================================================
    let mut opp_played_planes: [CardSet; NUM_PLAYERS] = [CardSet::EMPTY; NUM_PLAYERS];
    for trick in &state.tricks {
        for i in 0..trick.count as usize {
            let (p, c) = trick.cards[i];
            opp_played_planes[p as usize].insert(c);
        }
    }
    if let Some(ref trick) = state.current_trick {
        for i in 0..trick.count as usize {
            let (p, c) = trick.cards[i];
            opp_played_planes[p as usize].insert(c);
        }
    }
    for opp_offset in 1..NUM_PLAYERS {
        let opp_idx = ((player as usize + opp_offset) % NUM_PLAYERS) as u8;
        for c in opp_played_planes[opp_idx as usize].iter() {
            buf[o + c.0 as usize] = 1.0;
        }
        o += DECK_SIZE;
    }

    // --- Per-opponent tarok-void flags (3 dims) ---
    for opp_offset in 1..NUM_PLAYERS {
        let opp_idx = ((player as usize + opp_offset) % NUM_PLAYERS) as u8;
        if opp_tarok_void[opp_idx as usize] {
            buf[o] = 1.0;
        }
        o += 1;
    }

    // --- Per-opponent suit-void flags (3 × 4 = 12 dims) ---
    for opp_offset in 1..NUM_PLAYERS {
        let opp_idx = ((player as usize + opp_offset) % NUM_PLAYERS) as u8;
        let mask = opp_void[opp_idx as usize];
        for s in 0u8..4u8 {
            if mask & (1 << s) != 0 {
                buf[o + s as usize] = 1.0;
            }
        }
        o += 4;
    }

    // --- Live kings one-hot (4 dims, hearts/diamonds/clubs/spades) ---
    // A king is "live" if it has not appeared in any trick yet.  Unpicked
    // talon already enters played_cards via the plane above, but kings
    // cannot legally be in the talon groups' picked-group-retirement
    // anyway.  We compute liveness directly from played_cards + the
    // active trick so that the feature is agnostic to the Pagat
    // disappearing into the talon in weird edge cases.
    let mut king_played_mask: u8 = 0;
    let mut tarok_played_mask: u8 = 0; // bit 0 = pagat, bit 1 = mond, bit 2 = skis
    let check_card = |c: Card, kpm: &mut u8, tpm: &mut u8| {
        if c.is_king() {
            if let Some(s) = c.suit() {
                *kpm |= 1 << (s as u8);
            }
        }
        if c.card_type() == CardType::Tarok {
            match c.0 as usize {
                0 => *tpm |= 1,
                20 => *tpm |= 2,
                21 => *tpm |= 4,
                _ => {}
            }
        }
    };
    for trick in &state.tricks {
        for i in 0..trick.count as usize {
            check_card(trick.cards[i].1, &mut king_played_mask, &mut tarok_played_mask);
        }
    }
    if let Some(ref trick) = state.current_trick {
        for i in 0..trick.count as usize {
            check_card(trick.cards[i].1, &mut king_played_mask, &mut tarok_played_mask);
        }
    }
    for s in 0u8..4u8 {
        if king_played_mask & (1 << s) == 0 {
            buf[o + s as usize] = 1.0;
        }
    }
    o += LIVE_KINGS_SIZE;

    // --- Live trula one-hot (3 dims: pagat / mond / skis) ---
    for bit in 0u8..3u8 {
        if tarok_played_mask & (1 << bit) == 0 {
            buf[o + bit as usize] = 1.0;
        }
    }
    o += LIVE_TRULA_SIZE;

    // --- Called-king suit one-hot (4 dims, public after king-call) ---
    if let Some(k) = state.called_king {
        if let Some(s) = k.suit() {
            buf[o + s as usize] = 1.0;
        }
    }
    o += CALLED_KING_SIZE;

    debug_assert_eq!(o, STATE_SIZE);
}

/// Compute the set of unpicked talon cards and the picked group (if any).
///
/// A revealed group is "picked" once all its cards have left `state.talon`
/// (they were moved into the declarer's hand and possibly discarded).  All
/// other revealed groups are unpicked and publicly retired.
fn compute_talon_visibility(state: &GameState) -> (CardSet, Option<Vec<Card>>) {
    let mut unpicked = CardSet::EMPTY;
    let mut picked: Option<Vec<Card>> = None;
    for group in &state.talon_revealed {
        let still_in_talon = group.iter().any(|c| state.talon.contains(*c));
        if still_in_talon {
            for &c in group {
                unpicked.insert(c);
            }
        } else if picked.is_none() {
            picked = Some(group.clone());
        }
    }
    (unpicked, picked)
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
