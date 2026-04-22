/// State encoding — writes features directly into a numpy buffer.
///
/// v8 encoding (585 dims).  All card planes are grouped at the start in
/// a card-attention-friendly layout — every plane starts on a 54-aligned
/// offset so slicing is trivial.
///
///  - 0..53     : own hand
///  - 54..107   : opponent +1 belief probabilities
///  - 108..161  : opponent +2 belief probabilities
///  - 162..215  : opponent +3 belief probabilities
///  - 216..269  : own played cards (cards this player has played)   ← v8
///  - 270..323  : opponent +1 played cards
///  - 324..377  : opponent +2 played cards
///  - 378..431  : opponent +3 played cards  (declarer's plane also
///                includes publicly-retired unpicked talon cards)
///  - 432..485  : active trick plane (cards in the current trick)
///
///  Scalar tail (starts at 486):
///  - 486..489  : seat position relative to dealer (4 one-hot)
///  - 490..499  : contract one-hot (10)
///  - 500..502  : phase one-hot (3: bidding / trick_play / other)
///  - 503       : tricks_played / 12
///  - 504..508  : decision type one-hot (5)
///  - 509..517  : highest bid one-hot (9)
///  - 518..521  : passed players (4, dealer-relative)
///  - 522..525  : own-team announcements (4)
///  - 526..529  : opponent-team announcements (4)
///  - 530..534  : kontra levels (5, normalized)
///  - 535..537  : role one-hot (3: declarer / partner / opposition)
///  - 538..541  : partner relative seat (4, all-zero ⇒ unknown)
///  - 542..544  : centaur team points (mine/70, opp/70, current/20)
///  - 545..548  : trick leader relative seat (4)
///  - 549..552  : trick currently-winning relative seat (4)
///  - 553..558  : trick context (position + lead type/suit) (6)
///  - 559..561  : per-opponent tarok-void flags (3)
///  - 562..573  : per-opponent suit-void flags (12)
///  - 574..577  : live kings one-hot (4)
///  - 578..580  : live trula one-hot (pagat / mond / skis) (3)
///  - 581..584  : called-king suit one-hot (4)
///
/// Rule semantics:
///  - `talon_revealed` is public (Slovenian Tarok).  Unpicked talon
///    cards are publicly retired — they appear in the belief `known` set
///    and in the declarer's per-opp played plane (for counting).
///  - Forced-retention: taroks and kings from the picked talon group
///    cannot legally be discarded, so they are publicly known to still
///    be in declarer's hand.  The belief block pins these cards onto
///    the declarer's column (prob 1.0; 0 for other opponent columns).
///  - Role reports the acting player's own role (known post-bidding).
///    Partner identity (which seat holds the called king) is a separate
///    feature: partner_rel, populated only for players who actually
///    know (partner themselves always; everyone else after the king has
///    been played).
use crate::card::*;
use crate::game_state::*;
use crate::trick_eval::evaluate_trick;

// --- v9 layout constants (all card planes are 54-aligned) ---
pub const HAND_OFFSET: usize = 0;
/// Belief feature block offset (3 × 54 starting here).
pub const BELIEF_OFFSET: usize = 54;
const BELIEF_SIZE: usize = 3 * DECK_SIZE;
/// Own played-cards plane (v8).
pub const SELF_PLAYED_OFFSET: usize = BELIEF_OFFSET + BELIEF_SIZE; // 216
/// Own discarded-cards plane (v9, private to the declarer).
pub const SELF_DISCARDED_OFFSET: usize = SELF_PLAYED_OFFSET + DECK_SIZE; // 270
/// Per-opponent played planes start here (3 × 54).
pub const OPP_PLAYED_OFFSET: usize = SELF_DISCARDED_OFFSET + DECK_SIZE; // 324
const OPP_PLAYED_SIZE: usize = 3 * DECK_SIZE;
/// Active trick plane (cards on the table right now).
pub const ACTIVE_TRICK_OFFSET: usize = OPP_PLAYED_OFFSET + OPP_PLAYED_SIZE; // 486
/// Total card-plane block = 10 × 54.
pub const CARD_PLANES_SIZE: usize = 10 * DECK_SIZE; // 540
/// First scalar offset.
pub const SCALAR_OFFSET: usize = CARD_PLANES_SIZE; // 540
/// Contract one-hot offset.
pub const CONTRACT_OFFSET: usize = SCALAR_OFFSET + 4; // 544 (after seat-rel)
/// Contract one-hot feature length.
pub const CONTRACT_SIZE: usize = 10;

// Scalar-tail constituent sizes (kept for documentation / debug asserts)
const SCALAR_TAIL_SIZE: usize =
    4 + 10 + 3 + 1 + 5 + 9 + 4 + 8 + 5 + 3 + 4 + 3 + 4 + 4 + 6 + 3 + 12 + 4 + 3 + 4; // 99

/// v9 full state size.
pub const STATE_SIZE: usize = CARD_PLANES_SIZE + SCALAR_TAIL_SIZE; // 540 + 99 = 639
pub const ORACLE_EXTRA: usize = 3 * DECK_SIZE; // 162
pub const ORACLE_STATE_SIZE: usize = STATE_SIZE + ORACLE_EXTRA; // 801

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

    // --- Compute v5 public-talon sets up front (used by several blocks) ---
    let (unpicked_talon_set, picked_group) = compute_talon_visibility(state);
    let mut forced_retention = CardSet::EMPTY;
    if let Some(ref picked) = picked_group {
        for &c in picked {
            if c.card_type() == CardType::Tarok || c.is_king() {
                forced_retention.insert(c);
            }
        }
    }

    let hand = state.hands[player as usize];
    let my_team = state.get_team(player);
    let contract_opt = state.contract;

    // --- Card plane 0: own hand ---
    {
        let base = HAND_OFFSET;
        for c in hand.iter() {
            buf[base + c.0 as usize] = 1.0;
        }
    }

    // --- Running team card-point totals (for centaur block) ---
    let mut my_team_points: i32 = 0;
    let mut opp_team_points: i32 = 0;
    for (i, trick) in state.tricks.iter().enumerate() {
        let is_last = i == state.tricks.len() - 1;
        let tr = evaluate_trick(trick, is_last, contract_opt);
        let trick_cards_points: i32 = (0..trick.count as usize)
            .map(|j| trick.cards[j].1.points() as i32)
            .sum();
        if state.get_team(tr.winner) == my_team {
            my_team_points += trick_cards_points;
        } else {
            opp_team_points += trick_cards_points;
        }
    }

    // --- Belief block: cards known to be out of every opponent's hand ---
    let mut known = hand;
    known = known.union(state.played_cards);
    known = known.union(unpicked_talon_set);
    if let Some(ref trick) = state.current_trick {
        known = known.union(trick.played_cards_set());
    }

    // Suit-void and tarok-void inference from trick history.
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

    // --- Card planes 1..3: opponent belief probabilities ---
    let declarer_opt = state.declarer;
    for opp_offset in 1..NUM_PLAYERS {
        let opp_idx = ((player as usize + opp_offset) % NUM_PLAYERS) as u8;
        let base = BELIEF_OFFSET + (opp_offset - 1) * DECK_SIZE;
        let is_declarer_col = declarer_opt == Some(opp_idx);
        let void_mask = opp_void[opp_idx as usize];
        let tarok_void = opp_tarok_void[opp_idx as usize];
        for cidx in 0..DECK_SIZE {
            let c = Card(cidx as u8);
            if known.contains(c) {
                continue;
            }
            if forced_retention.contains(c) {
                if is_declarer_col {
                    buf[base + cidx] = 1.0;
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
                buf[base + cidx] = 1.0 / 3.0;
            }
        }
    }

    // --- Per-player played planes (self + 3 opponents) ---
    let mut played_by_seat: [CardSet; NUM_PLAYERS] = [CardSet::EMPTY; NUM_PLAYERS];
    for trick in &state.tricks {
        for i in 0..trick.count as usize {
            let (p, c) = trick.cards[i];
            played_by_seat[p as usize].insert(c);
        }
    }
    if let Some(ref trick) = state.current_trick {
        for i in 0..trick.count as usize {
            let (p, c) = trick.cards[i];
            played_by_seat[p as usize].insert(c);
        }
    }
    // Attribute unpicked-talon cards to declarer's plane (publicly retired).
    if let Some(decl) = state.declarer {
        for c in unpicked_talon_set.iter() {
            played_by_seat[decl as usize].insert(c);
        }
    }

    // Card plane 4: own played cards.
    {
        let base = SELF_PLAYED_OFFSET;
        for c in played_by_seat[player as usize].iter() {
            buf[base + c.0 as usize] = 1.0;
        }
    }
    // Card plane 5: own discarded cards (v9 — private to the declarer).
    if state.declarer == Some(player) {
        let base = SELF_DISCARDED_OFFSET;
        for c in state.put_down.iter() {
            buf[base + c.0 as usize] = 1.0;
        }
    }
    // Card planes 6..8: per-opponent played.
    for opp_offset in 1..NUM_PLAYERS {
        let opp_idx = ((player as usize + opp_offset) % NUM_PLAYERS) as u8;
        let base = OPP_PLAYED_OFFSET + (opp_offset - 1) * DECK_SIZE;
        for c in played_by_seat[opp_idx as usize].iter() {
            buf[base + c.0 as usize] = 1.0;
        }
    }

    // Card plane 8: active trick.
    if let Some(ref trick) = state.current_trick {
        let base = ACTIVE_TRICK_OFFSET;
        for i in 0..trick.count as usize {
            buf[base + trick.cards[i].1 .0 as usize] = 1.0;
        }
    }

    // ============= SCALAR TAIL (starts at SCALAR_OFFSET = 486) =============
    let mut o = SCALAR_OFFSET;

    // Seat position relative to dealer (4 one-hot).
    let rel_pos = ((player as usize + NUM_PLAYERS - state.dealer as usize) % NUM_PLAYERS) as usize;
    buf[o + rel_pos] = 1.0;
    o += 4;

    // Contract one-hot (10).
    debug_assert_eq!(o, CONTRACT_OFFSET);
    if let Some(c) = state.contract {
        let idx = c as usize;
        if idx < 10 {
            buf[o + idx] = 1.0;
        }
    }
    o += CONTRACT_SIZE;

    // Phase one-hot (3).
    match state.phase {
        Phase::Bidding => buf[o] = 1.0,
        Phase::TrickPlay => buf[o + 1] = 1.0,
        _ => buf[o + 2] = 1.0,
    }
    o += 3;

    // Tricks played / 12.
    buf[o] = state.tricks_played() as f32 / 12.0;
    o += 1;

    // Decision type (5 one-hot).
    if (decision_type as usize) < 5 {
        buf[o + decision_type as usize] = 1.0;
    }
    o += 5;

    // Highest bid (9 one-hot: no_bid + 8 contracts).
    let highest_bid = state
        .bids
        .iter()
        .filter_map(|b| b.contract)
        .max_by_key(|c| c.strength());
    match highest_bid {
        Some(c) => {
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
        None => buf[o] = 1.0,
    }
    o += 9;

    // Passed players (4 binary, dealer-relative order).
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

    // Announcements split by team (4 own + 4 opp).
    if state.declarer.is_some() {
        let mut own_ann: u8 = 0;
        let mut opp_ann: u8 = 0;
        for seat in 0..NUM_PLAYERS as u8 {
            let a = state.announcements[seat as usize];
            if state.get_team(seat) == my_team {
                own_ann |= a;
            } else {
                opp_ann |= a;
            }
        }
        for i in 0..4u8 {
            if own_ann & (1 << i) != 0 {
                buf[o + i as usize] = 1.0;
            }
            if opp_ann & (1 << i) != 0 {
                buf[o + 4 + i as usize] = 1.0;
            }
        }
    }
    o += 8;

    // Kontra levels (5 normalized).
    for i in 0..KontraTarget::NUM {
        let level = state.kontra_levels[i];
        buf[o + i] = (level.multiplier() as f32 - 1.0) / 7.0;
    }
    o += 5;

    // Role one-hot (3).
    if let Some(decl) = state.declarer {
        if player == decl {
            buf[o] = 1.0;
        } else if state.partner == Some(player) {
            buf[o + 1] = 1.0;
        } else if state.get_team(player) == Team::DeclarerTeam {
            buf[o + 1] = 1.0;
        } else {
            buf[o + 2] = 1.0;
        }
    }
    o += 3;

    // Partner relative seat (4; all-zero ⇒ unknown).
    if let Some(part) = state.partner {
        let known_partner = player == part
            || state.is_partner_revealed()
            || state.declarer == Some(part);
        if known_partner {
            let rel = ((part as usize + NUM_PLAYERS - player as usize) % NUM_PLAYERS) as usize;
            buf[o + rel] = 1.0;
        }
    }
    o += 4;

    // --- Centaur trick context: current trick summary ---
    let mut current_trick_points: i32 = 0;
    let mut current_leader: Option<u8> = None;
    let mut current_winner: Option<u8> = None;
    if let Some(ref trick) = state.current_trick {
        if trick.count > 0 {
            current_leader = Some(trick.cards[0].0);
            let lead_suit = {
                let lc = trick.cards[0].1;
                if lc.card_type() == CardType::Suit {
                    lc.suit()
                } else {
                    None
                }
            };
            let (mut best_p, mut best_c) = trick.cards[0];
            for j in 1..trick.count as usize {
                let (p, c) = trick.cards[j];
                if c.beats(best_c, lead_suit) {
                    best_p = p;
                    best_c = c;
                }
            }
            current_winner = Some(best_p);
            for j in 0..trick.count as usize {
                current_trick_points += trick.cards[j].1.points() as i32;
            }
        }
    }
    buf[o] = my_team_points as f32 / 70.0;
    buf[o + 1] = opp_team_points as f32 / 70.0;
    buf[o + 2] = current_trick_points as f32 / 20.0;
    o += 3;

    // Trick leader relative seat (4).
    if let Some(lead) = current_leader {
        let rel = ((lead as usize + NUM_PLAYERS - player as usize) % NUM_PLAYERS) as usize;
        buf[o + rel] = 1.0;
    }
    o += 4;

    // Trick currently-winning seat (4).
    if let Some(win) = current_winner {
        let rel = ((win as usize + NUM_PLAYERS - player as usize) % NUM_PLAYERS) as usize;
        buf[o + rel] = 1.0;
    }
    o += 4;

    // Trick context: position + lead type/suit (6).
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
    o += 6;

    // Per-opponent tarok-void flags (3).
    for opp_offset in 1..NUM_PLAYERS {
        let opp_idx = ((player as usize + opp_offset) % NUM_PLAYERS) as u8;
        if opp_tarok_void[opp_idx as usize] {
            buf[o] = 1.0;
        }
        o += 1;
    }

    // Per-opponent suit-void flags (12).
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

    // Live kings & live trula from trick + current-trick cards.
    let mut king_played_mask: u8 = 0;
    let mut tarok_played_mask: u8 = 0;
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
    o += 4;
    for bit in 0u8..3u8 {
        if tarok_played_mask & (1 << bit) == 0 {
            buf[o + bit as usize] = 1.0;
        }
    }
    o += 3;

    // Called-king suit one-hot (4).
    if let Some(k) = state.called_king {
        if let Some(s) = k.suit() {
            buf[o + s as usize] = 1.0;
        }
    }
    o += 4;

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
