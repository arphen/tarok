/// State encoding — writes features directly into a numpy buffer.
///
/// v7 encoding (531 dims):
///  - Header card planes (108 dims):
///      * 0..53   : own hand
///      * 54..107 : active trick plane (cards in the current trick)
///    (v6's forced-retention plane is dropped — those cards are already
///     pinned onto the declarer column of the belief block.)
///  - v1 base features tail (56 dims): position, contract, phase,
///    tricks_played, decision_type, highest_bid, passed_players,
///    announcements (own-team + opp-team), kontra, role, partner_rel.
///  - Centaur trick context (11 dims):
///      * team points: mine/70, opp/70, current_trick/20
///      * trick leader relative seat one-hot (4)
///      * trick currently-winning relative seat one-hot (4)
///  - Belief block 175..336 (162 dims): 3×54 opponent card likelihoods
///    with suit-void, tarok-void, unpicked-talon-retirement, and
///    forced-retention (declarer must-hold) constraints applied.
///  - Trick context 337..342 (6 dims): position + lead suit/type.
///  - Per-opponent played planes 343..504 (162 dims): 3×54 cards played.
///    Declarer's plane additionally includes publicly-retired unpicked
///    talon cards so the network can count remaining suits easily.
///  - Per-opponent tarok-void flags 505..507 (3 dims).
///  - Per-opponent suit-void flags 508..519 (12 dims).
///  - Live kings one-hot 520..523 (4 dims).
///  - Live trula one-hot 524..526 (3 dims): pagat / mond / skis.
///  - Called-king suit one-hot 527..530 (4 dims).
///
/// Rule semantics:
///  - `talon_revealed` is public (Slovenian Tarok).  Unpicked talon cards
///    are publicly retired — they appear in the belief `known` set and in
///    the declarer's per-opp played plane (for counting purposes).
///  - Forced-retention: taroks and kings from the picked talon group
///    cannot legally be discarded, so they are publicly known to still
///    be in declarer's hand.  The belief block pins these cards onto the
///    declarer's column (prob 1.0; 0 for other opponent columns).
///  - Role one-hot (208..210 of v6 equivalent) reports the acting
///    player's own role (known post-bidding).  Partner identity (which
///    seat holds the called king) is a separate feature: partner_rel,
///    populated only for players who actually know (partner themselves
///    always; everyone else after the king has been played).
use crate::card::*;
use crate::game_state::*;
use crate::trick_eval::evaluate_trick;

/// v1 base features size (header planes + base scalars + centaur block).
/// Belief block always starts immediately after this.
pub const V1_STATE_SIZE: usize = 175;
/// Contract one-hot feature offset in the flat state vector.
pub const CONTRACT_OFFSET: usize = 112;
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
/// v7 full state size.
pub const STATE_SIZE: usize = V1_STATE_SIZE
    + BELIEF_SIZE
    + TRICK_CTX_SIZE
    + OPP_PLAYED_SIZE
    + TAROK_VOID_SIZE
    + SUIT_VOID_SIZE
    + LIVE_KINGS_SIZE
    + LIVE_TRULA_SIZE
    + CALLED_KING_SIZE; // 531
pub const ORACLE_EXTRA: usize = 3 * DECK_SIZE; // 162
pub const ORACLE_STATE_SIZE: usize = STATE_SIZE + ORACLE_EXTRA; // 693

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

    // Current trick cards (54 binary).  Placed here (rather than after a
    // redundant global "played" plane) so the network sees the live table
    // state adjacent to its own hand.
    if let Some(ref trick) = state.current_trick {
        for i in 0..trick.count as usize {
            buf[o + trick.cards[i].1 .0 as usize] = 1.0;
        }
    }
    o += DECK_SIZE;

    // (v7: the forced-retention plane is gone — those cards are pinned
    //  onto the declarer's column of the belief block below, which is
    //  all the information it ever carried.)

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

    // Running team card-point totals (used by centaur block) +
    // tricks_played.  Team tricks-won count is omitted in v7 — it's
    // derivable and doesn't drive decisions.
    let my_team = state.get_team(player);
    let contract_opt = state.contract;
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

    // Announcements split by team (4 + 4 = 8 binary).  Own-team first,
    // opposition second.  Each 4-bit slot is {trula, kings, pagat, valat}.
    // During bidding the team partition is not yet defined — both slots
    // stay zero until declarer is known.
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

    // Kontra levels (5 features, normalized)
    for i in 0..KontraTarget::NUM {
        let level = state.kontra_levels[i];
        buf[o + i] = (level.multiplier() as f32 - 1.0) / 7.0;
    }
    o += 5;

    // Role one-hot (3 features: is_declarer, is_partner, is_opposition).
    // The acting player always knows their own role once bidding ends
    // (declarer is public; holding the called king ⇒ partner; else
    // opposition).  All-zero ⇒ bidding still in progress.
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
    o += 3;

    // Partner relative seat (4 one-hot; 0 = self is partner).  This
    // identifies *which seat* holds the called king.  Set only for
    // players who actually know:
    //   * the partner themselves (always — they hold the king),
    //   * any seat once the king has been publicly played,
    //   * a self-called-king declarer (partner == declarer).
    // Declarer cannot otherwise distinguish which opponent is the
    // partner until the king falls, so this stays all-zero for them.
    if let Some(part) = state.partner {
        let known = player == part
            || state.is_partner_revealed()
            || state.declarer == Some(part);
        if known {
            let rel = ((part as usize + NUM_PLAYERS - player as usize) % NUM_PLAYERS) as usize;
            buf[o + rel] = 1.0;
        }
    }
    o += 4;

    // --- v6 Centaur trick context (11 dims) ---
    // High-signal features for the early/mid-game (the centaur hands off
    // to PIMC before trick 9, so these help exactly the regime the NN is
    // asked to play in).

    // Team points running totals + current-trick value.
    let mut current_trick_points: i32 = 0;
    let mut current_leader: Option<u8> = None;
    let mut current_winner: Option<u8> = None;
    if let Some(ref trick) = state.current_trick {
        if trick.count > 0 {
            current_leader = Some(trick.cards[0].0);
            // "Currently winning" = best card under standard trick rules
            // over the populated slots.
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

    // Trick leader relative seat (4 one-hot, 0 = self, 1..3 = opponents).
    if let Some(lead) = current_leader {
        let rel = ((lead as usize + NUM_PLAYERS - player as usize) % NUM_PLAYERS) as usize;
        buf[o + rel] = 1.0;
    }
    o += 4;

    // Trick currently-winning seat (4 one-hot, relative).
    if let Some(win) = current_winner {
        let rel = ((win as usize + NUM_PLAYERS - player as usize) % NUM_PLAYERS) as usize;
        buf[o + rel] = 1.0;
    }
    o += 4;

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
    // v6 PER-OPPONENT PLAYED PLANES (3 × 54 = 162 dims)
    // Preserves "who played what" identity that a global played plane
    // would lose.  The declarer's plane additionally includes publicly-
    // retired unpicked talon cards — this lets the network count
    // remaining suits/taroks without a dedicated global played plane.
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
    // Attribute unpicked-talon cards to declarer's plane.  They are
    // publicly retired so encoding them as "declarer played" gives the
    // counting mechanism the right set without adding a redundant
    // global plane.  If no declarer yet (bidding), fall back to all
    // planes (unlikely but cheap).
    if let Some(decl) = state.declarer {
        for c in unpicked_talon_set.iter() {
            opp_played_planes[decl as usize].insert(c);
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
