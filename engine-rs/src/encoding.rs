/// State encoding — writes features directly into a numpy buffer.
///
/// v10 encoding (585 dims).  All card planes are grouped at the start in
/// a card-attention-friendly layout — every plane starts on a 54-aligned
/// offset so slicing is trivial.
///
///  - 0..53     : own hand
///  - 54..107   : opponent +1 belief marginals (cardinality-aware IPF)
///  - 108..161  : opponent +2 belief marginals
///  - 162..215  : opponent +3 belief marginals
///  - 216..269  : own played cards
///  - 270..323  : opponent +1 played cards
///  - 324..377  : opponent +2 played cards
///  - 378..431  : opponent +3 played cards  (declarer's plane also
///                includes publicly-retired unpicked talon cards)
///  - 432..485  : active trick plane (cards in the current trick)
///
///  Scalar tail (starts at 486, 99 dims):
///  - 486..489  : seat position relative to dealer (4 one-hot)
///  - 490..499  : contract one-hot (10)
///  - 500..502  : phase one-hot (3: bidding / trick_play / other)
///  - 503       : tricks_played / 12
///  - 504..508  : decision type one-hot (5)
///  - 509..517  : highest bid one-hot (9)
///  - 518..521  : passed players (4, dealer-relative)
///  - 522..526  : own-team announcements (5: trula, kings, pagat, king, valat)
///  - 527..531  : opponent-team announcements (5)
///  - 532..536  : kontra levels (5, normalized)
///  - 537..539  : role one-hot (3: declarer / partner / opposition)
///  - 540..543  : partner relative seat (4, all-zero ⇒ unknown)
///  - 544..546  : centaur team points (mine/70, opp/70, current/20)
///  - 547..550  : trick leader relative seat (4)
///  - 551..554  : trick currently-winning relative seat (4)
///  - 555..560  : trick context (position + lead type/suit) (6)
///  - 561..563  : per-opponent tarok-void flags (3)
///  - 564..575  : per-opponent suit-void flags (12)
///  - 576..579  : called-king suit one-hot (4)
///  - 580..584  : remaining-in-play counts (taroks/22, H/8, D/8, C/8, S/8)
///
/// v10 vs v9 changes (see docs/state_space_and_observation.md):
///  - Removed own-discarded plane (54 dims) — folded into belief `known`
///    set for the declarer's perspective.
///  - Removed live-kings (4) and live-trula (3) scalar blocks —
///    derivable from played planes.
///  - Added King Ultimo announcement slot (+2, own + opp).
///  - Added remaining-in-play counts block (+5).
///  - Belief planes use cardinality-aware IPF (Sinkhorn) normalisation
///    instead of a flat 1/3 prior.
///  - New tarok-void inference: suit-void opponent playing a non-tarok
///    non-lead-suit card is also tarok-void.
use crate::card::*;
use crate::game_state::*;
use crate::trick_eval::evaluate_trick;

// --- v10 layout constants (all card planes are 54-aligned) ---
pub const HAND_OFFSET: usize = 0;
/// Belief feature block offset (3 × 54 starting here).
pub const BELIEF_OFFSET: usize = 54;
const BELIEF_SIZE: usize = 3 * DECK_SIZE;
/// Own played-cards plane.
pub const SELF_PLAYED_OFFSET: usize = BELIEF_OFFSET + BELIEF_SIZE; // 216
/// Per-opponent played planes start here (3 × 54).
pub const OPP_PLAYED_OFFSET: usize = SELF_PLAYED_OFFSET + DECK_SIZE; // 270
const OPP_PLAYED_SIZE: usize = 3 * DECK_SIZE;
/// Active trick plane (cards on the table right now).
pub const ACTIVE_TRICK_OFFSET: usize = OPP_PLAYED_OFFSET + OPP_PLAYED_SIZE; // 432
/// Total card-plane block = 9 × 54.
pub const CARD_PLANES_SIZE: usize = 9 * DECK_SIZE; // 486
/// First scalar offset.
pub const SCALAR_OFFSET: usize = CARD_PLANES_SIZE; // 486
/// Contract one-hot offset.
pub const CONTRACT_OFFSET: usize = SCALAR_OFFSET + 4; // 490 (after seat-rel)
/// Contract one-hot feature length.
pub const CONTRACT_SIZE: usize = 10;

// Scalar-tail constituent sizes (kept for documentation / debug asserts).
// v10 tail: 4 + 10 + 3 + 1 + 5 + 9 + 4 + 10 + 5 + 3 + 4 + 3 + 4 + 4 + 6 + 3 + 12 + 4 + 5 = 99
const SCALAR_TAIL_SIZE: usize =
    4 + 10 + 3 + 1 + 5 + 9 + 4 + 10 + 5 + 3 + 4 + 3 + 4 + 4 + 6 + 3 + 12 + 4 + 5;

/// v10 full state size.
pub const STATE_SIZE: usize = CARD_PLANES_SIZE + SCALAR_TAIL_SIZE; // 486 + 99 = 585
pub const ORACLE_EXTRA: usize = 3 * DECK_SIZE; // 162
pub const ORACLE_STATE_SIZE: usize = STATE_SIZE + ORACLE_EXTRA; // 747

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
    // v10: declarer knows their own put_down — fold into the belief
    // `known` set so every opponent column is forced to 0 for those cards.
    // Only applied when the acting player IS the declarer; for other
    // seats `put_down` is private and must not leak.
    if state.declarer == Some(player) {
        known = known.union(state.put_down);
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
                                // Suit-void in lead suit AND did not play
                                // tarok — so at that moment tarok-void
                                // too (forced-follow rule). Tarok-void
                                // is monotone in time ⇒ still holds.
                                // (v10 addition.)
                                opp_void[p as usize] |= 1 << (ls as u8);
                                opp_tarok_void[p as usize] = true;
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

    // --- Card planes 1..3: opponent belief marginals (cardinality-aware IPF) ---
    //
    // Build a 3 × |U| matrix of feasibility weights, where U is the set
    // of cards whose location is unknown to the acting player.  Then
    // rescale rows (to opp remaining-slot counts) and columns (to 1.0
    // per card) iteratively until both marginals converge (Sinkhorn).
    //
    // Pre-seed rules:
    //   * pinned absent (void / declarer-pinned-elsewhere) → 0
    //   * forced-retention card on declarer column → 1 (pinned present)
    //   * otherwise → 1 (feasible, uniform starting weight)
    //
    // Row targets: hand_size_i − cards already pinned to opp i.
    // Column targets: 1 − pinned column mass (usually 1; 0 for pinned cards).
    let declarer_opt = state.declarer;

    // Compute each opponent's remaining hand size (free + pinned slots).
    let mut opp_hand_size: [usize; NUM_PLAYERS] = [0; NUM_PLAYERS];
    for seat in 0..NUM_PLAYERS {
        opp_hand_size[seat] = state.hands[seat].len() as usize;
    }

    // Collect unknown card indices.
    let mut unknown: Vec<usize> = Vec::with_capacity(DECK_SIZE);
    for cidx in 0..DECK_SIZE {
        if !known.contains(Card(cidx as u8)) {
            unknown.push(cidx);
        }
    }

    // Map relative opponent offset (0..=2) ↔ absolute seat.
    let opp_seats: [u8; 3] = [
        ((player as usize + 1) % NUM_PLAYERS) as u8,
        ((player as usize + 2) % NUM_PLAYERS) as u8,
        ((player as usize + 3) % NUM_PLAYERS) as u8,
    ];

    // Matrix M[row][col] where row = relative opponent index, col = pos in `unknown`.
    let mut m: Vec<[f32; 3]> = vec![[0.0; 3]; unknown.len()];
    // Free-slot row targets (hand size minus pinned-present count for that row).
    let mut row_target: [f32; 3] = [0.0; 3];
    for r in 0..3 {
        row_target[r] = opp_hand_size[opp_seats[r] as usize] as f32;
    }
    let mut col_pinned: Vec<bool> = vec![false; unknown.len()];

    for (col, &cidx) in unknown.iter().enumerate() {
        let c = Card(cidx as u8);
        // Forced retention: pin to declarer's relative column.
        if forced_retention.contains(c) {
            for r in 0..3 {
                if declarer_opt == Some(opp_seats[r]) {
                    m[col][r] = 1.0;
                    row_target[r] -= 1.0;
                }
            }
            col_pinned[col] = true;
            continue;
        }
        // Feasibility per row (void constraints).
        for r in 0..3 {
            let seat = opp_seats[r] as usize;
            let feasible = match c.card_type() {
                CardType::Tarok => !opp_tarok_void[seat],
                CardType::Suit => match c.suit() {
                    Some(s) => opp_void[seat] & (1 << (s as u8)) == 0,
                    None => true,
                },
            };
            if feasible {
                m[col][r] = 1.0;
            }
        }
    }

    // Iterative proportional fitting (Sinkhorn).  Fixed iterations —
    // deterministic; ~10 iters is ample for 3×|U| matrices this small.
    const IPF_ITERS: usize = 12;
    for _ in 0..IPF_ITERS {
        // Row scaling: each opponent's free cells should sum to their free slots.
        let mut row_sum: [f32; 3] = [0.0; 3];
        for col in 0..unknown.len() {
            if col_pinned[col] {
                continue;
            }
            for r in 0..3 {
                row_sum[r] += m[col][r];
            }
        }
        for r in 0..3 {
            if row_sum[r] > 1e-12 && row_target[r] > 0.0 {
                let s = row_target[r] / row_sum[r];
                for col in 0..unknown.len() {
                    if col_pinned[col] {
                        continue;
                    }
                    m[col][r] *= s;
                }
            } else if row_target[r] <= 0.0 {
                // No free slots — zero this row's free cells.
                for col in 0..unknown.len() {
                    if col_pinned[col] {
                        continue;
                    }
                    m[col][r] = 0.0;
                }
            }
        }
        // Column scaling: each unknown card has total probability 1.
        for col in 0..unknown.len() {
            if col_pinned[col] {
                continue;
            }
            let cs: f32 = m[col][0] + m[col][1] + m[col][2];
            if cs > 1e-12 {
                let s = 1.0 / cs;
                for r in 0..3 {
                    m[col][r] *= s;
                }
            }
            // If cs == 0 here, the column is infeasible under our
            // constraints; leave zeros (belief = 0 for all opponents).
            // The declarer-pinned case cannot hit this branch because
            // `col_pinned` is true and we skipped above.
        }
    }

    // Write belief planes.
    for r in 0..3 {
        let base = BELIEF_OFFSET + r * DECK_SIZE;
        for (col, &cidx) in unknown.iter().enumerate() {
            let v = m[col][r];
            if v > 0.0 {
                buf[base + cidx] = v;
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
    // (v10: no separate own-discarded plane; declarer's put_down is
    // folded into the belief `known` set above.)
    // Card planes 5..7: per-opponent played.
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

    // Announcements split by team (5 own + 5 opp): trula, kings, pagat,
    // king-ultimo, valat (v10).
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
        for i in 0..5u8 {
            if own_ann & (1 << i) != 0 {
                buf[o + i as usize] = 1.0;
            }
            if opp_ann & (1 << i) != 0 {
                buf[o + 5 + i as usize] = 1.0;
            }
        }
    }
    o += 10;

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

    // v10: live-kings and live-trula scalar blocks removed — fully
    // derivable from played planes via attention.  Instead we count
    // taroks and per-suit cards already played and emit their
    // remaining-in-play counts as normalised scalars (5 dims).
    let mut taroks_played: u32 = 0;
    let mut suit_played: [u32; 4] = [0; 4];
    let mut tally = |c: Card| {
        match c.card_type() {
            CardType::Tarok => taroks_played += 1,
            CardType::Suit => {
                if let Some(s) = c.suit() {
                    suit_played[s as usize] += 1;
                }
            }
        }
    };
    for trick in &state.tricks {
        for i in 0..trick.count as usize {
            tally(trick.cards[i].1);
        }
    }
    if let Some(ref trick) = state.current_trick {
        for i in 0..trick.count as usize {
            tally(trick.cards[i].1);
        }
    }

    // Called-king suit one-hot (4).
    if let Some(k) = state.called_king {
        if let Some(s) = k.suit() {
            buf[o + s as usize] = 1.0;
        }
    }
    o += 4;

    // Remaining-in-play counts (5): taroks/22, H/8, D/8, C/8, S/8.
    buf[o] = (22u32.saturating_sub(taroks_played)) as f32 / 22.0;
    for s in 0..4 {
        buf[o + 1 + s] = (8u32.saturating_sub(suit_played[s])) as f32 / 8.0;
    }
    o += 5;

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
