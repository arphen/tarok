/// 3-player Tarok ("tarok v treh") state encoding — v2 layout.
///
/// Mirrors the v10 4-player encoding ([`crate::encoding`]) feature-for-feature
/// where it makes sense, with these 3p adaptations:
///
///   * 3 seats instead of 4 (only 2 opponent planes / belief / void slots).
///   * No partner role: `partner_relative_seat` is replaced by a
///     `declarer_relative_seat` 3-dim one-hot. Combined with the acting
///     player's own role, every seat's role is fully determined.
///   * No king-calling: `called_king` one-hot block is dropped, and the
///     KingUltimo announcement bit (state `Announcement::KingUltimo`) is
///     intentionally not surfaced (the slot in the per-team announcement
///     blocks is reused for `BarvniValat` to keep 5 ann slots).
///   * Belief planes use the same cardinality-aware Sinkhorn IPF as v10,
///     pre-seeded with suit-void / tarok-void inference, declarer
///     `put_down` (when the acting player IS the declarer), and
///     forced-retention pins on the picked talon group.
///
/// Layout (513 dims total):
///
/// Card planes (8 × 54 = 432):
///   - 0..54     : own hand
///   - 54..108   : opp +1 belief marginal
///   - 108..162  : opp +2 belief marginal
///   - 162..216  : own played cards
///   - 216..270  : opp +1 played cards
///   - 270..324  : opp +2 played cards (declarer plane also includes
///                 publicly-retired unpicked talon cards)
///   - 324..378  : active trick
///   - 378..432  : visible (publicly-revealed) talon cards
///
/// Scalar tail (81 dims, starts at 432):
///   - 432..435  : seat position relative to dealer (3)
///   - 435..442  : contract one-hot (7)
///   - 442..445  : phase one-hot (3: bidding / trick_play / other)
///   - 445       : tricks_played / 16
///   - 446..451  : decision type one-hot (5; KING_CALL slot unused)
///   - 451..459  : highest bid one-hot (8 = no_bid + 7 contracts)
///   - 459..462  : passed players (3, dealer-relative)
///   - 462..467  : own-team announcements (5)
///   - 467..472  : opp-team announcements (5)
///   - 472..477  : kontra levels (5, normalized)
///   - 477..480  : own-role one-hot (3: declarer / partner-unused / opposition)
///   - 480..483  : declarer relative seat (3)
///   - 483..486  : centaur trick context (mine/70, opp/70, current/20)
///   - 486..489  : trick leader relative seat (3)
///   - 489..492  : trick currently-winning relative seat (3)
///   - 492..498  : trick context (count/3, lead-tarok flag, 4 lead-suit one-hot)
///   - 498..500  : per-opponent tarok-void flags (2)
///   - 500..508  : per-opponent suit-void flags (2 × 4)
///   - 508..513  : remaining-in-play counts (taroks/22, H/8, D/8, C/8, S/8)
///
/// Oracle extension (+108 = 2 × 54):
///   - 513..567  : opp +1 perfect hand
///   - 567..621  : opp +2 perfect hand
///
/// **Variant invariant**: `encode_state_3p` panics if `state.variant !=
/// Variant::ThreePlayer`.
use crate::card::*;
use crate::game_state::*;
use crate::trick_eval::evaluate_trick;

// --- Card plane offsets ---
pub const HAND_OFFSET_3P: usize = 0;
pub const BELIEF_OFFSET_3P: usize = HAND_OFFSET_3P + DECK_SIZE; // 54
const BELIEF_SIZE_3P: usize = 2 * DECK_SIZE; // 108
pub const SELF_PLAYED_OFFSET_3P: usize = BELIEF_OFFSET_3P + BELIEF_SIZE_3P; // 162
pub const OPP_PLAYED_OFFSET_3P: usize = SELF_PLAYED_OFFSET_3P + DECK_SIZE; // 216
const OPP_PLAYED_SIZE_3P: usize = 2 * DECK_SIZE; // 108
pub const ACTIVE_TRICK_OFFSET_3P: usize = OPP_PLAYED_OFFSET_3P + OPP_PLAYED_SIZE_3P; // 324
pub const TALON_VIS_OFFSET_3P: usize = ACTIVE_TRICK_OFFSET_3P + DECK_SIZE; // 378
pub const CARD_PLANES_SIZE_3P: usize = 8 * DECK_SIZE; // 432

// Back-compat: opp+1 / opp+2 individual played-plane offsets.
pub const OPP1_PLAYED_OFFSET_3P: usize = OPP_PLAYED_OFFSET_3P; // 216
pub const OPP2_PLAYED_OFFSET_3P: usize = OPP_PLAYED_OFFSET_3P + DECK_SIZE; // 270

// --- Scalar tail offsets ---
pub const SCALAR_OFFSET_3P: usize = CARD_PLANES_SIZE_3P; // 432
pub const SEAT_REL_OFFSET_3P: usize = SCALAR_OFFSET_3P; // 432
pub const CONTRACT_OFFSET_3P: usize = SEAT_REL_OFFSET_3P + 3; // 435
pub const CONTRACT_SIZE_3P: usize = 7;
const PHASE_OFFSET_3P: usize = CONTRACT_OFFSET_3P + CONTRACT_SIZE_3P; // 442
const TRICKS_PLAYED_OFFSET_3P: usize = PHASE_OFFSET_3P + 3; // 445
const DECISION_TYPE_OFFSET_3P: usize = TRICKS_PLAYED_OFFSET_3P + 1; // 446
const HIGHEST_BID_OFFSET_3P: usize = DECISION_TYPE_OFFSET_3P + 5; // 451
const HIGHEST_BID_SIZE_3P: usize = 1 + CONTRACT_SIZE_3P; // 8: no_bid + contracts
const PASSED_OFFSET_3P: usize = HIGHEST_BID_OFFSET_3P + HIGHEST_BID_SIZE_3P; // 459
const OWN_ANN_OFFSET_3P: usize = PASSED_OFFSET_3P + 3; // 462
const OPP_ANN_OFFSET_3P: usize = OWN_ANN_OFFSET_3P + 5; // 467
const KONTRA_OFFSET_3P: usize = OPP_ANN_OFFSET_3P + 5; // 472
const ROLE_OFFSET_3P: usize = KONTRA_OFFSET_3P + 5; // 477
const DECLARER_REL_OFFSET_3P: usize = ROLE_OFFSET_3P + 3; // 480
const CENTAUR_OFFSET_3P: usize = DECLARER_REL_OFFSET_3P + 3; // 483
const TRICK_LEADER_OFFSET_3P: usize = CENTAUR_OFFSET_3P + 3; // 486
const TRICK_WINNING_OFFSET_3P: usize = TRICK_LEADER_OFFSET_3P + 3; // 489
const TRICK_CTX_OFFSET_3P: usize = TRICK_WINNING_OFFSET_3P + 3; // 492
const TAROK_VOID_OFFSET_3P: usize = TRICK_CTX_OFFSET_3P + 6; // 498
const SUIT_VOID_OFFSET_3P: usize = TAROK_VOID_OFFSET_3P + 2; // 500
const REMAINING_OFFSET_3P: usize = SUIT_VOID_OFFSET_3P + 8; // 508

const SCALAR_TAIL_SIZE_3P: usize =
    3 + 7 + 3 + 1 + 5 + 8 + 3 + 5 + 5 + 5 + 3 + 3 + 3 + 3 + 3 + 6 + 2 + 8 + 5; // 81

pub const STATE_SIZE_3P: usize = CARD_PLANES_SIZE_3P + SCALAR_TAIL_SIZE_3P; // 513
pub const ORACLE_EXTRA_3P: usize = 2 * DECK_SIZE; // 108
pub const ORACLE_STATE_SIZE_3P: usize = STATE_SIZE_3P + ORACLE_EXTRA_3P; // 621

// Decision type codes match the 4p encoder so head-routing is shared.
pub const DT_BID_3P: u8 = 0;
pub const DT_KING_CALL_3P: u8 = 1; // unused in 3p; kept for parity
pub const DT_TALON_PICK_3P: u8 = 2;
pub const DT_CARD_PLAY_3P: u8 = 3;
pub const DT_ANNOUNCE_3P: u8 = 4;

// Action sizes for 3p heads.
pub const BID_ACTION_SIZE_3P: usize = 8; // pass + 7 contracts
pub const KING_ACTION_SIZE_3P: usize = 0; // 3p has no king-call decision
pub const TALON_ACTION_SIZE_3P: usize = 6;
pub const CARD_ACTION_SIZE_3P: usize = 54;
pub const ANNOUNCE_ACTION_SIZE_3P: usize = 10; // same as 4p

/// Maps 3p bid action index → Contract (None = pass).
pub const BID_IDX_TO_CONTRACT_3P: [Option<Contract>; BID_ACTION_SIZE_3P] = [
    None,                        // 0 = pass
    Some(Contract::Klop),        // 1
    Some(Contract::Berac),       // 2
    Some(Contract::SoloThree),   // 3
    Some(Contract::SoloTwo),     // 4
    Some(Contract::SoloOne),     // 5
    Some(Contract::Valat),       // 6
    Some(Contract::BarvniValat), // 7
];

/// Map a `Contract` to its index in the 3p one-hot (7 slots).
pub fn contract_index_3p(c: Contract) -> Option<usize> {
    match c {
        Contract::Klop => Some(0),
        Contract::Berac => Some(1),
        Contract::SoloThree => Some(2),
        Contract::SoloTwo => Some(3),
        Contract::SoloOne => Some(4),
        Contract::Valat => Some(5),
        Contract::BarvniValat => Some(6),
        _ => None,
    }
}

fn write_card_plane(buf: &mut [f32], offset: usize, set: CardSet) {
    for c in set.iter() {
        buf[offset + c.0 as usize] = 1.0;
    }
}

/// Compute the set of unpicked talon cards and the picked group (if any).
fn compute_talon_visibility_3p(state: &GameState) -> (CardSet, Option<Vec<Card>>) {
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

/// Encode a 3-player game state into a pre-zeroed buffer of length
/// `STATE_SIZE_3P` (or `ORACLE_STATE_SIZE_3P` if `include_oracle = true`).
///
/// Panics if the state's variant is not `Variant::ThreePlayer`.
pub fn encode_state_3p(
    buf: &mut [f32],
    state: &GameState,
    player: u8,
    decision_type: u8,
    include_oracle: bool,
) {
    assert!(
        state.variant == Variant::ThreePlayer,
        "encode_state_3p called on {:?} state",
        state.variant
    );
    let needed = if include_oracle {
        ORACLE_STATE_SIZE_3P
    } else {
        STATE_SIZE_3P
    };
    debug_assert!(buf.len() >= needed);

    let player_u = player as usize;
    const N: u8 = 3;
    const N_USIZE: usize = 3;
    const N_OPPS: usize = 2;

    let hand = state.hands[player_u];
    let my_team = state.get_team(player);
    let contract_opt = state.contract;

    // Public talon visibility (shared across blocks).
    let (unpicked_talon_set, picked_group) = compute_talon_visibility_3p(state);
    let mut forced_retention = CardSet::EMPTY;
    if let Some(ref picked) = picked_group {
        for &c in picked {
            // 3p has no king-call, but kings still cannot be discarded
            // back into the talon — keep the same forced-retention rule
            // as 4p so the belief module pins them to declarer.
            if c.card_type() == CardType::Tarok || c.is_king() {
                forced_retention.insert(c);
            }
        }
    }

    // --- Card plane: own hand ---
    write_card_plane(buf, HAND_OFFSET_3P, hand);

    // --- Per-seat played cards ---
    let mut played_by_seat: [CardSet; 4] = [CardSet::EMPTY; 4];
    for trick in state.tricks.iter() {
        for j in 0..trick.count as usize {
            let (p, c) = trick.cards[j];
            played_by_seat[p as usize].insert(c);
        }
    }
    if let Some(ref trick) = state.current_trick {
        for j in 0..trick.count as usize {
            let (p, c) = trick.cards[j];
            played_by_seat[p as usize].insert(c);
        }
    }
    // Attribute publicly-retired (unpicked) talon cards to the declarer's plane.
    if let Some(decl) = state.declarer {
        for c in unpicked_talon_set.iter() {
            played_by_seat[decl as usize].insert(c);
        }
    }
    write_card_plane(buf, SELF_PLAYED_OFFSET_3P, played_by_seat[player_u]);
    let opp_seats: [u8; N_OPPS] = [(player + 1) % N, (player + 2) % N];
    for (k, &opp) in opp_seats.iter().enumerate() {
        let off = OPP_PLAYED_OFFSET_3P + k * DECK_SIZE;
        write_card_plane(buf, off, played_by_seat[opp as usize]);
    }

    // --- Active trick plane ---
    if let Some(ref trick) = state.current_trick {
        let mut active = CardSet::EMPTY;
        for j in 0..trick.count as usize {
            active.insert(trick.cards[j].1);
        }
        write_card_plane(buf, ACTIVE_TRICK_OFFSET_3P, active);
    }

    // --- Visible talon plane (publicly revealed groups) ---
    let mut talon_visible = CardSet::EMPTY;
    for group in state.talon_revealed.iter() {
        for &c in group {
            talon_visible.insert(c);
        }
    }
    write_card_plane(buf, TALON_VIS_OFFSET_3P, talon_visible);

    // --- Suit-void / tarok-void inference (per opponent seat) ---
    let mut opp_void: [u8; 4] = [0; 4]; // suit-bitmask per absolute seat
    let mut opp_tarok_void: [bool; 4] = [false; 4];
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
                                // Forced-follow rule: suit-void AND not playing
                                // tarok ⇒ tarok-void at that moment (monotone
                                // in time, so still holds).
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

    // --- Belief block: cardinality-aware IPF over unknown cards ---
    let mut known = hand;
    known = known.union(state.played_cards);
    known = known.union(unpicked_talon_set);
    if let Some(ref trick) = state.current_trick {
        known = known.union(trick.played_cards_set());
    }
    // Declarer knows their own put_down.
    if state.declarer == Some(player) {
        known = known.union(state.put_down);
    }

    let declarer_opt = state.declarer;
    let mut unknown: Vec<usize> = Vec::with_capacity(DECK_SIZE);
    for cidx in 0..DECK_SIZE {
        if !known.contains(Card(cidx as u8)) {
            unknown.push(cidx);
        }
    }

    // Matrix m[col][row], row = relative-opp index (0..N_OPPS).
    let mut m: Vec<[f32; N_OPPS]> = vec![[0.0; N_OPPS]; unknown.len()];
    let mut row_target: [f32; N_OPPS] = [0.0; N_OPPS];
    for r in 0..N_OPPS {
        row_target[r] = state.hands[opp_seats[r] as usize].len() as f32;
    }
    let mut col_pinned: Vec<bool> = vec![false; unknown.len()];

    for (col, &cidx) in unknown.iter().enumerate() {
        let c = Card(cidx as u8);
        // Forced retention: pin to declarer (if declarer is one of our opps).
        if forced_retention.contains(c) {
            for r in 0..N_OPPS {
                if declarer_opt == Some(opp_seats[r]) {
                    m[col][r] = 1.0;
                    row_target[r] -= 1.0;
                }
            }
            col_pinned[col] = true;
            continue;
        }
        for r in 0..N_OPPS {
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

    const IPF_ITERS: usize = 12;
    for _ in 0..IPF_ITERS {
        // Row scaling.
        let mut row_sum: [f32; N_OPPS] = [0.0; N_OPPS];
        for col in 0..unknown.len() {
            if col_pinned[col] {
                continue;
            }
            for r in 0..N_OPPS {
                row_sum[r] += m[col][r];
            }
        }
        for r in 0..N_OPPS {
            if row_sum[r] > 1e-12 && row_target[r] > 0.0 {
                let s = row_target[r] / row_sum[r];
                for col in 0..unknown.len() {
                    if col_pinned[col] {
                        continue;
                    }
                    m[col][r] *= s;
                }
            } else if row_target[r] <= 0.0 {
                for col in 0..unknown.len() {
                    if col_pinned[col] {
                        continue;
                    }
                    m[col][r] = 0.0;
                }
            }
        }
        // Column scaling: each unknown card has total probability 1
        // distributed across the 2 opponents.
        for col in 0..unknown.len() {
            if col_pinned[col] {
                continue;
            }
            let cs: f32 = m[col][0] + m[col][1];
            if cs > 1e-12 {
                let s = 1.0 / cs;
                for r in 0..N_OPPS {
                    m[col][r] *= s;
                }
            }
        }
    }

    for r in 0..N_OPPS {
        let base = BELIEF_OFFSET_3P + r * DECK_SIZE;
        for (col, &cidx) in unknown.iter().enumerate() {
            let v = m[col][r];
            if v > 0.0 {
                buf[base + cidx] = v;
            }
        }
    }

    // --- Running team card-point totals (for centaur block) ---
    let mut my_team_points: i32 = 0;
    let mut opp_team_points: i32 = 0;
    for (i, trick) in state.tricks.iter().enumerate() {
        let is_last = i == state.tricks.len() - 1;
        let tr = evaluate_trick(trick, is_last, contract_opt);
        let trick_pts: i32 = (0..trick.count as usize)
            .map(|j| trick.cards[j].1.points() as i32)
            .sum();
        if state.get_team(tr.winner) == my_team {
            my_team_points += trick_pts;
        } else {
            opp_team_points += trick_pts;
        }
    }

    // ============= SCALAR TAIL =============
    let mut o = SCALAR_OFFSET_3P;

    // Seat position relative to dealer (3 one-hot).
    let seat_rel = ((player + N - state.dealer) % N) as usize;
    buf[o + seat_rel] = 1.0;
    o += 3;

    // Contract one-hot (7).
    debug_assert_eq!(o, CONTRACT_OFFSET_3P);
    if let Some(c) = state.contract {
        if let Some(idx) = contract_index_3p(c) {
            buf[o + idx] = 1.0;
        }
    }
    o += CONTRACT_SIZE_3P;

    // Phase one-hot (3).
    debug_assert_eq!(o, PHASE_OFFSET_3P);
    match state.phase {
        Phase::Bidding => buf[o] = 1.0,
        Phase::TrickPlay => buf[o + 1] = 1.0,
        _ => buf[o + 2] = 1.0,
    }
    o += 3;

    // Tricks played / 16.
    debug_assert_eq!(o, TRICKS_PLAYED_OFFSET_3P);
    buf[o] = (state.tricks_played() as f32) / 16.0;
    o += 1;

    // Decision type one-hot (5).
    debug_assert_eq!(o, DECISION_TYPE_OFFSET_3P);
    if (decision_type as usize) < 5 {
        buf[o + decision_type as usize] = 1.0;
    }
    o += 5;

    // Highest bid one-hot (8 = no_bid + 7 contracts).
    debug_assert_eq!(o, HIGHEST_BID_OFFSET_3P);
    let highest = state
        .bids
        .iter()
        .filter_map(|b| b.contract)
        .max_by_key(|c| c.strength());
    match highest {
        Some(c) => match contract_index_3p(c) {
            Some(idx) => buf[o + 1 + idx] = 1.0,
            None => buf[o] = 1.0, // shouldn't happen in 3p but stay safe
        },
        None => buf[o] = 1.0,
    }
    o += HIGHEST_BID_SIZE_3P;

    // Passed players (3, dealer-relative).
    debug_assert_eq!(o, PASSED_OFFSET_3P);
    let mut last_bid_per_player: [Option<Option<Contract>>; 4] = [None; 4];
    for b in state.bids.iter() {
        last_bid_per_player[b.player as usize] = Some(b.contract);
    }
    for p in 0..N_USIZE {
        if let Some(None) = last_bid_per_player[p] {
            let rel = ((p as u8 + N - state.dealer) % N) as usize;
            buf[o + rel] = 1.0;
        }
    }
    o += 3;

    // Announcements split by team (5 own + 5 opp).
    // Bit ordering matches `Announcement` enum:
    //   Trula=0, Kings=1, PagatUltimo=2, KingUltimo=3 (zero in 3p), Valat=4
    debug_assert_eq!(o, OWN_ANN_OFFSET_3P);
    if state.declarer.is_some() {
        let mut own_ann: u8 = 0;
        let mut opp_ann: u8 = 0;
        for seat in 0..N_USIZE as u8 {
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

    // Kontra levels (5 normalized to [0, 1]).
    debug_assert_eq!(o, KONTRA_OFFSET_3P);
    for i in 0..KontraTarget::NUM {
        let level = state.kontra_levels[i];
        buf[o + i] = (level.multiplier() as f32 - 1.0) / 7.0;
    }
    o += 5;

    // Own role one-hot (3 slots; partner slot kept for shape parity with 4p).
    debug_assert_eq!(o, ROLE_OFFSET_3P);
    let role_idx = match state.roles[player_u] {
        PlayerRole::Declarer => 0,
        PlayerRole::Partner => 1, // unused in 3p
        PlayerRole::Opponent => 2,
    };
    buf[o + role_idx] = 1.0;
    o += 3;

    // Declarer relative seat (3). Combined with own role + own seat-rel,
    // every seat's role becomes determined.
    debug_assert_eq!(o, DECLARER_REL_OFFSET_3P);
    if let Some(decl) = state.declarer {
        let rel = ((decl + N - player) % N) as usize;
        buf[o + rel] = 1.0;
    }
    o += 3;

    // Centaur trick context.
    debug_assert_eq!(o, CENTAUR_OFFSET_3P);
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

    // Trick leader relative seat (3).
    debug_assert_eq!(o, TRICK_LEADER_OFFSET_3P);
    if let Some(lead) = current_leader {
        let rel = ((lead + N - player) % N) as usize;
        buf[o + rel] = 1.0;
    }
    o += 3;

    // Trick currently-winning relative seat (3).
    debug_assert_eq!(o, TRICK_WINNING_OFFSET_3P);
    if let Some(win) = current_winner {
        let rel = ((win + N - player) % N) as usize;
        buf[o + rel] = 1.0;
    }
    o += 3;

    // Trick context: position + lead type/suit (6).
    debug_assert_eq!(o, TRICK_CTX_OFFSET_3P);
    if let Some(ref trick) = state.current_trick {
        if trick.count > 0 {
            buf[o] = trick.count as f32 / 3.0;
            let lead_card = trick.cards[0].1;
            if lead_card.card_type() == CardType::Tarok {
                buf[o + 1] = 1.0;
            } else if let Some(s) = lead_card.suit() {
                buf[o + 2 + s as usize] = 1.0;
            }
        }
    }
    o += 6;

    // Per-opponent tarok-void flags (2).
    debug_assert_eq!(o, TAROK_VOID_OFFSET_3P);
    for (k, &opp) in opp_seats.iter().enumerate() {
        if opp_tarok_void[opp as usize] {
            buf[o + k] = 1.0;
        }
    }
    o += 2;

    // Per-opponent suit-void flags (2 × 4).
    debug_assert_eq!(o, SUIT_VOID_OFFSET_3P);
    for (k, &opp) in opp_seats.iter().enumerate() {
        let mask = opp_void[opp as usize];
        for s in 0u8..4u8 {
            if mask & (1 << s) != 0 {
                buf[o + k * 4 + s as usize] = 1.0;
            }
        }
    }
    o += 8;

    // Remaining-in-play counts (5): taroks/22, H/8, D/8, C/8, S/8.
    debug_assert_eq!(o, REMAINING_OFFSET_3P);
    let mut taroks_played: u32 = 0;
    let mut suit_played: [u32; 4] = [0; 4];
    let mut tally = |c: Card| match c.card_type() {
        CardType::Tarok => taroks_played += 1,
        CardType::Suit => {
            if let Some(s) = c.suit() {
                suit_played[s as usize] += 1;
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
    buf[o] = (22u32.saturating_sub(taroks_played)) as f32 / 22.0;
    for s in 0..4 {
        buf[o + 1 + s] = (8u32.saturating_sub(suit_played[s])) as f32 / 8.0;
    }
    o += 5;

    debug_assert_eq!(o, STATE_SIZE_3P);

    // --- Oracle extension: opponent perfect hands ---
    if include_oracle {
        debug_assert!(buf.len() >= ORACLE_STATE_SIZE_3P);
        for (k, &opp) in opp_seats.iter().enumerate() {
            let off = STATE_SIZE_3P + k * DECK_SIZE;
            write_card_plane(buf, off, state.hands[opp as usize]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_size_constants() {
        assert_eq!(STATE_SIZE_3P, 513);
        assert_eq!(ORACLE_STATE_SIZE_3P, 621);
        assert_eq!(CARD_PLANES_SIZE_3P, 432);
        assert_eq!(SCALAR_TAIL_SIZE_3P, 81);
    }

    #[test]
    fn contract_index_round_trip() {
        assert_eq!(contract_index_3p(Contract::Klop), Some(0));
        assert_eq!(contract_index_3p(Contract::Berac), Some(1));
        assert_eq!(contract_index_3p(Contract::SoloThree), Some(2));
        assert_eq!(contract_index_3p(Contract::Valat), Some(5));
        assert_eq!(contract_index_3p(Contract::BarvniValat), Some(6));
        // 4p-only contracts must not have a 3p slot.
        assert_eq!(contract_index_3p(Contract::Three), None);
        assert_eq!(contract_index_3p(Contract::Solo), None);
    }
}
