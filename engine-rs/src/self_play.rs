//! Batched self-play runner using the [`BatchPlayer`] trait.
//!
//! Runs M concurrent games.  Pending decisions are grouped by player
//! and dispatched via [`BatchPlayer::batch_decide`], allowing neural
//! networks to do batched forward passes while heuristic bots loop.
//!
//! The Python side only needs to:
//!   1. Create player instances (NN model path or bot version)
//!   2. Call `run_self_play(n_games, concurrency, seat_config)`
//!   3. Receive a dict of batched numpy arrays (experiences)

use std::sync::Arc;

use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;

use crate::card::*;
use crate::encoding;
use crate::encoding_3p;
use crate::game_state::*;
use crate::legal_moves;
use crate::player::{
    card_suit_idx, BatchPlayer, DecisionContext, DecisionResult, DecisionType, BID_IDX_TO_CONTRACT,
    CARD_ACTION_SIZE, TALON_ACTION_SIZE,
};
use crate::scoring;

// -----------------------------------------------------------------------
// Variant-aware encoding dispatch helpers
// -----------------------------------------------------------------------

/// Public state vector length for a given variant. Used to size scratch
/// buffers and to flatten experiences on the Python side.
pub fn state_size_for(variant: Variant) -> usize {
    match variant {
        Variant::FourPlayer => encoding::STATE_SIZE,
        Variant::ThreePlayer => encoding_3p::STATE_SIZE_3P,
    }
}

/// Oracle (perfect-information) state vector length for a given variant.
pub fn oracle_state_size_for(variant: Variant) -> usize {
    match variant {
        Variant::FourPlayer => encoding::ORACLE_STATE_SIZE,
        Variant::ThreePlayer => encoding_3p::ORACLE_STATE_SIZE_3P,
    }
}

fn encode_state_dispatch(buf: &mut [f32], gs: &GameState, player: u8, dt: u8) {
    match gs.variant {
        Variant::FourPlayer => encoding::encode_state(buf, gs, player, dt),
        Variant::ThreePlayer => encoding_3p::encode_state_3p(buf, gs, player, dt, false),
    }
}

fn encode_oracle_state_dispatch(buf: &mut [f32], gs: &GameState, player: u8, dt: u8) {
    match gs.variant {
        Variant::FourPlayer => encoding::encode_oracle_state(buf, gs, player, dt),
        Variant::ThreePlayer => encoding_3p::encode_state_3p(buf, gs, player, dt, true),
    }
}

// -----------------------------------------------------------------------
// In-flight game
// -----------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GamePhase {
    Bid,
    KingCall,
    TalonPick,
    TrickPlay,
    Done,
}

struct InFlightGame {
    gs: GameState,
    dealer: u8,
    game_id: u32,
    phase: GamePhase,
    passed: [bool; 4],
    highest_bid: Option<Contract>,
    winning_bidder: Option<u8>,
    current_bidder: u8,
    bid_round: u8,
    trick_num: u8,
    trick_offset: u8,
    lead_player: u8,
    step_counter: u16,
    // Arena metadata
    initial_taroks: [u8; 4],
    bid_choices: [i8; 4],
    initial_hands: [CardSet; 4],
    initial_talon: CardSet,
    trace: GameTrace,
    // Reusable scratch buffers — allocated once per slot, filled in-place at each
    // decision point, then moved into RawExperience via replace(). This keeps the
    // per-decision heap allocation count to one (the fresh replacement buffer).
    state_buf: Vec<f32>,
    oracle_state_buf: Vec<f32>,
    legal_mask: Vec<f32>,
}

/// Internal tracking for a pending decision.
/// The actual state/oracle/mask buffers live in the owning `InFlightGame` slot
/// and are accessed in Steps 2 and 3 via `pd.slot`.
struct Pending {
    slot: usize,
    player: u8,
    decision_type: DecisionType,
    game_mode: u8,
}

// -----------------------------------------------------------------------
// Game trace — compact record of every decision for deterministic replay
// -----------------------------------------------------------------------

#[derive(Clone, Default)]
pub struct GameTrace {
    /// (player, action_idx) for each bid in chronological order
    pub bids: Vec<(u8, u8)>,
    /// (player, suit_action_idx) for king call, if any
    pub king_call: Option<(u8, u8)>,
    /// (player, group_idx) for talon pick, if any
    pub talon_pick: Option<(u8, u8)>,
    /// Card indices that were discarded after talon exchange
    pub put_down: Vec<u8>,
    /// (player, card_index) for every card played, in order
    pub cards_played: Vec<(u8, u8)>,
}

// -----------------------------------------------------------------------
// Experience record
// -----------------------------------------------------------------------

pub struct RawExperience {
    pub state: Vec<f32>,
    pub oracle_state: Vec<f32>,
    pub action: u16,
    pub log_prob: f32,
    pub value: f32,
    pub decision_type: u8,
    pub game_mode: u8,
    pub legal_mask: Vec<f32>,
    pub game_id: u32,
    pub step_in_game: u16,
    pub player: u8,
    // Reserved for optional trick-level shaping (currently disabled).
    // Kept at 0 so value targets are driven by final game scores.
    pub trick_points: i16,
}

pub struct GameResult {
    pub game_id: u32,
    pub scores: [i32; 4],
    /// Per-seat training reward signal (non-zero-sum). See
    /// [`crate::scoring::score_game_reward`]. Leaderboard/UI code should use
    /// `scores`; RL training code should use `reward_scores`.
    pub reward_scores: [i32; 4],
    pub experiences: Vec<RawExperience>,
    // Arena metadata
    pub contract: u8,
    pub declarer: i8,
    pub partner: i8,
    pub bid_contracts: [i8; 4],
    pub taroks_in_hand: [u8; 4],
    pub initial_hands: [CardSet; 4],
    pub initial_talon: CardSet,
    pub trace: GameTrace,
}

// -----------------------------------------------------------------------
// Batched self-play runner
// -----------------------------------------------------------------------

pub struct SelfPlayRunner {
    players: [Arc<dyn BatchPlayer>; 4],
    variant: Variant,
}

impl SelfPlayRunner {
    pub fn new(players: [Arc<dyn BatchPlayer>; 4]) -> Self {
        SelfPlayRunner { players, variant: Variant::FourPlayer }
    }

    /// Construct a runner for a specific variant. For 3-player games, slot 3
    /// of `players` is unused (the engine never asks player 3 to decide); pass
    /// any `Arc<dyn BatchPlayer>` as a phantom.
    pub fn new_with_variant(players: [Arc<dyn BatchPlayer>; 4], variant: Variant) -> Self {
        SelfPlayRunner { players, variant }
    }

    pub fn run(&self, n_games: u32, concurrency: usize) -> Vec<GameResult> {
        self.run_with_deck_seeds(n_games, concurrency, None)
    }

    /// Run `n_games` like [`run`], but optionally use per-game deck seeds.
    ///
    /// When `deck_seeds` is `Some(v)`, `v` must have exactly `n_games` entries
    /// and game `g` deals its hands from `SmallRng::seed_from_u64(v[g])`. This
    /// makes dealing deterministic and reproducible across seatings — the
    /// building block for duplicate training. Policy exploration still uses
    /// the players' own stochasticity, which is independent of the deal RNG.
    ///
    /// When `deck_seeds` is `None`, behavior is identical to [`run`] — dealing
    /// uses the process-default RNG.
    pub fn run_with_deck_seeds(
        &self,
        n_games: u32,
        concurrency: usize,
        deck_seeds: Option<Vec<u64>>,
    ) -> Vec<GameResult> {
        if let Some(ref seeds) = deck_seeds {
            assert_eq!(
                seeds.len(),
                n_games as usize,
                "deck_seeds length ({}) must equal n_games ({})",
                seeds.len(),
                n_games
            );
        }
        let mut rng = rand::rng();
        let mut results: Vec<GameResult> = Vec::with_capacity(n_games as usize);
        let mut slots: Vec<Option<InFlightGame>> = (0..concurrency).map(|_| None).collect();
        let mut next_game: u32 = 0;
        let mut active: usize = 0;
        let mut game_exps: Vec<Vec<RawExperience>> = Vec::new();

        // Seed initial games
        let n_initial = concurrency.min(n_games as usize);
        for i in 0..n_initial {
            slots[i] = Some(Self::new_game_dispatch(next_game, &mut rng, deck_seeds.as_deref(), self.variant));
            game_exps.push(Vec::new());
            next_game += 1;
            active += 1;
        }

        // Pre-compute unique player groups for efficient batching.
        // If seats 1,2,3 share the same Arc, their decisions are batched
        // into a single `batch_decide` call.
        let mut unique_players: Vec<Arc<dyn BatchPlayer>> = Vec::new();
        let mut seat_to_uid: [usize; 4] = [0; 4];
        for (seat, player) in self.players.iter().enumerate() {
            match unique_players.iter().position(|p| Arc::ptr_eq(p, player)) {
                Some(idx) => seat_to_uid[seat] = idx,
                None => {
                    seat_to_uid[seat] = unique_players.len();
                    unique_players.push(player.clone());
                }
            }
        }

        // Persistent per-unique-player index buffers reused each loop iteration.
        let n_unique = unique_players.len();
        let mut idx_bufs: Vec<Vec<usize>> = (0..n_unique).map(|_| Vec::new()).collect();

        while active > 0 {
            // Step 1: advance all games until they need a decision or finish
            let mut pending: Vec<Pending> = Vec::new();

            for (slot_idx, slot) in slots.iter_mut().enumerate() {
                let game = match slot.as_mut() {
                    Some(g) if g.phase != GamePhase::Done => g,
                    _ => continue,
                };

                loop {
                    match Self::advance_until_decision(game) {
                        Some(mut pd) => {
                            pd.slot = slot_idx;
                            pending.push(pd);
                            break;
                        }
                        None if game.phase == GamePhase::Done => {
                            let gid = game.game_id as usize;
                            let exps = std::mem::take(&mut game_exps[gid]);
                            results.push(Self::build_result(game, exps));
                            if next_game < n_games {
                                *game = Self::new_game_dispatch(next_game, &mut rng, deck_seeds.as_deref(), self.variant);
                                game_exps.push(Vec::new());
                                next_game += 1;
                            } else {
                                *slot = None;
                                active -= 1;
                            }
                            break;
                        }
                        None => break,
                    }
                }
            }

            if pending.is_empty() {
                continue;
            }

            // Step 2: dispatch decisions grouped by unique player
            let mut all_results: Vec<DecisionResult> = vec![
                DecisionResult {
                    action: 0,
                    log_prob: 0.0,
                    value: 0.0
                };
                pending.len()
            ];

            // Route pending indices into per-player index buffers (no alloc).
            for buf in idx_bufs.iter_mut() {
                buf.clear();
            }
            for (pi, pd) in pending.iter().enumerate() {
                idx_bufs[seat_to_uid[pd.player as usize]].push(pi);
            }

            for (uid, up) in unique_players.iter().enumerate() {
                let idx_buf = &idx_bufs[uid];
                if idx_buf.is_empty() {
                    continue;
                }

                // Build a borrowed context batch; this avoids deep-cloning GameState.
                let mut contexts: Vec<DecisionContext<'_>> = Vec::with_capacity(idx_buf.len());
                for &pi in idx_buf.iter() {
                    let pd = &pending[pi];
                    let game = slots[pd.slot].as_ref().unwrap();
                    contexts.push(DecisionContext {
                        gs: &game.gs,
                        player: pd.player,
                        decision_type: pd.decision_type,
                        legal_mask: game.legal_mask.clone(),
                        state_encoding: game.state_buf.clone(),
                    });
                }

                let batch_results = up.batch_decide(&contexts);

                for (i, &pi) in idx_buf.iter().enumerate() {
                    all_results[pi] = batch_results[i];
                }
            }

            // Step 3: record experiences and apply actions
            for (pi, pd) in pending.iter().enumerate() {
                let game = slots[pd.slot].as_mut().unwrap();
                let result = all_results[pi];

                let gid = game.game_id as usize;
                // Move the game's scratch buffers into RawExperience (zero clones),
                // replacing each with a fresh pre-allocated buffer for the next
                // decision in this slot. Buffer sizes track the game variant.
                let v = game.gs.variant;
                let state =
                    std::mem::replace(&mut game.state_buf, vec![0f32; state_size_for(v)]);
                let oracle_state = std::mem::replace(
                    &mut game.oracle_state_buf,
                    vec![0f32; oracle_state_size_for(v)],
                );
                let legal_mask =
                    std::mem::replace(&mut game.legal_mask, vec![0f32; CARD_ACTION_SIZE]);
                game_exps[gid].push(RawExperience {
                    state,
                    oracle_state,
                    action: result.action as u16,
                    log_prob: result.log_prob,
                    value: result.value,
                    decision_type: pd.decision_type as u8,
                    game_mode: pd.game_mode,
                    legal_mask,
                    game_id: game.game_id,
                    step_in_game: game.step_counter,
                    player: pd.player,
                    trick_points: 0,
                });
                game.step_counter += 1;

                Self::apply_action(
                    game,
                    pd.decision_type,
                    result.action,
                    &self.players[pd.player as usize],
                );
            }

            // Step 4: finalize newly-done games
            for slot in slots.iter_mut() {
                let game = match slot.as_mut() {
                    Some(g) if g.phase == GamePhase::Done => g,
                    _ => continue,
                };
                let gid = game.game_id as usize;
                let exps = std::mem::take(&mut game_exps[gid]);
                results.push(Self::build_result(game, exps));
                if next_game < n_games {
                    *game = Self::new_game_dispatch(next_game, &mut rng, deck_seeds.as_deref(), self.variant);
                    game_exps.push(Vec::new());
                    next_game += 1;
                } else {
                    *slot = None;
                    active -= 1;
                }
            }
        }

        results
    }

    // ------------------------------------------------------------------
    // Game lifecycle
    // ------------------------------------------------------------------

    /// Dispatch to [`new_game`] with a per-game deck RNG:
    /// - if `deck_seeds` is `Some`, deal from `SmallRng::seed_from_u64(deck_seeds[game_id])`.
    /// - otherwise, deal from the shared process RNG.
    fn new_game_dispatch(
        game_id: u32,
        shared_rng: &mut impl Rng,
        deck_seeds: Option<&[u64]>,
        variant: Variant,
    ) -> InFlightGame {
        match deck_seeds {
            Some(seeds) => {
                let mut per_game_rng = SmallRng::seed_from_u64(seeds[game_id as usize]);
                Self::new_game_with_variant(game_id, &mut per_game_rng, variant)
            }
            None => Self::new_game_with_variant(game_id, shared_rng, variant),
        }
    }

    fn new_game(game_id: u32, rng: &mut impl Rng) -> InFlightGame {
        Self::new_game_with_variant(game_id, rng, Variant::FourPlayer)
    }

    fn new_game_with_variant(
        game_id: u32,
        rng: &mut impl Rng,
        variant: Variant,
    ) -> InFlightGame {
        let n = variant.num_players() as u32;
        let dealer = (game_id % n) as u8;
        let mut gs = GameState::new_with_variant(variant, dealer);
        gs.deal(rng);
        let n_u8 = variant.num_players() as u8;
        let forehand = (dealer + 1) % n_u8;
        let first_bidder = (dealer + 2) % n_u8;
        let mut initial_taroks = [0u8; 4];
        let initial_hands = gs.hands;
        let initial_talon = gs.talon;
        for (pid, hand) in gs.hands.iter().enumerate() {
            initial_taroks[pid] = hand.tarok_count();
        }
        InFlightGame {
            gs,
            dealer,
            game_id,
            phase: GamePhase::Bid,
            // For 3p, slot 3 is a phantom seat: pre-mark as passed so the
            // bidding loop in `bid_step` doesn't treat it as an active bidder.
            // 4p uses all four slots.
            passed: {
                let mut p = [false; 4];
                if variant.is_three_player() {
                    p[3] = true;
                }
                p
            },
            highest_bid: None,
            winning_bidder: None,
            current_bidder: first_bidder,
            bid_round: 0,
            trick_num: 0,
            trick_offset: 0,
            lead_player: forehand,
            step_counter: 0,
            initial_taroks,
            bid_choices: [-1i8; 4],
            initial_hands,
            initial_talon,
            trace: GameTrace::default(),
            state_buf: vec![0f32; state_size_for(variant)],
            oracle_state_buf: vec![0f32; oracle_state_size_for(variant)],
            legal_mask: vec![0f32; CARD_ACTION_SIZE],
        }
    }

    fn score_game(gs: &GameState) -> [i32; 4] {
        scoring::score_game(gs)
    }

    fn game_mode_id(contract: Option<Contract>) -> u8 {
        match contract {
            Some(c) if c.is_solo() => 0,
            Some(Contract::Klop | Contract::Berac) => 1,
            Some(Contract::BarvniValat) => 3,
            _ => 2,
        }
    }

    fn build_result(game: &InFlightGame, exps: Vec<RawExperience>) -> GameResult {
        let scores = Self::score_game(&game.gs);
        let reward_scores = scoring::score_game_reward(&game.gs);
        let contract = game.gs.contract.map(|c| c as u8).unwrap_or(0);
        let declarer = game.gs.declarer.map(|d| d as i8).unwrap_or(-1);
        let partner = game.gs.partner.map(|p| p as i8).unwrap_or(-1);
        GameResult {
            game_id: game.game_id,
            scores,
            reward_scores,
            experiences: exps,
            contract,
            declarer,
            partner,
            bid_contracts: game.bid_choices,
            taroks_in_hand: game.initial_taroks,
            initial_hands: game.initial_hands,
            initial_talon: game.initial_talon,
            trace: game.trace.clone(),
        }
    }

    // ------------------------------------------------------------------
    // Advance game until a decision is needed
    // ------------------------------------------------------------------

    fn advance_until_decision(game: &mut InFlightGame) -> Option<Pending> {
        match game.phase {
            GamePhase::Bid => Self::bid_step(game),
            GamePhase::KingCall => Self::king_call_step(game),
            GamePhase::TalonPick => Self::talon_pick_step(game),
            GamePhase::TrickPlay => Self::trick_step(game),
            GamePhase::Done => None,
        }
    }

    // ------------------------------------------------------------------
    // Bidding
    // ------------------------------------------------------------------

    fn bid_step(game: &mut InFlightGame) -> Option<Pending> {
        let active_count = game.passed.iter().filter(|&&p| !p).count();
        if (active_count <= 1 && game.winning_bidder.is_some()) || active_count == 0 {
            Self::resolve_bidding(game);
            return Self::advance_until_decision(game);
        }
        if game.bid_round >= 20 {
            Self::resolve_bidding(game);
            return Self::advance_until_decision(game);
        }
        let bidder = game.current_bidder;
        if game.passed[bidder as usize] {
            Self::next_bidder(game);
            return Self::advance_until_decision(game);
        }

        game.state_buf.fill(0.0);
        encode_state_dispatch(&mut game.state_buf, &game.gs, bidder, encoding::DT_BID);
        game.oracle_state_buf.fill(0.0);
        encode_oracle_state_dispatch(
            &mut game.oracle_state_buf,
            &game.gs,
            bidder,
            encoding::DT_BID,
        );

        game.legal_mask.fill(0.0);
        game.legal_mask[0] = 1.0; // pass always legal
        let legal = game.gs.legal_bids(bidder);
        let bid_table: &[Option<Contract>] = match game.gs.variant {
            Variant::FourPlayer => &BID_IDX_TO_CONTRACT,
            Variant::ThreePlayer => &encoding_3p::BID_IDX_TO_CONTRACT_3P,
        };
        for contract in &legal {
            for (idx, mapped) in bid_table.iter().enumerate() {
                if *mapped == Some(*contract) {
                    game.legal_mask[idx] = 1.0;
                }
            }
        }

        Some(Pending {
            slot: 0,
            player: bidder,
            decision_type: DecisionType::Bid,
            game_mode: Self::game_mode_id(game.gs.contract),
        })
    }

    fn apply_bid(game: &mut InFlightGame, action_idx: usize) {
        let bidder = game.current_bidder;
        game.trace.bids.push((bidder, action_idx as u8));
        let contract = match game.gs.variant {
            Variant::FourPlayer => {
                if action_idx < BID_IDX_TO_CONTRACT.len() {
                    BID_IDX_TO_CONTRACT[action_idx]
                } else {
                    None
                }
            }
            Variant::ThreePlayer => {
                if action_idx < encoding_3p::BID_IDX_TO_CONTRACT_3P.len() {
                    encoding_3p::BID_IDX_TO_CONTRACT_3P[action_idx]
                } else {
                    None
                }
            }
        };

        match contract {
            None => {
                game.passed[bidder as usize] = true;
                game.gs.add_bid(bidder, None);
            }
            Some(c) => {
                let legal = game.gs.legal_bids(bidder);
                if legal.contains(&c) {
                    game.gs.add_bid(bidder, Some(c));
                    game.highest_bid = Some(c);
                    game.winning_bidder = Some(bidder);
                    game.bid_choices[bidder as usize] = c as i8;
                } else {
                    game.passed[bidder as usize] = true;
                    game.gs.add_bid(bidder, None);
                }
            }
        }
        game.bid_round += 1;
        Self::next_bidder(game);
    }

    fn next_bidder(game: &mut InFlightGame) {
        let n = game.gs.variant.num_players() as u8;
        for _ in 0..n {
            game.current_bidder = (game.current_bidder + 1) % n;
            if !game.passed[game.current_bidder as usize] {
                break;
            }
        }
    }

    fn resolve_bidding(game: &mut InFlightGame) {
        let n = game.gs.variant.num_players() as u8;
        if let (Some(bidder), Some(contract)) = (game.winning_bidder, game.highest_bid) {
            game.gs.declarer = Some(bidder);
            game.gs.contract = Some(contract);
            game.gs.roles[bidder as usize] = PlayerRole::Declarer;
            for i in 0..n {
                if i != bidder {
                    game.gs.roles[i as usize] = PlayerRole::Opponent;
                }
            }
            match contract {
                // No-talon contracts: straight to trick play.
                Contract::Berac
                | Contract::BarvniValat
                | Contract::Valat
                | Contract::Klop => {
                    game.gs.phase = Phase::TrickPlay;
                    game.phase = GamePhase::TrickPlay;
                    game.lead_player = (game.dealer + 1) % n;
                }
                _ if contract.is_solo() => {
                    let tc = contract.talon_cards();
                    if tc > 0 {
                        game.gs.phase = Phase::TalonExchange;
                        game.phase = GamePhase::TalonPick;
                    } else {
                        game.gs.phase = Phase::TrickPlay;
                        game.phase = GamePhase::TrickPlay;
                        game.lead_player = (game.dealer + 1) % n;
                    }
                }
                _ => {
                    // 4p partner contracts (Three/Two/One): king calling next.
                    // The 3p variant never reaches this arm because its biddable
                    // contracts are exhausted by the matches above.
                    debug_assert!(
                        game.gs.variant.has_king_call(),
                        "reached king-call branch in a variant that has no kings",
                    );
                    game.gs.phase = Phase::KingCalling;
                    game.phase = GamePhase::KingCall;
                }
            }
        } else {
            // Klop (nobody bid)
            game.gs.contract = Some(Contract::Klop);
            for i in 0..n as usize {
                game.gs.roles[i] = PlayerRole::Opponent;
            }
            game.gs.phase = Phase::TrickPlay;
            game.phase = GamePhase::TrickPlay;
            game.lead_player = (game.dealer + 1) % n;
        }
    }

    // ------------------------------------------------------------------
    // King Call
    // ------------------------------------------------------------------

    fn king_call_step(game: &mut InFlightGame) -> Option<Pending> {
        let declarer = game.gs.declarer?;
        let callable = game.gs.callable_kings();
        if callable.is_empty() {
            Self::transition_after_king_call(game);
            return Self::advance_until_decision(game);
        }

        game.state_buf.fill(0.0);
        encode_state_dispatch(
            &mut game.state_buf,
            &game.gs,
            declarer,
            encoding::DT_KING_CALL,
        );
        game.oracle_state_buf.fill(0.0);
        encode_oracle_state_dispatch(
            &mut game.oracle_state_buf,
            &game.gs,
            declarer,
            encoding::DT_KING_CALL,
        );

        game.legal_mask.fill(0.0);
        for &card in &callable {
            if let Some(suit_idx) = card_suit_idx(card.0) {
                game.legal_mask[suit_idx] = 1.0;
            }
        }

        Some(Pending {
            slot: 0,
            player: declarer,
            decision_type: DecisionType::KingCall,
            game_mode: Self::game_mode_id(game.gs.contract),
        })
    }

    fn apply_king_call(game: &mut InFlightGame, action_idx: usize) {
        let declarer = match game.gs.declarer {
            Some(d) => d,
            None => return,
        };
        game.trace.king_call = Some((declarer, action_idx as u8));
        let callable = game.gs.callable_kings();
        let chosen = callable
            .iter()
            .find(|c| card_suit_idx(c.0) == Some(action_idx))
            .copied()
            .unwrap_or(callable[0]);

        game.gs.called_king = Some(chosen);

        // Find partner (4p only — king-call phase is only entered when variant has kings)
        let n = game.gs.variant.num_players() as u8;
        for p in 0..n {
            if p != declarer && game.gs.hands[p as usize].contains(chosen) {
                game.gs.partner = Some(p);
                game.gs.roles[p as usize] = PlayerRole::Partner;
                break;
            }
        }

        Self::transition_after_king_call(game);
    }

    fn transition_after_king_call(game: &mut InFlightGame) {
        let contract = game.gs.contract.unwrap_or(Contract::Three);
        let tc = contract.talon_cards();
        let n = game.gs.variant.num_players() as u8;
        if tc > 0 {
            game.gs.phase = Phase::TalonExchange;
            game.phase = GamePhase::TalonPick;
        } else {
            game.gs.phase = Phase::TrickPlay;
            game.phase = GamePhase::TrickPlay;
            game.lead_player = (game.dealer + 1) % n;
        }
    }

    // ------------------------------------------------------------------
    // Talon Pick
    // ------------------------------------------------------------------

    fn talon_pick_step(game: &mut InFlightGame) -> Option<Pending> {
        let declarer = game.gs.declarer?;
        let contract = game.gs.contract?;
        let tc = contract.talon_cards() as usize;
        let n = game.gs.variant.num_players() as u8;
        if tc == 0 {
            game.gs.phase = Phase::TrickPlay;
            game.phase = GamePhase::TrickPlay;
            game.lead_player = (game.dealer + 1) % n;
            return Self::advance_until_decision(game);
        }

        let talon_cards = game.gs.talon.to_vec();
        let group_size = tc;
        let num_groups = talon_cards.len() / group_size.max(1);

        // Build revealed groups
        let groups: Vec<Vec<Card>> = (0..num_groups)
            .map(|i| talon_cards[i * group_size..(i + 1) * group_size].to_vec())
            .collect();
        game.gs.talon_revealed = groups;

        game.state_buf.fill(0.0);
        encode_state_dispatch(
            &mut game.state_buf,
            &game.gs,
            declarer,
            encoding::DT_TALON_PICK,
        );
        game.oracle_state_buf.fill(0.0);
        encode_oracle_state_dispatch(
            &mut game.oracle_state_buf,
            &game.gs,
            declarer,
            encoding::DT_TALON_PICK,
        );

        game.legal_mask.fill(0.0);
        for i in 0..num_groups.min(TALON_ACTION_SIZE) {
            game.legal_mask[i] = 1.0;
        }

        Some(Pending {
            slot: 0,
            player: declarer,
            decision_type: DecisionType::TalonPick,
            game_mode: Self::game_mode_id(game.gs.contract),
        })
    }

    fn apply_talon_pick(game: &mut InFlightGame, action_idx: usize, player: &Arc<dyn BatchPlayer>) {
        let declarer = match game.gs.declarer {
            Some(d) => d,
            None => return,
        };
        let contract = game.gs.contract.unwrap_or(Contract::Three);
        let tc = contract.talon_cards() as usize;
        if tc == 0 {
            return;
        }

        let talon_cards = game.gs.talon.to_vec();
        let group_size = tc;
        let num_groups = talon_cards.len() / group_size.max(1);

        let pick_idx = action_idx.min(num_groups.saturating_sub(1));
        game.trace.talon_pick = Some((declarer, pick_idx as u8));
        let start = pick_idx * group_size;
        let end = (start + group_size).min(talon_cards.len());
        let picked: Vec<Card> = talon_cards[start..end].to_vec();

        // Add picked cards to hand
        for &card in &picked {
            game.gs.hands[declarer as usize].insert(card);
        }
        // Remove from talon
        for &card in &picked {
            game.gs.talon.remove(card);
        }

        // Discard: let the player choose, or fall back to a generic heuristic
        if let Some(discards) = player.choose_discards(&game.gs, declarer, tc) {
            for card in &discards {
                game.gs.hands[declarer as usize].remove(*card);
                game.gs.put_down.insert(*card);
            }
        } else {
            // Default: discard low non-king, non-tarok cards
            let hand = game.gs.hands[declarer as usize];
            let mut discardable: Vec<Card> = hand
                .iter()
                .filter(|c| !c.is_king() && c.card_type() != CardType::Tarok)
                .collect();
            if discardable.len() < tc {
                discardable = hand.iter().filter(|c| !c.is_king()).collect();
            }
            discardable.sort_by_key(|c| c.0);
            for &card in discardable.iter().take(tc) {
                game.gs.hands[declarer as usize].remove(card);
                game.gs.put_down.insert(card);
            }
        }

        // Record discarded cards in trace
        game.trace.put_down = game.gs.put_down.iter().map(|c| c.0).collect();

        let n = game.gs.variant.num_players() as u8;
        game.gs.phase = Phase::TrickPlay;
        game.phase = GamePhase::TrickPlay;
        game.lead_player = (game.dealer + 1) % n;
    }

    // ------------------------------------------------------------------
    // Trick Play
    // ------------------------------------------------------------------

    fn trick_step(game: &mut InFlightGame) -> Option<Pending> {
        let total_tricks = game.gs.variant.tricks_per_game() as u8;
        let n = game.gs.variant.num_players() as u8;
        if game.trick_num >= total_tricks {
            game.phase = GamePhase::Done;
            return None;
        }

        if game.trick_offset == 0 {
            game.gs.start_trick(game.lead_player);
        }

        let player = (game.lead_player + game.trick_offset) % n;
        game.gs.current_player = player;

        game.state_buf.fill(0.0);
        encode_state_dispatch(
            &mut game.state_buf,
            &game.gs,
            player,
            encoding::DT_CARD_PLAY,
        );
        game.oracle_state_buf.fill(0.0);
        encode_oracle_state_dispatch(
            &mut game.oracle_state_buf,
            &game.gs,
            player,
            encoding::DT_CARD_PLAY,
        );

        let ctx = legal_moves::MoveCtx::from_state(&game.gs, player);
        let legal = legal_moves::generate_legal_moves(&ctx);

        game.legal_mask.fill(0.0);
        for card in legal.iter() {
            game.legal_mask[card.0 as usize] = 1.0;
        }

        Some(Pending {
            slot: 0,
            player,
            decision_type: DecisionType::CardPlay,
            game_mode: Self::game_mode_id(game.gs.contract),
        })
    }

    fn apply_trick_card(game: &mut InFlightGame, action_idx: usize) {
        let n = game.gs.variant.num_players() as u8;
        let total_tricks = game.gs.variant.tricks_per_game() as u8;
        let player = (game.lead_player + game.trick_offset) % n;
        let card = Card(action_idx as u8);

        let actual_card;
        if !game.gs.hands[player as usize].contains(card) {
            // Fallback: play first legal card
            actual_card = game.gs.hands[player as usize].iter().next().unwrap_or(card);
            game.gs.play_card(player, actual_card);
        } else {
            actual_card = card;
            game.gs.play_card(player, card);
        }
        game.trace.cards_played.push((player, actual_card.0));

        game.trick_offset += 1;

        if game.trick_offset >= n {
            let (winner, _points) = game.gs.finish_trick();
            game.lead_player = winner;
            game.trick_num += 1;
            game.trick_offset = 0;

            if game.trick_num >= total_tricks {
                game.phase = GamePhase::Done;
            }
        }
    }

    // ------------------------------------------------------------------
    // Action dispatch
    // ------------------------------------------------------------------

    fn apply_action(
        game: &mut InFlightGame,
        dt: DecisionType,
        action_idx: usize,
        player: &Arc<dyn BatchPlayer>,
    ) {
        match dt {
            DecisionType::Bid => Self::apply_bid(game, action_idx),
            DecisionType::KingCall => Self::apply_king_call(game, action_idx),
            DecisionType::TalonPick => Self::apply_talon_pick(game, action_idx, player),
            DecisionType::CardPlay => Self::apply_trick_card(game, action_idx),
        }
    }
}
