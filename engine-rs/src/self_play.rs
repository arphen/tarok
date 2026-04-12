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

use rand::Rng;

use crate::card::*;
use crate::encoding;
use crate::game_state::*;
use crate::legal_moves;
use crate::player::{
    BatchPlayer, DecisionContext, DecisionResult, DecisionType,
    BID_ACTION_SIZE, KING_ACTION_SIZE, TALON_ACTION_SIZE, CARD_ACTION_SIZE,
    BID_IDX_TO_CONTRACT, card_suit_idx,
};
use crate::scoring;

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
}

/// Internal tracking for a pending decision.
struct Pending {
    slot: usize,
    player: u8,
    decision_type: DecisionType,
    state_buf: Vec<f32>,
    legal_mask: Vec<f32>,
}

// -----------------------------------------------------------------------
// Experience record
// -----------------------------------------------------------------------

pub struct RawExperience {
    pub state: Vec<f32>,
    pub action: u16,
    pub log_prob: f32,
    pub value: f32,
    pub decision_type: u8,
    pub legal_mask: Vec<f32>,
    pub game_id: u32,
    pub step_in_game: u16,
    pub player: u8,
}

pub struct GameResult {
    pub game_id: u32,
    pub scores: [i32; 4],
    pub experiences: Vec<RawExperience>,
    // Arena metadata
    pub contract: u8,
    pub declarer: i8,
    pub partner: i8,
    pub bid_contracts: [i8; 4],
    pub taroks_in_hand: [u8; 4],
}

// -----------------------------------------------------------------------
// Batched self-play runner
// -----------------------------------------------------------------------

pub struct SelfPlayRunner {
    players: [Arc<dyn BatchPlayer>; 4],
}

impl SelfPlayRunner {
    pub fn new(players: [Arc<dyn BatchPlayer>; 4]) -> Self {
        SelfPlayRunner { players }
    }

    pub fn run(
        &self,
        n_games: u32,
        concurrency: usize,
    ) -> Vec<GameResult> {
        let mut rng = rand::rng();
        let mut results: Vec<GameResult> = Vec::with_capacity(n_games as usize);
        let mut slots: Vec<Option<InFlightGame>> = (0..concurrency).map(|_| None).collect();
        let mut next_game: u32 = 0;
        let mut active: usize = 0;
        let mut game_exps: Vec<Vec<RawExperience>> = Vec::new();

        // Seed initial games
        let n_initial = concurrency.min(n_games as usize);
        for i in 0..n_initial {
            slots[i] = Some(Self::new_game(next_game, &mut rng));
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
                                *game = Self::new_game(next_game, &mut rng);
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
            let mut all_results: Vec<DecisionResult> =
                vec![DecisionResult { action: 0, log_prob: 0.0, value: 0.0 }; pending.len()];

            for (uid, up) in unique_players.iter().enumerate() {
                let group: Vec<usize> = pending
                    .iter()
                    .enumerate()
                    .filter(|(_, pd)| seat_to_uid[pd.player as usize] == uid)
                    .map(|(i, _)| i)
                    .collect();

                if group.is_empty() {
                    continue;
                }

                let contexts: Vec<DecisionContext> = group
                    .iter()
                    .map(|&pi| {
                        let pd = &pending[pi];
                        let game = slots[pd.slot].as_ref().unwrap();
                        DecisionContext {
                            gs: game.gs.clone(),
                            player: pd.player,
                            decision_type: pd.decision_type,
                            legal_mask: pd.legal_mask.clone(),
                            state_encoding: pd.state_buf.clone(),
                        }
                    })
                    .collect();

                let batch_results = up.batch_decide(&contexts);

                for (i, &pi) in group.iter().enumerate() {
                    all_results[pi] = batch_results[i];
                }
            }

            // Step 3: record experiences and apply actions
            for (pi, pd) in pending.iter().enumerate() {
                let game = slots[pd.slot].as_mut().unwrap();
                let result = all_results[pi];

                let action_size = pd.decision_type.action_size();
                let gid = game.game_id as usize;
                game_exps[gid].push(RawExperience {
                    state: pd.state_buf.clone(),
                    action: result.action as u16,
                    log_prob: result.log_prob,
                    value: result.value,
                    decision_type: pd.decision_type as u8,
                    legal_mask: pd.legal_mask[..action_size].to_vec(),
                    game_id: game.game_id,
                    step_in_game: game.step_counter,
                    player: pd.player,
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
                    *game = Self::new_game(next_game, &mut rng);
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

    fn new_game(game_id: u32, rng: &mut impl Rng) -> InFlightGame {
        let dealer = (game_id % 4) as u8;
        let mut gs = GameState::new(dealer);
        gs.deal(rng);
        let first = (dealer + 1) % 4;
        let mut initial_taroks = [0u8; 4];
        for (pid, hand) in gs.hands.iter().enumerate() {
            initial_taroks[pid] = hand.tarok_count();
        }
        InFlightGame {
            gs,
            dealer,
            game_id,
            phase: GamePhase::Bid,
            passed: [false; 4],
            highest_bid: None,
            winning_bidder: None,
            current_bidder: first,
            bid_round: 0,
            trick_num: 0,
            trick_offset: 0,
            lead_player: first,
            step_counter: 0,
            initial_taroks,
            bid_choices: [-1i8; 4],
        }
    }

    fn score_game(gs: &GameState) -> [i32; 4] {
        scoring::score_game(gs)
    }

    fn build_result(game: &InFlightGame, exps: Vec<RawExperience>) -> GameResult {
        let scores = Self::score_game(&game.gs);
        let contract = game.gs.contract.map(|c| c as u8).unwrap_or(0);
        let declarer = game.gs.declarer.map(|d| d as i8).unwrap_or(-1);
        let partner = game.gs.partner.map(|p| p as i8).unwrap_or(-1);
        GameResult {
            game_id: game.game_id,
            scores,
            experiences: exps,
            contract,
            declarer,
            partner,
            bid_contracts: game.bid_choices,
            taroks_in_hand: game.initial_taroks,
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

        let mut state_buf = vec![0f32; encoding::STATE_SIZE];
        encoding::encode_state(&mut state_buf, &game.gs, bidder, encoding::DT_BID);

        let mut mask = vec![0f32; BID_ACTION_SIZE];
        mask[0] = 1.0; // pass always legal
        let legal = game.gs.legal_bids(bidder);
        for contract in &legal {
            for (idx, mapped) in BID_IDX_TO_CONTRACT.iter().enumerate() {
                if *mapped == Some(*contract) {
                    mask[idx] = 1.0;
                }
            }
        }

        Some(Pending {
            slot: 0,
            player: bidder,
            decision_type: DecisionType::Bid,
            state_buf,
            legal_mask: mask,
        })
    }

    fn apply_bid(game: &mut InFlightGame, action_idx: usize) {
        let bidder = game.current_bidder;
        let contract = if action_idx < BID_IDX_TO_CONTRACT.len() {
            BID_IDX_TO_CONTRACT[action_idx]
        } else {
            None
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
        for _ in 0..4 {
            game.current_bidder = (game.current_bidder + 1) % 4;
            if !game.passed[game.current_bidder as usize] {
                break;
            }
        }
    }

    fn resolve_bidding(game: &mut InFlightGame) {
        if let (Some(bidder), Some(contract)) = (game.winning_bidder, game.highest_bid) {
            game.gs.declarer = Some(bidder);
            game.gs.contract = Some(contract);
            game.gs.roles[bidder as usize] = PlayerRole::Declarer;
            for i in 0..4u8 {
                if i != bidder {
                    game.gs.roles[i as usize] = PlayerRole::Opponent;
                }
            }
            match contract {
                Contract::Berac | Contract::BarvniValat => {
                    game.gs.phase = Phase::TrickPlay;
                    game.phase = GamePhase::TrickPlay;
                    game.lead_player = (game.dealer + 1) % 4;
                }
                Contract::Klop => {
                    game.gs.phase = Phase::TrickPlay;
                    game.phase = GamePhase::TrickPlay;
                    game.lead_player = (game.dealer + 1) % 4;
                }
                _ if contract.is_solo() => {
                    let tc = contract.talon_cards();
                    if tc > 0 {
                        game.gs.phase = Phase::TalonExchange;
                        game.phase = GamePhase::TalonPick;
                    } else {
                        game.gs.phase = Phase::TrickPlay;
                        game.phase = GamePhase::TrickPlay;
                        game.lead_player = (game.dealer + 1) % 4;
                    }
                }
                _ => {
                    game.gs.phase = Phase::KingCalling;
                    game.phase = GamePhase::KingCall;
                }
            }
        } else {
            // Klop (nobody bid)
            game.gs.contract = Some(Contract::Klop);
            for i in 0..4 {
                game.gs.roles[i] = PlayerRole::Opponent;
            }
            game.gs.phase = Phase::TrickPlay;
            game.phase = GamePhase::TrickPlay;
            game.lead_player = (game.dealer + 1) % 4;
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

        let mut state_buf = vec![0f32; encoding::STATE_SIZE];
        encoding::encode_state(&mut state_buf, &game.gs, declarer, encoding::DT_KING_CALL);

        let mut mask = vec![0f32; KING_ACTION_SIZE];
        for &card in &callable {
            if let Some(suit_idx) = card_suit_idx(card.0) {
                mask[suit_idx] = 1.0;
            }
        }

        Some(Pending {
            slot: 0,
            player: declarer,
            decision_type: DecisionType::KingCall,
            state_buf,
            legal_mask: mask,
        })
    }

    fn apply_king_call(game: &mut InFlightGame, action_idx: usize) {
        let declarer = match game.gs.declarer {
            Some(d) => d,
            None => return,
        };
        let callable = game.gs.callable_kings();
        let chosen = callable
            .iter()
            .find(|c| card_suit_idx(c.0) == Some(action_idx))
            .copied()
            .unwrap_or(callable[0]);

        game.gs.called_king = Some(chosen);

        // Find partner
        for p in 0..4u8 {
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
        if tc > 0 {
            game.gs.phase = Phase::TalonExchange;
            game.phase = GamePhase::TalonPick;
        } else {
            game.gs.phase = Phase::TrickPlay;
            game.phase = GamePhase::TrickPlay;
            game.lead_player = (game.dealer + 1) % 4;
        }
    }

    // ------------------------------------------------------------------
    // Talon Pick
    // ------------------------------------------------------------------

    fn talon_pick_step(game: &mut InFlightGame) -> Option<Pending> {
        let declarer = game.gs.declarer?;
        let contract = game.gs.contract?;
        let tc = contract.talon_cards() as usize;
        if tc == 0 {
            game.gs.phase = Phase::TrickPlay;
            game.phase = GamePhase::TrickPlay;
            game.lead_player = (game.dealer + 1) % 4;
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

        let mut state_buf = vec![0f32; encoding::STATE_SIZE];
        encoding::encode_state(&mut state_buf, &game.gs, declarer, encoding::DT_TALON_PICK);

        let mut mask = vec![0f32; TALON_ACTION_SIZE];
        for i in 0..num_groups.min(TALON_ACTION_SIZE) {
            mask[i] = 1.0;
        }

        Some(Pending {
            slot: 0,
            player: declarer,
            decision_type: DecisionType::TalonPick,
            state_buf,
            legal_mask: mask,
        })
    }

    fn apply_talon_pick(
        game: &mut InFlightGame,
        action_idx: usize,
        player: &Arc<dyn BatchPlayer>,
    ) {
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

        game.gs.phase = Phase::TrickPlay;
        game.phase = GamePhase::TrickPlay;
        game.lead_player = (game.dealer + 1) % 4;
    }

    // ------------------------------------------------------------------
    // Trick Play
    // ------------------------------------------------------------------

    fn trick_step(game: &mut InFlightGame) -> Option<Pending> {
        if game.trick_num >= 12 {
            game.phase = GamePhase::Done;
            return None;
        }

        if game.trick_offset == 0 {
            game.gs.start_trick(game.lead_player);
        }

        let player = (game.lead_player + game.trick_offset) % 4;
        game.gs.current_player = player;

        let mut state_buf = vec![0f32; encoding::STATE_SIZE];
        encoding::encode_state(&mut state_buf, &game.gs, player, encoding::DT_CARD_PLAY);

        let ctx = legal_moves::MoveCtx::from_state(&game.gs, player);
        let legal = legal_moves::generate_legal_moves(&ctx);

        let mut mask = vec![0f32; CARD_ACTION_SIZE];
        for card in legal.iter() {
            mask[card.0 as usize] = 1.0;
        }

        Some(Pending {
            slot: 0,
            player,
            decision_type: DecisionType::CardPlay,
            state_buf,
            legal_mask: mask,
        })
    }

    fn apply_trick_card(game: &mut InFlightGame, action_idx: usize) {
        let player = (game.lead_player + game.trick_offset) % 4;
        let card = Card(action_idx as u8);

        if !game.gs.hands[player as usize].contains(card) {
            // Fallback: play first legal card
            if let Some(first) = game.gs.hands[player as usize].iter().next() {
                game.gs.play_card(player, first);
            }
        } else {
            game.gs.play_card(player, card);
        }

        game.trick_offset += 1;

        if game.trick_offset >= 4 {
            let (winner, _points) = game.gs.finish_trick();
            game.lead_player = winner;
            game.trick_num += 1;
            game.trick_offset = 0;

            if game.trick_num >= 12 {
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
