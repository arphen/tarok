/// Expert game generator using v5-only bots for training data.
///
/// Generates games where all four players use the v5 (strongest) heuristic
/// bot. This provides the purest expert demonstrations for imitation
/// learning from a single consistent play style.
///
/// Uses Rayon parallelism to process games across all CPU cores, since v5's
/// belief tracking (`CardTracker`) makes each game significantly more
/// expensive than simpler bot versions.

use rayon::prelude::*;
use rand::prelude::*;
use rand::rng;

use crate::card::*;
use crate::encoding;
use crate::game_state::*;
use crate::legal_moves;
use crate::scoring;
use crate::stockskis_v5;
use crate::trick_eval;

/// Batch result from expert game generation.
pub struct ExpertBatch {
    pub states: Vec<f32>,
    pub oracle_states: Vec<f32>,
    pub decision_types: Vec<u8>,
    pub actions: Vec<u16>,
    pub rewards: Vec<f32>,
    pub legal_masks: Vec<u8>,
    pub state_size: usize,
    pub oracle_state_size: usize,
}

struct ExpertExp {
    state: [f32; encoding::STATE_SIZE],
    oracle_state: [f32; encoding::ORACLE_STATE_SIZE],
    decision_type: u8,
    action: u16,
    player: u8,
}

/// Generate expert experiences from v5-only bot games.
///
/// Uses Rayon to parallelize across CPU cores. Each game is independent,
/// so results are generated per-game and merged at the end.
pub fn generate_expert_batch_v5(
    num_games: usize,
    include_oracle: bool,
) -> ExpertBatch {
    // Play all games in parallel, collecting per-game results
    let per_game_results: Vec<GameResult> = (0..num_games)
        .into_par_iter()
        .map(|_| play_single_game(include_oracle))
        .collect();

    // Merge all per-game results into a single flat batch.
    // Pre-compute total size to avoid reallocation.
    let total_exps: usize = per_game_results.iter().map(|g| g.rewards.len()).sum();

    let mut all_states = Vec::with_capacity(total_exps * encoding::STATE_SIZE);
    let mut all_oracle = Vec::with_capacity(if include_oracle {
        total_exps * encoding::ORACLE_STATE_SIZE
    } else {
        0
    });
    let mut all_dt = Vec::with_capacity(total_exps);
    let mut all_actions = Vec::with_capacity(total_exps);
    let mut all_rewards = Vec::with_capacity(total_exps);
    let mut all_masks = Vec::with_capacity(total_exps * DECK_SIZE);

    for g in per_game_results {
        all_states.extend(g.states);
        all_oracle.extend(g.oracle_states);
        all_dt.extend(g.decision_types);
        all_actions.extend(g.actions);
        all_rewards.extend(g.rewards);
        all_masks.extend(g.legal_masks);
    }

    ExpertBatch {
        states: all_states,
        oracle_states: all_oracle,
        decision_types: all_dt,
        actions: all_actions,
        rewards: all_rewards,
        legal_masks: all_masks,
        state_size: encoding::STATE_SIZE,
        oracle_state_size: encoding::ORACLE_STATE_SIZE,
    }
}

/// Per-game output collected during parallel execution.
struct GameResult {
    states: Vec<f32>,
    oracle_states: Vec<f32>,
    decision_types: Vec<u8>,
    actions: Vec<u16>,
    rewards: Vec<f32>,
    legal_masks: Vec<u8>,
}

/// Play one complete game with v5 bots and return flattened results.
fn play_single_game(include_oracle: bool) -> GameResult {
    let mut r = rng();
    let mut exps: Vec<ExpertExp> = Vec::with_capacity(60);
    let mut state = GameState::new(r.random_range(0..NUM_PLAYERS as u8));

    // Deal
    let mut deck = build_deck();
    deck.shuffle(&mut r);
    for (i, &card) in deck.iter().enumerate() {
        if i < 48 {
            state.hands[i / 12].insert(card);
        } else {
            state.talon.insert(card);
        }
    }
    state.phase = Phase::Bidding;

    let mut decision_masks: Vec<Vec<u8>> = Vec::with_capacity(60);

    // --- Bidding ---
    let mut highest: Option<Contract> = None;
    let mut winning_player: Option<u8> = None;
    let mut bidder = state.current_bidder;
    let forehand = state.forehand();

    for _seat in 0..NUM_PLAYERS {
        let is_fh = bidder == forehand;

        let mut exp_state = [0.0f32; encoding::STATE_SIZE];
        encoding::encode_state(&mut exp_state, &state, bidder, encoding::DT_BID);
        let mut exp_oracle = [0.0f32; encoding::ORACLE_STATE_SIZE];
        if include_oracle {
            encoding::encode_oracle_state(&mut exp_oracle, &state, bidder, encoding::DT_BID);
        }

        let bid_mask_arr = state.legal_bid_mask(bidder);
        let bid_mask: Vec<u8> = bid_mask_arr.to_vec();

        let bid = stockskis_v5::evaluate_bid_v5(state.hands[bidder as usize], highest)
            .filter(|&c| {
                if c == Contract::Three && !is_fh {
                    return false;
                }
                match highest {
                    Some(h) if is_fh => c.strength() >= h.strength(),
                    Some(h) => c.strength() > h.strength(),
                    None => true,
                }
            });
        let action: u16;

        match bid {
            Some(contract) => {
                action = Contract::BIDDABLE
                    .iter()
                    .position(|&c| c == contract)
                    .map(|i| (i + 1) as u16)
                    .unwrap_or(0);
                state.bids.push(Bid {
                    player: bidder,
                    contract: Some(contract),
                });
                highest = Some(contract);
                winning_player = Some(bidder);
            }
            None => {
                action = 0;
                state.bids.push(Bid {
                    player: bidder,
                    contract: None,
                });
            }
        }

        exps.push(ExpertExp {
            state: exp_state,
            oracle_state: exp_oracle,
            decision_type: encoding::DT_BID,
            action,
            player: bidder,
        });
        decision_masks.push(bid_mask);

        bidder = (bidder + 1) % NUM_PLAYERS as u8;
    }

    // Resolve bidding — forehand wins ties
    if let Some(h) = highest {
        let forehand_bid = state
            .bids
            .iter()
            .find(|b| b.player == forehand && b.contract == Some(h));
        if forehand_bid.is_some() {
            winning_player = Some(forehand);
        }
    }

    match (winning_player, highest) {
        (Some(p), Some(c)) => {
            state.declarer = Some(p);
            state.contract = Some(c);
            state.roles[p as usize] = PlayerRole::Declarer;
            for i in 0..NUM_PLAYERS {
                if i != p as usize {
                    state.roles[i] = PlayerRole::Opponent;
                }
            }
            if c.is_berac() {
                state.phase = Phase::TrickPlay;
                state.current_player = (state.dealer + 1) % NUM_PLAYERS as u8;
            } else if c.is_solo() {
                handle_talon(
                    &mut state,
                    &mut exps,
                    &mut decision_masks,
                    include_oracle,
                );
                state.phase = Phase::TrickPlay;
                state.current_player = (state.dealer + 1) % NUM_PLAYERS as u8;
            } else {
                handle_king_call(
                    &mut state,
                    &mut exps,
                    &mut decision_masks,
                    include_oracle,
                );
                handle_talon(
                    &mut state,
                    &mut exps,
                    &mut decision_masks,
                    include_oracle,
                );
                state.phase = Phase::TrickPlay;
                state.current_player = (state.dealer + 1) % NUM_PLAYERS as u8;
            }
        }
        _ => {
            state.contract = Some(Contract::Klop);
            state.phase = Phase::TrickPlay;
            for i in 0..NUM_PLAYERS {
                state.roles[i] = PlayerRole::Opponent;
            }
            state.current_player = (state.dealer + 1) % NUM_PLAYERS as u8;
        }
    }

    // --- Trick play ---
    let mut lead_player = state.current_player;
    for trick_num in 0..TRICKS_PER_GAME {
        state.current_trick = Some(Trick::new(lead_player));

        for offset in 0..4u8 {
            let player = (lead_player + offset) % NUM_PLAYERS as u8;
            state.current_player = player;

            let mut exp_state = [0.0f32; encoding::STATE_SIZE];
            encoding::encode_state(&mut exp_state, &state, player, encoding::DT_CARD_PLAY);
            let mut exp_oracle = [0.0f32; encoding::ORACLE_STATE_SIZE];
            if include_oracle {
                encoding::encode_oracle_state(
                    &mut exp_oracle,
                    &state,
                    player,
                    encoding::DT_CARD_PLAY,
                );
            }

            let ctx = legal_moves::MoveCtx::from_state(&state, player);
            let legal = legal_moves::generate_legal_moves(&ctx);
            let mut card_mask = vec![0u8; DECK_SIZE];
            for c in legal.iter() {
                card_mask[c.0 as usize] = 1;
            }

            let card = stockskis_v5::choose_card_v5(state.hands[player as usize], &state, player);
            let action = card.0 as u16;

            exps.push(ExpertExp {
                state: exp_state,
                oracle_state: exp_oracle,
                decision_type: encoding::DT_CARD_PLAY,
                action,
                player,
            });
            decision_masks.push(card_mask);

            state.hands[player as usize].remove(card);
            if let Some(ref mut trick) = state.current_trick {
                trick.play(player, card);
            }
            state.played_cards.insert(card);
        }

        let trick = state.current_trick.take().unwrap();
        let is_last = trick_num == TRICKS_PER_GAME - 1;
        let result = trick_eval::evaluate_trick(&trick, is_last, state.contract);
        lead_player = result.winner;
        state.tricks.push(trick);
    }

    state.phase = Phase::Scoring;
    let scores = scoring::score_game(&state);

    // Flatten into per-game result
    let n = exps.len();
    let mut states = Vec::with_capacity(n * encoding::STATE_SIZE);
    let mut oracle_states = Vec::with_capacity(if include_oracle {
        n * encoding::ORACLE_STATE_SIZE
    } else {
        0
    });
    let mut decision_types = Vec::with_capacity(n);
    let mut actions = Vec::with_capacity(n);
    let mut rewards = Vec::with_capacity(n);
    let mut legal_masks = Vec::new();

    debug_assert_eq!(exps.len(), decision_masks.len());
    for (exp, mask) in exps.iter().zip(decision_masks.iter()) {
        let reward = scores[exp.player as usize] as f32 / 100.0;
        decision_types.push(exp.decision_type);
        actions.push(exp.action);
        rewards.push(reward);
        states.extend_from_slice(&exp.state);
        legal_masks.extend(mask);
        if include_oracle {
            oracle_states.extend_from_slice(&exp.oracle_state);
        }
    }

    GameResult {
        states,
        oracle_states,
        decision_types,
        actions,
        rewards,
        legal_masks,
    }
}

// -----------------------------------------------------------------------
// King call + talon helpers
// -----------------------------------------------------------------------

fn handle_king_call(
    state: &mut GameState,
    exps: &mut Vec<ExpertExp>,
    masks: &mut Vec<Vec<u8>>,
    include_oracle: bool,
) {
    let declarer = state.declarer.unwrap();
    let hand = state.hands[declarer as usize];

    let king = match stockskis_v5::choose_king_v5(hand) {
        Some(k) => k,
        None => return,
    };

    let mut exp_state = [0.0f32; encoding::STATE_SIZE];
    encoding::encode_state(&mut exp_state, state, declarer, encoding::DT_KING_CALL);
    let mut exp_oracle = [0.0f32; encoding::ORACLE_STATE_SIZE];
    if include_oracle {
        encoding::encode_oracle_state(&mut exp_oracle, state, declarer, encoding::DT_KING_CALL);
    }

    let mut king_mask = vec![0u8; 4];
    for s in Suit::ALL {
        let k = Card::suit_card(s, SuitRank::King);
        if !hand.contains(k) {
            king_mask[s as usize] = 1;
        }
    }
    if king_mask.iter().all(|&m| m == 0) {
        for s in Suit::ALL {
            let q = Card::suit_card(s, SuitRank::Queen);
            if !hand.contains(q) {
                king_mask[s as usize] = 1;
            }
        }
    }

    let action = king.suit().map_or(0, |s| s as u16);
    exps.push(ExpertExp {
        state: exp_state,
        oracle_state: exp_oracle,
        decision_type: encoding::DT_KING_CALL,
        action,
        player: declarer,
    });
    masks.push(king_mask);

    state.called_king = Some(king);
    for p in 0..NUM_PLAYERS {
        if p != declarer as usize && state.hands[p].contains(king) {
            state.partner = Some(p as u8);
            state.roles[p] = PlayerRole::Partner;
            break;
        }
    }
}

fn handle_talon(
    state: &mut GameState,
    exps: &mut Vec<ExpertExp>,
    masks: &mut Vec<Vec<u8>>,
    include_oracle: bool,
) {
    let contract = state.contract.unwrap();
    let talon_cards = contract.talon_cards();
    if talon_cards == 0 {
        return;
    }

    let declarer = state.declarer.unwrap();
    let hand = state.hands[declarer as usize];

    let talon_vec: Vec<Card> = state.talon.iter().collect();
    let group_size = (6 / (6 / talon_cards as usize)).max(1);
    let mut groups: Vec<Vec<Card>> = Vec::new();
    for chunk in talon_vec.chunks(group_size) {
        groups.push(chunk.to_vec());
    }
    state.talon_revealed = groups.clone();

    let mut exp_state = [0.0f32; encoding::STATE_SIZE];
    encoding::encode_state(&mut exp_state, state, declarer, encoding::DT_TALON_PICK);
    let mut exp_oracle = [0.0f32; encoding::ORACLE_STATE_SIZE];
    if include_oracle {
        encoding::encode_oracle_state(&mut exp_oracle, state, declarer, encoding::DT_TALON_PICK);
    }

    let mut talon_mask = vec![0u8; 6];
    for i in 0..groups.len() {
        talon_mask[i] = 1;
    }

    let group_idx = stockskis_v5::choose_talon_group_v5(&groups, hand, state.called_king);
    let action = group_idx as u16;

    exps.push(ExpertExp {
        state: exp_state,
        oracle_state: exp_oracle,
        decision_type: encoding::DT_TALON_PICK,
        action,
        player: declarer,
    });
    masks.push(talon_mask);

    let picked = &groups[group_idx];
    for &card in picked {
        state.hands[declarer as usize].insert(card);
    }

    let discards = stockskis_v5::choose_discards_v5(
        state.hands[declarer as usize],
        talon_cards as usize,
        state.called_king,
    );
    for card in discards {
        state.hands[declarer as usize].remove(card);
        state.put_down.insert(card);
    }
}
