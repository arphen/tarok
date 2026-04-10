/// Expert game generator using v2, v3, and v5 bots for training data.
///
/// Generates games where v2, v3, and v5 bots play against each other in
/// rotation. This provides the strongest expert demonstrations for
/// imitation learning, mixing the conservative precision of v5 with the
/// more aggressive styles of v2 and v3.

use rand::prelude::*;
use rand::rng;

use crate::card::*;
use crate::encoding;
use crate::expert_games::ExpertBatch;
use crate::game_state::*;
use crate::legal_moves;
use crate::scoring;
use crate::stockskis_v2;
use crate::stockskis_v3;
use crate::stockskis_v5;
use crate::trick_eval;

struct ExpertExp {
    state: [f32; encoding::STATE_SIZE],
    oracle_state: [f32; encoding::ORACLE_STATE_SIZE],
    decision_type: u8,
    action: u16,
    player: u8,
}

/// Which bot version each player uses, cycling through v2/v3/v5.
#[derive(Clone, Copy)]
enum BotV {
    V2,
    V3,
    V5,
}

const BOT_CYCLE: [BotV; 3] = [BotV::V2, BotV::V3, BotV::V5];

/// Generate expert experiences from v2/v3/v5 bot games.
///
/// Games cycle through three configurations:
///   0: all-v2, 1: all-v3, 2: all-v5
/// This gives a balanced mix of play styles for the neural network to
/// learn from.
pub fn generate_expert_batch_v2v3v5(
    num_games: usize,
    include_oracle: bool,
) -> ExpertBatch {
    let mut r = rng();
    let mut all_states: Vec<f32> = Vec::new();
    let mut all_oracle: Vec<f32> = Vec::new();
    let mut all_dt: Vec<u8> = Vec::new();
    let mut all_actions: Vec<u16> = Vec::new();
    let mut all_rewards: Vec<f32> = Vec::new();
    let mut all_masks: Vec<u8> = Vec::new();

    for game_idx in 0..num_games {
        let bot = BOT_CYCLE[game_idx % 3];
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

            let bid = evaluate_bid(bot, state.hands[bidder as usize], highest)
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
                        bot,
                    );
                    state.phase = Phase::TrickPlay;
                    state.current_player = (state.dealer + 1) % NUM_PLAYERS as u8;
                } else {
                    handle_king_call(
                        &mut state,
                        &mut exps,
                        &mut decision_masks,
                        include_oracle,
                        bot,
                    );
                    handle_talon(
                        &mut state,
                        &mut exps,
                        &mut decision_masks,
                        include_oracle,
                        bot,
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

                let card = choose_card(bot, state.hands[player as usize], &state, player);
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

        debug_assert_eq!(exps.len(), decision_masks.len());
        for (exp, mask) in exps.iter().zip(decision_masks.iter()) {
            let reward = scores[exp.player as usize] as f32 / 100.0;
            all_dt.push(exp.decision_type);
            all_actions.push(exp.action);
            all_rewards.push(reward);
            all_states.extend_from_slice(&exp.state);
            all_masks.extend(mask);
            if include_oracle {
                all_oracle.extend_from_slice(&exp.oracle_state);
            }
        }
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

// -----------------------------------------------------------------------
// Bot dispatch helpers
// -----------------------------------------------------------------------

fn evaluate_bid(bot: BotV, hand: CardSet, highest: Option<Contract>) -> Option<Contract> {
    match bot {
        BotV::V2 => stockskis_v2::evaluate_bid_v2(hand, highest),
        BotV::V3 => stockskis_v3::evaluate_bid_v3(hand, highest),
        BotV::V5 => stockskis_v5::evaluate_bid_v5(hand, highest),
    }
}

fn choose_card(bot: BotV, hand: CardSet, state: &GameState, player: u8) -> Card {
    match bot {
        BotV::V2 => stockskis_v2::choose_card_v2(hand, state, player),
        BotV::V3 => stockskis_v3::choose_card_v3(hand, state, player),
        BotV::V5 => stockskis_v5::choose_card_v5(hand, state, player),
    }
}

fn choose_king(bot: BotV, hand: CardSet) -> Option<Card> {
    match bot {
        BotV::V2 => stockskis_v2::choose_king_v2(hand),
        BotV::V3 => stockskis_v3::choose_king_v3(hand),
        BotV::V5 => stockskis_v5::choose_king_v5(hand),
    }
}

fn choose_talon_group(bot: BotV, groups: &[Vec<Card>], hand: CardSet, called_king: Option<Card>) -> usize {
    match bot {
        BotV::V2 => stockskis_v2::choose_talon_group_v2(groups, hand, called_king),
        BotV::V3 => stockskis_v3::choose_talon_group_v3(groups, hand, called_king),
        BotV::V5 => stockskis_v5::choose_talon_group_v5(groups, hand, called_king),
    }
}

fn choose_discards(bot: BotV, hand: CardSet, n: usize, called_king: Option<Card>) -> Vec<Card> {
    match bot {
        BotV::V2 => stockskis_v2::choose_discards_v2(hand, n, called_king),
        BotV::V3 => stockskis_v3::choose_discards_v3(hand, n, called_king),
        BotV::V5 => stockskis_v5::choose_discards_v5(hand, n, called_king),
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
    bot: BotV,
) {
    let declarer = state.declarer.unwrap();
    let hand = state.hands[declarer as usize];

    let king = match choose_king(bot, hand) {
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
    bot: BotV,
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

    let group_idx = choose_talon_group(bot, &groups, hand, state.called_king);
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

    let discards = choose_discards(
        bot,
        state.hands[declarer as usize],
        talon_cards as usize,
        state.called_king,
    );
    for card in discards {
        state.hands[declarer as usize].remove(card);
        state.put_down.insert(card);
    }
}
