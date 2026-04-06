/// Warmup experience generator — plays random games and collects
/// (state, decision_type, reward) tuples for value network pre-training.
///
/// The key insight: even from random play, the VALUE FUNCTION can learn
/// which game states tend to lead to positive or negative outcomes.
/// This gives the critic a head start before self-play begins.

use rand::prelude::*;
use rand::rng;

use crate::card::*;
use crate::encoding;
use crate::game_state::*;
use crate::legal_moves;
use crate::trick_eval;

/// A single warmup experience: state encoding + metadata.
struct WarmupExp {
    state: [f32; encoding::STATE_SIZE],
    oracle_state: [f32; encoding::ORACLE_STATE_SIZE],
    decision_type: u8,
    player: u8,
}

/// Generate warmup experiences from random games.
///
/// Returns flattened arrays:
///   states:       (total_decisions, STATE_SIZE)
///   oracle_states: (total_decisions, ORACLE_STATE_SIZE)
///   decision_types: (total_decisions,)
///   rewards:      (total_decisions,)
///
/// Each decision point in each game produces one experience.
/// The reward is the final game score (divided by 100) for the player
/// who made that decision.
pub fn generate_warmup_batch(
    num_games: usize,
    include_oracle: bool,
) -> WarmupBatch {
    let mut r = rng();
    let mut all_states: Vec<f32> = Vec::new();
    let mut all_oracle: Vec<f32> = Vec::new();
    let mut all_dt: Vec<u8> = Vec::new();
    let mut all_rewards: Vec<f32> = Vec::new();

    for _ in 0..num_games {
        let mut exps = Vec::with_capacity(60); // ~48 card plays + bids
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

        // --- Bidding ---
        let mut passed = [false; NUM_PLAYERS];
        let mut highest: Option<Contract> = None;
        let mut winning_player: Option<u8> = None;
        let mut bidder = state.current_bidder;

        for _round in 0..20 {
            let active_count = passed.iter().filter(|&&p| !p).count();
            if active_count <= 1 && winning_player.is_some() {
                break;
            }
            if active_count == 0 {
                break;
            }

            // Record bid decision experience
            let mut exp_state = [0.0f32; encoding::STATE_SIZE];
            encoding::encode_state(&mut exp_state, &state, bidder, encoding::DT_BID);
            let mut exp_oracle = [0.0f32; encoding::ORACLE_STATE_SIZE];
            if include_oracle {
                encoding::encode_oracle_state(&mut exp_oracle, &state, bidder, encoding::DT_BID);
            }
            exps.push(WarmupExp {
                state: exp_state,
                oracle_state: exp_oracle,
                decision_type: encoding::DT_BID,
                player: bidder,
            });

            // Random bid: 50% pass, 50% bid something legal
            let do_pass = r.random_bool(0.6);
            if do_pass || highest == Some(Contract::Berac) {
                passed[bidder as usize] = true;
                state.bids.push(Bid { player: bidder, contract: None });
            } else {
                // Pick a random legal contract (stronger than current highest)
                let legal: Vec<Contract> = Contract::BIDDABLE
                    .iter()
                    .copied()
                    .filter(|c| highest.map_or(true, |h| c.strength() > h.strength()))
                    .collect();
                if legal.is_empty() {
                    passed[bidder as usize] = true;
                    state.bids.push(Bid { player: bidder, contract: None });
                } else {
                    let &choice = legal.choose(&mut r).unwrap();
                    state.bids.push(Bid { player: bidder, contract: Some(choice) });
                    highest = Some(choice);
                    winning_player = Some(bidder);
                }
            }

            // Next bidder
            loop {
                bidder = (bidder + 1) % NUM_PLAYERS as u8;
                if !passed[bidder as usize] {
                    break;
                }
                // Check if we've gone all the way around
                let active = passed.iter().filter(|&&p| !p).count();
                if active <= 1 {
                    break;
                }
            }
        }

        // Resolve bidding
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
                    // Solo: handle talon if needed, then trick play
                    handle_talon_random(&mut state, &mut r, &mut exps, include_oracle);
                    state.phase = Phase::TrickPlay;
                    state.current_player = (state.dealer + 1) % NUM_PLAYERS as u8;
                } else {
                    // Non-solo: king calling + talon
                    handle_king_call_random(&mut state, &mut r, &mut exps, include_oracle);
                    handle_talon_random(&mut state, &mut r, &mut exps, include_oracle);
                    state.phase = Phase::TrickPlay;
                    state.current_player = (state.dealer + 1) % NUM_PLAYERS as u8;
                }
            }
            _ => {
                // All passed → Klop
                state.contract = Some(Contract::Klop);
                state.phase = Phase::TrickPlay;
                for i in 0..NUM_PLAYERS {
                    state.roles[i] = PlayerRole::Opponent;
                }
                state.current_player = (state.dealer + 1) % NUM_PLAYERS as u8;
            }
        }

        // --- Skip announcements for warmup (random play) ---

        // --- Trick play ---
        let mut lead_player = state.current_player;
        for trick_num in 0..TRICKS_PER_GAME {
            state.current_trick = Some(Trick::new(lead_player));

            for offset in 0..4u8 {
                let player = (lead_player + offset) % NUM_PLAYERS as u8;
                state.current_player = player;

                // Record card play decision experience
                let mut exp_state = [0.0f32; encoding::STATE_SIZE];
                encoding::encode_state(&mut exp_state, &state, player, encoding::DT_CARD_PLAY);
                let mut exp_oracle = [0.0f32; encoding::ORACLE_STATE_SIZE];
                if include_oracle {
                    encoding::encode_oracle_state(
                        &mut exp_oracle, &state, player, encoding::DT_CARD_PLAY,
                    );
                }
                exps.push(WarmupExp {
                    state: exp_state,
                    oracle_state: exp_oracle,
                    decision_type: encoding::DT_CARD_PLAY,
                    player,
                });

                // Pick a random legal card
                let ctx = legal_moves::MoveCtx::from_state(&state, player);
                let legal = legal_moves::generate_legal_moves(&ctx);
                let legal_vec: Vec<Card> = legal.iter().collect();
                let &card = legal_vec.choose(&mut r).unwrap();

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

        // Score the game
        state.phase = Phase::Scoring;
        let scores = crate::scoring::score_game(&state);

        // Assign rewards to all experiences
        for exp in &exps {
            let reward = scores[exp.player as usize] as f32 / 100.0;
            all_dt.push(exp.decision_type);
            all_rewards.push(reward);
            all_states.extend_from_slice(&exp.state);
            if include_oracle {
                all_oracle.extend_from_slice(&exp.oracle_state);
            }
        }
    }

    WarmupBatch {
        states: all_states,
        oracle_states: all_oracle,
        decision_types: all_dt,
        rewards: all_rewards,
        state_size: encoding::STATE_SIZE,
        oracle_state_size: encoding::ORACLE_STATE_SIZE,
    }
}

/// Handle random king calling.
fn handle_king_call_random(
    state: &mut GameState,
    r: &mut impl Rng,
    exps: &mut Vec<WarmupExp>,
    include_oracle: bool,
) {
    let declarer = state.declarer.unwrap();
    let hand = state.hands[declarer as usize];

    // Find callable kings (kings NOT in declarer's hand)
    let mut callable = Vec::new();
    for s in Suit::ALL {
        let king = Card::suit_card(s, SuitRank::King);
        if !hand.contains(king) {
            callable.push(king);
        }
    }
    if callable.is_empty() {
        // Has all kings — use queens
        for s in Suit::ALL {
            let queen = Card::suit_card(s, SuitRank::Queen);
            if !hand.contains(queen) {
                callable.push(queen);
            }
        }
    }

    if !callable.is_empty() {
        // Record king call decision
        let mut exp_state = [0.0f32; encoding::STATE_SIZE];
        encoding::encode_state(&mut exp_state, state, declarer, encoding::DT_KING_CALL);
        let mut exp_oracle = [0.0f32; encoding::ORACLE_STATE_SIZE];
        if include_oracle {
            encoding::encode_oracle_state(&mut exp_oracle, state, declarer, encoding::DT_KING_CALL);
        }
        exps.push(WarmupExp {
            state: exp_state,
            oracle_state: exp_oracle,
            decision_type: encoding::DT_KING_CALL,
            player: declarer,
        });

        let &king = callable.choose(r).unwrap();
        state.called_king = Some(king);

        // Identify partner
        for p in 0..NUM_PLAYERS {
            if p != declarer as usize && state.hands[p].contains(king) {
                state.partner = Some(p as u8);
                state.roles[p] = PlayerRole::Partner;
                break;
            }
        }
    }
}

/// Handle random talon exchange.
fn handle_talon_random(
    state: &mut GameState,
    r: &mut impl Rng,
    exps: &mut Vec<WarmupExp>,
    include_oracle: bool,
) {
    let contract = state.contract.unwrap();
    let talon_cards = contract.talon_cards();
    if talon_cards == 0 {
        return;
    }

    let declarer = state.declarer.unwrap();

    // Reveal talon groups
    let talon_vec: Vec<Card> = state.talon.iter().collect();
    let group_size = (6 / (6 / talon_cards as usize)).max(1);
    let mut groups: Vec<Vec<Card>> = Vec::new();
    for chunk in talon_vec.chunks(group_size) {
        groups.push(chunk.to_vec());
    }
    state.talon_revealed = groups.clone();

    // Record talon pick decision
    let mut exp_state = [0.0f32; encoding::STATE_SIZE];
    encoding::encode_state(&mut exp_state, state, declarer, encoding::DT_TALON_PICK);
    let mut exp_oracle = [0.0f32; encoding::ORACLE_STATE_SIZE];
    if include_oracle {
        encoding::encode_oracle_state(&mut exp_oracle, state, declarer, encoding::DT_TALON_PICK);
    }
    exps.push(WarmupExp {
        state: exp_state,
        oracle_state: exp_oracle,
        decision_type: encoding::DT_TALON_PICK,
        player: declarer,
    });

    // Pick a random group
    let group_idx = r.random_range(0..groups.len());
    let picked = &groups[group_idx];
    for &card in picked {
        state.hands[declarer as usize].insert(card);
    }

    // Discard: pick cheapest non-king, non-tarok cards
    let hand = state.hands[declarer as usize];
    let mut discardable: Vec<Card> = hand
        .iter()
        .filter(|c| !c.is_king() && c.card_type() != CardType::Tarok)
        .collect();
    if discardable.len() < talon_cards as usize {
        // If not enough non-tarok non-king, allow taroks
        discardable = hand.iter().filter(|c| !c.is_king()).collect();
    }
    discardable.sort_by_key(|c| c.points());
    for &card in discardable.iter().take(talon_cards as usize) {
        state.hands[declarer as usize].remove(card);
        state.put_down.insert(card);
    }
}

/// Batch result from warmup generation.
pub struct WarmupBatch {
    pub states: Vec<f32>,
    pub oracle_states: Vec<f32>,
    pub decision_types: Vec<u8>,
    pub rewards: Vec<f32>,
    pub state_size: usize,
    pub oracle_state_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warmup_batch_produces_experiences() {
        let batch = generate_warmup_batch(10, false);
        // Each game has at least 48 card plays + some bids
        assert!(batch.rewards.len() >= 10 * 48, "Got {} exps", batch.rewards.len());
        assert_eq!(batch.states.len(), batch.rewards.len() * encoding::STATE_SIZE);
        assert_eq!(batch.decision_types.len(), batch.rewards.len());
        // Oracle should be empty when not requested
        assert!(batch.oracle_states.is_empty());
    }

    #[test]
    fn warmup_with_oracle() {
        let batch = generate_warmup_batch(5, true);
        assert_eq!(
            batch.oracle_states.len(),
            batch.rewards.len() * encoding::ORACLE_STATE_SIZE
        );
    }
}
