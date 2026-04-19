use rand::prelude::*;
use rand::rng;
/// Lightweight arena runner for bot-vs-bot mass simulation.
///
/// Unlike [`SelfPlayRunner`], this module does NOT collect training
/// experiences — it only records per-game scores and metadata (contract,
/// declarer, partner).  This keeps memory usage constant regardless of
/// the number of games, making it suitable for 100K+ game arenas.
///
/// Uses Rayon parallelism: each game is fully independent, so all CPU
/// cores are utilised without any synchronisation overhead.
use rayon::prelude::*;

use crate::card::*;
use crate::game_state::*;
use crate::scoring;
use crate::stockskis_v5;
use crate::stockskis_v6;
use crate::trick_eval;

/// Per-bot version dispatch tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BotVersion {
    V5,
    V6,
}

/// Lightweight result for a single arena game.
pub struct ArenaGameResult {
    pub scores: [i32; 4],
    pub contract: u8,            // Contract as u8 (0=Klop..9=BarvniValat)
    pub declarer: i8,            // -1 if nobody (Klop)
    pub partner: i8,             // -1 if none
    pub bid_contracts: [i8; 4],  // Per-seat chosen bid in this auction (-1=pass)
    pub taroks_in_hand: [u8; 4], // Tarok count in dealt 12-card hand
}

/// Run `n_games` arena games in parallel using Rayon.
///
/// `versions` specifies the bot version for each of the 4 seats.
pub fn run_arena_batch(n_games: u32, versions: [BotVersion; 4]) -> Vec<ArenaGameResult> {
    (0..n_games)
        .into_par_iter()
        .map(|game_idx| play_one_game(game_idx, &versions))
        .collect()
}

// -----------------------------------------------------------------------
// Helpers — pick bid / king / talon / card for a given bot version
// -----------------------------------------------------------------------

fn bot_bid(v: BotVersion, hand: CardSet, highest: Option<Contract>) -> Option<Contract> {
    match v {
        BotVersion::V5 => stockskis_v5::evaluate_bid_v5(hand, highest),
        BotVersion::V6 => stockskis_v6::evaluate_bid_v6(hand, highest),
    }
}

fn bot_king(v: BotVersion, hand: CardSet) -> Option<Card> {
    match v {
        BotVersion::V5 => stockskis_v5::choose_king_v5(hand),
        BotVersion::V6 => stockskis_v6::choose_king_v6(hand),
    }
}

fn bot_talon(
    v: BotVersion,
    groups: &[Vec<Card>],
    hand: CardSet,
    called_king: Option<Card>,
) -> usize {
    match v {
        BotVersion::V5 => stockskis_v5::choose_talon_group_v5(groups, hand, called_king),
        BotVersion::V6 => stockskis_v6::choose_talon_group_v6(groups, hand, called_king),
    }
}

fn bot_discards(v: BotVersion, hand: CardSet, n: usize, called_king: Option<Card>) -> Vec<Card> {
    match v {
        BotVersion::V5 => stockskis_v5::choose_discards_v5(hand, n, called_king),
        BotVersion::V6 => stockskis_v6::choose_discards_v6(hand, n, called_king),
    }
}

fn bot_card(v: BotVersion, hand: CardSet, state: &GameState, player: u8) -> Card {
    match v {
        BotVersion::V5 => stockskis_v5::choose_card_v5(hand, state, player),
        BotVersion::V6 => stockskis_v6::choose_card_v6(hand, state, player),
    }
}

// -----------------------------------------------------------------------
// Play one complete game
// -----------------------------------------------------------------------

fn play_one_game(game_idx: u32, versions: &[BotVersion; 4]) -> ArenaGameResult {
    let mut r = rng();
    let dealer = (game_idx % 4) as u8;
    let mut state = GameState::new(dealer);

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
    let mut taroks_in_hand = [0u8; 4];
    for (pid, hand) in state.hands.iter().enumerate() {
        taroks_in_hand[pid] = hand.tarok_count();
    }
    state.phase = Phase::Bidding;

    // --- Bidding ---
    let mut highest: Option<Contract> = None;
    let mut winning_player: Option<u8> = None;
    let mut bid_contracts = [-1i8; 4];
    let mut bidder = state.current_bidder;
    let forehand = state.forehand();

    for _ in 0..NUM_PLAYERS {
        let is_fh = bidder == forehand;
        let bid = bot_bid(
            versions[bidder as usize],
            state.hands[bidder as usize],
            highest,
        )
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

        match bid {
            Some(contract) => {
                state.bids.push(Bid {
                    player: bidder,
                    contract: Some(contract),
                });
                bid_contracts[bidder as usize] = contract as i8;
                highest = Some(contract);
                winning_player = Some(bidder);
            }
            None => {
                state.bids.push(Bid {
                    player: bidder,
                    contract: None,
                });
            }
        }

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

    let mut result_declarer: i8 = -1;
    let mut result_partner: i8 = -1;
    let mut result_contract: u8 = 0; // Klop

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
            result_declarer = p as i8;
            result_contract = c as u8;

            if c.is_berac() || c.is_barvni_valat() {
                // Skip to trick play
            } else if c.is_solo() {
                handle_talon(&mut state, versions);
            } else {
                handle_king_call(&mut state, versions);
                if state.partner.is_some() {
                    result_partner = state.partner.unwrap() as i8;
                }
                handle_talon(&mut state, versions);
            }

            state.phase = Phase::TrickPlay;
            state.current_player = (state.dealer + 1) % NUM_PLAYERS as u8;
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

            let card = bot_card(
                versions[player as usize],
                state.hands[player as usize],
                &state,
                player,
            );

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

    ArenaGameResult {
        scores,
        contract: result_contract,
        declarer: result_declarer,
        partner: result_partner,
        bid_contracts,
        taroks_in_hand,
    }
}

// -----------------------------------------------------------------------
// King call
// -----------------------------------------------------------------------

fn handle_king_call(state: &mut GameState, versions: &[BotVersion; 4]) {
    let declarer = match state.declarer {
        Some(d) => d,
        None => return,
    };
    let hand = state.hands[declarer as usize];
    let king = match bot_king(versions[declarer as usize], hand) {
        Some(k) => k,
        None => return,
    };

    state.called_king = Some(king);
    for p in 0..NUM_PLAYERS {
        if p != declarer as usize && state.hands[p].contains(king) {
            state.partner = Some(p as u8);
            state.roles[p] = PlayerRole::Partner;
            break;
        }
    }
}

// -----------------------------------------------------------------------
// Talon exchange
// -----------------------------------------------------------------------

fn handle_talon(state: &mut GameState, versions: &[BotVersion; 4]) {
    let contract = match state.contract {
        Some(c) => c,
        None => return,
    };
    let talon_cards = contract.talon_cards() as usize;
    if talon_cards == 0 {
        return;
    }

    let declarer = match state.declarer {
        Some(d) => d,
        None => return,
    };
    let hand = state.hands[declarer as usize];

    let talon_vec: Vec<Card> = state.talon.iter().collect();
    let group_size = (6 / (6 / talon_cards)).max(1);
    let mut groups: Vec<Vec<Card>> = Vec::new();
    for chunk in talon_vec.chunks(group_size) {
        groups.push(chunk.to_vec());
    }
    state.talon_revealed = groups.clone();

    let group_idx = bot_talon(
        versions[declarer as usize],
        &groups,
        hand,
        state.called_king,
    );
    let picked = &groups[group_idx.min(groups.len().saturating_sub(1))];
    for &card in picked {
        state.hands[declarer as usize].insert(card);
    }

    let discards = bot_discards(
        versions[declarer as usize],
        state.hands[declarer as usize],
        talon_cards,
        state.called_king,
    );
    for card in discards {
        state.hands[declarer as usize].remove(card);
        state.put_down.insert(card);
    }
}
