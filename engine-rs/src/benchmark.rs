/// Benchmark: StockŠkis v1 vs v2 vs v3 vs v4 head-to-head.
///
/// Plays thousands of games with different bot configurations and
/// reports win rates, average scores, and per-contract stats.

use rand::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

use crate::card::*;
use crate::game_state::*;
use crate::scoring;
use crate::stockskis;
use crate::stockskis_v2;
use crate::stockskis_v3;
use crate::stockskis_v4;
use crate::trick_eval;

/// Which bot version a player uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BotVersion {
    V1,
    V2,
    V3,
    V4,
}

impl std::fmt::Display for BotVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BotVersion::V1 => write!(f, "V1"),
            BotVersion::V2 => write!(f, "V2"),
            BotVersion::V3 => write!(f, "V3"),
            BotVersion::V4 => write!(f, "V4"),
        }
    }
}

/// Stats for a single bot version across all games in a config.
#[derive(Debug, Default)]
pub struct VersionStats {
    pub games: u32,
    pub wins: u32,
    pub total_score: i64,
    pub declarer_games: u32,
    pub declarer_wins: u32,
}

/// Per-contract stats.
#[derive(Debug, Default)]
struct ContractStats {
    played: u32,
    wins: u32,
    total_score: i64,
}

/// Run the full benchmark suite.
pub fn run_benchmark(num_games: usize) {
    let configs: Vec<(&str, [BotVersion; 4])> = vec![
        ("V1 vs V1 vs V1 vs V1", [BotVersion::V1, BotVersion::V1, BotVersion::V1, BotVersion::V1]),
        ("V2 vs V2 vs V2 vs V2", [BotVersion::V2, BotVersion::V2, BotVersion::V2, BotVersion::V2]),
        ("V3 vs V3 vs V3 vs V3", [BotVersion::V3, BotVersion::V3, BotVersion::V3, BotVersion::V3]),
        ("V4 vs V4 vs V4 vs V4", [BotVersion::V4, BotVersion::V4, BotVersion::V4, BotVersion::V4]),
        ("V2 vs V1 vs V1 vs V1", [BotVersion::V2, BotVersion::V1, BotVersion::V1, BotVersion::V1]),
        ("V1 vs V2 vs V2 vs V2", [BotVersion::V1, BotVersion::V2, BotVersion::V2, BotVersion::V2]),
        ("V3 vs V1 vs V1 vs V1", [BotVersion::V3, BotVersion::V1, BotVersion::V1, BotVersion::V1]),
        ("V1 vs V3 vs V3 vs V3", [BotVersion::V1, BotVersion::V3, BotVersion::V3, BotVersion::V3]),
        ("V3 vs V2 vs V2 vs V2", [BotVersion::V3, BotVersion::V2, BotVersion::V2, BotVersion::V2]),
        ("V2 vs V3 vs V3 vs V3", [BotVersion::V2, BotVersion::V3, BotVersion::V3, BotVersion::V3]),
        ("V3 vs V2 vs V1 vs V1", [BotVersion::V3, BotVersion::V2, BotVersion::V1, BotVersion::V1]),
        ("V4 vs V3 vs V3 vs V3", [BotVersion::V4, BotVersion::V3, BotVersion::V3, BotVersion::V3]),
        ("V3 vs V4 vs V4 vs V4", [BotVersion::V3, BotVersion::V4, BotVersion::V4, BotVersion::V4]),
        ("V4 vs V2 vs V2 vs V2", [BotVersion::V4, BotVersion::V2, BotVersion::V2, BotVersion::V2]),
        ("V2 vs V4 vs V4 vs V4", [BotVersion::V2, BotVersion::V4, BotVersion::V4, BotVersion::V4]),
        ("V4 vs V1 vs V1 vs V1", [BotVersion::V4, BotVersion::V1, BotVersion::V1, BotVersion::V1]),
        ("V1 vs V4 vs V4 vs V4", [BotVersion::V1, BotVersion::V4, BotVersion::V4, BotVersion::V4]),
    ];

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║      StockŠkis Bot Benchmark: V1 vs V2 vs V3 vs V4         ║");
    println!("║      {} games per configuration                          ║", num_games);
    println!("╚══════════════════════════════════════════════════════════════╝");

    // Aggregate stats across all configs
    let mut global_stats: HashMap<BotVersion, VersionStats> = HashMap::new();
    global_stats.insert(BotVersion::V1, VersionStats::default());
    global_stats.insert(BotVersion::V2, VersionStats::default());
    global_stats.insert(BotVersion::V3, VersionStats::default());
    global_stats.insert(BotVersion::V4, VersionStats::default());

    for (config_name, versions) in &configs {
        let t0 = Instant::now();
        let mut version_stats: HashMap<BotVersion, VersionStats> = HashMap::new();
        let mut contract_stats: HashMap<String, ContractStats> = HashMap::new();
        let mut errors = 0u32;

        for g in 0..num_games {
            let result = play_one_game(*versions, g);
            match result {
                Some((contract, declarer, scores)) => {
                    let contract_name = format!("{:?}", contract);

                    // Compute per-player score differentials (player - avg of others)
                    let score_diffs: [i64; 4] = std::array::from_fn(|p| {
                        let others_sum: i64 = (0..4).filter(|&o| o != p).map(|o| scores[o] as i64).sum();
                        scores[p] as i64 - others_sum / 3
                    });

                    for p in 0..4 {
                        let ver = versions[p];
                        let entry = version_stats.entry(ver).or_default();
                        entry.games += 1;
                        entry.total_score += score_diffs[p];
                        if scores[p] > 0 {
                            entry.wins += 1;
                        }
                        if Some(p as u8) == declarer {
                            entry.declarer_games += 1;
                            if scores[p] > 0 {
                                entry.declarer_wins += 1;
                            }
                        }

                        // Global stats
                        let gentry = global_stats.get_mut(&ver).unwrap();
                        gentry.games += 1;
                        gentry.total_score += score_diffs[p];
                        if scores[p] > 0 { gentry.wins += 1; }
                        if Some(p as u8) == declarer {
                            gentry.declarer_games += 1;
                            if scores[p] > 0 { gentry.declarer_wins += 1; }
                        }
                    }

                    // Contract stats for declarer
                    if let Some(decl) = declarer {
                        let key = format!("{}:{}", versions[decl as usize], contract_name);
                        let cs = contract_stats.entry(key).or_default();
                        cs.played += 1;
                        cs.total_score += scores[decl as usize] as i64;
                        if scores[decl as usize] > 0 { cs.wins += 1; }
                    }
                }
                None => {
                    errors += 1;
                }
            }
        }

        let elapsed = t0.elapsed();
        let completed = num_games as u32 - errors;
        let gps = completed as f64 / elapsed.as_secs_f64();

        println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  {} ({} games, {:.0} games/s, {:.2}s)", config_name, completed, gps, elapsed.as_secs_f64());
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        if errors > 0 {
            println!("  {} errors", errors);
        }

        println!("  {:<8} {:>6} {:>6} {:>8} {:>10} {:>10} {:>8}",
            "Bot", "Games", "Wins", "WinRate", "ScoreDiff", "DeclGames", "DeclWR");
        println!("  {}", "-".repeat(60));

        let mut sorted_versions: Vec<_> = version_stats.iter().collect();
        sorted_versions.sort_by_key(|(v, _)| match v {
            BotVersion::V1 => 1,
            BotVersion::V2 => 2,
            BotVersion::V3 => 3,
            BotVersion::V4 => 4,
        });

        for (ver, s) in &sorted_versions {
            let wr = if s.games > 0 { s.wins as f64 / s.games as f64 * 100.0 } else { 0.0 };
            let avg = if s.games > 0 { s.total_score as f64 / s.games as f64 } else { 0.0 };
            let dwr = if s.declarer_games > 0 { s.declarer_wins as f64 / s.declarer_games as f64 * 100.0 } else { 0.0 };
            println!("  {:<8} {:>6} {:>6} {:>7.1}% {:>+10.1} {:>10} {:>7.1}%",
                format!("{}", ver), s.games, s.wins, wr, avg, s.declarer_games, dwr);
        }
    }

    // Print global summary
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                     GLOBAL SUMMARY                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("  {:<8} {:>8} {:>8} {:>8} {:>10} {:>10} {:>8}",
        "Bot", "Games", "Wins", "WinRate", "ScoreDiff", "DeclGames", "DeclWR");
    println!("  {}", "-".repeat(64));

    for ver in [BotVersion::V1, BotVersion::V2, BotVersion::V3, BotVersion::V4] {
        let s = &global_stats[&ver];
        let wr = if s.games > 0 { s.wins as f64 / s.games as f64 * 100.0 } else { 0.0 };
        let avg = if s.games > 0 { s.total_score as f64 / s.games as f64 } else { 0.0 };
        let dwr = if s.declarer_games > 0 { s.declarer_wins as f64 / s.declarer_games as f64 * 100.0 } else { 0.0 };
        println!("  {:<8} {:>8} {:>8} {:>7.1}% {:>+10.1} {:>10} {:>7.1}%",
            format!("{}", ver), s.games, s.wins, wr, avg, s.declarer_games, dwr);
    }
    println!();
}

/// Play a single game with the given bot versions. Returns (contract, declarer, scores).
fn play_one_game(
    versions: [BotVersion; 4],
    game_idx: usize,
) -> Option<(Contract, Option<u8>, [i32; NUM_PLAYERS])> {
    let mut r = rand::rngs::SmallRng::seed_from_u64(game_idx as u64 * 7 + 42);
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
    state.phase = Phase::Bidding;

    // --- Bidding (single round, forehand last with priority) ---
    let mut highest: Option<Contract> = None;
    let mut winning_player: Option<u8> = None;
    let mut bidder = state.current_bidder;
    let forehand = state.forehand();

    for _seat in 0..NUM_PLAYERS {
        let is_fh = bidder == forehand;

        let bid = match versions[bidder as usize] {
            BotVersion::V1 => stockskis::evaluate_bid(state.hands[bidder as usize], highest),
            BotVersion::V2 => stockskis_v2::evaluate_bid_v2(state.hands[bidder as usize], highest),
            BotVersion::V3 => stockskis_v3::evaluate_bid_v3(state.hands[bidder as usize], highest),
            BotVersion::V4 => stockskis_v4::evaluate_bid_v4(state.hands[bidder as usize], highest),
        }
        .filter(|&c| {
            if c == Contract::Three && !is_fh { return false; }
            match highest {
                Some(h) if is_fh => c.strength() >= h.strength(),
                Some(h) => c.strength() > h.strength(),
                None => true,
            }
        });

        match bid {
            Some(contract) => {
                state.bids.push(Bid { player: bidder, contract: Some(contract) });
                highest = Some(contract);
                winning_player = Some(bidder);
            }
            None => {
                state.bids.push(Bid { player: bidder, contract: None });
            }
        }

        bidder = (bidder + 1) % NUM_PLAYERS as u8;
    }

    // Resolve bidding — forehand wins ties
    if let Some(h) = highest {
        let forehand_bid = state.bids.iter().find(|b| b.player == forehand && b.contract == Some(h));
        if forehand_bid.is_some() {
            winning_player = Some(forehand);
        }
    }

    // Resolve bidding
    match (winning_player, highest) {
        (Some(p), Some(c)) => {
            state.declarer = Some(p);
            state.contract = Some(c);
            state.roles[p as usize] = PlayerRole::Declarer;
            for i in 0..NUM_PLAYERS {
                if i != p as usize { state.roles[i] = PlayerRole::Opponent; }
            }
            if c.is_berac() {
                state.phase = Phase::TrickPlay;
                state.current_player = (state.dealer + 1) % NUM_PLAYERS as u8;
            } else if c.is_solo() {
                handle_talon(&mut state, versions);
                state.phase = Phase::TrickPlay;
                state.current_player = (state.dealer + 1) % NUM_PLAYERS as u8;
            } else {
                handle_king_call(&mut state, versions);
                handle_talon(&mut state, versions);
                state.phase = Phase::TrickPlay;
                state.current_player = (state.dealer + 1) % NUM_PLAYERS as u8;
            }
        }
        _ => {
            state.contract = Some(Contract::Klop);
            state.phase = Phase::TrickPlay;
            for i in 0..NUM_PLAYERS { state.roles[i] = PlayerRole::Opponent; }
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

            let card = match versions[player as usize] {
                BotVersion::V1 => stockskis::choose_card(state.hands[player as usize], &state, player),
                BotVersion::V2 => stockskis_v2::choose_card_v2(state.hands[player as usize], &state, player),
                BotVersion::V3 => stockskis_v3::choose_card_v3(state.hands[player as usize], &state, player),
                BotVersion::V4 => stockskis_v4::choose_card_v4(state.hands[player as usize], &state, player),
            };

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

    Some((state.contract.unwrap(), state.declarer, scores))
}

/// Handle king calling with the appropriate bot version.
fn handle_king_call(state: &mut GameState, versions: [BotVersion; 4]) {
    let declarer = state.declarer.unwrap();
    let hand = state.hands[declarer as usize];

    let king = match versions[declarer as usize] {
        BotVersion::V1 => stockskis::choose_king(hand),
        BotVersion::V2 => stockskis_v2::choose_king_v2(hand),
        BotVersion::V3 => stockskis_v3::choose_king_v3(hand),
        BotVersion::V4 => stockskis_v4::choose_king_v4(hand),
    };

    if let Some(king) = king {
        state.called_king = Some(king);
        for p in 0..NUM_PLAYERS {
            if p != declarer as usize && state.hands[p].contains(king) {
                state.partner = Some(p as u8);
                state.roles[p] = PlayerRole::Partner;
                break;
            }
        }
    }
}

/// Handle talon exchange with the appropriate bot version.
fn handle_talon(state: &mut GameState, versions: [BotVersion; 4]) {
    let contract = state.contract.unwrap();
    let talon_cards = contract.talon_cards();
    if talon_cards == 0 { return; }

    let declarer = state.declarer.unwrap();
    let hand = state.hands[declarer as usize];

    // Reveal talon groups
    let talon_vec: Vec<Card> = state.talon.iter().collect();
    let group_size = (6 / (6 / talon_cards as usize)).max(1);
    let mut groups: Vec<Vec<Card>> = Vec::new();
    for chunk in talon_vec.chunks(group_size) {
        groups.push(chunk.to_vec());
    }
    state.talon_revealed = groups.clone();

    let group_idx = match versions[declarer as usize] {
        BotVersion::V1 => stockskis::choose_talon_group(&groups, state.called_king),
        BotVersion::V2 => stockskis_v2::choose_talon_group_v2(&groups, hand, state.called_king),
        BotVersion::V3 => stockskis_v3::choose_talon_group_v3(&groups, hand, state.called_king),
        BotVersion::V4 => stockskis_v4::choose_talon_group_v4(&groups, hand, state.called_king),
    };

    // Pick the group
    let picked = &groups[group_idx];
    for &card in picked {
        state.hands[declarer as usize].insert(card);
        state.talon.remove(card);
    }

    // Put remaining talon cards into put_down (they go to declarer score pile)
    for card in state.talon.iter() {
        state.put_down.insert(card);
    }

    // Discard
    let discards = match versions[declarer as usize] {
        BotVersion::V1 => stockskis::choose_discards(
            state.hands[declarer as usize],
            talon_cards as usize,
            state.called_king,
        ),
        BotVersion::V2 => stockskis_v2::choose_discards_v2(
            state.hands[declarer as usize],
            talon_cards as usize,
            state.called_king,
        ),
        BotVersion::V3 => stockskis_v3::choose_discards_v3(
            state.hands[declarer as usize],
            talon_cards as usize,
            state.called_king,
        ),
        BotVersion::V4 => stockskis_v4::choose_discards_v4(
            state.hands[declarer as usize],
            talon_cards as usize,
            state.called_king,
        ),
    };

    for card in discards {
        state.hands[declarer as usize].remove(card);
        state.put_down.insert(card);
    }
}

/// Run a single config and return stats for player 0 only.
/// Used by Python to measure "solo player" performance against opponents.
pub fn run_eval_config(versions: [BotVersion; 4], num_games: usize) -> VersionStats {
    let mut stats = VersionStats::default();
    for g in 0..num_games {
        if let Some((_contract, declarer, scores)) = play_one_game(versions, g) {
            // Score differential: player 0 minus average of opponents
            let opp_avg = (scores[1] as i64 + scores[2] as i64 + scores[3] as i64) / 3;
            stats.games += 1;
            stats.total_score += scores[0] as i64 - opp_avg;
            if scores[0] > 0 { stats.wins += 1; }
            if declarer == Some(0) {
                stats.declarer_games += 1;
                if scores[0] > 0 { stats.declarer_wins += 1; }
            }
        }
    }
    stats
}
