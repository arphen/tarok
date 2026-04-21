//! Focused comparison: PIMC vs alpha-mu on the SAME endgame positions.
//!
//! This isolates the two endgame solvers from any surrounding bot heuristic
//! (m6, lustrek, etc).  Each trial:
//!
//!   1. Deals 48 cards (12 per player, no talon exchange).
//!   2. Sets roles + contract directly (declarer + partner vs 2 opponents,
//!      contract = Three).
//!   3. Plays tricks 1..=8 with uniform-random legal moves (seeded RNG, same
//!      for both branches → identical endgame snapshot).
//!   4. Plays tricks 9..=12 TWICE from that snapshot:
//!        - Branch P: declarer seat uses `pimc_choose_card`
//!        - Branch A: declarer seat uses `alpha_mu_choose_card`
//!      Non-declarer seats play uniform-random legal moves with an RNG
//!      seeded identically at the snapshot, so their move *distributions*
//!      are matched (exact plays can still diverge once declarer diverges,
//!      but the input stream of randomness is the same).
//!   5. Records declarer-team raw card points from the last 4 tricks.
//!
//! Run with:
//!
//! ```bash
//! cargo test --release -p tarok-engine \
//!     --test endgame_pimc_vs_alpha_mu -- --nocapture
//! ```
//!
//! Tweak NUM_TRIALS / NUM_WORLDS / ALPHA_MU_DEPTH via env vars:
//!
//! ```bash
//! ENDGAME_TRIALS=400 ENDGAME_WORLDS=100 ENDGAME_ALPHA_DEPTH=2 \
//!   cargo test --release -p tarok-engine \
//!     --test endgame_pimc_vs_alpha_mu -- --nocapture
//! ```

use rand::prelude::*;
use rand::rngs::SmallRng;
use rand::SeedableRng;

use tarok_engine::alpha_mu::alpha_mu_choose_card;
use tarok_engine::card::*;
use tarok_engine::game_state::*;
use tarok_engine::legal_moves;
use tarok_engine::pimc::pimc_choose_card;
use tarok_engine::trick_eval;

const DEFAULT_TRIALS: u32 = 120;
const DEFAULT_WORLDS: u32 = 100;
const DEFAULT_ALPHA_DEPTH: usize = 2;
const SNAPSHOT_TRICK: usize = 8; // run solvers for tricks 9..=12

// Only test team-vs-team contracts so declarer/opponent value signal is
// meaningful for the solver.
const CONTRACT: Contract = Contract::Three;

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

/// Deal a fresh 48-card game, 12 per player, no talon.
fn deal_random<R: Rng>(rng: &mut R) -> [CardSet; NUM_PLAYERS] {
    let mut deck: Vec<Card> = build_deck().to_vec();
    deck.shuffle(rng);
    let mut hands = [CardSet::EMPTY; NUM_PLAYERS];
    for (i, &c) in deck.iter().enumerate().take(48) {
        hands[i / 12].insert(c);
    }
    hands
}

/// Pick a uniformly-random legal card for `player` in `gs`.
fn random_legal<R: Rng>(gs: &GameState, player: u8, rng: &mut R) -> Card {
    let ctx = legal_moves::MoveCtx::from_state(gs, player);
    let legal = legal_moves::generate_legal_moves(&ctx);
    let v: Vec<Card> = legal.iter().collect();
    assert!(!v.is_empty(), "no legal moves for player {player}");
    v[rng.random_range(0..v.len())]
}

/// Apply one played card, finishing the trick (and rotating lead) if complete.
/// Returns the (winner, points) if a trick just finished.
fn apply_card(gs: &mut GameState, player: u8, card: Card) -> Option<(u8, u8)> {
    gs.hands[player as usize].remove(card);
    gs.played_cards.insert(card);
    if let Some(ref mut t) = gs.current_trick {
        t.play(player, card);
    }

    let complete = gs.current_trick.as_ref().map(|t| t.is_complete()).unwrap_or(false);
    if !complete {
        gs.current_player = (player + 1) % NUM_PLAYERS as u8;
        return None;
    }

    // Finish the trick
    let trick = gs.current_trick.take().expect("current trick");
    let is_last = gs.tricks.len() == 11;
    let result = trick_eval::evaluate_trick(&trick, is_last, gs.contract);
    gs.tricks.push(trick);

    let winner = result.winner;
    let points = result.points;

    if gs.tricks.len() < TRICKS_PER_GAME {
        gs.current_trick = Some(Trick::new(winner));
        gs.current_player = winner;
    }
    Some((winner, points))
}

/// Build an initial game state ready for trick play, with random deal +
/// fixed roles.
///
/// - dealer = 0, forehand (and declarer) = 1, partner = 3, opponents = 0 & 2
/// - contract = Three
fn build_initial_state<R: Rng>(rng: &mut R) -> GameState {
    let mut gs = GameState::new(0);
    gs.hands = deal_random(rng);
    gs.contract = Some(CONTRACT);
    gs.declarer = Some(1);
    gs.partner = Some(3);
    gs.roles[0] = PlayerRole::Opponent;
    gs.roles[1] = PlayerRole::Declarer;
    gs.roles[2] = PlayerRole::Opponent;
    gs.roles[3] = PlayerRole::Partner;
    gs.phase = Phase::TrickPlay;
    let lead = gs.forehand(); // = 1
    gs.current_trick = Some(Trick::new(lead));
    gs.current_player = lead;
    gs
}

/// Play tricks until `gs.tricks.len() == up_to_trick`, using uniform-random
/// legal moves for every seat.
fn play_until<R: Rng>(gs: &mut GameState, up_to_trick: usize, rng: &mut R) {
    while gs.tricks.len() < up_to_trick {
        let p = gs.current_player;
        let c = random_legal(gs, p, rng);
        apply_card(gs, p, c);
    }
}

/// Finish the game (tricks 9..=12) using `solver` for the declarer seat and
/// random-legal for every other seat.  Returns declarer-team raw points from
/// the tricks played *after* entering this function.
fn finish_with<R, F>(gs: &mut GameState, declarer: u8, mut solver: F, rng: &mut R) -> i32
where
    R: Rng,
    F: FnMut(&GameState, u8) -> Card,
{
    let mut dec_points: i32 = 0;
    while gs.tricks.len() < TRICKS_PER_GAME {
        let p = gs.current_player;
        let card = if p == declarer {
            solver(gs, p)
        } else {
            random_legal(gs, p, rng)
        };
        if let Some((winner, pts)) = apply_card(gs, p, card) {
            if gs.get_team(winner) == Team::DeclarerTeam {
                dec_points += pts as i32;
            }
        }
    }
    dec_points
}

#[test]
fn pimc_vs_alpha_mu_endgame_comparison() {
    let num_trials = env_u32("ENDGAME_TRIALS", DEFAULT_TRIALS);
    let num_worlds = env_u32("ENDGAME_WORLDS", DEFAULT_WORLDS);
    let alpha_depth = env_usize("ENDGAME_ALPHA_DEPTH", DEFAULT_ALPHA_DEPTH);
    let base_seed: u64 = std::env::var("ENDGAME_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0xC0FFEE_u64);

    println!(
        "\nPIMC vs alpha-mu endgame comparison"
    );
    println!(
        "  trials={num_trials}  worlds={num_worlds}  alpha_depth={alpha_depth}  \
         snapshot_trick={SNAPSHOT_TRICK}  seed=0x{base_seed:x}"
    );

    let declarer: u8 = 1;

    let mut pimc_sum: i64 = 0;
    let mut alpha_sum: i64 = 0;
    let mut pimc_wins: u32 = 0;
    let mut alpha_wins: u32 = 0;
    let mut ties: u32 = 0;
    let mut valid_trials: u32 = 0;
    let mut diffs: Vec<i32> = Vec::with_capacity(num_trials as usize);

    for trial in 0..num_trials {
        // Separate RNG streams, all derived from trial+base_seed:
        //   setup_rng     — dealing + tricks 1..=8 (shared by both branches)
        //   opp_rng       — opponent random picks during the endgame
        //                   (reseeded identically before each branch)
        let trial_seed = base_seed.wrapping_add(trial as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut setup_rng = SmallRng::seed_from_u64(trial_seed);

        let mut gs0 = build_initial_state(&mut setup_rng);
        play_until(&mut gs0, SNAPSHOT_TRICK, &mut setup_rng);

        // Abort trials where the declarer happens to be out of cards before
        // the endgame is reached (shouldn't happen with 12/hand + trick=8,
        // but defend against it).
        if gs0.tricks.len() != SNAPSHOT_TRICK {
            continue;
        }

        let opp_seed = trial_seed ^ 0xDEAD_BEEF_CAFE_F00D_u64;

        // --- Branch P: PIMC ---
        let mut gs_p = gs0.clone();
        let mut opp_rng_p = SmallRng::seed_from_u64(opp_seed);
        let p_points = finish_with(
            &mut gs_p,
            declarer,
            |s, p| pimc_choose_card(s, p, num_worlds),
            &mut opp_rng_p,
        );

        // --- Branch A: alpha-mu ---
        let mut gs_a = gs0.clone();
        let mut opp_rng_a = SmallRng::seed_from_u64(opp_seed);
        let a_points = finish_with(
            &mut gs_a,
            declarer,
            |s, p| alpha_mu_choose_card(s, p, num_worlds, alpha_depth),
            &mut opp_rng_a,
        );

        valid_trials += 1;
        pimc_sum += p_points as i64;
        alpha_sum += a_points as i64;
        let diff = a_points - p_points;
        diffs.push(diff);

        match diff.cmp(&0) {
            std::cmp::Ordering::Greater => alpha_wins += 1,
            std::cmp::Ordering::Less => pimc_wins += 1,
            std::cmp::Ordering::Equal => ties += 1,
        }
    }

    assert!(valid_trials > 0, "no valid trials ran");

    let n = valid_trials as f64;
    let pimc_mean = pimc_sum as f64 / n;
    let alpha_mean = alpha_sum as f64 / n;
    let mean_diff = alpha_mean - pimc_mean;
    let var_diff = diffs
        .iter()
        .map(|&d| {
            let x = d as f64 - mean_diff;
            x * x
        })
        .sum::<f64>()
        / n;
    let std_diff = var_diff.sqrt();
    let se_diff = if n > 1.0 { std_diff / n.sqrt() } else { 0.0 };

    println!("\n--- Results ({valid_trials} valid trials, declarer-team raw points over last 4 tricks) ---");
    println!("  PIMC       mean = {pimc_mean:+7.2}");
    println!("  alpha-mu   mean = {alpha_mean:+7.2}");
    println!("  alpha-mu − PIMC = {mean_diff:+7.2}  (std={std_diff:.2}, se={se_diff:.2})");
    println!("  win counts:  alpha-mu={alpha_wins}  pimc={pimc_wins}  ties={ties}");

    // Informational only — don't fail the test on ordering.  This test exists
    // to surface the gap, not to gate CI.
    // If you want a hard regression assertion, add one here.
}
