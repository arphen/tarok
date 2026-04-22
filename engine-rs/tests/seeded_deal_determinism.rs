//! Duplicate-RL Phase 1: verify that seeded dealing is deterministic across
//! independent runs when the same `deck_seeds` list is supplied.
//!
//! Uses cheap heuristic bots in all four seats so no NN model is required.
//! The only thing we assert is that `initial_hands` and `initial_talon` are
//! byte-identical between the two runs for every game_id. Policy outputs may
//! still differ (the bots are allowed their own randomness), but the deal
//! itself — the thing duplicate pairing depends on — must be reproducible.

use std::sync::Arc;

use tarok_engine::player::BatchPlayer;
use tarok_engine::player_bot::try_make_bot_by_seat_label;
use tarok_engine::self_play::SelfPlayRunner;

fn make_bot_players(label: &str) -> [Arc<dyn BatchPlayer>; 4] {
    let bot = Arc::new(try_make_bot_by_seat_label(label).expect("bot seat label"));
    [bot.clone(), bot.clone(), bot.clone(), bot.clone()] as [Arc<dyn BatchPlayer>; 4]
}

#[test]
fn seeded_deals_are_deterministic_across_runs() {
    let seeds: Vec<u64> = (0..8).map(|i| 0xDEAD_BEEF_0000_0000u64 ^ i as u64).collect();
    let players_a = make_bot_players("bot_v5");
    let players_b = make_bot_players("bot_v5");

    let runner_a = SelfPlayRunner::new(players_a);
    let runner_b = SelfPlayRunner::new(players_b);

    let results_a = runner_a.run_with_deck_seeds(8, 4, Some(seeds.clone()));
    let results_b = runner_b.run_with_deck_seeds(8, 4, Some(seeds));

    assert_eq!(results_a.len(), 8);
    assert_eq!(results_b.len(), 8);

    // Sort both by game_id for a stable comparison (order isn't guaranteed
    // across the concurrent slot machinery).
    let mut a: Vec<_> = results_a.iter().collect();
    let mut b: Vec<_> = results_b.iter().collect();
    a.sort_by_key(|r| r.game_id);
    b.sort_by_key(|r| r.game_id);

    for (ga, gb) in a.iter().zip(b.iter()) {
        assert_eq!(ga.game_id, gb.game_id);
        assert_eq!(
            ga.initial_hands, gb.initial_hands,
            "hands must match for game {}",
            ga.game_id
        );
        assert_eq!(
            ga.initial_talon, gb.initial_talon,
            "talon must match for game {}",
            ga.game_id
        );
    }
}

#[test]
fn seeded_deals_differ_across_different_seeds() {
    let seeds_a: Vec<u64> = (0..4).map(|i| i as u64).collect();
    let seeds_b: Vec<u64> = (0..4).map(|i| (i as u64) ^ 0xFFFF_FFFF).collect();

    let runner_a = SelfPlayRunner::new(make_bot_players("bot_v5"));
    let runner_b = SelfPlayRunner::new(make_bot_players("bot_v5"));

    let ra = runner_a.run_with_deck_seeds(4, 2, Some(seeds_a));
    let rb = runner_b.run_with_deck_seeds(4, 2, Some(seeds_b));

    let mut same_hand_count = 0usize;
    for (a, b) in ra.iter().zip(rb.iter()) {
        if a.initial_hands == b.initial_hands {
            same_hand_count += 1;
        }
    }
    assert!(
        same_hand_count < ra.len(),
        "expected at least one different deal across different seed sets"
    );
}
