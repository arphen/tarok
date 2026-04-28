//! Phase A foundation tests for 3-player Tarok ("tarok v treh").
//!
//! These tests exercise the variant-aware infrastructure added for the new
//! game mode:
//!   * `Variant` enum + `Variant::*` constants
//!   * `GameState::new_with_variant`
//!   * Variant-aware `deal()` (16/16/16 + 6 talon)
//!   * Variant-aware `legal_bids()` (no Three/Two/One/Solo; Valat & graded
//!     solos available)
//!   * `Contract::Valat` is biddable in 3p only
//!   * `Contract::base_value_for(Variant::ThreePlayer)` returns the 3p table
//!
//! Note: these tests share a single test process with the rest of the
//! engine's cargo test suite via `--test`, but each integration test file
//! gets its own binary, so the process-global variant guard is fresh here.
//! We always reset before locking ThreePlayer to be defensive.

use rand::SeedableRng;
use rand::rngs::StdRng;
use std::sync::{Mutex, MutexGuard};
use tarok_engine::game_state::{
    Contract, GameState, Variant, reset_process_variant_for_tests,
};

// All tests in this file mutate the process-global variant guard, so we
// serialize them with a single mutex. The mutex guard is held for the
// lifetime of each test by being returned from the helper / acquired at the
// top of each test fn.
static GUARD_TEST_LOCK: Mutex<()> = Mutex::new(());

fn acquire_guard() -> MutexGuard<'static, ()> {
    let g = GUARD_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    reset_process_variant_for_tests();
    g
}

fn fresh_three_player(dealer: u8) -> GameState {
    GameState::new_with_variant(Variant::ThreePlayer, dealer)
}

#[test]
fn variant_constants_three_player() {
    let _g = acquire_guard();
    assert_eq!(Variant::ThreePlayer.num_players(), 3);
    assert_eq!(Variant::ThreePlayer.hand_size(), 16);
    assert_eq!(Variant::ThreePlayer.talon_size(), 6);
    assert_eq!(Variant::ThreePlayer.tricks_per_game(), 16);
    assert!(!Variant::ThreePlayer.has_king_call());
    assert!(Variant::ThreePlayer.is_three_player());

    assert_eq!(Variant::FourPlayer.num_players(), 4);
    assert_eq!(Variant::FourPlayer.hand_size(), 12);
    assert_eq!(Variant::FourPlayer.talon_size(), 6);
    assert_eq!(Variant::FourPlayer.tricks_per_game(), 12);
    assert!(Variant::FourPlayer.has_king_call());
    assert!(!Variant::FourPlayer.is_three_player());
}

#[test]
fn three_player_biddable_contracts_match_spec() {
    let _g = acquire_guard();
    let biddable = Variant::ThreePlayer.biddable_contracts();
    // Klop is implicit (all-pass), Valat/BarvniValat are routed via
    // announcement-style outcomes, so the active 3p bid list is exactly:
    //   [SoloThree, SoloTwo, SoloOne, Berac]
    assert_eq!(biddable.len(), 4);
    assert!(biddable.contains(&Contract::SoloThree));
    assert!(biddable.contains(&Contract::SoloTwo));
    assert!(biddable.contains(&Contract::SoloOne));
    assert!(biddable.contains(&Contract::Berac));
    assert!(!biddable.contains(&Contract::Valat));
    // Definitely NOT in 3p: 4p-only contracts.
    assert!(!biddable.contains(&Contract::Three));
    assert!(!biddable.contains(&Contract::Two));
    assert!(!biddable.contains(&Contract::One));
    assert!(!biddable.contains(&Contract::Solo));
}

#[test]
fn three_player_deal_distributes_16_16_16_plus_6_talon() {
    let _g = acquire_guard();
    let mut state = fresh_three_player(0);
    let mut rng = StdRng::seed_from_u64(42);
    state.deal(&mut rng);

    assert_eq!(state.hands[0].len(), 16, "seat 0 hand size");
    assert_eq!(state.hands[1].len(), 16, "seat 1 hand size");
    assert_eq!(state.hands[2].len(), 16, "seat 2 hand size");
    // Seat 3 is unused in 3p — must remain empty.
    assert_eq!(state.hands[3].len(), 0, "seat 3 must be empty in 3p");
    assert_eq!(state.talon.len(), 6, "talon size");

    // Total cards dealt = 16*3 + 6 = 54 (full deck).
    let total = state.hands[0].len()
        + state.hands[1].len()
        + state.hands[2].len()
        + state.hands[3].len()
        + state.talon.len();
    assert_eq!(total, 54);
}

#[test]
fn three_player_legal_bids_excludes_4p_only_contracts() {
    let _g = acquire_guard();
    let mut state = fresh_three_player(0);
    let mut rng = StdRng::seed_from_u64(123);
    state.deal(&mut rng);

    // Forehand bids first (no prior bids, no highest).
    let forehand = state.forehand();
    let bids = state.legal_bids(forehand);

    // Forehand should see the full 3p ladder and nothing 4p-only.
    for c in &bids {
        assert!(
            !matches!(
                c,
                Contract::Three | Contract::Two | Contract::One | Contract::Solo
            ),
            "4p-only contract {:?} leaked into 3p legal_bids",
            c
        );
    }
    // The 3p ladder should be fully present (no prior bid → all biddable).
    assert!(bids.contains(&Contract::SoloThree));
    assert!(bids.contains(&Contract::SoloTwo));
    assert!(bids.contains(&Contract::SoloOne));
    assert!(bids.contains(&Contract::Berac));
    assert!(!bids.contains(&Contract::Valat));
}

#[test]
fn three_player_base_values_distinct_from_4p() {
    let _g = acquire_guard();
    // The 3p table differs (compressed) from 4p; user can tune.
    assert_eq!(Contract::SoloThree.base_value_for(Variant::ThreePlayer), 20);
    assert_eq!(Contract::SoloTwo.base_value_for(Variant::ThreePlayer), 30);
    assert_eq!(Contract::SoloOne.base_value_for(Variant::ThreePlayer), 40);
    assert_eq!(Contract::Berac.base_value_for(Variant::ThreePlayer), 70);
    assert_eq!(Contract::Valat.base_value_for(Variant::ThreePlayer), 250);
    assert_eq!(
        Contract::BarvniValat.base_value_for(Variant::ThreePlayer),
        125
    );

    // 4p backward-compatible call still returns 4p table.
    assert_eq!(Contract::SoloThree.base_value(), 40);
    assert_eq!(Contract::Solo.base_value(), 80);
    assert_eq!(Contract::Berac.base_value(), 70);
}

#[test]
fn contract_valat_round_trips_through_u8() {
    let _g = acquire_guard();
    assert_eq!(Contract::from_u8(10), Some(Contract::Valat));
    assert_eq!(Contract::Valat as u8, 10);
}

// Variant-guard tests must not run concurrently with other tests in this
// binary that lock the variant. We use a `Mutex` to serialize the two-step
// "lock-then-check" sequences. The `#[should_panic]` test catches the panic
// from inside `catch_unwind` so the mutex is always released.

#[test]
fn process_variant_guard_allows_same_variant() {
    let _g = acquire_guard();
    let _a = GameState::new_with_variant(Variant::ThreePlayer, 0);
    // Same variant: must not panic.
    let _b = GameState::new_with_variant(Variant::ThreePlayer, 1);
}

#[test]
fn process_variant_guard_rejects_mixed_variants() {
    let _g = acquire_guard();
    let _a = GameState::new_with_variant(Variant::ThreePlayer, 0);
    // Different variant in same process: must panic. We catch the unwind so
    // the test can clean up afterwards.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        GameState::new_with_variant(Variant::FourPlayer, 0)
    }));
    assert!(result.is_err(), "expected panic on mixed variant");
}
