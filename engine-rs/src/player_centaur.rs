//! Centaur (hybrid NN + endgame solver) player for Tarok.
//!
//! Uses the neural network for early/mid-game decisions (bidding, king
//! calling, talon exchange, and tricks before the handoff point), then
//! switches to a search-based endgame solver where the information set
//! is nearly collapsed and search is practically exact.
//!
//! The endgame solver is configurable:
//! - **PIMC** (default): Perfect-Information Monte Carlo — samples worlds
//!   and DD-solves each independently.  Fast, but susceptible to
//!   non-locality (strategy fusion is not enforced).
//! - **AlphaMu**: The αμ algorithm — enforces strategy fusion via Pareto
//!   fronts of per-world outcome vectors.  Stronger but slower.
//!
//! During training the solver-decided experiences are tagged with
//! `log_prob = NaN` so the PPO trainer can skip them while still
//! benefiting from the solver-optimal terminal rewards flowing back
//! through the NN's earlier decisions.

use crate::alpha_mu;
use crate::bots::centaur_bidding;
use crate::bots::stockskis_v3_3p;
use crate::card::{Card, CardSet, CardType};
use crate::game_state::{Contract, GameState, Variant};
use crate::legal_moves;
use crate::pimc;
use crate::player::*;
use crate::player_nn::NeuralNetPlayer;
use rayon::prelude::*;
use tch::Device;

/// Default trick number at which the endgame solver takes over card play
/// (0-indexed, so 8 means the solver plays tricks 9–12, i.e. the last 4).
pub const DEFAULT_HANDOFF_TRICK: usize = 8;

/// Default number of worlds to sample per decision.
pub const DEFAULT_NUM_WORLDS: u32 = 100;

/// Default αμ search depth (number of Max moves to look ahead).
pub const DEFAULT_ALPHA_MU_DEPTH: usize = 2;

// -----------------------------------------------------------------------
// Endgame policy
// -----------------------------------------------------------------------

/// Which search algorithm to use for endgame card play.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndgamePolicy {
    /// PIMC: sample worlds, DD-solve each independently.
    Pimc,
    /// αμ: sample worlds, enforce strategy fusion via Pareto fronts.
    AlphaMu { max_depth: usize },
}

impl EndgamePolicy {
    pub fn from_name(name: &str, alpha_mu_depth: usize) -> Self {
        match name {
            "alpha_mu" | "alphamu" => EndgamePolicy::AlphaMu {
                max_depth: alpha_mu_depth,
            },
            _ => EndgamePolicy::Pimc,
        }
    }
}

// -----------------------------------------------------------------------
// CentaurBot
// -----------------------------------------------------------------------

pub struct CentaurBot {
    nn: NeuralNetPlayer,
    handoff_trick: usize,
    num_worlds: u32,
    endgame_policy: EndgamePolicy,
    /// When `Some(salt)`, the endgame solver is seeded deterministically
    /// from a canonical fingerprint of the visible game state XORed with
    /// `salt`. Used by duplicate-RL so that active and shadow tables
    /// reaching the same state sample the same worlds → PIMC noise
    /// cancels in the reward difference. When `None` (the default),
    /// the solver uses fresh randomness per decision.
    deterministic_seed_salt: Option<u64>,
}

impl CentaurBot {
    pub fn new(
        model_path: &str,
        device: Device,
        explore_rate: f64,
        handoff_trick: usize,
        num_worlds: u32,
        endgame_policy: EndgamePolicy,
    ) -> Self {
        CentaurBot {
            nn: NeuralNetPlayer::new(model_path, device, explore_rate),
            handoff_trick,
            num_worlds,
            endgame_policy,
            deterministic_seed_salt: None,
        }
    }

    /// Enable deterministic seeding of the endgame solver. The salt is
    /// XORed into the state fingerprint so different training runs can
    /// still produce independent streams while any single run remains
    /// reproducible and duplicate-invariant.
    pub fn with_deterministic_seed(mut self, salt: u64) -> Self {
        self.deterministic_seed_salt = Some(salt);
        self
    }

    /// Should this card-play decision be handled by PIMC?
    fn use_pimc(&self, ctx: &DecisionContext<'_>) -> bool {
        if ctx.decision_type != DecisionType::CardPlay {
            return false;
        }
        // Klop is individual scoring — PIMC's declarer-vs-opponent
        // maximisation doesn't apply.  Keep the NN for Klop.
        if ctx.gs.contract == Some(Contract::Klop) {
            return false;
        }
        // Berač declarer uses a deterministic heuristic (see batch_decide);
        // skip PIMC so it doesn't overwrite that override.
        if ctx.gs.contract == Some(Contract::Berac)
            && ctx.gs.declarer == Some(ctx.player)
        {
            return false;
        }
        ctx.gs.tricks_played() >= self.handoff_trick
    }
}

impl BatchPlayer for CentaurBot {
    fn batch_decide(&self, contexts: &[DecisionContext<'_>]) -> Vec<DecisionResult> {
        if contexts.is_empty() {
            return Vec::new();
        }

        // Let the NN decide everything first (gives us value estimates
        // for all positions and correct actions for non-PIMC decisions).
        let mut results = self.nn.batch_decide(contexts);

        // Override bidding-phase decisions with the m6 heuristic (4p only).
        //
        // For 4p, the NN bid/king/talon heads are not trained when bidding
        // is handled heuristically — log_prob is set to NaN so the PPO
        // trainer skips the policy-gradient update for these steps while
        // still flowing terminal rewards back through the card-play heads.
        //
        // For 3p, the NN handles bid / king / talon-pick / discard. We
        // *do* want the bid head to learn (e.g. that Berač is risky); for
        // that to work the duplicate-RL signal must produce a non-zero
        // gradient on bid steps, which requires `shadow_source` to be a
        // heuristic bot (e.g. `bot_v3_3p`) rather than a near-copy of the
        // learner — see training-lab/configs/three-player-duplicate.yaml.
        for (i, ctx) in contexts.iter().enumerate() {
            // 3p: leave bid / king / talon decisions to the NN so the
            // policy gradient can shape them.
            if matches!(ctx.gs.variant, Variant::ThreePlayer)
                && !matches!(ctx.decision_type, DecisionType::CardPlay)
            {
                continue;
            }
            let action = match ctx.decision_type {
                DecisionType::Bid => {
                    let hand = ctx.gs.hands[ctx.player as usize];
                    let highest = ctx.gs.bids.iter()
                        .filter_map(|b| b.contract)
                        .max_by_key(|c| c.strength());
                    let chosen = centaur_bidding::evaluate_bid_centaur(hand, highest);
                    let a = contract_to_bid_action(chosen);
                    if ctx.legal_mask.get(a).map_or(false, |&v| v > 0.5) { a } else { 0 }
                }
                DecisionType::KingCall => {
                    let hand = ctx.gs.hands[ctx.player as usize];
                    let chosen = centaur_bidding::choose_king_centaur(hand);
                    match chosen.and_then(|c| card_suit_idx(c.0)) {
                        Some(idx) if ctx.legal_mask.get(idx).map_or(false, |&v| v > 0.5) => idx,
                        _ => ctx.legal_mask.iter().position(|&v| v > 0.5).unwrap_or(0),
                    }
                }
                DecisionType::TalonPick => {
                    let hand = ctx.gs.hands[ctx.player as usize];
                    let chosen = centaur_bidding::choose_talon_group_centaur(
                        &ctx.gs.talon_revealed, hand, ctx.gs.called_king,
                    );
                    if ctx.legal_mask.get(chosen).map_or(false, |&v| v > 0.5) {
                        chosen
                    } else {
                        ctx.legal_mask.iter().position(|&v| v > 0.5).unwrap_or(0)
                    }
                }
                DecisionType::CardPlay => continue,
            };
            results[i] = DecisionResult {
                action,
                log_prob: f32::NAN,
                value: results[i].value,
            };
        }

        // Berač declarer cardplay override.
        //
        // Berač is won by the declarer iff they take ZERO tricks. The
        // optimal heuristic for the declarer is well-known and we
        // hardcode it here so the NN doesn't have to learn the inverted
        // objective (every other contract rewards taking tricks):
        //
        //   * On the very first lead of the game, lead the HIGHEST tarok
        //     in hand (declarer always leads in berač). This dumps the
        //     dangerous high tarok while opponents still hold higher ones.
        //   * Otherwise (whether leading later or following) play the
        //     HIGHEST legal card that does NOT become the trick's
        //     best-card-so-far — i.e. the highest card that will not win
        //     the trick. If every legal card would win, play the LOWEST
        //     legal card to minimise the damage on the next trick.
        //
        // This runs for the declarer seat only; defenders keep using the
        // NN (their objective — making the declarer win at least one
        // trick — is the same "take tricks" objective the NN trains on
        // everywhere else, so the NN can handle it).
        for (i, ctx) in contexts.iter().enumerate() {
            if ctx.decision_type != DecisionType::CardPlay {
                continue;
            }
            if ctx.gs.contract != Some(Contract::Berac) {
                continue;
            }
            if ctx.gs.declarer != Some(ctx.player) {
                continue;
            }
            if let Some(action) = berac_declarer_play(ctx) {
                if std::env::var("CENTAUR_BERAC_DEBUG").is_ok() {
                    let legal_ok = ctx.legal_mask.get(action).map_or(false, |&v| v > 0.5);
                    let nn_action = results[i].action;
                    eprintln!(
                        "[BERAC] seat={} trick={} count={} nn_action={} our_action={} legal_ok={}",
                        ctx.player,
                        ctx.gs.tricks_played(),
                        ctx.gs.current_trick.as_ref().map_or(0, |t| t.count),
                        nn_action,
                        action,
                        legal_ok,
                    );
                }
                results[i] = DecisionResult {
                    action,
                    log_prob: f32::NAN,
                    value: results[i].value,
                };
            }
        }

        // Override late-game card plays with the endgame solver.
        //
        // Each PIMC/alpha-mu call is independent and, under deterministic
        // seeding, a pure function of (state, viewer, salt). Running the
        // per-context overrides in parallel preserves determinism and lets
        // endgame-heavy batches saturate all cores instead of serialising
        // behind a single dispatch thread.
        let overrides: Vec<Option<(usize, usize)>> = contexts
            .par_iter()
            .enumerate()
            .map(|(i, ctx)| {
                if !self.use_pimc(ctx) {
                    return None;
                }
                let card = match (self.endgame_policy, self.deterministic_seed_salt) {
                    (EndgamePolicy::Pimc, None) => {
                        pimc::pimc_choose_card(ctx.gs, ctx.player, self.num_worlds)
                    }
                    (EndgamePolicy::Pimc, Some(salt)) => {
                        let seed = state_fingerprint(ctx.gs, ctx.player) ^ salt;
                        pimc::pimc_choose_card_with_seed(
                            ctx.gs,
                            ctx.player,
                            self.num_worlds,
                            seed,
                        )
                    }
                    (EndgamePolicy::AlphaMu { max_depth }, None) => alpha_mu::alpha_mu_choose_card(
                        ctx.gs,
                        ctx.player,
                        self.num_worlds,
                        max_depth,
                    ),
                    (EndgamePolicy::AlphaMu { max_depth }, Some(salt)) => {
                        let seed = state_fingerprint(ctx.gs, ctx.player) ^ salt;
                        alpha_mu::alpha_mu_choose_card_with_seed(
                            ctx.gs,
                            ctx.player,
                            self.num_worlds,
                            max_depth,
                            seed,
                        )
                    }
                };
                Some((i, card.0 as usize))
            })
            .collect();

        for entry in overrides.into_iter().flatten() {
            let (i, action) = entry;
            results[i] = DecisionResult {
                action,
                // NaN sentinel: PPO should skip this experience's policy
                // loss while still using the game's terminal reward.
                log_prob: f32::NAN,
                // Keep the NN's value estimate — useful for GAE on the
                // preceding (NN-decided) time steps.
                value: results[i].value,
            };
        }

        results
    }

    fn choose_discards(
        &self,
        gs: &GameState,
        player: u8,
        must_discard: usize,
    ) -> Option<Vec<crate::card::Card>> {
        let hand = gs.hands[player as usize];
        match gs.variant {
            Variant::FourPlayer => Some(centaur_bidding::choose_discards_centaur(
                hand,
                must_discard,
                gs.called_king,
            )),
            Variant::ThreePlayer => Some(stockskis_v3_3p::choose_discards_v3_3p(
                hand,
                must_discard,
            )),
        }
    }

    fn name(&self) -> &str {
        "centaur"
    }
}

// -----------------------------------------------------------------------
// Berač declarer cardplay heuristic
// -----------------------------------------------------------------------

/// Decide a Berač declarer card play. Returns the action index (== card.0
/// as usize) or `None` if the legal moves are empty (which should never
/// happen during a real game; falls back to the NN-chosen action then).
///
/// Strategy:
///   1. First lead of the game: lead the highest tarok in hand.
///   2. Otherwise: pick the highest legal card that does not beat the
///      current trick-best (i.e. would not win the trick). If no legal
///      card avoids winning, pick the lowest legal card.
fn berac_declarer_play(ctx: &DecisionContext<'_>) -> Option<usize> {
    let move_ctx = legal_moves::MoveCtx::from_state(ctx.gs, ctx.player);
    let legal = legal_moves::generate_legal_moves(&move_ctx);
    if legal.is_empty() {
        return None;
    }

    let trick_count = ctx.gs.current_trick.as_ref().map_or(0, |t| t.count);
    let is_first_lead =
        ctx.gs.tricks_played() == 0 && trick_count == 0;

    // (1) First lead → highest tarok if any.
    if is_first_lead {
        if let Some(card) = highest_tarok(legal) {
            return Some(card.0 as usize);
        }
        // No taroks at all: fall through to "highest legal card" logic.
    }

    let best_card = ctx.gs.current_trick.as_ref().and_then(|t| t.best_card());
    let lead_suit = ctx.gs.current_trick.as_ref().and_then(|t| t.lead_suit());

    // (2a) When LEADING (no card on the table yet), every legal card we
    //      lead becomes "best" — there is no way to "not win" a lead.
    //      Pick the highest legal card; this dumps high cards while
    //      opponents still hold higher ones.
    if best_card.is_none() {
        return highest_card(legal).map(|c| c.0 as usize);
    }

    // (2b) Following: prefer the highest legal card that does NOT beat the
    //      current best. Beating-or-not is determined by `Card::beats`
    //      with the recorded lead suit.
    let best = best_card.unwrap();
    let mut losing: Option<Card> = None;
    let mut winning: Option<Card> = None;
    for c in legal.iter() {
        if c.beats(best, lead_suit) {
            // would-win candidate — track the LOWEST so we can fall back.
            winning = match winning {
                None => Some(c),
                Some(prev) => Some(if c.beats(prev, lead_suit) { prev } else { c }),
            };
        } else {
            // would-lose candidate — track the HIGHEST.
            losing = match losing {
                None => Some(c),
                Some(prev) => Some(if c.beats(prev, lead_suit) { c } else { prev }),
            };
        }
    }
    let chosen = losing.or(winning)?;
    Some(chosen.0 as usize)
}

/// Highest tarok in `set`, or `None` if no taroks. Skis (value=22) wins
/// over Mond (21) in tarok-vs-tarok comparisons.
fn highest_tarok(set: CardSet) -> Option<Card> {
    let taroks = set.taroks();
    let mut best: Option<Card> = None;
    for c in taroks.iter() {
        match best {
            None => best = Some(c),
            Some(prev) => {
                if c.beats(prev, None) {
                    best = Some(c);
                }
            }
        }
    }
    best
}

/// Highest card in `set` under tarok-trumps-suit ordering. Among suit
/// cards of different suits, ranks them as if each is the lead — i.e.
/// returns the highest-rank suit card overall, with ties broken by value.
/// Used for "I'm leading, dump my biggest card".
fn highest_card(set: CardSet) -> Option<Card> {
    if let Some(t) = highest_tarok(set) {
        return Some(t);
    }
    let mut best: Option<Card> = None;
    for c in set.iter() {
        match best {
            None => best = Some(c),
            Some(prev) => {
                // Both are suit cards (no taroks left); compare by value
                // (King > Queen > Knight > Jack > 10..1). For different
                // suits we just pick the higher value — when leading,
                // either choice "wins its own suit" so value rank is
                // what matters.
                let cv = c.value();
                let pv = prev.value();
                if c.card_type() == CardType::Suit
                    && prev.card_type() == CardType::Suit
                    && cv > pv
                {
                    best = Some(c);
                }
            }
        }
    }
    best
}


// -----------------------------------------------------------------------
// Deterministic state fingerprint (for duplicate-RL PIMC seeding)
// -----------------------------------------------------------------------

/// SplitMix64 finalizer — stable, deterministic, no external deps.
#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn mix(acc: u64, x: u64) -> u64 {
    splitmix64(acc ^ splitmix64(x))
}

/// Hash every decision-relevant bit of the visible game state into a u64.
///
/// Two game states that reach this function with identical values produce
/// the same fingerprint. Different states — even ones that differ only in
/// played-card order — produce different fingerprints with overwhelming
/// probability (SplitMix64 avalanche).
///
/// This is the seed source for deterministic PIMC under duplicate-RL: when
/// active and shadow tables reach identical visible states, they pick the
/// same worlds and thus the same card, so the PIMC sampling noise
/// disappears from `R_active − R_shadow`.
pub fn state_fingerprint(gs: &GameState, viewer: u8) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325; // FNV offset; arbitrary nonzero start

    h = mix(h, viewer as u64);

    // Contract, declarer, partner — identifies who is maximizing what.
    let contract_tag: u64 = match gs.contract {
        None => 0,
        Some(c) => 1 + c as u64,
    };
    h = mix(h, contract_tag);
    h = mix(h, gs.declarer.map(|p| p as u64 + 1).unwrap_or(0));
    h = mix(h, gs.partner.map(|p| p as u64 + 1).unwrap_or(0));

    // Roles — declarer/partner/opponent per seat.
    let mut roles_word: u64 = 0;
    for (i, &r) in gs.roles.iter().enumerate() {
        roles_word |= (r as u64 & 0xF) << (i * 4);
    }
    h = mix(h, roles_word);

    // All cards played so far (order-independent by construction).
    h = mix(h, gs.played_cards.0);

    // Viewer's own hand — narrows the world-sampling space.
    h = mix(h, gs.hands[viewer as usize].0);

    // Current trick: lead + played cards in play order.
    if let Some(ref t) = gs.current_trick {
        h = mix(h, t.lead_player as u64);
        h = mix(h, t.count as u64);
        for i in 0..t.count as usize {
            let (p, c) = t.cards[i];
            h = mix(h, ((p as u64) << 8) | c.0 as u64);
        }
    } else {
        h = mix(h, 0xFFFF_FFFF_FFFF_FFFF); // distinguish "no trick" from trick with count=0
    }

    // Tricks played: count + who won each (viewer knows this from public play).
    h = mix(h, gs.tricks_played() as u64);
    for (i, trick) in gs.tricks.iter().enumerate() {
        h = mix(h, (i as u64) << 32 | trick.lead_player as u64);
        for j in 0..trick.count as usize {
            let (p, c) = trick.cards[j];
            h = mix(h, ((p as u64) << 16) | ((j as u64) << 8) | c.0 as u64);
        }
    }

    // Ensure no zero output (zero ^ salt = salt, which would make salt
    // trivially recoverable; not a security issue, but mildly untidy).
    splitmix64(h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game_state::*;

    #[test]
    fn pimc_threshold_not_reached() {
        let gs = GameState::new(0);
        // tricks_played() == 0, handoff_trick == 8 → NN should handle this
        let ctx = DecisionContext {
            gs: &gs,
            player: 0,
            decision_type: DecisionType::CardPlay,
            legal_mask: vec![],
            state_encoding: vec![],
        };
        // Manually test the predicate (can't construct a full CentaurBot
        // without a real TorchScript model in unit tests).
        let handoff = 8usize;
        let should_pimc = ctx.decision_type == DecisionType::CardPlay
            && ctx.gs.contract != Some(Contract::Klop)
            && ctx.gs.tricks_played() >= handoff;
        assert!(!should_pimc);
    }

    #[test]
    fn pimc_skips_klop() {
        let mut gs = GameState::new(0);
        gs.contract = Some(Contract::Klop);
        let ctx = DecisionContext {
            gs: &gs,
            player: 0,
            decision_type: DecisionType::CardPlay,
            legal_mask: vec![],
            state_encoding: vec![],
        };
        let handoff = 0usize; // even with handoff at 0
        let should_pimc = ctx.decision_type == DecisionType::CardPlay
            && ctx.gs.contract != Some(Contract::Klop)
            && ctx.gs.tricks_played() >= handoff;
        assert!(!should_pimc);
    }

    #[test]
    fn pimc_skips_non_card_decisions() {
        let mut gs = GameState::new(0);
        gs.contract = Some(Contract::Three);
        let ctx = DecisionContext {
            gs: &gs,
            player: 0,
            decision_type: DecisionType::Bid,
            legal_mask: vec![],
            state_encoding: vec![],
        };
        let handoff = 0usize;
        let should_pimc = ctx.decision_type == DecisionType::CardPlay
            && ctx.gs.contract != Some(Contract::Klop)
            && ctx.gs.tricks_played() >= handoff;
        assert!(!should_pimc);
    }

    // -------- state_fingerprint --------

    #[test]
    fn fingerprint_is_stable_for_identical_state() {
        let mut gs = GameState::new(0);
        gs.contract = Some(Contract::Three);
        gs.declarer = Some(1);
        gs.partner = Some(3);
        gs.played_cards = crate::card::CardSet(0xDEAD_BEEF_1234_5678);
        gs.hands[0] = crate::card::CardSet(0xAAAA_5555_AAAA_5555);

        let a = state_fingerprint(&gs, 0);
        let b = state_fingerprint(&gs, 0);
        assert_eq!(a, b, "fingerprint must be pure");
    }

    #[test]
    fn fingerprint_differs_across_viewers() {
        let mut gs = GameState::new(0);
        gs.contract = Some(Contract::Three);
        gs.played_cards = crate::card::CardSet(0x1111_2222_3333_4444);
        gs.hands[0] = crate::card::CardSet(0xFF);
        gs.hands[1] = crate::card::CardSet(0xFF00);

        let a = state_fingerprint(&gs, 0);
        let b = state_fingerprint(&gs, 1);
        assert_ne!(a, b, "different viewer → different fingerprint");
    }

    #[test]
    fn fingerprint_differs_when_played_cards_differ() {
        let mut gs = GameState::new(0);
        gs.contract = Some(Contract::Three);
        gs.played_cards = crate::card::CardSet(0x1);
        let a = state_fingerprint(&gs, 0);

        gs.played_cards = crate::card::CardSet(0x2);
        let b = state_fingerprint(&gs, 0);

        assert_ne!(a, b, "different played cards → different fingerprint");
    }

    #[test]
    fn fingerprint_is_nonzero() {
        let gs = GameState::new(0);
        assert_ne!(state_fingerprint(&gs, 0), 0);
    }
}
