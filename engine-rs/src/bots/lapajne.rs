/// Lapajne heuristic bot — implementation of Lapajne (2011) MCTS algorithm.
///
/// Reference: Mitja Lapajne, "Program za igranje taroka z uporabo drevesnega
/// preiskovanja Monte-Carlo", BSc thesis, FRI Ljubljana, 2011.
///
/// Algorithms implemented verbatim from the paper:
///
/// - §5.2  MCTS tree: UCT selection (C=0.5), 1-node-per-sim expansion,
///         heuristic rollout, multi-player backpropagation.
/// - §5.1  Determinization layer: sample opponent hands from consistent
///         worlds (belief-constrained by observed void-suits).
/// - §5.3  Bidding heuristic: tarok/king/trula thresholds.
/// - §5.4  Talon selection: maximize suit voiding when discarding.
/// - §6.1.3 Rollout heuristics: three tactical rules applied during
///         simulations to produce more realistic play.
///
/// Card play combines §5.1 + §5.2 (Algorithm 2 in the paper):
///   for each determinized world → build MCTS tree → collect root visit
///   counts → pick card with most total visits across all worlds.
use crate::card::*;
use crate::game_state::*;
use crate::legal_moves;
use crate::scoring;
use crate::bots::stockskis_v1;
use rand::prelude::IndexedRandom;
use rand::seq::SliceRandom;
use std::sync::atomic::{AtomicUsize, Ordering};

// ─────────────────────────────────────────────────────────────────────────────
// Configurable parameters
// ─────────────────────────────────────────────────────────────────────────────

/// Number of random determinizations per move (worlds).
const DEFAULT_MC_WORLDS: usize = 20;
/// UCT simulations per determinized world tree.
const DEFAULT_MC_SIMS: usize = 20;
/// UCT exploration constant — C=0.5 recommended by paper (§6.1.1).
const UCT_C: f64 = 0.5;

static MC_WORLDS_CFG: AtomicUsize = AtomicUsize::new(DEFAULT_MC_WORLDS);
static MC_SIMS_CFG: AtomicUsize = AtomicUsize::new(DEFAULT_MC_SIMS);

#[inline]
pub fn set_mc_worlds(worlds: usize) -> usize {
    let clamped = worlds.clamp(1, 512);
    MC_WORLDS_CFG.store(clamped, Ordering::Relaxed);
    clamped
}

#[inline]
pub fn mc_worlds() -> usize {
    MC_WORLDS_CFG.load(Ordering::Relaxed)
}

#[inline]
pub fn set_mc_sims(sims: usize) -> usize {
    let clamped = sims.clamp(1, 4096);
    MC_SIMS_CFG.store(clamped, Ordering::Relaxed);
    clamped
}

#[inline]
pub fn mc_sims() -> usize {
    MC_SIMS_CFG.load(Ordering::Relaxed)
}

// ─────────────────────────────────────────────────────────────────────────────
// Bidding — §5.3 (Lapajne 2011)
// ─────────────────────────────────────────────────────────────────────────────

/// Bidding heuristic from §5.3 of the paper.
///
/// Three: (≥6 taroks AND ≥1 king AND ≥1 trula) OR ≥10 taroks
/// Two:   (≥7 taroks AND ≥2 kings AND ≥1 trula) OR ≥10 taroks
/// One:   (≥8 taroks AND ≥2 kings AND ≥1 trula) OR ≥10 taroks
/// Otherwise: pass.
pub fn evaluate_bid_lapajne(
    hand: CardSet,
    highest_so_far: Option<Contract>,
) -> Option<Contract> {
    let taroks = hand.tarok_count() as usize;
    let kings = hand.king_count() as usize;
    let has_trula = (hand.0 & TRULA_MASK) != 0;

    // Check lowest to highest; last assignment wins (= highest qualifying).
    let mut best: Option<Contract> = None;
    if (taroks >= 6 && kings >= 1 && has_trula) || taroks >= 10 {
        best = Some(Contract::Three);
    }
    if (taroks >= 7 && kings >= 2 && has_trula) || taroks >= 10 {
        best = Some(Contract::Two);
    }
    if (taroks >= 8 && kings >= 2 && has_trula) || taroks >= 10 {
        best = Some(Contract::One);
    }

    if let Some(contract) = best {
        if let Some(h) = highest_so_far {
            if contract.strength() <= h.strength() {
                return None;
            }
        }
        Some(contract)
    } else {
        None
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// King calling — unchanged from v1
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
pub fn choose_king_lapajne(hand: CardSet) -> Option<Card> {
    stockskis_v1::choose_king_v1(hand)
}

// ─────────────────────────────────────────────────────────────────────────────
// Talon selection — §5.4 (Lapajne 2011)
// ─────────────────────────────────────────────────────────────────────────────

/// Choose the talon group that maximises the number of suits voided when
/// discarding.  Tie-break: more taroks in the group, then more points.
///
/// "Izbrana je tista karta / tisti par ali trojček kart, ki pri zalaganju
/// omogoča, da se znebimo čimveč barv."  (§5.4)
pub fn choose_talon_group_lapajne(
    groups: &[Vec<Card>],
    hand: CardSet,
    called_king: Option<Card>,
) -> usize {
    if groups.is_empty() {
        return 0;
    }
    let called_suit = called_king.and_then(|k| k.suit());
    let must_discard = groups[0].len();

    let mut best_idx = 0usize;
    let mut best_score = (i32::MIN, 0u8, i32::MIN);

    for (i, group) in groups.iter().enumerate() {
        let mut combined = hand;
        for &card in group {
            combined.insert(card);
        }

        // Count cards per voidable suit, sort ascending, greedily void.
        let mut suit_counts: Vec<usize> = Suit::ALL
            .iter()
            .filter(|&&s| Some(s) != called_suit)
            .map(|&s| combined.suit_count(s) as usize)
            .filter(|&n| n > 0)
            .collect();
        suit_counts.sort_unstable();

        let mut remaining = must_discard;
        let mut suits_voided = 0i32;
        for &count in &suit_counts {
            if count <= remaining {
                suits_voided += 1;
                remaining -= count;
            } else {
                break;
            }
        }

        let group_taroks = group
            .iter()
            .filter(|c| c.card_type() == CardType::Tarok)
            .count() as u8;
        let group_points: i32 = group.iter().map(|c| c.points() as i32).sum();

        let score = (suits_voided, group_taroks, group_points);
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }

    best_idx
}

// ─────────────────────────────────────────────────────────────────────────────
// Discarding — unchanged from v1
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
pub fn choose_discards_lapajne(
    hand: CardSet,
    must_discard: usize,
    called_king: Option<Card>,
) -> Vec<Card> {
    stockskis_v1::choose_discards_v1(hand, must_discard, called_king)
}

// ─────────────────────────────────────────────────────────────────────────────
// MCTS tree node
// ─────────────────────────────────────────────────────────────────────────────

struct MctsNode {
    visits: u32,
    /// Cumulative utility from **root_player**'s perspective.
    value: f64,
    /// Expanded child moves: (card played, index into arena vec).
    children: Vec<(Card, usize)>,
    /// Moves not yet expanded.  Pre-shuffled so pop() gives random picks.
    untried: Vec<Card>,
    /// Whose turn it is at this node.
    player: u8,
}

// ─────────────────────────────────────────────────────────────────────────────
// Card play — Algorithm 2 (Lapajne 2011): MCTS over determinizations
// ─────────────────────────────────────────────────────────────────────────────

/// Main card-play entry point.
///
/// Combines §5.1 determinization sampling with §5.2 MCTS tree search.
/// For each sampled world build a UCT tree, sum root visit counts across
/// worlds, return the card with the most total visits.
pub fn choose_card_lapajne(hand: CardSet, state: &GameState, player: u8) -> Card {
    let ctx = legal_moves::MoveCtx::from_state(state, player);
    let legal = legal_moves::generate_legal_moves(&ctx);
    if legal.is_empty() {
        return hand.iter().next().unwrap_or(Card(0));
    }
    let legal_cards: Vec<Card> = legal.iter().collect();
    if legal_cards.len() == 1 {
        return legal_cards[0];
    }

    let worlds = mc_worlds();
    let sims = mc_sims();
    let mut visit_totals = vec![0u32; legal_cards.len()];
    let mut rng = rand::rng();

    for _ in 0..worlds {
        let world = sample_world(state, player, &mut rng);
        let counts = mcts_for_world(&world, player, &legal_cards, sims, &mut rng);
        for (i, &v) in counts.iter().enumerate() {
            visit_totals[i] += v;
        }
    }

    // Pick card with most visits; break ties in favour of v1's recommendation.
    let v1_card = stockskis_v1::choose_card_v1(hand, state, player);
    let max_visits = *visit_totals.iter().max().unwrap_or(&0);
    legal_cards
        .iter()
        .copied()
        .zip(visit_totals.iter().copied())
        .filter(|&(_, v)| v == max_visits)
        .max_by_key(|&(c, _)| if c == v1_card { 1i32 } else { 0i32 })
        .map(|(c, _)| c)
        .unwrap_or(legal_cards[0])
}

/// Build a UCT MCTS tree for one determinized world and return visit counts
/// for each of `root_moves` (indexed to match that slice).
fn mcts_for_world(
    world: &GameState,
    root_player: u8,
    root_moves: &[Card],
    n_sims: usize,
    rng: &mut impl rand::Rng,
) -> Vec<u32> {
    let mut arena: Vec<MctsNode> =
        Vec::with_capacity(n_sims.min(256) * 3 + root_moves.len() + 1);
    let mut root_untried = root_moves.to_vec();
    root_untried.shuffle(rng);
    arena.push(MctsNode {
        visits: 0,
        value: 0.0,
        children: Vec::new(),
        untried: root_untried,
        player: root_player,
    });

    for _ in 0..n_sims {
        let mut sim_state = world.clone();
        let path = mcts_select_expand(&mut arena, &mut sim_state, rng);
        if path.is_empty() {
            continue;
        }
        let result = rollout_heuristic(&mut sim_state, root_player) as f64;
        mcts_backprop(&mut arena, &path, result, root_player, &world.roles);
    }

    // Collect visit counts for root's direct children by root_moves index.
    let mut counts = vec![0u32; root_moves.len()];
    for &(card, child_idx) in &arena[0].children {
        if let Some(i) = root_moves.iter().position(|&c| c == card) {
            counts[i] = arena[child_idx].visits;
        }
    }
    counts
}

/// UCT tree traversal + single-node expansion (§3.3.3–3.3.4).
///
/// Walks the tree from the root using the UCT selection rule, expands one
/// unexplored move when a leaf is reached, and applies all moves to `state`
/// so the caller can run a rollout from there.
///
/// Returns the path as `(node_idx, card_played)` pairs for backprop.
fn mcts_select_expand(
    arena: &mut Vec<MctsNode>,
    state: &mut GameState,
    rng: &mut impl rand::Rng,
) -> Vec<(usize, Card)> {
    let mut path: Vec<(usize, Card)> = Vec::new();
    let mut current = 0usize;

    loop {
        if state.tricks_played() >= TRICKS_PER_GAME {
            break;
        }
        if state.current_trick.is_none() {
            state.start_trick(state.current_player);
        }

        // ── Expansion: node has untried moves ─────────────────────────────
        if !arena[current].untried.is_empty() {
            let card = arena[current].untried.pop().unwrap(); // pre-shuffled
            let node_player = arena[current].player;
            path.push((current, card));

            state.play_card(node_player, card);
            finish_current_trick_if_complete(state);

            if state.tricks_played() >= TRICKS_PER_GAME {
                break;
            }
            if state.current_trick.is_none() {
                state.start_trick(state.current_player);
            }

            let next_player = current_turn_player(state);
            let next_ctx = legal_moves::MoveCtx::from_state(state, next_player);
            let mut next_legal: Vec<Card> =
                legal_moves::generate_legal_moves(&next_ctx).iter().collect();
            next_legal.shuffle(rng);

            let child_idx = arena.len();
            arena.push(MctsNode {
                visits: 0,
                value: 0.0,
                children: Vec::new(),
                untried: next_legal,
                player: next_player,
            });
            arena[current].children.push((card, child_idx));
            break; // rollout from newly created leaf
        }

        // ── No untried moves: UCT selection (§3.3.3) ─────────────────────
        if arena[current].children.is_empty() {
            break;
        }

        let ln_parent = (arena[current].visits.max(1) as f64).ln();
        let node_player = arena[current].player;
        // Root player is always stored in arena[0].player.
        let root_player = arena[0].player;
        let is_my_team = same_team_by_roles(root_player, node_player, &state.roles);

        // Clone children list to avoid simultaneous borrow.
        let children: Vec<(Card, usize)> = arena[current].children.clone();
        let (best_card, best_child) = children
            .into_iter()
            .max_by(|(_, ai), (_, bi)| {
                let av = uct_value(&arena[*ai], ln_parent, is_my_team);
                let bv = uct_value(&arena[*bi], ln_parent, is_my_team);
                av.partial_cmp(&bv).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        path.push((current, best_card));
        state.play_card(node_player, best_card);
        finish_current_trick_if_complete(state);
        current = best_child;
    }

    path
}

/// UCT value for a child node (§3.3.3, equation 3.1 with C=0.5).
///
/// Sign is flipped when the selecting player is on the opposing team so
/// opponents minimise the root player's utility (multi-player UCT).
#[inline]
fn uct_value(node: &MctsNode, ln_parent_visits: f64, is_my_team: bool) -> f64 {
    if node.visits == 0 {
        return f64::INFINITY;
    }
    let mean = node.value / node.visits as f64;
    let adjusted = if is_my_team { mean } else { -mean };
    adjusted + UCT_C * (ln_parent_visits / node.visits as f64).sqrt()
}

/// Backpropagation (§3.3.6 / Algorithm 8).
///
/// Teammates get +result; opponents get -result (all from root_player's
/// perspective).
fn mcts_backprop(
    arena: &mut Vec<MctsNode>,
    path: &[(usize, Card)],
    result: f64,
    _root_player: u8,
    _roles: &[PlayerRole; NUM_PLAYERS],
) {
    for &(node_idx, _) in path {
        arena[node_idx].visits += 1;
        // Store values consistently in root-player utility space.
        // Team/opponent perspective is handled in UCT selection via `is_my_team`.
        arena[node_idx].value += result;
    }
}

/// True when `a` and `b` are on the same team.
#[inline]
fn same_team_by_roles(a: u8, b: u8, roles: &[PlayerRole; NUM_PLAYERS]) -> bool {
    if a == b {
        return true;
    }
    matches!(
        (roles[a as usize], roles[b as usize]),
        (PlayerRole::Declarer, PlayerRole::Partner)
            | (PlayerRole::Partner, PlayerRole::Declarer)
            | (PlayerRole::Opponent, PlayerRole::Opponent)
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Heuristic rollout — §6.1.3 tactical rules
// ─────────────────────────────────────────────────────────────────────────────

/// Complete the game from the current state using tactical heuristics.
fn rollout_heuristic(state: &mut GameState, root_player: u8) -> i32 {
    loop {
        if state.tricks_played() >= TRICKS_PER_GAME {
            break;
        }
        if state.current_trick.is_none() {
            state.start_trick(state.current_player);
        }
        let cur = current_turn_player(state);
        let ctx = legal_moves::MoveCtx::from_state(state, cur);
        let legal = legal_moves::generate_legal_moves(&ctx);
        if legal.is_empty() {
            break;
        }
        let chosen = choose_rollout_card(state, cur, legal);
        state.play_card(cur, chosen);
        finish_current_trick_if_complete(state);
    }
    utility_for_root(state, root_player)
}

/// Choose a card for rollout using the three tactical rules from §6.1.3.
///
/// Rule 3: If Mond is on the table and we have Škis → capture it.
/// Rule 1: If last to play, others all played suit, we must play tarok
///         → discard the smallest tarok.
/// Rule 2: If last to play, must play tarok → play smallest winning tarok;
///         if none wins, play smallest tarok.
/// Default: v1 heuristic.
fn choose_rollout_card(state: &GameState, player: u8, legal: CardSet) -> Card {
    let trick = state.current_trick.as_ref();
    let mond = Card::tarok(MOND);
    let skis = Card::tarok(SKIS);

    // Rule 3: capture Mond with Škis.
    if let Some(t) = trick {
        let mond_on_table = t.cards[..t.count as usize]
            .iter()
            .any(|&(_, c)| c == mond);
        if mond_on_table && legal.contains(skis) {
            return skis;
        }
    }

    // Rules 1 & 2: last player forced to tarok.
    if let Some(t) = trick {
        let is_last = t.count as usize == NUM_PLAYERS - 1;
        if is_last {
            let all_legal_taroks = legal.iter().all(|c| c.card_type() == CardType::Tarok);
            if all_legal_taroks {
                // Rule 1: all others played suit.
                let all_suit = t.cards[..t.count as usize]
                    .iter()
                    .all(|&(_, c)| c.card_type() == CardType::Suit);
                if all_suit {
                    if let Some(smallest) = legal.taroks().iter().min_by_key(|c| c.value()) {
                        return smallest;
                    }
                }
                // Rule 2: smallest winning tarok, or smallest tarok.
                let smallest_winner = legal
                    .taroks()
                    .iter()
                    .filter(|&c| tarok_beats_trick(c, t))
                    .min_by_key(|c| c.value());
                let smallest = legal.taroks().iter().min_by_key(|c| c.value());
                if let Some(winner) = smallest_winner.or(smallest) {
                    return winner;
                }
            }
        }
    }

    // Default: v1 heuristic.
    let hand = state.hands[player as usize];
    let v1 = stockskis_v1::choose_card_v1(hand, state, player);
    if legal.contains(v1) { v1 } else { legal.iter().next().unwrap_or(v1) }
}

/// True when `candidate` tarok beats the current best card in `trick`.
fn tarok_beats_trick(candidate: Card, trick: &Trick) -> bool {
    let Some(lead) = trick.lead_card() else {
        return true;
    };
    if lead.card_type() == CardType::Suit {
        return true; // any tarok beats a suit-card lead
    }
    let best_val = trick.cards[..trick.count as usize]
        .iter()
        .filter(|&&(_, c)| c.card_type() == CardType::Tarok)
        .map(|&(_, c)| c.value())
        .max()
        .unwrap_or(0);
    candidate.value() > best_val
}

// ─────────────────────────────────────────────────────────────────────────────
// Determinization (§5.1)
// ─────────────────────────────────────────────────────────────────────────────

fn sample_world(state: &GameState, player: u8, rng: &mut impl rand::Rng) -> GameState {
    let mut world = state.clone();

    // Start from cards still unseen to the root player.
    let mut unseen: Vec<Card> = build_deck()
        .into_iter()
        .filter(|&c| {
            !state.hands[player as usize].contains(c)
                && !state.played_cards.contains(c)
                && !state.put_down.contains(c)
        })
        .collect();
    unseen.shuffle(rng);

    // Keep root player's hand fixed; re-sample all other hands.
    world.hands[player as usize] = state.hands[player as usize];
    let mut remaining = [0usize; NUM_PLAYERS];
    let mut target_assign = 0usize;
    for p in 0..NUM_PLAYERS as u8 {
        if p == player {
            continue;
        }
        world.hands[p as usize] = CardSet::EMPTY;
        let need = state.hands[p as usize].len() as usize;
        remaining[p as usize] = need;
        target_assign += need;
    }

    // Public-belief approximation: infer void suits from observed trick play.
    let void_suits = infer_void_suits(state);

    let mut assigned = 0usize;
    let mut talon = CardSet::EMPTY;
    for &card in unseen.iter() {
        if assigned >= target_assign {
            talon.insert(card);
            continue;
        }

        let mut candidates: Vec<u8> = Vec::new();
        for p in 0..NUM_PLAYERS as u8 {
            if p == player || remaining[p as usize] == 0 {
                continue;
            }
            let blocked = card
                .suit()
                .map(|s| void_suits[p as usize][s as usize])
                .unwrap_or(false);
            if !blocked {
                candidates.push(p);
            }
        }

        let chosen = if !candidates.is_empty() {
            candidates.choose(rng).copied()
        } else {
            (0..NUM_PLAYERS as u8)
                .filter(|&p| p != player && remaining[p as usize] > 0)
                .collect::<Vec<u8>>()
                .choose(rng)
                .copied()
        };

        if let Some(p) = chosen {
            world.hands[p as usize].insert(card);
            remaining[p as usize] -= 1;
            assigned += 1;
        } else {
            talon.insert(card);
        }
    }

    world.talon = talon;

    reconcile_roles(&mut world);
    world
}

fn infer_void_suits(state: &GameState) -> [[bool; 4]; NUM_PLAYERS] {
    let mut voids = [[false; 4]; NUM_PLAYERS];

    for trick in state.tricks.iter() {
        mark_trick_voids(trick, &mut voids);
    }
    if let Some(trick) = &state.current_trick {
        mark_trick_voids(trick, &mut voids);
    }

    voids
}

fn mark_trick_voids(trick: &Trick, voids: &mut [[bool; 4]; NUM_PLAYERS]) {
    let Some(lead) = trick.lead_card() else {
        return;
    };
    let Some(lead_suit) = lead.suit() else {
        return;
    };

    for i in 1..trick.count as usize {
        let (p, c) = trick.cards[i];
        if c.suit() != Some(lead_suit) {
            voids[p as usize][lead_suit as usize] = true;
        }
    }
}

fn reconcile_roles(state: &mut GameState) {
    let Some(contract) = state.contract else {
        return;
    };

    if contract == Contract::Klop {
        for r in state.roles.iter_mut() {
            *r = PlayerRole::Opponent;
        }
        state.partner = None;
        return;
    }

    let Some(declarer) = state.declarer else {
        return;
    };

    for r in state.roles.iter_mut() {
        *r = PlayerRole::Opponent;
    }
    state.roles[declarer as usize] = PlayerRole::Declarer;

    if contract.is_solo() || contract.is_berac() || contract.is_barvni_valat() {
        state.partner = None;
        return;
    }

    state.partner = None;
    if let Some(called_king) = state.called_king {
        for p in 0..NUM_PLAYERS as u8 {
            if p != declarer && state.hands[p as usize].contains(called_king) {
                state.partner = Some(p);
                state.roles[p as usize] = PlayerRole::Partner;
                break;
            }
        }
    }
}

fn current_turn_player(state: &GameState) -> u8 {
    if let Some(trick) = &state.current_trick {
        (trick.lead_player + trick.count) % NUM_PLAYERS as u8
    } else {
        state.current_player
    }
}

fn finish_current_trick_if_complete(state: &mut GameState) {
    if state
        .current_trick
        .as_ref()
        .map_or(false, |t| t.is_complete())
    {
        let (winner, _points) = state.finish_trick();
        state.current_player = winner;
    }
}

fn utility_for_root(state: &GameState, root_player: u8) -> i32 {
    let scores = scoring::score_game(state);
    let Some(contract) = state.contract else {
        return scores[root_player as usize];
    };

    if contract == Contract::Klop {
        return scores[root_player as usize];
    }

    let Some(declarer) = state.declarer else {
        return scores[root_player as usize];
    };
    let decl_score = scores[declarer as usize];
    let root_team = state.get_team(root_player);
    let decl_team = state.get_team(declarer);
    if root_team == decl_team {
        decl_score
    } else {
        -decl_score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lapajne_choose_card_returns_legal_move() {
        let mut gs = GameState::new(0);
        gs.phase = Phase::TrickPlay;
        gs.contract = Some(Contract::Three);
        gs.declarer = Some(0);
        gs.roles = [
            PlayerRole::Declarer,
            PlayerRole::Opponent,
            PlayerRole::Opponent,
            PlayerRole::Opponent,
        ];

        // Build a tiny synthetic trick-play state with 3 cards per player.
        gs.hands = [CardSet::EMPTY; 4];
        gs.hands[0].insert(Card::suit_card(Suit::Hearts, SuitRank::King));
        gs.hands[0].insert(Card::suit_card(Suit::Clubs, SuitRank::Jack));
        gs.hands[0].insert(Card::tarok(5));

        gs.hands[1].insert(Card::suit_card(Suit::Hearts, SuitRank::Queen));
        gs.hands[1].insert(Card::suit_card(Suit::Spades, SuitRank::King));
        gs.hands[1].insert(Card::tarok(6));

        gs.hands[2].insert(Card::suit_card(Suit::Hearts, SuitRank::Pip1));
        gs.hands[2].insert(Card::suit_card(Suit::Diamonds, SuitRank::Jack));
        gs.hands[2].insert(Card::tarok(7));

        gs.hands[3].insert(Card::suit_card(Suit::Hearts, SuitRank::Jack));
        gs.hands[3].insert(Card::suit_card(Suit::Clubs, SuitRank::Queen));
        gs.hands[3].insert(Card::tarok(8));

        gs.current_player = 0;
        gs.start_trick(0);

        let hand = gs.hands[0];
        let legal = legal_moves::generate_legal_moves(&legal_moves::MoveCtx::from_state(&gs, 0));
        let chosen = choose_card_lapajne(hand, &gs, 0);

        assert!(legal.contains(chosen));
    }

    #[test]
    fn lapajne_respects_single_legal_card() {
        let mut gs = GameState::new(0);
        gs.phase = Phase::TrickPlay;
        gs.contract = Some(Contract::Three);
        gs.declarer = Some(0);

        gs.hands = [CardSet::EMPTY; 4];
        let forced = Card::suit_card(Suit::Hearts, SuitRank::King);
        gs.hands[0].insert(forced);
        gs.hands[0].insert(Card::tarok(5));
        gs.hands[0].insert(Card::suit_card(Suit::Clubs, SuitRank::Jack));

        gs.hands[1].insert(Card::suit_card(Suit::Hearts, SuitRank::Queen));
        gs.hands[2].insert(Card::suit_card(Suit::Hearts, SuitRank::Pip1));
        gs.hands[3].insert(Card::suit_card(Suit::Hearts, SuitRank::Jack));

        gs.current_player = 0;
        gs.start_trick(1);
        gs.play_card(1, Card::suit_card(Suit::Hearts, SuitRank::Queen));

        let chosen = choose_card_lapajne(gs.hands[0], &gs, 0);
        assert_eq!(chosen, forced);
    }

    #[test]
    fn lapajne_bidding_paper_thresholds() {
        // 8 taroks + 2 kings + 1 trula → should bid One (§5.3)
        let mut hand = CardSet::EMPTY;
        for t in 5u8..=12 {
            hand.insert(Card::tarok(t));
        }
        hand.insert(Card::suit_card(Suit::Hearts, SuitRank::King));
        hand.insert(Card::suit_card(Suit::Clubs, SuitRank::King));
        hand.insert(Card::tarok(MOND));
        assert_eq!(evaluate_bid_lapajne(hand, None), Some(Contract::One));
    }

    #[test]
    fn lapajne_bidding_weak_hand_passes() {
        let mut hand = CardSet::EMPTY;
        for t in 5u8..=8 {
            hand.insert(Card::tarok(t));
        }
        hand.insert(Card::suit_card(Suit::Hearts, SuitRank::Pip1));
        assert_eq!(evaluate_bid_lapajne(hand, None), None);
    }

    #[test]
    fn lapajne_skis_captures_mond() {
        // §6.1.3 Rule 3: Škis must capture Mond when on the table.
        let mut gs = GameState::new(0);
        gs.phase = Phase::TrickPlay;
        gs.contract = Some(Contract::Three);
        gs.declarer = Some(0);
        gs.roles = [
            PlayerRole::Declarer,
            PlayerRole::Opponent,
            PlayerRole::Opponent,
            PlayerRole::Opponent,
        ];

        gs.hands = [CardSet::EMPTY; 4];
        gs.hands[0].insert(Card::tarok(SKIS));
        gs.hands[0].insert(Card::tarok(15));
        gs.hands[1].insert(Card::tarok(MOND));
        gs.hands[2].insert(Card::tarok(10));
        gs.hands[3].insert(Card::tarok(9));

        gs.current_player = 1;
        gs.start_trick(1);
        gs.play_card(1, Card::tarok(MOND));
        gs.play_card(2, Card::tarok(10));
        gs.play_card(3, Card::tarok(9));

        let legal =
            legal_moves::generate_legal_moves(&legal_moves::MoveCtx::from_state(&gs, 0));
        let chosen = choose_rollout_card(&gs, 0, legal);
        assert_eq!(chosen, Card::tarok(SKIS), "Škis must capture Mond");
    }

    #[test]
    fn lapajne_talon_prefers_tarok_tiebreak() {
        // §5.4: when suit-void count ties, prefer group with more taroks.
        let mut hand = CardSet::EMPTY;
        hand.insert(Card::suit_card(Suit::Hearts, SuitRank::Pip1));
        hand.insert(Card::suit_card(Suit::Hearts, SuitRank::Pip2));
        hand.insert(Card::suit_card(Suit::Spades, SuitRank::Pip1));
        hand.insert(Card::suit_card(Suit::Spades, SuitRank::Pip2));
        hand.insert(Card::suit_card(Suit::Spades, SuitRank::Pip3));
        // must_discard = 1 for single-card groups
        // Both groups let us void diamonds (0 diamonds in hand, irrelevant)
        // Actually: with must_discard=1 and no small suit, both void 0 suits.
        // Tie-break on tarok count: group_b has a tarok.
        let group_a = vec![Card::suit_card(Suit::Hearts, SuitRank::Pip3)];
        let group_b = vec![Card::tarok(5)];
        let best = choose_talon_group_lapajne(&[group_a, group_b], hand, None);
        assert_eq!(best, 1, "group with tarok wins tie-break");
    }

    #[test]
    fn lapajne_uct_opponent_prefers_lower_root_utility() {
        // If node values are stored in root-utility space, opponent selection
        // should prefer *lower* root utility.
        let ln_parent = (10.0f64).ln();
        let good_for_root = MctsNode {
            visits: 10,
            value: 40.0,
            children: Vec::new(),
            untried: Vec::new(),
            player: 0,
        };
        let bad_for_root = MctsNode {
            visits: 10,
            value: -40.0,
            children: Vec::new(),
            untried: Vec::new(),
            player: 0,
        };

        let score_good = uct_value(&good_for_root, ln_parent, false);
        let score_bad = uct_value(&bad_for_root, ln_parent, false);
        assert!(
            score_bad > score_good,
            "opponent should prefer branches with lower root utility"
        );
    }
}
