/// Double-dummy alpha-beta solver for Tarok endgames.
///
/// Operates on a lightweight `DDState` that captures only what matters for
/// the remaining tricks: hands, current trick state, roles, and contract.
/// The solver maximises declarer-team raw card points; opponents minimise.
use crate::card::*;
use crate::game_state::*;
use crate::legal_moves;

// -----------------------------------------------------------------------
// DDState — compact endgame position
// -----------------------------------------------------------------------

#[derive(Clone)]
pub struct DDState {
    pub hands: [CardSet; NUM_PLAYERS],
    pub lead_player: u8,
    pub trick: [(u8, Card); 4],
    pub trick_count: u8,
    /// Completed tricks since entering the DD solver.
    pub tricks_completed: usize,
    /// Trick index in the full game where the endgame started.
    pub game_trick_offset: usize,
    /// Raw card points won by declarer team in the endgame.
    pub decl_points: i32,
    /// Raw card points accumulated per player (used for Klop objective).
    pub player_points: [i32; NUM_PLAYERS],
    /// Number of tricks won per player (used for Berac objective).
    pub tricks_won: [u32; NUM_PLAYERS],
    pub roles: [PlayerRole; NUM_PLAYERS],
    pub contract: Option<Contract>,
    /// Number of seats actively playing (3 for ThreePlayer, 4 for
    /// FourPlayer). Determines trick width and modular player rotation.
    /// The fixed-size hands/roles/trick arrays remain length 4; for 3p
    /// the slot at index 3 is unused (empty hand, default role).
    pub num_players: u8,
    /// Total tricks in the full game (12 for 4p, 16 for 3p). Used to
    /// detect the last trick (which changes Škis-capture rules).
    pub tricks_per_game: u8,
}

/// What metric PIMC/DD should optimise for. All variants express the score
/// so that "higher is better for `viewer`" — `solve_all_moves_viewer` uses
/// this to compute a consistent ranking signal for world aggregation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DDObjective {
    /// Team contracts: declarer team maximises raw card points.
    DeclarerTeamPoints,
    /// Berac: declarer minimises own tricks won; opponents maximise.
    DeclarerTricks,
    /// Klop: viewer minimises own captured card points; everyone else is
    /// modelled adversarially (standard PIMC approximation).
    ViewerCardPoints,
}

impl DDState {
    #[inline]
    pub fn norm_player(player: u8) -> u8 {
        (player as usize % NUM_PLAYERS) as u8
    }

    /// Build a DDState from the current game position.
    ///
    /// `num_players` is 3 (ThreePlayer) or 4 (FourPlayer). For 3p the
    /// 4th slot of the fixed-size arrays must be unused (empty hand,
    /// no entries in the trick).
    pub fn new(
        hands: [CardSet; NUM_PLAYERS],
        current_trick: Option<&Trick>,
        current_player: u8,
        tricks_played: usize,
        roles: [PlayerRole; NUM_PLAYERS],
        contract: Option<Contract>,
        num_players: u8,
        tricks_per_game: u8,
    ) -> Self {
        debug_assert!(num_players == 3 || num_players == 4,
            "DDState supports only 3 or 4 players, got {num_players}");
        let n = num_players as usize;
        let (trick, trick_count, lead_player) = if let Some(ct) = current_trick {
            let lp = (ct.lead_player as usize % n) as u8;
            (ct.cards, ct.count, lp)
        } else {
            ([(0, Card(0)); 4], 0, (current_player as usize % n) as u8)
        };
        let mut state = DDState {
            hands,
            lead_player,
            trick,
            trick_count,
            tricks_completed: 0,
            game_trick_offset: tricks_played,
            decl_points: 0,
            player_points: [0; NUM_PLAYERS],
            tricks_won: [0; NUM_PLAYERS],
            roles,
            contract,
            num_players,
            tricks_per_game,
        };
        // Some callers can hand us a fully populated current trick
        // (count==num_players) that wasn't flushed yet. Resolve it
        // immediately so trick indexing always stays within bounds.
        if state.trick_count >= num_players {
            state.resolve_completed_trick();
        }
        state
    }

    /// Build directly from a GameState (variant inferred from `gs.variant`).
    pub fn from_game_state(gs: &GameState) -> Self {
        Self::new(
            gs.hands,
            gs.current_trick.as_ref(),
            gs.current_player,
            gs.tricks_played(),
            gs.roles,
            gs.contract,
            gs.variant.num_players() as u8,
            gs.variant.tricks_per_game() as u8,
        )
    }

    #[inline]
    pub fn current_player(&self) -> u8 {
        ((self.lead_player as usize + self.trick_count as usize)
            % self.num_players as usize) as u8
    }

    #[inline]
    fn is_terminal(&self) -> bool {
        self.hands.iter().all(|h| h.is_empty()) && self.trick_count == 0
    }

    #[inline]
    fn game_trick_index(&self) -> usize {
        self.game_trick_offset + self.tricks_completed
    }

    #[inline]
    fn is_game_last_trick(&self) -> bool {
        self.game_trick_index() + 1 == self.tricks_per_game as usize
    }

    fn get_team(&self, player: u8) -> Team {
        match self.roles[Self::norm_player(player) as usize] {
            PlayerRole::Declarer | PlayerRole::Partner => Team::DeclarerTeam,
            PlayerRole::Opponent => Team::OpponentTeam,
        }
    }

    /// Resolve `self.trick` when it contains `num_players` cards.
    fn resolve_completed_trick(&mut self) {
        let n = self.num_players as usize;
        if (self.trick_count as usize) < n {
            return;
        }

        // Inline the trick winner / points so we don't depend on the
        // 4-player-hardcoded `trick_eval` paths. We replicate the
        // standard winner rule (taroks beat suit cards; among same
        // type, higher value wins) and the BarvniValat rule (suit
        // cards beat taroks). PagatWinsTrulaTrick is impossible in 3p
        // (declarer team has at most one of trula in hand) and uses
        // the same winner rule in 4p (handled below).

        // Lead suit (None when a tarok leads).
        let lead_card = self.trick[0].1;
        let lead_is_tarok = lead_card.card_type() == CardType::Tarok;
        let lead_suit = if lead_is_tarok { None } else { lead_card.suit() };

        // Trula-pagat special: all three trula cards present (4p only;
        // in 3p with 3-card tricks this never holds).
        let mut trula_set = CardSet::EMPTY;
        for i in 0..n {
            trula_set.insert(self.trick[i].1);
        }
        let pagat_trula = (trula_set.0 & TRULA_MASK) == TRULA_MASK;

        let winner: u8;
        if pagat_trula {
            // Find player who played pagat.
            let pagat = Card::tarok(PAGAT);
            let mut w = 0u8;
            for i in 0..n {
                if self.trick[i].1 == pagat {
                    w = self.trick[i].0;
                    break;
                }
            }
            winner = w;
        } else if self.contract == Some(Contract::BarvniValat) {
            // BarvniValat: any suit card beats all taroks; among suits
            // the lead suit ranks highest, ties broken by higher value.
            let mut has_suit = false;
            for i in 0..n {
                if self.trick[i].1.card_type() == CardType::Suit {
                    has_suit = true;
                    break;
                }
            }
            if has_suit {
                let mut best_player = 0u8;
                let mut best_card: Option<Card> = None;
                for i in 0..n {
                    let (player, card) = self.trick[i];
                    if card.card_type() != CardType::Suit {
                        continue;
                    }
                    match best_card {
                        None => {
                            best_player = player;
                            best_card = Some(card);
                        }
                        Some(bc) => {
                            if card.suit() == bc.suit() {
                                if card.value() > bc.value() {
                                    best_player = player;
                                    best_card = Some(card);
                                }
                            } else if card.suit() == lead_suit {
                                best_player = player;
                                best_card = Some(card);
                            }
                        }
                    }
                }
                winner = best_player;
            } else {
                winner = self.standard_trick_winner(lead_suit);
            }
        } else {
            winner = self.standard_trick_winner(lead_suit);
        }

        let points: i32 = (0..n).map(|i| self.trick[i].1.points() as i32).sum();
        if self.get_team(winner) == Team::DeclarerTeam {
            self.decl_points += points;
        }
        self.player_points[winner as usize] += points;
        self.tricks_won[winner as usize] += 1;

        self.tricks_completed += 1;
        self.lead_player = winner;
        self.trick_count = 0;
        self.trick = [(0, Card(0)); 4];
    }

    /// Standard "highest tarok / highest lead-suit card wins" rule
    /// applied across the first `num_players` entries of the trick.
    #[inline]
    fn standard_trick_winner(&self, lead_suit: Option<Suit>) -> u8 {
        let n = self.num_players as usize;
        let (mut best_player, mut best_card) = self.trick[0];
        for i in 1..n {
            let (player, card) = self.trick[i];
            if card.beats(best_card, lead_suit) {
                best_player = player;
                best_card = card;
            }
        }
        best_player
    }

    /// Compute legal moves for `player` from the current DD position.
    pub fn legal_moves(&self, player: u8) -> CardSet {
        let p = Self::norm_player(player);
        let hand = self.hands[p as usize];
        if hand.is_empty() {
            return CardSet::EMPTY;
        }

        let (lead_card, lead_suit, best_card, trick_cards_set) = if self.trick_count > 0 {
            let lc = self.trick[0].1;
            let ls = if lc.card_type() == CardType::Tarok {
                None
            } else {
                lc.suit()
            };
            let mut best = lc;
            let mut tcs = CardSet::EMPTY;
            for i in 0..self.trick_count as usize {
                let c = self.trick[i].1;
                tcs.insert(c);
                if c.beats(best, ls) {
                    best = c;
                }
            }
            (Some(lc), ls, Some(best), tcs)
        } else {
            (None, None, None, CardSet::EMPTY)
        };

        let ctx = legal_moves::MoveCtx {
            hand,
            lead_card,
            lead_suit,
            best_card,
            contract_name: contract_name_str(self.contract),
            is_last_trick: self.is_game_last_trick(),
            trick_cards: trick_cards_set,
        };

        legal_moves::generate_legal_moves(&ctx)
    }

    /// Play `card` for `player`, returning the resulting state.
    /// If the trick completes, it is resolved (winner + points).
    fn play_card(&self, player: u8, card: Card) -> DDState {
        let mut next = self.clone();
        let n = self.num_players;
        if next.trick_count >= n {
            next.resolve_completed_trick();
        }
        let p = Self::norm_player(player);
        next.hands[p as usize].remove(card);
        next.trick[next.trick_count as usize] = (p, card);
        next.trick_count += 1;

        if next.trick_count == n {
            next.resolve_completed_trick();
        }

        next
    }
}

// -----------------------------------------------------------------------
// Solver
// -----------------------------------------------------------------------

/// Solve the position, returning optimal declarer-team raw card points.
pub fn solve(state: &DDState) -> i32 {
    alpha_beta(state, i32::MIN + 1, i32::MAX - 1)
}

fn alpha_beta(state: &DDState, mut alpha: i32, mut beta: i32) -> i32 {
    if state.is_terminal() {
        return state.decl_points;
    }

    let player = state.current_player();
    let legal = state.legal_moves(player);
    if legal.is_empty() {
        return state.decl_points;
    }

    let maximizing = state.get_team(player) == Team::DeclarerTeam;
    let ordered = ordered_moves(legal, maximizing);

    if maximizing {
        let mut best = i32::MIN + 1;
        for card in ordered {
            let child = state.play_card(player, card);
            let val = alpha_beta(&child, alpha, beta);
            if val > best {
                best = val;
            }
            if val > alpha {
                alpha = val;
            }
            if alpha >= beta {
                break;
            }
        }
        best
    } else {
        let mut best = i32::MAX - 1;
        for card in ordered {
            let child = state.play_card(player, card);
            let val = alpha_beta(&child, alpha, beta);
            if val < best {
                best = val;
            }
            if val < beta {
                beta = val;
            }
            if alpha >= beta {
                break;
            }
        }
        best
    }
}

/// Order legal moves for alpha-beta: maximizer tries high-value / high-point
/// cards first (they tend to produce the best score), minimizer tries low
/// cards first. Card id is used as a stable tiebreaker so ordering is
/// deterministic — this preserves the seeded-PIMC determinism guarantee.
#[inline]
fn ordered_moves(legal: CardSet, maximizing: bool) -> Vec<Card> {
    // Hand sizes here are ≤ 4 (endgame), so allocation + sort is cheap.
    let mut moves: Vec<Card> = legal.iter().collect();
    if maximizing {
        moves.sort_by_key(|c| {
            // Sort key: higher points first, then higher value, then lower id.
            // We invert by using (-points, -value, id).
            (-(c.points() as i32), -(c.value() as i32), c.0 as i32)
        });
    } else {
        moves.sort_by_key(|c| (c.points() as i32, c.value() as i32, c.0 as i32));
    }
    moves
}

/// Solve all legal moves for the current player.
/// Returns `(card, optimal_decl_raw_points)` for each legal move.
pub fn solve_all_moves(state: &DDState) -> Vec<(Card, i32)> {
    let player = state.current_player();
    let legal = state.legal_moves(player);
    let mut results = Vec::with_capacity(legal.len() as usize);
    for card in legal.iter() {
        let child = state.play_card(player, card);
        let val = solve(&child);
        results.push((card, val));
    }
    results
}

// -----------------------------------------------------------------------
// Objective-aware solver (viewer utility)
// -----------------------------------------------------------------------

/// Compute the viewer's terminal utility for the given objective.
/// Higher is always better for `viewer`.
#[inline]
fn viewer_utility(state: &DDState, viewer: u8, obj: DDObjective) -> i32 {
    let v = DDState::norm_player(viewer) as usize;
    match obj {
        DDObjective::DeclarerTeamPoints => {
            if state.get_team(viewer) == Team::DeclarerTeam {
                state.decl_points
            } else {
                -state.decl_points
            }
        }
        DDObjective::DeclarerTricks => {
            // Declarer wants 0 tricks. Find declarer.
            let declarer = (0..state.num_players as usize)
                .find(|&i| state.roles[i] == PlayerRole::Declarer)
                .unwrap_or(0) as u8;
            let decl_tricks = state.tricks_won[DDState::norm_player(declarer) as usize] as i32;
            if DDState::norm_player(viewer) == DDState::norm_player(declarer) {
                -decl_tricks
            } else {
                decl_tricks
            }
        }
        DDObjective::ViewerCardPoints => {
            // Klop: viewer wants 0 card points. Lower viewer points = higher utility.
            -state.player_points[v]
        }
    }
}

/// Should `player` maximise `viewer`'s utility at a decision node?
#[inline]
fn player_maximises_for_viewer(
    state: &DDState,
    player: u8,
    viewer: u8,
    obj: DDObjective,
) -> bool {
    let p = DDState::norm_player(player);
    let v = DDState::norm_player(viewer);
    match obj {
        DDObjective::DeclarerTeamPoints => {
            // Same-team players push decl_points the same direction as viewer.
            state.get_team(p) == state.get_team(v)
        }
        DDObjective::DeclarerTricks => {
            // Declarer alone vs everyone else.
            let is_declarer = |pp: u8| state.roles[pp as usize] == PlayerRole::Declarer;
            is_declarer(p) == is_declarer(v)
        }
        DDObjective::ViewerCardPoints => {
            // Klop: only the viewer maximises own utility; others are
            // modelled adversarially.
            p == v
        }
    }
}

fn alpha_beta_viewer(
    state: &DDState,
    viewer: u8,
    obj: DDObjective,
    mut alpha: i32,
    mut beta: i32,
) -> i32 {
    if state.is_terminal() {
        return viewer_utility(state, viewer, obj);
    }

    let player = state.current_player();
    let legal = state.legal_moves(player);
    if legal.is_empty() {
        return viewer_utility(state, viewer, obj);
    }

    let maximizing = player_maximises_for_viewer(state, player, viewer, obj);
    let ordered = ordered_moves(legal, maximizing);

    if maximizing {
        let mut best = i32::MIN + 1;
        for card in ordered {
            let child = state.play_card(player, card);
            let val = alpha_beta_viewer(&child, viewer, obj, alpha, beta);
            if val > best {
                best = val;
            }
            if val > alpha {
                alpha = val;
            }
            if alpha >= beta {
                break;
            }
        }
        best
    } else {
        let mut best = i32::MAX - 1;
        for card in ordered {
            let child = state.play_card(player, card);
            let val = alpha_beta_viewer(&child, viewer, obj, alpha, beta);
            if val < best {
                best = val;
            }
            if val < beta {
                beta = val;
            }
            if alpha >= beta {
                break;
            }
        }
        best
    }
}

/// Solve all legal moves for the current player under a viewer-centric
/// objective. Each value in the result is viewer utility (higher is better
/// for `viewer`).
pub fn solve_all_moves_viewer(
    state: &DDState,
    viewer: u8,
    obj: DDObjective,
) -> Vec<(Card, i32)> {
    let player = state.current_player();
    let legal = state.legal_moves(player);
    let mut results = Vec::with_capacity(legal.len() as usize);
    for card in legal.iter() {
        let child = state.play_card(player, card);
        let val = alpha_beta_viewer(&child, viewer, obj, i32::MIN + 1, i32::MAX - 1);
        results.push((card, val));
    }
    results
}

// -----------------------------------------------------------------------
// Berač survival solver
// -----------------------------------------------------------------------
//
// Specialised DD for Berač: declarer wants 0 tricks, opponents want ≥1. The
// tree collapses to a Boolean game:
//
//   utility = 1  ↔  declarer won 0 tricks when hands empty
//   utility = 0  ↔  declarer has already won at least one trick
//
// Once declarer picks up any trick, no further play matters — we return 0
// immediately. This aborts the vast majority of branches very early, making
// PIMC at full-hand positions practical.

/// Can the declarer survive from `state` without winning any trick, under
/// perfect play by both sides? `declarer` is the seat id in the full game.
///
/// Internally this is α-β on a {0,1} utility from the declarer's POV, with
/// an early-out as soon as declarer is credited with a trick.
pub fn solve_berac_survival(state: &DDState, declarer: u8) -> bool {
    let d = DDState::norm_player(declarer);
    // Already doomed? Prefix check before any work.
    if state.tricks_won[d as usize] > 0 {
        return false;
    }
    berac_ab(state, d, 0, 1) == 1
}

fn berac_ab(state: &DDState, declarer: u8, mut alpha: i32, mut beta: i32) -> i32 {
    // Hard fail-fast: declarer already took a trick anywhere in this line.
    if state.tricks_won[declarer as usize] > 0 {
        return 0;
    }
    if state.is_terminal() {
        // Survived all tricks with 0 declarer wins.
        return 1;
    }

    let player = state.current_player();
    let legal = state.legal_moves(player);
    if legal.is_empty() {
        return if state.tricks_won[declarer as usize] == 0 { 1 } else { 0 };
    }

    // Declarer maximises survival (wants 1); everyone else minimises (wants 0).
    let maximizing = DDState::norm_player(player) == declarer;

    // Move ordering is relatively unimportant in a {0,1} game — the α-β
    // window collapses on first success/failure. We reuse the generic
    // ordering so declarer tries "safe-looking" (low-points/low-value)
    // cards first, opponents try high cards first.
    let ordered = ordered_moves(legal, !maximizing);

    if maximizing {
        let mut best = 0;
        for card in ordered {
            let child = state.play_card(player, card);
            let val = berac_ab(&child, declarer, alpha, beta);
            if val > best {
                best = val;
            }
            if val > alpha {
                alpha = val;
            }
            if alpha >= beta {
                break;
            }
        }
        best
    } else {
        let mut best = 1;
        for card in ordered {
            let child = state.play_card(player, card);
            let val = berac_ab(&child, declarer, alpha, beta);
            if val < best {
                best = val;
            }
            if val < beta {
                beta = val;
            }
            if alpha >= beta {
                break;
            }
        }
        best
    }
}

/// For each legal move the declarer can play from `state`, return
/// `(card, 1 if a surviving line exists after this move else 0)`.
pub fn berac_solve_all_moves(state: &DDState, declarer: u8) -> Vec<(Card, i32)> {
    let d = DDState::norm_player(declarer);
    let player = state.current_player();
    debug_assert_eq!(DDState::norm_player(player), d,
        "berac_solve_all_moves must be called at a declarer decision node");
    let legal = state.legal_moves(player);
    let mut results = Vec::with_capacity(legal.len() as usize);
    for card in legal.iter() {
        let child = state.play_card(player, card);
        let val = berac_ab(&child, d, 0, 1);
        results.push((card, val));
    }
    results
}

// -----------------------------------------------------------------------
// Helper
// -----------------------------------------------------------------------

fn contract_name_str(contract: Option<Contract>) -> Option<&'static str> {
    contract.map(|c| match c {
        Contract::Klop => "klop",
        Contract::Three => "three",
        Contract::Two => "two",
        Contract::One => "one",
        Contract::SoloThree => "solo_three",
        Contract::SoloTwo => "solo_two",
        Contract::SoloOne => "solo_one",
        Contract::Solo => "solo",
        Contract::Berac => "berac",
        Contract::BarvniValat => "barvni_valat",
        Contract::Valat => "valat",
    })
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a 1-trick endgame (each player has 1 card).
    fn one_trick_state(
        cards: [(u8, Card); 4],
        lead: u8,
        roles: [PlayerRole; NUM_PLAYERS],
        contract: Option<Contract>,
    ) -> DDState {
        let mut hands = [CardSet::EMPTY; NUM_PLAYERS];
        for &(p, c) in &cards {
            hands[p as usize].insert(c);
        }
        DDState::new(hands, None, lead, 11, roles, contract, 4, 12)
    }

    #[test]
    fn single_trick_declarer_wins() {
        // Declarer leads Mond (21), opponents have low taroks, partner has tarok 5
        let roles = [
            PlayerRole::Declarer,
            PlayerRole::Opponent,
            PlayerRole::Partner,
            PlayerRole::Opponent,
        ];
        let state = one_trick_state(
            [
                (0, Card::tarok(MOND)),
                (1, Card::tarok(3)),
                (2, Card::tarok(5)),
                (3, Card::tarok(7)),
            ],
            0,
            roles,
            Some(Contract::Three),
        );
        let val = solve(&state);
        // Mond(5) + three taroks(1+1+1) = 8 raw points, declarer team wins
        assert_eq!(val, 8);
    }

    #[test]
    fn single_trick_opponent_wins() {
        // Opponent 1 leads Škis, declarer has low tarok
        let roles = [
            PlayerRole::Opponent,
            PlayerRole::Declarer,
            PlayerRole::Opponent,
            PlayerRole::Partner,
        ];
        let state = one_trick_state(
            [
                (0, Card::tarok(SKIS)),
                (1, Card::tarok(2)),
                (2, Card::tarok(4)),
                (3, Card::tarok(6)),
            ],
            0,
            roles,
            Some(Contract::Three),
        );
        let val = solve(&state);
        // Škis(5) + taroks = 8, but opponent wins → decl gets 0
        assert_eq!(val, 0);
    }

    #[test]
    fn solve_all_moves_returns_legal() {
        let roles = [
            PlayerRole::Declarer,
            PlayerRole::Opponent,
            PlayerRole::Partner,
            PlayerRole::Opponent,
        ];
        // Declarer has 2 taroks, 2 tricks left
        let mut hands = [CardSet::EMPTY; 4];
        hands[0].insert(Card::tarok(MOND));
        hands[0].insert(Card::tarok(10));
        hands[1].insert(Card::tarok(3));
        hands[1].insert(Card::tarok(4));
        hands[2].insert(Card::tarok(5));
        hands[2].insert(Card::tarok(6));
        hands[3].insert(Card::tarok(7));
        hands[3].insert(Card::tarok(8));

        let state = DDState::new(hands, None, 0, 10, roles, Some(Contract::Three), 4, 12);
        let moves = solve_all_moves(&state);
        assert_eq!(moves.len(), 2);
        // Both moves should be taroks from declarer's hand
        for (card, _val) in &moves {
            assert!(hands[0].contains(*card));
        }
    }
}
