/// Double-dummy alpha-beta solver for Tarok endgames.
///
/// Operates on a lightweight `DDState` that captures only what matters for
/// the remaining tricks: hands, current trick state, roles, and contract.
/// The solver maximises declarer-team raw card points; opponents minimise.
use crate::card::*;
use crate::game_state::*;
use crate::legal_moves;
use crate::trick_eval;

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
    /// Trick index in the full 12-trick game where the endgame started.
    pub game_trick_offset: usize,
    /// Raw card points won by declarer team in the endgame.
    pub decl_points: i32,
    /// Raw card points accumulated per player (used for Klop objective).
    pub player_points: [i32; NUM_PLAYERS],
    /// Number of tricks won per player (used for Berac objective).
    pub tricks_won: [u32; NUM_PLAYERS],
    pub roles: [PlayerRole; NUM_PLAYERS],
    pub contract: Option<Contract>,
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
    pub fn new(
        hands: [CardSet; NUM_PLAYERS],
        current_trick: Option<&Trick>,
        current_player: u8,
        tricks_played: usize,
        roles: [PlayerRole; NUM_PLAYERS],
        contract: Option<Contract>,
    ) -> Self {
        let (trick, trick_count, lead_player) = if let Some(ct) = current_trick {
            (ct.cards, ct.count, Self::norm_player(ct.lead_player))
        } else {
            ([(0, Card(0)); 4], 0, Self::norm_player(current_player))
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
        };
        // Some callers can hand us a fully populated current trick
        // (count==4) that wasn't flushed yet. Resolve it immediately so
        // trick indexing always stays within 0..=3.
        if state.trick_count >= 4 {
            state.resolve_completed_trick();
        }
        state
    }

    /// Build directly from a GameState.
    pub fn from_game_state(gs: &GameState) -> Self {
        Self::new(
            gs.hands,
            gs.current_trick.as_ref(),
            gs.current_player,
            gs.tricks_played(),
            gs.roles,
            gs.contract,
        )
    }

    #[inline]
    pub fn current_player(&self) -> u8 {
        ((self.lead_player as usize + self.trick_count as usize) % NUM_PLAYERS) as u8
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
        self.game_trick_index() == TRICKS_PER_GAME - 1
    }

    fn get_team(&self, player: u8) -> Team {
        match self.roles[Self::norm_player(player) as usize] {
            PlayerRole::Declarer | PlayerRole::Partner => Team::DeclarerTeam,
            PlayerRole::Opponent => Team::OpponentTeam,
        }
    }

    /// Resolve `self.trick` when it contains 4 cards.
    fn resolve_completed_trick(&mut self) {
        if self.trick_count < 4 {
            return;
        }

        let mut t = Trick::new(self.lead_player);
        for i in 0..4 {
            t.play(self.trick[i].0, self.trick[i].1);
        }
        let result = trick_eval::evaluate_trick(&t, self.is_game_last_trick(), self.contract);

        let points: i32 = (0..4).map(|i| self.trick[i].1.points() as i32).sum();
        let winner = Self::norm_player(result.winner);
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
        if next.trick_count >= 4 {
            next.resolve_completed_trick();
        }
        let p = Self::norm_player(player);
        next.hands[p as usize].remove(card);
        next.trick[next.trick_count as usize] = (p, card);
        next.trick_count += 1;

        if next.trick_count == 4 {
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

    if maximizing {
        let mut best = i32::MIN + 1;
        for card in legal.iter() {
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
        for card in legal.iter() {
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
            let declarer = (0..NUM_PLAYERS)
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

    if maximizing {
        let mut best = i32::MIN + 1;
        for card in legal.iter() {
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
        for card in legal.iter() {
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
        DDState::new(hands, None, lead, 11, roles, contract)
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

        let state = DDState::new(hands, None, 0, 10, roles, Some(Contract::Three));
        let moves = solve_all_moves(&state);
        assert_eq!(moves.len(), 2);
        // Both moves should be taroks from declarer's hand
        for (card, _val) in &moves {
            assert!(hands[0].contains(*card));
        }
    }
}
