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
    pub roles: [PlayerRole; NUM_PLAYERS],
    pub contract: Option<Contract>,
}

impl DDState {
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
            (ct.cards, ct.count, ct.lead_player)
        } else {
            ([(0, Card(0)); 4], 0, current_player)
        };
        DDState {
            hands,
            lead_player,
            trick,
            trick_count,
            tricks_completed: 0,
            game_trick_offset: tricks_played,
            decl_points: 0,
            roles,
            contract,
        }
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
        match self.roles[player as usize] {
            PlayerRole::Declarer | PlayerRole::Partner => Team::DeclarerTeam,
            PlayerRole::Opponent => Team::OpponentTeam,
        }
    }

    /// Compute legal moves for `player` from the current DD position.
    pub fn legal_moves(&self, player: u8) -> CardSet {
        let hand = self.hands[player as usize];
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
        next.hands[player as usize].remove(card);
        next.trick[next.trick_count as usize] = (player, card);
        next.trick_count += 1;

        if next.trick_count == 4 {
            // Build a Trick and resolve it
            let mut t = Trick::new(next.lead_player);
            for i in 0..4 {
                t.play(next.trick[i].0, next.trick[i].1);
            }
            let result =
                trick_eval::evaluate_trick(&t, next.is_game_last_trick(), next.contract);

            // Accumulate raw card points for the winning team
            let points: i32 = (0..4).map(|i| next.trick[i].1.points() as i32).sum();
            if next.get_team(result.winner) == Team::DeclarerTeam {
                next.decl_points += points;
            }

            next.tricks_completed += 1;
            next.lead_player = result.winner;
            next.trick_count = 0;
            next.trick = [(0, Card(0)); 4];
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
