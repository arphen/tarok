/// PyO3 bindings — exposes the Rust engine to Python.
///
/// The API is designed to be a drop-in replacement for the Python engine.
/// Python code calls these functions with plain ints/lists and gets back
/// plain lists/dicts.

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use rand::prelude::*;
use rand::rng;

use crate::card::*;
// use crate::double_dummy;
use crate::encoding;
use crate::game_state::*;
use crate::legal_moves;
// use crate::pimc;
use crate::scoring;
use crate::trick_eval;
use crate::warmup;
use serde_json;

// -----------------------------------------------------------------------
// PyGameState — the main Python-visible object
// -----------------------------------------------------------------------

#[pyclass(name = "RustGameState")]
pub struct PyGameState {
    pub state: GameState,
}

#[pymethods]
impl PyGameState {
    #[new]
    #[pyo3(signature = (dealer=0))]
    fn new(dealer: u8) -> Self {
        PyGameState {
            state: GameState::new(dealer),
        }
    }

    // -- Card dealing --

    fn deal(&mut self) {
        let mut deck = build_deck();
        deck.shuffle(&mut rng());
        // Deal 12 cards to each player, 6 to talon
        for (i, &card) in deck.iter().enumerate() {
            if i < 48 {
                self.state.hands[i / 12].insert(card);
            } else {
                self.state.talon.insert(card);
            }
        }
        self.state.phase = Phase::Bidding;
    }

    fn deal_hands(&mut self, hands: Vec<Vec<u8>>, talon: Vec<u8>) -> PyResult<()> {
        if hands.len() != NUM_PLAYERS {
            return Err(pyo3::exceptions::PyValueError::new_err("Need 4 hands"));
        }
        for (i, hand) in hands.iter().enumerate() {
            let mut cs = CardSet::EMPTY;
            for &idx in hand {
                cs.insert(Card(idx));
            }
            self.state.hands[i] = cs;
        }
        let mut ts = CardSet::EMPTY;
        for &idx in &talon {
            ts.insert(Card(idx));
        }
        self.state.talon = ts;
        self.state.phase = Phase::Bidding;
        Ok(())
    }

    // -- Phase / state getters --

    #[getter]
    fn phase(&self) -> u8 {
        self.state.phase as u8
    }

    #[setter]
    fn set_phase(&mut self, v: u8) {
        self.state.phase = match v {
            0 => Phase::Dealing,
            1 => Phase::Bidding,
            2 => Phase::KingCalling,
            3 => Phase::TalonExchange,
            4 => Phase::Announcements,
            5 => Phase::TrickPlay,
            6 => Phase::Scoring,
            _ => Phase::Finished,
        };
    }

    #[getter]
    fn dealer(&self) -> u8 {
        self.state.dealer
    }

    #[setter]
    fn set_dealer(&mut self, v: u8) {
        self.state.dealer = v;
    }

    #[getter]
    fn current_player(&self) -> u8 {
        self.state.current_player
    }

    #[setter]
    fn set_current_player(&mut self, v: u8) {
        self.state.current_player = v;
    }

    #[getter]
    fn declarer(&self) -> Option<u8> {
        self.state.declarer
    }

    #[setter]
    fn set_declarer(&mut self, v: Option<u8>) {
        self.state.declarer = v;
    }

    #[getter]
    fn partner(&self) -> Option<u8> {
        self.state.partner
    }

    #[setter]
    fn set_partner(&mut self, v: Option<u8>) {
        self.state.partner = v;
    }

    #[getter]
    fn contract(&self) -> Option<u8> {
        self.state.contract.map(|c| c as u8)
    }

    #[setter]
    fn set_contract(&mut self, v: Option<u8>) {
        self.state.contract = v.and_then(Contract::from_u8);
    }

    #[getter]
    fn tricks_played(&self) -> usize {
        self.state.tricks_played()
    }

    #[getter]
    fn is_last_trick(&self) -> bool {
        self.state.is_last_trick()
    }

    #[getter]
    fn is_partner_revealed(&self) -> bool {
        self.state.is_partner_revealed()
    }

    // -- Hand access --

    fn hand(&self, player: u8) -> Vec<u8> {
        self.state.hands[player as usize]
            .iter()
            .map(|c| c.0)
            .collect()
    }

    fn hand_size(&self, player: u8) -> u32 {
        self.state.hands[player as usize].len()
    }

    fn remove_card(&mut self, player: u8, card_idx: u8) {
        self.state.hands[player as usize].remove(Card(card_idx));
    }

    // -- Bidding --

    fn add_bid(&mut self, player: u8, contract: Option<u8>) {
        self.state.bids.push(Bid {
            player,
            contract: contract.and_then(Contract::from_u8),
        });
    }

    fn legal_bids(&self, player: u8) -> Vec<Option<u8>> {
        // Keep a single source of truth for legality in GameState (training + live).
        let mut result: Vec<Option<u8>> = vec![None]; // pass always legal
        for c in self.state.legal_bids(player) {
            result.push(Some(c as u8));
        }
        result
    }

    // -- King calling --

    fn set_called_king(&mut self, card_idx: u8) {
        self.state.called_king = Some(Card(card_idx));
    }

    fn callable_kings(&self) -> Vec<u8> {
        let declarer = self.state.declarer.unwrap() as usize;
        let hand = self.state.hands[declarer];
        let mut kings = Vec::new();
        for s in Suit::ALL {
            let king = Card::suit_card(s, SuitRank::King);
            if !hand.contains(king) {
                kings.push(king.0);
            }
        }
        if kings.is_empty() {
            // Has all 4 kings — use queens
            for s in Suit::ALL {
                let queen = Card::suit_card(s, SuitRank::Queen);
                if !hand.contains(queen) {
                    kings.push(queen.0);
                }
            }
        }
        kings
    }

    // -- Talon --

    fn set_talon_revealed(&mut self, groups: Vec<Vec<u8>>) {
        self.state.talon_revealed = groups
            .into_iter()
            .map(|g| g.into_iter().map(Card).collect())
            .collect();
    }

    fn add_put_down(&mut self, card_idx: u8) {
        self.state.put_down.insert(Card(card_idx));
    }

    fn add_to_hand(&mut self, player: u8, card_idx: u8) {
        self.state.hands[player as usize].insert(Card(card_idx));
    }

    /// Return talon card indices.
    fn talon(&self) -> Vec<u8> {
        self.state.talon.iter().map(|c| c.0).collect()
    }

    /// Remove a card from the talon.
    fn remove_from_talon(&mut self, card_idx: u8) {
        self.state.talon.remove(Card(card_idx));
    }

    // -- Announcements --

    fn announce(&mut self, player: u8, announcement: u8) {
        self.state.announcements[player as usize] |= 1 << announcement;
    }

    fn set_kontra_level(&mut self, target: u8, level: u8) {
        let kl = match level {
            0 => KontraLevel::None,
            1 => KontraLevel::Kontra,
            2 => KontraLevel::Re,
            3 => KontraLevel::Sub,
            _ => KontraLevel::None,
        };
        if (target as usize) < KontraTarget::NUM {
            self.state.kontra_levels[target as usize] = kl;
        }
    }

    // -- Roles --

    fn set_role(&mut self, player: u8, role: u8) {
        self.state.roles[player as usize] = match role {
            0 => PlayerRole::Declarer,
            1 => PlayerRole::Partner,
            _ => PlayerRole::Opponent,
        };
    }

    fn get_role(&self, player: u8) -> u8 {
        self.state.roles[player as usize] as u8
    }

    fn get_team(&self, player: u8) -> u8 {
        self.state.get_team(player) as u8
    }

    // -- Trick play --

    fn start_trick(&mut self, lead_player: u8) {
        self.state.current_trick = Some(Trick::new(lead_player));
    }

    fn play_card(&mut self, player: u8, card_idx: u8) {
        let card = Card(card_idx);
        self.state.hands[player as usize].remove(card);
        if let Some(ref mut trick) = self.state.current_trick {
            trick.play(player, card);
        }
        self.state.played_cards.insert(card);
    }

    fn finish_trick(&mut self) -> (u8, u8) {
        let trick = self.state.current_trick.take().expect("No current trick");
        let is_last = self.state.tricks.len() == 11;
        let result = trick_eval::evaluate_trick(&trick, is_last, self.state.contract);
        self.state.tricks.push(trick);
        (result.winner, result.points)
    }

    fn is_trick_complete(&self) -> bool {
        self.state
            .current_trick
            .as_ref()
            .map_or(false, |t| t.is_complete())
    }

    fn current_trick_cards(&self) -> Vec<(u8, u8)> {
        match &self.state.current_trick {
            Some(t) => (0..t.count as usize)
                .map(|i| (t.cards[i].0, t.cards[i].1 .0))
                .collect(),
            None => Vec::new(),
        }
    }

    // -- Legal moves --

    fn legal_plays(&self, player: u8) -> Vec<u8> {
        let ctx = legal_moves::MoveCtx::from_state(&self.state, player);
        let legal = legal_moves::generate_legal_moves(&ctx);
        legal.iter().map(|c| c.0).collect()
    }

    /// Return legal plays as a 54-element binary mask (numpy array).
    fn legal_plays_mask<'py>(&self, py: Python<'py>, player: u8) -> Bound<'py, PyArray1<f32>> {
        let ctx = legal_moves::MoveCtx::from_state(&self.state, player);
        let legal = legal_moves::generate_legal_moves(&ctx);
        let mut buf = [0.0f32; DECK_SIZE];
        encoding::encode_legal_mask(&mut buf, legal);
        PyArray1::from_slice(py, &buf)
    }

    // -- Encoding --

    /// Encode state as a numpy array (STATE_SIZE=450).
    fn encode_state<'py>(
        &self,
        py: Python<'py>,
        player: u8,
        decision_type: u8,
    ) -> Bound<'py, PyArray1<f32>> {
        let mut buf = [0.0f32; encoding::STATE_SIZE];
        encoding::encode_state(&mut buf, &self.state, player, decision_type);
        PyArray1::from_slice(py, &buf)
    }

    /// Encode oracle state as a numpy array (ORACLE_STATE_SIZE=612).
    fn encode_oracle_state<'py>(
        &self,
        py: Python<'py>,
        player: u8,
        decision_type: u8,
    ) -> Bound<'py, PyArray1<f32>> {
        let mut buf = [0.0f32; encoding::ORACLE_STATE_SIZE];
        encoding::encode_oracle_state(&mut buf, &self.state, player, decision_type);
        PyArray1::from_slice(py, &buf)
    }

    // -- Scoring --

    fn score_game(&self) -> [i32; 4] {
        scoring::score_game(&self.state)
    }

    /// Return a JSON string with full scoring breakdown for UI display.
    fn score_game_breakdown_json(&self) -> String {
        let breakdown = scoring::score_game_breakdown(&self.state);
        serde_json::to_string(&breakdown).expect("breakdown serialization failed")
    }

    // -- Card utilities --

    #[staticmethod]
    fn card_label(card_idx: u8) -> String {
        Card(card_idx).label()
    }

    #[staticmethod]
    fn card_points(card_idx: u8) -> u8 {
        Card(card_idx).points()
    }

    #[staticmethod]
    fn card_beats(a: u8, b: u8, lead_suit: Option<u8>) -> bool {
        Card(a).beats(Card(b), lead_suit.and_then(Suit::from_u8))
    }

    #[staticmethod]
    fn compute_card_points(card_indices: Vec<u8>) -> i32 {
        let cards: Vec<Card> = card_indices.into_iter().map(Card).collect();
        scoring::compute_card_points(&cards)
    }

    #[staticmethod]
    fn deck_size() -> usize {
        DECK_SIZE
    }

    // -- Batch simulation --

    /// Run N random games and return total card points per player (for benchmarking).
    #[staticmethod]
    fn bench_random_games(n: usize) -> [i64; 4] {
        let mut totals = [0i64; 4];
        for _ in 0..n {
            let scores = play_random_game();
            for p in 0..NUM_PLAYERS {
                totals[p] += scores[p] as i64;
            }
        }
        totals
    }

    // -- StockŠkis V5 decision functions (exposed for Python V5 player) --

    /// V5 bidding: returns the chosen contract as Option<u8>, or None for pass.
    fn v5_choose_bid(&self, player: u8) -> Option<u8> {
        let hand = self.state.hands[player as usize];
        let highest = self.state.bids.iter()
            .filter_map(|b| b.contract)
            .max_by_key(|c| c.strength());
        crate::bots::stockskis_v5::evaluate_bid_v5(hand, highest).map(|c| c as u8)
    }

    /// V5 king calling: returns the card index to call.
    fn v5_choose_king(&self, player: u8) -> Option<u8> {
        let hand = self.state.hands[player as usize];
        crate::bots::stockskis_v5::choose_king_v5(hand).map(|c| c.0)
    }

    /// V5 talon group selection: returns the group index (0 or 1, or 0-5).
    fn v5_choose_talon_group(&self, player: u8, groups: Vec<Vec<u8>>) -> usize {
        let hand = self.state.hands[player as usize];
        let groups_cards: Vec<Vec<Card>> = groups.iter()
            .map(|g| g.iter().map(|&idx| Card(idx)).collect())
            .collect();
        crate::bots::stockskis_v5::choose_talon_group_v5(
            &groups_cards, hand, self.state.called_king
        )
    }

    /// V5 discard selection: returns card indices to discard.
    fn v5_choose_discards(&self, player: u8, must_discard: usize) -> Vec<u8> {
        let hand = self.state.hands[player as usize];
        crate::bots::stockskis_v5::choose_discards_v5(hand, must_discard, self.state.called_king)
            .iter().map(|c| c.0).collect()
    }

    /// V5 card play: returns the card index to play.
    fn v5_choose_card(&self, player: u8) -> u8 {
        let hand = self.state.hands[player as usize];
        crate::bots::stockskis_v5::choose_card_v5(hand, &self.state, player).0
    }

    // -- StockŠkis M6 decision functions --

    fn m6_choose_bid(&self, player: u8) -> Option<u8> {
        let hand = self.state.hands[player as usize];
        let highest = self.state.bids.iter()
            .filter_map(|b| b.contract)
            .max_by_key(|c| c.strength());
        crate::bots::stockskis_m6::evaluate_bid_m6(hand, highest).map(|c| c as u8)
    }

    fn m6_choose_king(&self, player: u8) -> Option<u8> {
        let hand = self.state.hands[player as usize];
        crate::bots::stockskis_m6::choose_king_m6(hand).map(|c| c.0)
    }

    fn m6_choose_talon_group(&self, player: u8, groups: Vec<Vec<u8>>) -> usize {
        let hand = self.state.hands[player as usize];
        let groups_cards: Vec<Vec<Card>> = groups.iter()
            .map(|g| g.iter().map(|&idx| Card(idx)).collect())
            .collect();
        crate::bots::stockskis_m6::choose_talon_group_m6(
            &groups_cards, hand, self.state.called_king
        )
    }

    fn m6_choose_discards(&self, player: u8, must_discard: usize) -> Vec<u8> {
        let hand = self.state.hands[player as usize];
        crate::bots::stockskis_m6::choose_discards_m6(hand, must_discard, self.state.called_king)
            .iter().map(|c| c.0).collect()
    }

    fn m6_choose_card(&self, player: u8) -> u8 {
        let hand = self.state.hands[player as usize];
        crate::bots::stockskis_m6::choose_card_m6(hand, &self.state, player).0
    }

    // -- POŽRL (Domen Požrl, 2021 thesis) decision functions --

    fn pozrl_choose_bid(&self, player: u8) -> Option<u8> {
        let hand = self.state.hands[player as usize];
        let highest = self
            .state
            .bids
            .iter()
            .filter_map(|b| b.contract)
            .max_by_key(|c| c.strength());
        crate::bots::stockskis_pozrl::evaluate_bid_pozrl(hand, highest).map(|c| c as u8)
    }

    fn pozrl_choose_king(&self, player: u8) -> Option<u8> {
        let hand = self.state.hands[player as usize];
        crate::bots::stockskis_pozrl::choose_king_pozrl(hand).map(|c| c.0)
    }

    fn pozrl_choose_talon_group(&self, player: u8, groups: Vec<Vec<u8>>) -> usize {
        let hand = self.state.hands[player as usize];
        let groups_cards: Vec<Vec<Card>> = groups
            .iter()
            .map(|g| g.iter().map(|&idx| Card(idx)).collect())
            .collect();
        crate::bots::stockskis_pozrl::choose_talon_group_pozrl(
            &groups_cards,
            hand,
            self.state.called_king,
        )
    }

    fn pozrl_choose_discards(&self, player: u8, must_discard: usize) -> Vec<u8> {
        let hand = self.state.hands[player as usize];
        crate::bots::stockskis_pozrl::choose_discards_pozrl(hand, must_discard, self.state.called_king)
            .iter()
            .map(|c| c.0)
            .collect()
    }

    fn pozrl_choose_card(&self, player: u8) -> u8 {
        let hand = self.state.hands[player as usize];
        crate::bots::stockskis_pozrl::choose_card_pozrl(hand, &self.state, player).0
    }
}

/// Play a single random game (for benchmarking throughput).
fn play_random_game() -> [i32; 4] {
    let mut r = rng();
    let mut state = GameState::new(0);

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

    // Skip bidding; assign Klop
    state.contract = Some(Contract::Klop);
    state.phase = Phase::TrickPlay;
    for i in 0..NUM_PLAYERS {
        state.roles[i] = PlayerRole::Opponent;
    }

    // Play 12 tricks
    let mut lead_player = 1u8; // forehand
    for trick_num in 0..12 {
        state.current_trick = Some(Trick::new(lead_player));

        for offset in 0..4 {
            let player = (lead_player + offset) % NUM_PLAYERS as u8;
            let ctx = legal_moves::MoveCtx::from_state(&state, player);
            let legal = legal_moves::generate_legal_moves(&ctx);

            // Pick a random legal card
            let legal_vec: Vec<Card> = legal.iter().collect();
            let &card = legal_vec.choose(&mut r).unwrap();

            state.hands[player as usize].remove(card);
            state.current_trick.as_mut().unwrap().play(player, card);
            state.played_cards.insert(card);
        }

        let trick = state.current_trick.take().unwrap();
        let is_last = trick_num == 11;
        let result = trick_eval::evaluate_trick(&trick, is_last, state.contract);
        lead_player = result.winner;
        state.tricks.push(trick);
    }

    state.phase = Phase::Scoring;
    scoring::score_game(&state)
}

// -----------------------------------------------------------------------
// Module registration
// -----------------------------------------------------------------------

/// Generate warmup experiences from random games (value pre-training).
///
/// Returns dict with numpy arrays:
///   states:        (N, STATE_SIZE) float32
///   oracle_states: (N, ORACLE_STATE_SIZE) float32  — empty if include_oracle=False
///   decision_types: (N,) uint8
///   rewards:       (N,) float32
#[pyfunction]
#[pyo3(signature = (num_games, include_oracle=true))]
fn generate_warmup_data(
    py: Python<'_>,
    num_games: usize,
    include_oracle: bool,
) -> PyResult<PyObject> {
    let batch = warmup::generate_warmup_batch(num_games, include_oracle);
    let n = batch.rewards.len();

    let dict = pyo3::types::PyDict::new(py);
    // Reshape states to (N, STATE_SIZE)
    let states = numpy::PyArray2::<f32>::from_vec2(
        py,
        &batch
            .states
            .chunks(batch.state_size)
            .map(|c| c.to_vec())
            .collect::<Vec<_>>(),
    )
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("states: {e}")))?;
    dict.set_item("states", states)?;

    if include_oracle && !batch.oracle_states.is_empty() {
        let oracle = numpy::PyArray2::<f32>::from_vec2(
            py,
            &batch
                .oracle_states
                .chunks(batch.oracle_state_size)
                .map(|c| c.to_vec())
                .collect::<Vec<_>>(),
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("oracle: {e}")))?;
        dict.set_item("oracle_states", oracle)?;
    } else {
        dict.set_item("oracle_states", py.None())?;
    }

    dict.set_item(
        "decision_types",
        numpy::PyArray1::<u8>::from_vec(py, batch.decision_types),
    )?;
    dict.set_item(
        "rewards",
        numpy::PyArray1::<f32>::from_vec(py, batch.rewards),
    )?;
    dict.set_item("num_experiences", n)?;

    Ok(dict.into())
}

// -----------------------------------------------------------------------
// Standalone functions — bridge for Python engine replacement
// -----------------------------------------------------------------------

/// Evaluate a completed trick and return the winner player index.
///
/// Args:
///   cards: list of (player_idx, card_idx) tuples representing the 4-card trick
///   is_last_trick: whether this is the 12th trick
///   contract: optional contract id (u8), None for klop
///
/// Returns: winner player index (u8)
#[pyfunction]
#[pyo3(signature = (cards, is_last_trick=false, contract=None))]
fn evaluate_trick_winner(cards: Vec<(u8, u8)>, is_last_trick: bool, contract: Option<u8>) -> PyResult<u8> {
    if cards.len() != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err("Trick must have exactly 4 cards"));
    }
    let mut trick = Trick::new(cards[0].0);
    for &(player, card_idx) in &cards {
        trick.play(player, Card(card_idx));
    }
    let contract_enum = contract.and_then(Contract::from_u8);
    let result = trick_eval::evaluate_trick(&trick, is_last_trick, contract_enum);
    Ok(result.winner)
}

/// Compute the legal plays for a player given their hand and the current trick state.
///
/// Args:
///   hand: list of card indices in the player's hand
///   trick_cards: list of (player_idx, card_idx) tuples already played in the trick (may be empty)
///   contract: optional contract id (u8)
///   is_last_trick: whether this is the 12th trick
///
/// Returns: list of card indices that are legal to play
#[pyfunction]
#[pyo3(signature = (hand, trick_cards, contract=None, is_last_trick=false))]
fn compute_legal_plays(hand: Vec<u8>, trick_cards: Vec<(u8, u8)>, contract: Option<u8>, is_last_trick: bool) -> Vec<u8> {
    let mut hand_set = CardSet::EMPTY;
    for &idx in &hand {
        hand_set.insert(Card(idx));
    }

    let contract_enum = contract.and_then(Contract::from_u8);
    let contract_name: Option<&'static str> = match contract_enum {
        Some(Contract::Klop) => Some("klop"),
        Some(Contract::Berac) => Some("berac"),
        Some(Contract::BarvniValat) => Some("barvni_valat"),
        Some(Contract::Three) => Some("three"),
        Some(Contract::Two) => Some("two"),
        Some(Contract::One) => Some("one"),
        Some(Contract::SoloThree) => Some("solo_three"),
        Some(Contract::SoloTwo) => Some("solo_two"),
        Some(Contract::SoloOne) => Some("solo_one"),
        Some(Contract::Solo) => Some("solo"),
        None => None,
    };

    let (lead_card, lead_suit, best_card, trick_card_set) = if !trick_cards.is_empty() {
        let first_card = Card(trick_cards[0].1);
        let ls = first_card.suit();
        let mut best = first_card;
        let mut tcs = CardSet::EMPTY;
        for &(_, card_idx) in &trick_cards {
            let c = Card(card_idx);
            tcs.insert(c);
            if c.beats(best, ls) {
                best = c;
            }
        }
        (Some(first_card), ls, Some(best), tcs)
    } else {
        (None, None, None, CardSet::EMPTY)
    };

    let ctx = legal_moves::MoveCtx {
        hand: hand_set,
        lead_card,
        lead_suit,
        best_card,
        contract_name,
        is_last_trick,
        trick_cards: trick_card_set,
    };

    let legal = legal_moves::generate_legal_moves(&ctx);
    legal.iter().map(|c| c.0).collect()
}

// -----------------------------------------------------------------------
// Self-play with tch-rs (pure-Rust NN inference, zero GIL)
// -----------------------------------------------------------------------

/// Run batched self-play with a TorchScript model entirely in Rust.
///
/// Returns a dict of numpy arrays:
///   states:         (N, STATE_SIZE) float32
///   actions:        (N,) uint16
///   log_probs:      (N,) float32
///   values:         (N,) float32
///   decision_types: (N,) uint8
///   legal_masks:    list of N float32 arrays (variable length per dt)
///   rewards:        (N,) float32  — per-step reward (based on final game score)
///   game_ids:       (N,) uint32
///   players:        (N,) uint8   — which seat made this decision
///   n_games:        int
///   n_experiences:  int
///   scores:         (n_games, 4) int32
///   oracle_states:  (N, ORACLE_STATE_SIZE) float32, only when `include_oracle_states=True`
///   initial_hands:  (n_games, 4, 12) uint8, only when `include_replay_data=True`
///   initial_talon:  (n_games, 6) uint8, only when `include_replay_data=True`
///   traces:         list[dict], only when `include_replay_data=True`
#[pyfunction]
#[pyo3(signature = (n_games, concurrency=64, model_path=None, explore_rate=0.05, seat_config="nn,nn,nn,nn", include_replay_data=true, include_oracle_states=false, lapajne_mc_worlds=None, lapajne_mc_sims=None, centaur_handoff_trick=None, centaur_pimc_worlds=None, centaur_endgame_solver=None, centaur_alpha_mu_depth=None))]
fn run_self_play(
    py: Python<'_>,
    n_games: u32,
    concurrency: usize,
    model_path: Option<&str>,
    explore_rate: f64,
    seat_config: &str,
    include_replay_data: bool,
    include_oracle_states: bool,
    lapajne_mc_worlds: Option<usize>,
    lapajne_mc_sims: Option<usize>,
    centaur_handoff_trick: Option<usize>,
    centaur_pimc_worlds: Option<u32>,
    centaur_endgame_solver: Option<&str>,
    centaur_alpha_mu_depth: Option<usize>,
) -> PyResult<PyObject> {
    use std::collections::HashMap;
    use std::sync::Arc;
    use crate::self_play::{GameResult, SelfPlayRunner};
    use crate::player::{BatchPlayer, CARD_ACTION_SIZE};
    use crate::player_bot::{try_make_bot_by_seat_label, SUPPORTED_BOT_SEAT_LABELS};
    use crate::player_nn::NeuralNetPlayer;
    use crate::player_centaur::{CentaurBot, EndgamePolicy, DEFAULT_HANDOFF_TRICK, DEFAULT_NUM_WORLDS, DEFAULT_ALPHA_MU_DEPTH};

    if let Some(sims) = lapajne_mc_sims {
        crate::bots::lapajne::set_mc_sims(sims);
    }
    if let Some(worlds) = lapajne_mc_worlds {
        crate::bots::lapajne::set_mc_worlds(worlds);
    }

    // Parse seat_config: "nn,nn,nn,nn" or "nn,bot_v5,bot_v5,bot_v5" etc.
    let seat_labels: Vec<&str> = seat_config.split(',').map(|s| s.trim()).collect();
    if seat_labels.len() != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "seat_config must have exactly 4 comma-separated entries (e.g. 'nn,bot_v5,bot_v5,bot_v5')"
        ));
    }

    let needs_nn = seat_labels.iter().any(|&s| s == "nn");
    let needs_centaur = seat_labels.iter().any(|&s| s == "centaur");
    if (needs_nn || needs_centaur) && model_path.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "model_path is required when seat_config contains 'nn' or 'centaur'"
        ));
    }

    // Build shared player instances — one per unique type
    let nn_player: Option<Arc<dyn BatchPlayer>> = if needs_nn {
        Some(Arc::new(NeuralNetPlayer::new(
            model_path.unwrap(),
            tch::Device::Cpu,
            explore_rate,
        )))
    } else {
        None
    };

    let centaur_player: Option<Arc<dyn BatchPlayer>> = if needs_centaur {
        let policy = EndgamePolicy::from_name(
            centaur_endgame_solver.unwrap_or("pimc"),
            centaur_alpha_mu_depth.unwrap_or(DEFAULT_ALPHA_MU_DEPTH),
        );
        Some(Arc::new(CentaurBot::new(
            model_path.unwrap(),
            tch::Device::Cpu,
            explore_rate,
            centaur_handoff_trick.unwrap_or(DEFAULT_HANDOFF_TRICK),
            centaur_pimc_worlds.unwrap_or(DEFAULT_NUM_WORLDS),
            policy,
        )))
    } else {
        None
    };

    // Cache for path-based NN opponents (loaded once per unique path)
    let mut path_players: HashMap<String, Arc<dyn BatchPlayer>> = HashMap::new();
    // Cache for heuristic seat labels so each bot type is constructed once.
    let mut heuristic_players: HashMap<String, Arc<dyn BatchPlayer>> = HashMap::new();

    let mut players: Vec<Arc<dyn BatchPlayer>> = Vec::with_capacity(4);
    for &label in &seat_labels {
        let player: Arc<dyn BatchPlayer> = if label == "centaur" {
            centaur_player.as_ref().unwrap().clone()
        } else if label == "nn" {
            nn_player.as_ref().unwrap().clone()
        } else if label.ends_with(".pt") || label.contains('/') || label.contains('\\') {
            // Path-based frozen NN checkpoint — cached by path
            if !path_players.contains_key(label) {
                let opp_player = Arc::new(NeuralNetPlayer::new(
                    label,
                    tch::Device::Cpu,
                    0.0, // frozen opponents play greedily
                ));
                path_players.insert(label.to_string(), opp_player);
            }
            path_players[label].clone()
        } else if let Some(existing) = heuristic_players.get(label) {
            existing.clone()
        } else if let Some(bot) = try_make_bot_by_seat_label(label) {
            let bot_arc: Arc<dyn BatchPlayer> = Arc::new(bot);
            heuristic_players.insert(label.to_string(), bot_arc.clone());
            bot_arc
        } else {
            let supported = SUPPORTED_BOT_SEAT_LABELS
                .iter()
                .map(|s| format!("'{}'", s))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!(
                    "Unknown seat type '{}'. Use 'nn', 'centaur', {}, or a .pt path.",
                    label, supported
                ),
            ));
        };
        players.push(player);
    }

    let players_arr: [Arc<dyn BatchPlayer>; 4] = [
        players[0].clone(),
        players[1].clone(),
        players[2].clone(),
        players[3].clone(),
    ];

    // Training should only learn from learner seats (labels "nn" or "centaur").
    // We still run full games with all seats, but only emit learner experiences.
    let learner_seat_mask: [bool; 4] = std::array::from_fn(|i| {
        matches!(seat_labels[i], "nn" | "centaur")
    });

    // Release GIL — the entire self-play loop runs in pure Rust
    let results: Vec<GameResult> = py.allow_threads(|| {
        let runner = SelfPlayRunner::new(players_arr);
        runner.run(n_games, concurrency)
    });

    // Flatten results into numpy arrays
    let total_games = results.len();
    let state_size = encoding::STATE_SIZE;

    let total_exp: usize = results
        .iter()
        .map(|r| {
            r.experiences
                .iter()
                .filter(|exp| {
                    let si = exp.player as usize;
                    si < 4 && learner_seat_mask[si]
                })
                .count()
        })
        .sum();

    let mut all_states = Vec::with_capacity(total_exp * state_size);
    let mut all_actions = Vec::with_capacity(total_exp);
    let mut all_log_probs = Vec::with_capacity(total_exp);
    let mut all_values = Vec::with_capacity(total_exp);
    let mut all_dt = Vec::with_capacity(total_exp);
    let mut all_game_modes = Vec::with_capacity(total_exp);
    let mut all_rewards = Vec::with_capacity(total_exp);
    let mut all_game_ids = Vec::with_capacity(total_exp);
    let mut all_players = Vec::with_capacity(total_exp);
    let mut all_masks: Vec<f32> = Vec::with_capacity(total_exp * CARD_ACTION_SIZE);
    let mut all_oracle_states: Vec<f32> = if include_oracle_states {
        Vec::with_capacity(total_exp * encoding::ORACLE_STATE_SIZE)
    } else {
        Vec::new()
    };
    let mut all_scores: Vec<[i32; 4]> = Vec::with_capacity(total_games);
    let mut all_contracts: Vec<u8> = Vec::with_capacity(total_games);
    let mut all_declarers: Vec<i8> = Vec::with_capacity(total_games);
    let mut all_partners: Vec<i8> = Vec::with_capacity(total_games);
    let mut all_bid_contracts: Vec<[i8; 4]> = Vec::with_capacity(total_games);
    let mut all_taroks_in_hand: Vec<[u8; 4]> = Vec::with_capacity(total_games);
    // Replay payload is optional because training never reads it, but arena
    // analytics and future deterministic replay tools still can.
    let mut all_initial_hands: Vec<u8> = if include_replay_data {
        Vec::with_capacity(total_games * 4 * 12)
    } else {
        Vec::new()
    };
    let mut all_initial_talon: Vec<u8> = if include_replay_data {
        Vec::with_capacity(total_games * 6)
    } else {
        Vec::new()
    };

    for result in &results {
        all_scores.push(result.scores);
        all_contracts.push(result.contract);
        all_declarers.push(result.declarer);
        all_partners.push(result.partner);
        all_bid_contracts.push(result.bid_contracts);
        all_taroks_in_hand.push(result.taroks_in_hand);

        if include_replay_data {
            // Flatten initial hands: 4 players × 12 cards
            for hand in &result.initial_hands {
                for card in hand.iter() {
                    all_initial_hands.push(card.0);
                }
            }
            // Flatten initial talon: 6 cards
            for card in result.initial_talon.iter() {
                all_initial_talon.push(card.0);
            }
        }

        for exp in &result.experiences {
            let si = exp.player as usize;
            if si >= 4 || !learner_seat_mask[si] {
                continue;
            }
            all_states.extend_from_slice(&exp.state);
            if include_oracle_states {
                all_oracle_states.extend_from_slice(&exp.oracle_state);
            }
            all_actions.push(exp.action);
            all_log_probs.push(exp.log_prob);
            all_values.push(exp.value);
            all_dt.push(exp.decision_type);
            all_game_modes.push(exp.game_mode);
            all_masks.extend_from_slice(&exp.legal_mask);
            all_game_ids.push(exp.game_id);
            all_players.push(exp.player);
            all_rewards.push(0.0f32);
        }
    }

    let dict = pyo3::types::PyDict::new(py);

    let states_arr = PyArray1::<f32>::from_vec(py, all_states)
        .reshape([total_exp, state_size])
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("states: {e}")))?;
    dict.set_item("states", states_arr)?;
    dict.set_item("actions", numpy::PyArray1::<u16>::from_vec(py, all_actions))?;
    dict.set_item("log_probs", numpy::PyArray1::<f32>::from_vec(py, all_log_probs))?;
    dict.set_item("values", numpy::PyArray1::<f32>::from_vec(py, all_values))?;
    dict.set_item("decision_types", numpy::PyArray1::<u8>::from_vec(py, all_dt))?;
    dict.set_item("game_modes", numpy::PyArray1::<u8>::from_vec(py, all_game_modes))?;
    dict.set_item("rewards", numpy::PyArray1::<f32>::from_vec(py, all_rewards))?;
    dict.set_item("game_ids", numpy::PyArray1::<u32>::from_vec(py, all_game_ids))?;
    dict.set_item("players", numpy::PyArray1::<u8>::from_vec(py, all_players))?;
    if include_oracle_states {
        let oracle_arr = PyArray1::<f32>::from_vec(py, all_oracle_states)
            .reshape([total_exp, encoding::ORACLE_STATE_SIZE])
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("oracle_states: {e}")))?;
        dict.set_item("oracle_states", oracle_arr)?;
    }

    let masks_arr = PyArray1::<f32>::from_vec(py, all_masks)
        .reshape([total_exp, CARD_ACTION_SIZE])
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("legal_masks: {e}")))?;
    dict.set_item("legal_masks", masks_arr)?;

    let scores_flat: Vec<i32> = all_scores.iter().flat_map(|s| s.iter().copied()).collect();
    let scores_arr = numpy::PyArray1::<i32>::from_vec(py, scores_flat)
        .reshape([total_games, 4])
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("scores: {e}")))?;
    dict.set_item("scores", scores_arr)?;

    // Arena metadata arrays
    dict.set_item("contracts", numpy::PyArray1::<u8>::from_vec(py, all_contracts))?;
    dict.set_item("declarers", numpy::PyArray1::<i8>::from_vec(py, all_declarers))?;
    dict.set_item("partners", numpy::PyArray1::<i8>::from_vec(py, all_partners))?;
    let flat_bid_contracts: Vec<i8> = all_bid_contracts.iter().flat_map(|b| b.iter().copied()).collect();
    let bid_contracts_arr = PyArray1::<i8>::from_vec(py, flat_bid_contracts)
        .reshape([total_games, 4])
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("bid_contracts: {e}")))?;
    dict.set_item("bid_contracts", bid_contracts_arr)?;
    let flat_taroks: Vec<u8> = all_taroks_in_hand.iter().flat_map(|t| t.iter().copied()).collect();
    let taroks_arr = PyArray1::<u8>::from_vec(py, flat_taroks)
        .reshape([total_games, 4])
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("taroks_in_hand: {e}")))?;
    dict.set_item("taroks_in_hand", taroks_arr)?;

    if include_replay_data {
        // Initial hands: (n_games, 4, 12) u8 — card indices for replay
        let hands_arr = PyArray1::<u8>::from_vec(py, all_initial_hands)
            .reshape([total_games, 4, 12])
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("initial_hands: {e}")))?;
        dict.set_item("initial_hands", hands_arr)?;
        // Initial talon: (n_games, 6) u8
        let talon_arr = PyArray1::<u8>::from_vec(py, all_initial_talon)
            .reshape([total_games, 6])
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("initial_talon: {e}")))?;
        dict.set_item("initial_talon", talon_arr)?;

        // Per-game traces: list of dicts for deterministic replay
        let traces_list = pyo3::types::PyList::empty(py);
        for result in &results {
            let trace_dict = pyo3::types::PyDict::new(py);
            let bids: Vec<(u8, u8)> = result.trace.bids.clone();
            let bids_list = pyo3::types::PyList::new(
                py,
                bids.iter().map(|(p, a)| pyo3::types::PyTuple::new(py, &[*p, *a]).unwrap()),
            )?;
            trace_dict.set_item("bids", bids_list)?;

            match result.trace.king_call {
                Some((p, a)) => {
                    trace_dict.set_item("king_call", pyo3::types::PyTuple::new(py, &[p, a])?)?;
                }
                None => {
                    trace_dict.set_item("king_call", py.None())?;
                }
            }

            match result.trace.talon_pick {
                Some((p, a)) => {
                    trace_dict.set_item("talon_pick", pyo3::types::PyTuple::new(py, &[p, a])?)?;
                }
                None => {
                    trace_dict.set_item("talon_pick", py.None())?;
                }
            }

            let put_down_list = pyo3::types::PyList::new(
                py,
                result.trace.put_down.iter().map(|&c| c),
            )?;
            trace_dict.set_item("put_down", put_down_list)?;

            let cards_list = pyo3::types::PyList::new(
                py,
                result.trace.cards_played.iter().map(|(p, c)| pyo3::types::PyTuple::new(py, &[*p, *c]).unwrap()),
            )?;
            trace_dict.set_item("cards_played", cards_list)?;

            trace_dict.set_item("dealer", (result.game_id % 4) as u8)?;

            traces_list.append(trace_dict)?;
        }
        dict.set_item("traces", traces_list)?;
    }

    dict.set_item("n_games", total_games)?;
    dict.set_item("n_experiences", total_exp)?;

    Ok(dict.into())
}

/// Run a large-scale bot-vs-bot arena.  Returns per-game scores +
/// contract & declarer metadata with zero training-data overhead.
///
/// `seat_config`: comma-separated, e.g. `"bot_v5,bot_v6,bot_v5,bot_v5"`.
/// Supports `bot_lapajne`, `bot_v5`, and `bot_v6` (no NN seats — use
/// [`run_self_play`] for that).
///
/// Returns a dict with:
///   - `scores`: numpy (n_games, 4) int32
///   - `contracts`: numpy (n_games,) uint8
///   - `declarers`: numpy (n_games,) int8
///   - `partners`: numpy (n_games,) int8
///   - `bid_contracts`: numpy (n_games, 4) int8
///   - `taroks_in_hand`: numpy (n_games, 4) uint8
///   - `n_games`: int
#[pyfunction]
#[pyo3(signature = (n_games, seat_config="bot_v5,bot_v5,bot_v5,bot_v5"))]
fn run_arena_games(
    py: Python<'_>,
    n_games: u32,
    seat_config: &str,
) -> PyResult<PyObject> {
    use crate::arena::{self, BotVersion};

    let seat_labels: Vec<&str> = seat_config.split(',').map(|s| s.trim()).collect();
    if seat_labels.len() != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "seat_config must have exactly 4 comma-separated entries",
        ));
    }

    let mut versions = [BotVersion::V5; 4];
    for (i, &label) in seat_labels.iter().enumerate() {
        versions[i] = match label {
            "bot_lapajne" => BotVersion::Lapajne,
            "bot_lustrek" => BotVersion::Lustrek,
            "bot_v1"     => BotVersion::V1,
            "bot_v3"     => BotVersion::V3,
            "bot_v5"     => BotVersion::V5,
            "bot_v6"     => BotVersion::V6,
            "bot_m6"     => BotVersion::M6,
            "bot_m8"     => BotVersion::M8,
            "bot_pozrl"  => BotVersion::Pozrl,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "run_arena_games only supports heuristic bots, got '{other}'"
                )));
            }
        };
    }

    // Release GIL — entire computation runs in pure Rust with Rayon
    let results = py.allow_threads(|| arena::run_arena_batch(n_games, versions));

    let total = results.len();
    let mut flat_scores: Vec<i32> = Vec::with_capacity(total * 4);
    let mut contracts: Vec<u8> = Vec::with_capacity(total);
    let mut declarers: Vec<i8> = Vec::with_capacity(total);
    let mut partners: Vec<i8> = Vec::with_capacity(total);
    let mut flat_bid_contracts: Vec<i8> = Vec::with_capacity(total * 4);
    let mut flat_taroks_in_hand: Vec<u8> = Vec::with_capacity(total * 4);

    for r in &results {
        flat_scores.extend_from_slice(&r.scores);
        contracts.push(r.contract);
        declarers.push(r.declarer);
        partners.push(r.partner);
        flat_bid_contracts.extend_from_slice(&r.bid_contracts);
        flat_taroks_in_hand.extend_from_slice(&r.taroks_in_hand);
    }

    let dict = pyo3::types::PyDict::new(py);
    let scores_arr = PyArray1::<i32>::from_vec(py, flat_scores)
        .reshape([total, 4])
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("scores: {e}")))?;
    dict.set_item("scores", scores_arr)?;
    dict.set_item("contracts", numpy::PyArray1::<u8>::from_vec(py, contracts))?;
    dict.set_item("declarers", numpy::PyArray1::<i8>::from_vec(py, declarers))?;
    dict.set_item("partners", numpy::PyArray1::<i8>::from_vec(py, partners))?;
    let bid_contracts_arr = PyArray1::<i8>::from_vec(py, flat_bid_contracts)
        .reshape([total, 4])
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("bid_contracts: {e}")))?;
    dict.set_item("bid_contracts", bid_contracts_arr)?;
    let taroks_in_hand_arr = PyArray1::<u8>::from_vec(py, flat_taroks_in_hand)
        .reshape([total, 4])
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("taroks_in_hand: {e}")))?;
    dict.set_item("taroks_in_hand", taroks_in_hand_arr)?;
    dict.set_item("n_games", total)?;

    Ok(dict.into())
}

// -----------------------------------------------------------------------
// Double-dummy solver bindings
// -----------------------------------------------------------------------

/// Solve a position from a RustGameState, returning optimal card points.
///
// Commented out: dd_solve depends on double_dummy module which is not available
// /// Returns a dict with:
// ///   - decl_points: optimal declarer team card points
// ///   - opp_points:  optimal opponent team card points
// ///   - best_move:   optimal card index (u8) or None
// ///   - nodes:       search nodes explored
// #[pyfunction]
// fn dd_solve(py: Python<'_>, gs: &PyGameState) -> PyResult<PyObject> {
//     let result = py.allow_threads(|| {
//         double_dummy::solve(&gs.state)
//     });
//
//     let dict = pyo3::types::PyDict::new(py);
//     dict.set_item("decl_points", result.decl_points)?;
//     dict.set_item("opp_points", result.opp_points)?;
//     dict.set_item("best_move", result.best_move.map(|c| c.0))?;
//     dict.set_item("nodes", result.nodes)?;
//
//     Ok(dict.into())
// }

// Commented out: dd_solve_all_moves depends on double_dummy module which is not available
// /// Solve all legal moves from a position, returning DD value per move.
// ///
// /// Returns a list of dicts, each with:
// ///   - card: card index (u8)
// ///   - decl_points: declarer team card points if this card is played
// ///   - opp_points:  opponent team card points if this card is played
// ///   - nodes:       search nodes for this subtree
// #[pyfunction]
// fn dd_solve_all_moves(py: Python<'_>, gs: &PyGameState) -> PyResult<PyObject> {
//     let results = py.allow_threads(|| {
//         double_dummy::solve_all_moves(&gs.state)
//     });
//
//     let list = pyo3::types::PyList::empty(py);
//     for (card, result) in results {
//         let dict = pyo3::types::PyDict::new(py);
//         dict.set_item("card", card.0)?;
//         dict.set_item("card_label", card.label())?;
//         dict.set_item("decl_points", result.decl_points)?;
//         dict.set_item("opp_points", result.opp_points)?;
//         dict.set_item("nodes", result.nodes)?;
//         list.append(dict)?;
//     }
//
//     Ok(list.into())
// }

// Commented out: pimc_solve depends on pimc module which is not available
// /// PIMC (Perfect Information Monte Carlo) solver.
// ///
// /// Samples `num_worlds` consistent deals from the viewer's perspective,
// /// DD-solves each, and returns aggregated results per legal move.
// ///
// /// Returns a dict with:
// ///   - best_move: card index (u8) or None
// ///   - worlds_sampled: number of worlds
// ///   - moves: list of dicts, each with:
// ///       - card: card index (u8)
// ///       - card_label: human-readable label
// ///       - avg_decl_points: average declarer points across worlds
// ///       - win_count: worlds where this was the best move
// ///       - sample_count: total worlds sampled for this move
// #[pyfunction]
// #[pyo3(signature = (gs, viewer, num_worlds=100))]
// fn pimc_solve(py: Python<'_>, gs: &PyGameState, viewer: u8, num_worlds: u32) -> PyResult<PyObject> {
//     let result = py.allow_threads(|| {
//         pimc::pimc_solve(&gs.state, viewer, num_worlds)
//     });
//
//     let dict = pyo3::types::PyDict::new(py);
//     dict.set_item("best_move", result.best_move.map(|c| c.0))?;
//     dict.set_item("worlds_sampled", result.worlds_sampled)?;
//
//     let moves_list = pyo3::types::PyList::empty(py);
//     for m in &result.moves {
//         let mdict = pyo3::types::PyDict::new(py);
//         mdict.set_item("card", m.card.0)?;
//         mdict.set_item("card_label", m.card.label())?;
//         mdict.set_item("avg_decl_points", m.avg_decl_points)?;
//         mdict.set_item("win_count", m.win_count)?;
//         mdict.set_item("sample_count", m.sample_count)?;
//         moves_list.append(mdict)?;
//     }
//     dict.set_item("moves", moves_list)?;
//
//     Ok(dict.into())
// }

// Commented out: generate_dd_training_data and related functions depend on double_dummy module
// /// Generate DD-labeled training data from random games.
// ///
// /// Plays `num_games` random games. At each trick-play decision point,
// /// runs the DD solver to produce perfect value labels and per-move
// /// policy targets.
// ///
// /// Returns a dict with numpy arrays:
// ///   - states:         (N, STATE_SIZE) float32 — encoded game states
// ///   - oracle_states:  (N, ORACLE_STATE_SIZE) float32 — oracle-encoded states
// ///   - dd_values:      (N,) float32 — DD declarer card points (normalized)
// ///   - dd_best_moves:  (N,) u8 — DD optimal card index
// ///   - dd_move_values: (N, 54) float32 — per-move DD values (0 for illegal)
// ///   - decision_types: (N,) u8
// ///   - legal_masks:    (N, 54) u8
// ///   - num_experiences: int
// #[pyfunction]
// #[pyo3(signature = (num_games, include_oracle=true))]
// fn generate_dd_training_data(
//     py: Python<'_>,
//     num_games: usize,
//     include_oracle: bool,
// ) -> PyResult<PyObject> {
//     let batch = py.allow_threads(|| {
//         generate_dd_batch(num_games, include_oracle)
//     });
//
//     let n_exp = batch.dd_values.len();
//     let dict = pyo3::types::PyDict::new(py);
//
//     // States
//     let states = PyArray1::<f32>::from_vec(py, batch.states)
//         .reshape([n_exp, encoding::STATE_SIZE])
//         .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("states reshape: {e}")))?;
//     dict.set_item("states", states)?;
//
//     // Oracle states
//     if include_oracle && !batch.oracle_states.is_empty() {
//         let oracle = PyArray1::<f32>::from_vec(py, batch.oracle_states)
//             .reshape([n_exp, encoding::ORACLE_STATE_SIZE])
//             .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("oracle reshape: {e}")))?;
//         dict.set_item("oracle_states", oracle)?;
//     } else {
//         dict.set_item("oracle_states", py.None())?;
//     }
//
//     // DD labels
//     dict.set_item("dd_values", numpy::PyArray1::<f32>::from_vec(py, batch.dd_values))?;
//     dict.set_item("dd_best_moves", numpy::PyArray1::<u8>::from_vec(py, batch.dd_best_moves))?;
//
//     let move_values = PyArray1::<f32>::from_vec(py, batch.dd_move_values)
//         .reshape([n_exp, DECK_SIZE])
//         .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("move_values reshape: {e}")))?;
//     dict.set_item("dd_move_values", move_values)?;
//
//     dict.set_item("decision_types", numpy::PyArray1::<u8>::from_vec(py, batch.decision_types))?;
//
//     let masks = PyArray1::<u8>::from_vec(py, batch.legal_masks)
//         .reshape([n_exp, DECK_SIZE])
//         .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("legal_masks reshape: {e}")))?;
//     dict.set_item("legal_masks", masks)?;
//
//     dict.set_item("num_experiences", n_exp)?;
//
//     Ok(dict.into())
// }
//
// /// Internal batch data structure for DD training data.
// struct DDBatch {
//     states: Vec<f32>,
//     oracle_states: Vec<f32>,
//     dd_values: Vec<f32>,
//     dd_best_moves: Vec<u8>,
//     dd_move_values: Vec<f32>,  // flat: N * 54
//     decision_types: Vec<u8>,
//     legal_masks: Vec<u8>,      // flat: N * 54
// }
//
// /// Generate a batch of DD-labeled experiences from random games.
// fn generate_dd_batch(num_games: usize, include_oracle: bool) -> DDBatch {
//     use rayon::prelude::*;
//
//     let per_game: Vec<DDBatch> = (0..num_games)
//         .into_par_iter()
//         .map(|_| generate_dd_single_game(include_oracle))
//         .collect();
//
//     // Merge all batches
//     let total: usize = per_game.iter().map(|b| b.dd_values.len()).sum();
//     let mut merged = DDBatch {
//         states: Vec::with_capacity(total * encoding::STATE_SIZE),
//         oracle_states: if include_oracle {
//             Vec::with_capacity(total * encoding::ORACLE_STATE_SIZE)
//         } else {
//             Vec::new()
//         },
//         dd_values: Vec::with_capacity(total),
//         dd_best_moves: Vec::with_capacity(total),
//         dd_move_values: Vec::with_capacity(total * DECK_SIZE),
//         decision_types: Vec::with_capacity(total),
//         legal_masks: Vec::with_capacity(total * DECK_SIZE),
//     };
//
//     for batch in per_game {
//         merged.states.extend(batch.states);
//         merged.oracle_states.extend(batch.oracle_states);
//         merged.dd_values.extend(batch.dd_values);
//         merged.dd_best_moves.extend(batch.dd_best_moves);
//         merged.dd_move_values.extend(batch.dd_move_values);
//         merged.decision_types.extend(batch.decision_types);
//         merged.legal_masks.extend(batch.legal_masks);
//     }
//
//     merged
// }
//
// /// Generate DD-labeled experiences from a single random game.
// fn generate_dd_single_game(include_oracle: bool) -> DDBatch {
//     use rand::prelude::*;
//
//     let mut batch = DDBatch {
//         states: Vec::new(),
//         oracle_states: Vec::new(),
//         dd_values: Vec::new(),
//         dd_best_moves: Vec::new(),
//         dd_move_values: Vec::new(),
//         decision_types: Vec::new(),
//         legal_masks: Vec::new(),
//     };
//
//     let mut r = rand::rng();
//     let mut gs = GameState::new(r.random_range(0..NUM_PLAYERS as u8));
//
//     // Deal randomly
//     let mut deck = build_deck();
//     deck.shuffle(&mut r);
//     for (i, &card) in deck.iter().enumerate() {
//         if i < 48 {
//             gs.hands[i / 12].insert(card);
//         } else {
//             gs.talon.insert(card);
//         }
//     }
//
//     // Quick random bidding to get to trick play
//     gs.phase = Phase::TrickPlay;
//     gs.contract = Some(Contract::Three); // Default to Three for training
//     gs.declarer = Some(0);
//     gs.roles[0] = PlayerRole::Declarer;
//     gs.roles[1] = PlayerRole::Opponent;
//     gs.roles[2] = PlayerRole::Partner; // simplified: P2 is always partner
//     gs.roles[3] = PlayerRole::Opponent;
//
//     // Simple talon exchange: give first 3 talon cards to declarer, rest to opponents
//     let talon_cards: Vec<Card> = gs.talon.iter().collect();
//     if talon_cards.len() >= 3 {
//         for &c in &talon_cards[0..3] {
//             gs.hands[0].insert(c);
//             gs.talon.remove(c);
//         }
//         // Put down 3 cards from declarer's hand (lowest value)
//         let mut hand_cards: Vec<Card> = gs.hands[0].iter().collect();
//         hand_cards.sort_by_key(|c| c.points());
//         for &c in hand_cards.iter().take(3) {
//             gs.hands[0].remove(c);
//             gs.put_down.insert(c);
//         }
//     }
//
//     // Now play tricks with DD guidance
//     let mut lead_player = gs.forehand();
//     gs.current_player = lead_player;
//
//     for trick_num in 0..TRICKS_PER_GAME {
//         gs.current_trick = Some(Trick::new(lead_player));
//
//         for offset in 0..4u8 {
//             let player = (lead_player + offset) % NUM_PLAYERS as u8;
//             gs.current_player = player;
//
//             // Encode state for this decision
//             let mut state_buf = [0.0f32; encoding::STATE_SIZE];
//             encoding::encode_state(&mut state_buf, &gs, player, encoding::DT_CARD_PLAY);
//             batch.states.extend_from_slice(&state_buf);
//
//             if include_oracle {
//                 let mut oracle_buf = [0.0f32; encoding::ORACLE_STATE_SIZE];
//                 encoding::encode_oracle_state(&mut oracle_buf, &gs, player, encoding::DT_CARD_PLAY);
//                 batch.oracle_states.extend_from_slice(&oracle_buf);
//             }
//
//             batch.decision_types.push(encoding::DT_CARD_PLAY);
//
//             // Legal mask
//             let ctx = crate::legal_moves::MoveCtx::from_state(&gs, player);
//             let legal_set = crate::legal_moves::generate_legal_moves(&ctx);
//             let mut mask = [0u8; DECK_SIZE];
//             for c in legal_set.iter() {
//                 mask[c.0 as usize] = 1;
//             }
//             batch.legal_masks.extend_from_slice(&mask);
//
//             // DD solve all moves
//             let dd_results = double_dummy::solve_all_moves(&gs);
//
//             // Find best move and build move value vector
//             let mut move_values = [0.0f32; DECK_SIZE];
//             let mut best_card = Card(0);
//             let is_declarer_team = matches!(
//                 gs.roles[player as usize],
//                 PlayerRole::Declarer | PlayerRole::Partner
//             );
//             let mut best_val = if is_declarer_team {
//                 i32::MIN
//             } else {
//                 i32::MAX
//             };
//
//             for (card, result) in &dd_results {
//                 // Normalize DD value to [0, 1] range (0-70 card points)
//                 move_values[card.0 as usize] = result.decl_points as f32 / 70.0;
//
//                 if is_declarer_team {
//                     if result.decl_points > best_val {
//                         best_val = result.decl_points;
//                         best_card = *card;
//                     }
//                 } else if result.decl_points < best_val {
//                     best_val = result.decl_points;
//                     best_card = *card;
//                 }
//             }
//
//             batch.dd_values.push(best_val as f32 / 70.0);
//             batch.dd_best_moves.push(best_card.0);
//             batch.dd_move_values.extend_from_slice(&move_values);
//
//             // Play the DD-optimal move (so subsequent positions are from optimal play)
//             gs.hands[player as usize].remove(best_card);
//             if let Some(ref mut trick) = gs.current_trick {
//                 trick.play(player, best_card);
//             }
//             gs.played_cards.insert(best_card);
//         }
//
//         let trick = gs.current_trick.take().unwrap();
//         let is_last = trick_num == TRICKS_PER_GAME - 1;
//         let result = crate::trick_eval::evaluate_trick(&trick, is_last, gs.contract);
//         lead_player = result.winner;
//         gs.tricks.push(trick);
//     }
//
//     batch
// }

/// Build the full 54-card deck as a Vec.
fn build_deck() -> Vec<Card> {
    crate::card::build_deck().to_vec()
}

/// Compute GAE and returns over a key-sorted trajectory stream.
///
/// Inputs must be 1D arrays of equal length and sorted so that all steps of
/// the same trajectory key are contiguous.
#[pyfunction]
#[pyo3(signature = (values, rewards, traj_keys, gamma=0.99, gae_lambda=0.95))]
fn compute_gae<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f32>,
    rewards: PyReadonlyArray1<'py, f32>,
    traj_keys: PyReadonlyArray1<'py, i64>,
    gamma: f32,
    gae_lambda: f32,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)> {
    let values = values.as_slice()?;
    let rewards = rewards.as_slice()?;
    let traj_keys = traj_keys.as_slice()?;

    let n = values.len();
    if rewards.len() != n || traj_keys.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "values, rewards, and traj_keys must have identical length",
        ));
    }

    let mut advantages = vec![0.0f32; n];
    let mut returns = vec![0.0f32; n];

    for idx in (0..n).rev() {
        let is_last = idx == n - 1 || traj_keys[idx] != traj_keys[idx + 1];
        let next_value = if is_last { 0.0 } else { values[idx + 1] };
        let next_gae = if is_last { 0.0 } else { advantages[idx + 1] };

        let delta = rewards[idx] + gamma * next_value - values[idx];
        let gae = delta + gamma * gae_lambda * next_gae;

        advantages[idx] = gae;
        returns[idx] = gae + values[idx];
    }

    Ok((PyArray1::from_vec(py, advantages), PyArray1::from_vec(py, returns)))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGameState>()?;
    m.add_function(wrap_pyfunction!(generate_warmup_data, m)?)?;
    m.add_function(wrap_pyfunction!(generate_expert_data, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_trick_winner, m)?)?;
    m.add_function(wrap_pyfunction!(compute_legal_plays, m)?)?;
    m.add_function(wrap_pyfunction!(compute_gae, m)?)?;
    m.add_function(wrap_pyfunction!(run_self_play, m)?)?;
    m.add_function(wrap_pyfunction!(run_arena_games, m)?)?;
    // Commented out: DD and PIMC functions depend on unavailable modules
    // m.add_function(wrap_pyfunction!(dd_solve, m)?)?;
    // m.add_function(wrap_pyfunction!(dd_solve_all_moves, m)?)?;
    // m.add_function(wrap_pyfunction!(generate_dd_training_data, m)?)?;
    // m.add_function(wrap_pyfunction!(pimc_solve, m)?)?;

    // Expose constants
    m.add("STATE_SIZE", encoding::STATE_SIZE)?;
    m.add("ORACLE_STATE_SIZE", encoding::ORACLE_STATE_SIZE)?;
    m.add("CONTRACT_OFFSET", encoding::CONTRACT_OFFSET)?;
    m.add("CONTRACT_SIZE", encoding::CONTRACT_SIZE)?;
    m.add("BELIEF_OFFSET", encoding::BELIEF_OFFSET)?;
    m.add("DECK_SIZE", DECK_SIZE)?;
    m.add("NUM_PLAYERS", NUM_PLAYERS)?;

    // Decision type constants
    m.add("DT_BID", encoding::DT_BID)?;
    m.add("DT_KING_CALL", encoding::DT_KING_CALL)?;
    m.add("DT_TALON_PICK", encoding::DT_TALON_PICK)?;
    m.add("DT_CARD_PLAY", encoding::DT_CARD_PLAY)?;
    m.add("DT_ANNOUNCE", encoding::DT_ANNOUNCE)?;

    // Phase constants
    m.add("PHASE_DEALING", Phase::Dealing as u8)?;
    m.add("PHASE_BIDDING", Phase::Bidding as u8)?;
    m.add("PHASE_KING_CALLING", Phase::KingCalling as u8)?;
    m.add("PHASE_TALON_EXCHANGE", Phase::TalonExchange as u8)?;
    m.add("PHASE_ANNOUNCEMENTS", Phase::Announcements as u8)?;
    m.add("PHASE_TRICK_PLAY", Phase::TrickPlay as u8)?;
    m.add("PHASE_SCORING", Phase::Scoring as u8)?;
    m.add("PHASE_FINISHED", Phase::Finished as u8)?;

    // Contract constants
    m.add("CONTRACT_KLOP", Contract::Klop as u8)?;
    m.add("CONTRACT_THREE", Contract::Three as u8)?;
    m.add("CONTRACT_TWO", Contract::Two as u8)?;
    m.add("CONTRACT_ONE", Contract::One as u8)?;
    m.add("CONTRACT_SOLO_THREE", Contract::SoloThree as u8)?;
    m.add("CONTRACT_SOLO_TWO", Contract::SoloTwo as u8)?;
    m.add("CONTRACT_SOLO_ONE", Contract::SoloOne as u8)?;
    m.add("CONTRACT_SOLO", Contract::Solo as u8)?;
    m.add("CONTRACT_BERAC", Contract::Berac as u8)?;
    m.add("CONTRACT_BARVNI_VALAT", Contract::BarvniValat as u8)?;

    // Team constants
    m.add("TEAM_DECLARER", Team::DeclarerTeam as u8)?;
    m.add("TEAM_OPPONENT", Team::OpponentTeam as u8)?;

    Ok(())
}

/// Generate expert data from v5-only bot games.
///
/// Releases the GIL during Rust computation and uses Rayon parallelism
/// for significantly better throughput with the more expensive v5 bot.
#[pyfunction]
#[pyo3(signature = (num_games, include_oracle=false))]
fn generate_expert_data(py: Python<'_>, num_games: usize, include_oracle: bool) -> PyResult<PyObject> {
    // Release the GIL while running the CPU-intensive Rust computation
    let batch = py.allow_threads(|| {
        crate::expert_games_v5::generate_expert_batch_v5(num_games, include_oracle)
    });
    let n_exp = batch.rewards.len();

    let dict = pyo3::types::PyDict::new(py);

    // Use from_slice + reshape for zero-copy numpy array creation
    let states = PyArray1::<f32>::from_vec(py, batch.states)
        .reshape([n_exp, batch.state_size])
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("states reshape: {e}")))?;
    dict.set_item("states", states)?;

    if include_oracle && !batch.oracle_states.is_empty() {
        let oracle = PyArray1::<f32>::from_vec(py, batch.oracle_states)
            .reshape([n_exp, batch.oracle_state_size])
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("oracle reshape: {e}")))?;
        dict.set_item("oracle_states", oracle)?;
    } else {
        dict.set_item("oracle_states", py.None())?;
    }

    dict.set_item("decision_types", numpy::PyArray1::<u8>::from_vec(py, batch.decision_types))?;
    dict.set_item("actions", numpy::PyArray1::<u16>::from_vec(py, batch.actions))?;
    dict.set_item("rewards", numpy::PyArray1::<f32>::from_vec(py, batch.rewards))?;
    dict.set_item("legal_masks", numpy::PyArray1::<u8>::from_vec(py, batch.legal_masks))?;
    dict.set_item("num_experiences", n_exp)?;

    Ok(dict.into())
}
