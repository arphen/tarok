/// PyO3 bindings — exposes the Rust engine to Python.
///
/// The API is designed to be a drop-in replacement for the Python engine.
/// Python code calls these functions with plain ints/lists and gets back
/// plain lists/dicts.

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use rand::prelude::*;
use rand::rng;

use crate::card::*;
use crate::encoding;
use crate::expert_games;
use crate::game_state::*;
use crate::legal_moves;
use crate::scoring;
use crate::trick_eval;
use crate::warmup;

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
        let mut result: Vec<Option<u8>> = vec![None]; // can always pass
        let forehand = (self.state.dealer + 1) % NUM_PLAYERS as u8;
        let is_forehand = player == forehand;
        let highest = self
            .state
            .bids
            .iter()
            .filter_map(|b| b.contract)
            .max_by_key(|c| c.strength());

        for c in Contract::BIDDABLE {
            // THREE is forehand-only
            if c == Contract::Three && !is_forehand {
                continue;
            }
            if let Some(h) = highest {
                if is_forehand {
                    // Forehand can match (>=) the current highest
                    if c.strength() >= h.strength() {
                        result.push(Some(c as u8));
                    }
                } else {
                    // Others must strictly outbid (>)
                    if c.strength() > h.strength() {
                        result.push(Some(c as u8));
                    }
                }
            } else {
                result.push(Some(c as u8));
            }
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
        crate::stockskis_v5::evaluate_bid_v5(hand, highest).map(|c| c as u8)
    }

    /// V5 king calling: returns the card index to call.
    fn v5_choose_king(&self, player: u8) -> Option<u8> {
        let hand = self.state.hands[player as usize];
        crate::stockskis_v5::choose_king_v5(hand).map(|c| c.0)
    }

    /// V5 talon group selection: returns the group index (0 or 1, or 0-5).
    fn v5_choose_talon_group(&self, player: u8, groups: Vec<Vec<u8>>) -> usize {
        let hand = self.state.hands[player as usize];
        let groups_cards: Vec<Vec<Card>> = groups.iter()
            .map(|g| g.iter().map(|&idx| Card(idx)).collect())
            .collect();
        crate::stockskis_v5::choose_talon_group_v5(
            &groups_cards, hand, self.state.called_king
        )
    }

    /// V5 discard selection: returns card indices to discard.
    fn v5_choose_discards(&self, player: u8, must_discard: usize) -> Vec<u8> {
        let hand = self.state.hands[player as usize];
        crate::stockskis_v5::choose_discards_v5(hand, must_discard, self.state.called_king)
            .iter().map(|c| c.0).collect()
    }

    /// V5 card play: returns the card index to play.
    fn v5_choose_card(&self, player: u8) -> u8 {
        let hand = self.state.hands[player as usize];
        crate::stockskis_v5::choose_card_v5(hand, &self.state, player).0
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

/// Generate expert experiences from StockŠkis bot games (imitation learning).
///
/// Returns dict with numpy arrays:
///   states:         (N, STATE_SIZE) float32
///   oracle_states:  (N, ORACLE_STATE_SIZE) float32  — None if include_oracle=False
///   decision_types: (N,) uint8
///   actions:        (N,) uint16  — the action index the bot chose
///   rewards:        (N,) float32 — final game reward for the acting player
///   legal_masks:    (N, action_size) uint8  — varies per decision type, flattened
///   num_experiences: int
#[pyfunction]
#[pyo3(signature = (num_games, include_oracle=true))]
fn generate_expert_data(
    py: Python<'_>,
    num_games: usize,
    include_oracle: bool,
) -> PyResult<PyObject> {
    let batch = expert_games::generate_expert_batch(num_games, include_oracle);
    let n = batch.rewards.len();

    let dict = pyo3::types::PyDict::new(py);

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
        "actions",
        numpy::PyArray1::<u16>::from_vec(py, batch.actions),
    )?;
    dict.set_item(
        "rewards",
        numpy::PyArray1::<f32>::from_vec(py, batch.rewards),
    )?;
    dict.set_item(
        "legal_masks",
        numpy::PyArray1::<u8>::from_vec(py, batch.legal_masks),
    )?;
    dict.set_item("num_experiences", n)?;

    Ok(dict.into())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGameState>()?;
    m.add_function(wrap_pyfunction!(generate_warmup_data, m)?)?;
    m.add_function(wrap_pyfunction!(generate_expert_data, m)?)?;

    // Expose constants
    m.add("STATE_SIZE", encoding::STATE_SIZE)?;
    m.add("ORACLE_STATE_SIZE", encoding::ORACLE_STATE_SIZE)?;
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

    // Benchmark function
    m.add_function(wrap_pyfunction!(py_run_benchmark, m)?)?;
    m.add_function(wrap_pyfunction!(py_eval_vs_bots, m)?)?;
    m.add_function(wrap_pyfunction!(py_generate_expert_data_v2v3, m)?)?;
    m.add_function(wrap_pyfunction!(py_generate_expert_data_v2v3v5, m)?)?;
    m.add_function(wrap_pyfunction!(py_generate_expert_data_v5, m)?)?;

    Ok(())
}

/// Run StockŠkis v1/v2/v3/v4 benchmark from Python.
#[pyfunction]
#[pyo3(signature = (num_games=1000))]
fn py_run_benchmark(num_games: usize) {
    crate::benchmark::run_benchmark(num_games);
}

/// Evaluate "player 0" (stub — scores come from heuristic play) against bot opponents.
/// Returns a dict with win_rate for each bot version opponent.
/// `bot_version`: 1, 2, or 3 — the opponent bot version to test against.
/// `num_games`: games to play per configuration.
/// The "player 0" in this context uses the same bot version just to measure baseline;
/// the caller (Python) replaces p0 with NN decisions.
#[pyfunction]
#[pyo3(signature = (num_games=500))]
fn py_eval_vs_bots(py: Python<'_>, num_games: usize) -> PyResult<PyObject> {
    use crate::benchmark::{run_eval_config, BotVersion};

    let dict = pyo3::types::PyDict::new(py);

    // Test each opponent version: NN (player 0) vs 3x V_i
    // Since we can't run the NN here, we return stats for heuristic baselines
    for (label, versions) in [
        ("vs_v1", [BotVersion::V1, BotVersion::V1, BotVersion::V1, BotVersion::V1]),
        ("vs_v2", [BotVersion::V2, BotVersion::V2, BotVersion::V2, BotVersion::V2]),
        ("vs_v3", [BotVersion::V3, BotVersion::V3, BotVersion::V3, BotVersion::V3]),
        ("vs_v4", [BotVersion::V4, BotVersion::V4, BotVersion::V4, BotVersion::V4]),
        ("vs_v5", [BotVersion::V5, BotVersion::V5, BotVersion::V5, BotVersion::V5]),
    ] {
        let stats = run_eval_config(versions, num_games);
        let inner = pyo3::types::PyDict::new(py);
        inner.set_item("games", stats.games)?;
        inner.set_item("wins", stats.wins)?;
        inner.set_item("win_rate", if stats.games > 0 { stats.wins as f64 / stats.games as f64 } else { 0.0 })?;
        inner.set_item("avg_score", if stats.games > 0 { stats.total_score as f64 / stats.games as f64 } else { 0.0 })?;
        inner.set_item("declarer_games", stats.declarer_games)?;
        inner.set_item("declarer_wins", stats.declarer_wins)?;
        inner.set_item("declarer_wr", if stats.declarer_games > 0 { stats.declarer_wins as f64 / stats.declarer_games as f64 } else { 0.0 })?;
        dict.set_item(label, inner)?;
    }

    Ok(dict.into())
}

/// Generate expert data from v2 and v3 bots playing against each other
/// (mixed tables for richer training signal).
#[pyfunction]
#[pyo3(signature = (num_games, include_oracle=false))]
fn py_generate_expert_data_v2v3(py: Python<'_>, num_games: usize, include_oracle: bool) -> PyResult<PyObject> {
    let batch = crate::expert_games_v2v3::generate_expert_batch_v2v3(num_games, include_oracle);
    let n_exp = batch.rewards.len();

    let dict = pyo3::types::PyDict::new(py);

    let states = numpy::PyArray2::<f32>::from_vec2(
        py,
        &batch.states.chunks(batch.state_size).map(|c| c.to_vec()).collect::<Vec<_>>(),
    ).map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("states: {e}")))?;
    dict.set_item("states", states)?;

    if include_oracle && !batch.oracle_states.is_empty() {
        let oracle = numpy::PyArray2::<f32>::from_vec2(
            py,
            &batch.oracle_states.chunks(batch.oracle_state_size).map(|c| c.to_vec()).collect::<Vec<_>>(),
        ).map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("oracle: {e}")))?;
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

/// Generate expert data from v2, v3, and v5 bots playing against each other
/// (mixed tables with strongest bot for richer training signal).
#[pyfunction]
#[pyo3(signature = (num_games, include_oracle=false))]
fn py_generate_expert_data_v2v3v5(py: Python<'_>, num_games: usize, include_oracle: bool) -> PyResult<PyObject> {
    let batch = crate::expert_games_v2v3v5::generate_expert_batch_v2v3v5(num_games, include_oracle);
    let n_exp = batch.rewards.len();

    let dict = pyo3::types::PyDict::new(py);

    let states = numpy::PyArray2::<f32>::from_vec2(
        py,
        &batch.states.chunks(batch.state_size).map(|c| c.to_vec()).collect::<Vec<_>>(),
    ).map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("states: {e}")))?;
    dict.set_item("states", states)?;

    if include_oracle && !batch.oracle_states.is_empty() {
        let oracle = numpy::PyArray2::<f32>::from_vec2(
            py,
            &batch.oracle_states.chunks(batch.oracle_state_size).map(|c| c.to_vec()).collect::<Vec<_>>(),
        ).map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("oracle: {e}")))?;
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

/// Generate expert data from v5-only bot games.
///
/// Releases the GIL during Rust computation and uses Rayon parallelism
/// for significantly better throughput with the more expensive v5 bot.
#[pyfunction]
#[pyo3(signature = (num_games, include_oracle=false))]
fn py_generate_expert_data_v5(py: Python<'_>, num_games: usize, include_oracle: bool) -> PyResult<PyObject> {
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
