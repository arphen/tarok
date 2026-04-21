export interface CardData {
  card_type: 'tarok' | 'suit';
  value: number;
  suit: 'hearts' | 'diamonds' | 'clubs' | 'spades' | null;
  label: string;
  points: number;
}

export type TrickCard = [number, CardData];

export interface PlayerCardInfo {
  void_suits: string[];
  highest_tarok: number | null;
  lowest_tarok: number | null;
  taroks_played_count: number;
}

export interface CardTracker {
  remaining_by_group: Record<string, CardData[]>;
  remaining_count: number;
  player_info: Record<string, PlayerCardInfo>;
}

export interface MatchInfo {
  round_num: number;
  total_rounds: number;
  cumulative_scores: Record<string, number>;
  caller_counts: Record<string, number>;
  called_counts: Record<string, number>;
  round_history: RoundResult[];
}

export interface RoundResult {
  round: number;
  scores: Record<string, number>;
  contract: number | null;
  declarer: number | null;
  partner: number | null;
}

export interface ShadowHint {
  decision_type: 'bid' | 'card' | 'king' | 'talon' | 'discard';
  hint: {
    contract?: number | null;
    card_type?: string;
    value?: number;
    suit?: string | null;
    label?: string;
    group_index?: number;
    cards?: CardData[];
  };
}

export interface GameState {
  phase: string;
  hand: CardData[];
  hand_sizes: number[];
  talon_groups: CardData[][] | null;
  bids: { player: number; contract: number | null }[];
  contract: number | null;
  declarer: number | null;
  called_king: CardData | null;
  partner_revealed: boolean;
  partner: number | null;
  current_trick: TrickCard[];
  tricks_played: number;
  current_player: number;
  scores: Record<string, number> | null;
  legal_plays: CardData[];
  legal_bids: (number | null)[] | null;
  callable_kings: CardData[] | null;
  must_discard: number;
  player_names: string[];
  card_tracker: CardTracker | null;
  match_info: MatchInfo | null;
  hands: Record<string, CardData[]> | null;
}

export interface GameEvent {
  event: string;
  data: Record<string, unknown>;
  state: GameState;
}

export interface ContractStat {
  played: number;
  decl_played: number;
  decl_won: number;
  decl_win_rate: number;
  decl_avg_score: number;
  def_played: number;
  def_won: number;
  def_win_rate: number;
  def_avg_score: number;
}

export interface SnapshotInfo {
  filename: string;
  episode: number;
  session: number;
  win_rate: number;
  avg_reward: number;
  games_per_second: number;
}

export interface TrainingMetrics {
  run_id: string;
  episode: number;
  total_episodes: number;
  session: number;
  total_sessions: number;
  avg_reward: number;
  avg_loss: number;
  avg_placement: number;
  entropy: number;
  value_loss: number;
  policy_loss: number;
  games_per_second: number;
  bid_rate: number;
  klop_rate: number;
  solo_rate: number;
  contract_stats: Record<string, ContractStat>;
  history_offset: number;
  reward_history: number[];
  avg_placement_history: number[];
  loss_history: number[];
  bid_rate_history: number[];
  klop_rate_history: number[];
  solo_rate_history: number[];
  contract_win_rate_history: Record<string, number[]>;
  session_avg_score_history: number[];
  stockskis_place_history: number[];
  snapshots: SnapshotInfo[];
  tarok_count_bids?: Record<string, Record<string, number>>;
  // Per-opponent avg placement histories
  placement_selfplay_history: number[];
  placement_hof_history: number[];
  placement_v5_history: number[];
}

export const CONTRACT_NAMES: Record<number, string> = {
  '-99': 'Klop',
  3: 'Three',
  2: 'Two',
  1: 'One',
  '-3': 'Solo Three',
  '-2': 'Solo Two',
  '-1': 'Solo One',
  0: 'Solo',
  '-100': 'Berač',
  '-101': 'Barvni Valat',
};

export const SUIT_SYMBOLS: Record<string, string> = {
  hearts: '♥',
  diamonds: '♦',
  clubs: '♣',
  spades: '♠',
};

// ---- Spectator types ----

export type SpectatorTrickCard = [number, CardData];

export interface CompletedTrick {
  lead_player: number;
  cards: SpectatorTrickCard[];
  winner: number;
}

export interface SpectatorState {
  phase: string;
  hands: CardData[][];          // all 4 hands visible
  hand_sizes: number[];
  talon_groups: CardData[][] | null;
  bids: { player: number; contract: number | null }[];
  contract: number | null;
  declarer: number | null;
  called_king: CardData | null;
  partner_revealed: boolean;
  partner: number | null;
  current_trick: SpectatorTrickCard[];
  tricks_played: number;
  current_player: number;
  scores: Record<string, number> | null;
  player_names: string[];
  completed_tricks: CompletedTrick[];
  roles: Record<string, string>;
  announcements: Record<string, string[]>;
  kontra_levels: Record<string, string>;
  put_down: CardData[];
  score_breakdown: ScoreBreakdown | null;
  trick_summary: TrickSummaryEntry[] | null;
  dealer: number | null;
}

export interface ScoreBreakdownLine {
  label: string;
  value?: number;
  detail?: string;
}

export interface ScoreBreakdown {
  contract: string | number;
  mode: string;
  declarer_won?: boolean;
  declarer_points?: number;
  opponent_points?: number;
  explanation?: string;
  lines: ScoreBreakdownLine[];
}

export interface TrickSummaryCard {
  player: number;
  label: string;
  points: number;
}

export interface TrickSummaryEntry {
  trick_num: number;
  lead_player: number;
  cards: TrickSummaryCard[];
  winner: number;
  card_points: number;
}

export interface SpectatorEvent {
  event: string;
  data: Record<string, unknown>;
  state: SpectatorState;
}
