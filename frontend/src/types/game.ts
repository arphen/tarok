export interface CardData {
  card_type: 'tarok' | 'suit';
  value: number;
  suit: 'hearts' | 'diamonds' | 'clubs' | 'spades' | null;
  label: string;
  points: number;
}

export type TrickCard = [number, CardData];

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
  episode: number;
  total_episodes: number;
  session: number;
  total_sessions: number;
  avg_reward: number;
  avg_loss: number;
  win_rate: number;
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
  win_rate_history: number[];
  loss_history: number[];
  bid_rate_history: number[];
  klop_rate_history: number[];
  solo_rate_history: number[];
  contract_win_rate_history: Record<string, number[]>;
  session_avg_score_history: number[];
  stockskis_place_history: number[];
  snapshots: SnapshotInfo[];
  tarok_count_bids?: Record<string, Record<string, number>>;
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

// ---- Evolution types ----

export interface EvoIndividual {
  index: number;
  hparams: Record<string, number>;
  fitness: number;
  win_rate: number;
  reward_trend: number;
}

export interface EvoGenStats {
  gen: number;
  nevals: number;
  avg: number;
  std: number;
  min: number;
  max: number;
}

export interface EvoProgress {
  generation: number;
  total_generations: number;
  evaluating_index: number;
  evaluating_total: number;
  phase: 'idle' | 'evaluating' | 'selecting' | 'done' | 'starting';
  elapsed_seconds: number;
  gen_stats: EvoGenStats[];
  population: EvoIndividual[];
  hall_of_fame: EvoIndividual[];
  best_fitness: number;
  best_hparams: Record<string, number>;
}

// ---- Breeding types ----

export interface BreedIndividual {
  index: number;
  profile: Record<string, number>;
  fitness: number;
  win_rate: number;
  avg_reward: number;
  bid_rate: number;
  solo_rate: number;
}

export interface BreedGenStats {
  cycle: number;
  gen: number;
  avg: number;
  std: number;
  min: number;
  max: number;
}

export interface BreedCycleSummary {
  cycle: number;
  best_fitness: number;
  best_profile: Record<string, number>;
  refine_win_rate: number;
  refine_avg_reward: number;
}

export interface BreedProgress {
  phase: 'idle' | 'warmup' | 'breeding' | 'evaluating' | 'refining' | 'done' | 'starting';
  cycle: number;
  total_cycles: number;
  generation: number;
  total_generations: number;
  evaluating_index: number;
  evaluating_total: number;
  warmup_session: number;
  warmup_total_sessions: number;
  refine_session: number;
  refine_total_sessions: number;
  elapsed_seconds: number;
  model_name: string;
  population: BreedIndividual[];
  hall_of_fame: BreedIndividual[];
  best_fitness: number;
  best_profile: Record<string, number>;
  gen_stats: BreedGenStats[];
  cycle_summaries: BreedCycleSummary[];
}

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
  score_breakdown: ScoreBreakdown | null;
  trick_summary: TrickSummaryEntry[] | null;
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
