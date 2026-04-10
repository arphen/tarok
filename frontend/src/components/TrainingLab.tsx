import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, BarChart, Bar, Cell, Area, AreaChart,
  ReferenceLine,
} from 'recharts';
import './TrainingLab.css';

interface Props { onBack: () => void }

interface EvalPoint {
  step: number;
  label: string;
  program?: string;
  vs_v1?: number;
  vs_v2?: number;
  vs_v3?: number;
  vs_v4?: number;
  vs_v5?: number;
  avg_score_v1?: number;
  avg_score_v2?: number;
  avg_score_v3?: number;
  avg_score_v4?: number;
  avg_score_v5?: number;
  loss: number;
  experiences?: number;
  games?: number;
  [key: string]: unknown;
}

interface Snapshot {
  filename: string;
  display_name: string;
  model_hash: string;
}

interface PopulationMember {
  index: number;
  label: string;
  fitness: number;
  batch_avg_reward: number;
  batch_win_rate: number;
  vs_v1?: number;
  vs_v2?: number;
  vs_v3?: number;
  vs_v4?: number;
  vs_v5?: number;
  avg_score_v1?: number;
  avg_score_v2?: number;
  avg_score_v3?: number;
  avg_score_v4?: number;
  avg_score_v5?: number;
  loss: number;
  games: number;
  status: string;
  copied_from: number | null;
  mutations: number;
  survival_count: number;
  model_hash: string;
  hparams: Record<string, number>;
}

interface GenerationPoint {
  generation: number;
  avg_fitness: number;
  min_fitness: number;
  max_fitness: number;
  avg_v3: number;
  best_index: number;
  best_label: string;
  best_vs_v1?: number;
  best_vs_v2?: number;
  best_vs_v3?: number;
  best_vs_v4?: number;
  best_vs_v5?: number;
  best_batch_reward: number;
}

interface CheckpointOption {
  filename: string;
  episode: number;
  session?: number;
  win_rate: number;
  model_name?: string;
  is_hof?: boolean;
}

interface LabState {
  phase: string;
  has_network: boolean;
  hidden_size: number;
  persona: { first_name?: string; last_name?: string; age?: number } | null;
  model_hash: string;
  display_name: string;
  active_program: string;
  eval_history: EvalPoint[];
  training_sessions_done: number;
  total_training_sessions: number;
  current_loss: number;
  expert_games_generated: number;
  expert_experiences: number;
  self_play_games: number;
  self_play_sessions: number;
  running: boolean;
  error: string | null;
  snapshots: Snapshot[];
  // Per-session self-play metrics
  sp_win_rate: number;
  sp_avg_reward: number;
  sp_avg_score: number;
  sp_bid_rate: number;
  sp_klop_rate: number;
  sp_solo_rate: number;
  sp_games_per_second: number;
  sp_win_rate_history: number[];
  sp_avg_reward_history: number[];
  sp_avg_score_history: number[];
  sp_loss_history: number[];
  sp_bid_rate_history: number[];
  sp_klop_rate_history: number[];
  sp_solo_rate_history: number[];
  sp_min_score_history: number[];
  sp_max_score_history: number[];
  // PBT
  pbt_enabled: boolean;
  pbt_generation: number;
  pbt_total_generations: number;
  pbt_population_size: number;
  pbt_member_index: number;
  pbt_member_total: number;
  population: PopulationMember[];
  generation_history: GenerationPoint[];
  population_events: Array<{ generation: number; target: number; source: number; hparams: Record<string, number> }>;
}

const API = '';

const PHASE_LABELS: Record<string, { text: string; color: string; icon: string }> = {
  idle: { text: 'Ready', color: '#666', icon: '⏸' },
  evaluating: { text: 'Playing Evaluation Games...', color: '#4a9eff', icon: '🔍' },
  training: { text: 'Imitation Learning...', color: '#4caf50', icon: '📚' },
  self_play: { text: 'Self-Play Training...', color: '#e040fb', icon: '🎮' },
  exploiting: { text: 'PBT Exploit / Explore...', color: '#ff9800', icon: '🧬' },
  done: { text: 'Training Complete!', color: '#d4a843', icon: '✅' },
};

const tooltipStyle = {
  contentStyle: { background: '#1a1b1e', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, fontSize: 13 },
  labelStyle: { color: '#aaa' },
};

const ALL_BOT_VERSIONS = ['v1', 'v2', 'v3', 'v3.2', 'v4', 'v5'] as const;
const BOT_COLORS: Record<string, string> = {
  v1: '#4caf50', v2: '#2196f3', v3: '#9c27b0', 'v3.2': '#e91e63', v4: '#ff9800', v5: '#f44336',
};
const BOT_LABELS: Record<string, string> = {
  v1: 'V1', v2: 'V2', v3: 'V3', 'v3.2': 'V3.2', v4: 'V4', v5: 'V5',
};
/** Sanitize bot version for use as Recharts dataKey (dots break nested access) */
const sanitizeKey = (v: string) => v.replace(/\./g, '_');

export default function TrainingLab({ onBack }: Props) {
  const [state, setState] = useState<LabState | null>(null);
  const [checkpoints, setCheckpoints] = useState<CheckpointOption[]>([]);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState('');
  // Imitation learning config
  const [expertGames, setExpertGames] = useState(500000);
  const [expertSource, setExpertSource] = useState("v2v3v5");
  const [evalBots, setEvalBots] = useState<string[]>(['v1', 'v2', 'v3']);
  const [numRounds, setNumRounds] = useState(10);
  const [evalGames, setEvalGames] = useState(100);
  const [hiddenSize, setHiddenSize] = useState(256);
  const [ilLearningRate, setIlLearningRate] = useState(0.001);
  // Self-play config
  const [spSessions, setSpSessions] = useState(50);
  const [spGamesPerSession, setSpGamesPerSession] = useState(20);
  const [spEvalInterval, setSpEvalInterval] = useState(5);
  const [spLearningRate, setSpLearningRate] = useState(0.0003);
  const [spFspRatio, setSpFspRatio] = useState(0.3);
  const [pbtEnabled, setPbtEnabled] = useState(true);
  const [pbtPopulationSize, setPbtPopulationSize] = useState(4);
  const [pbtTopRatio, setPbtTopRatio] = useState(0.25);
  const [pbtBottomRatio, setPbtBottomRatio] = useState(0.25);
  const [pbtMutationScale, setPbtMutationScale] = useState(1.0);
  const [timeLimitMinutes, setTimeLimitMinutes] = useState(5);
  const [smoothing, setSmoothing] = useState(0.6);

  const [tab, setTab] = useState<'progress' | 'population' | 'live' | 'winrate' | 'scores' | 'hof'>('progress');

  // Poll state
  const poll = useCallback(async () => {
    try {
      const res = await fetch(`${API}/api/lab/state`);
      const data = await res.json();
      setState(data);
    } catch { /* server not up */ }
  }, []);

  const loadCheckpoints = useCallback(async () => {
    try {
      const res = await fetch(`${API}/api/checkpoints`);
      const data = await res.json();
      const items: CheckpointOption[] = data.checkpoints ?? [];
      setCheckpoints(items);
      if (!selectedCheckpoint && items.length > 0) {
        setSelectedCheckpoint(items[0].filename);
      }
    } catch { /* server not up */ }
  }, [selectedCheckpoint]);

  useEffect(() => {
    poll();
    const id = setInterval(poll, 1500);
    return () => clearInterval(id);
  }, [poll]);

  useEffect(() => {
    loadCheckpoints();
  }, [loadCheckpoints]);

  const createNetwork = async () => {
    await fetch(`${API}/api/lab/create`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ hidden_size: hiddenSize }),
    });
    poll();
  };

  const loadCheckpoint = async () => {
    if (!selectedCheckpoint) return;
    await fetch(`${API}/api/lab/load`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ checkpoint: selectedCheckpoint }),
    });
    await poll();
  };

  const startImitation = async () => {
    await fetch(`${API}/api/lab/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        expert_games: expertGames,
        expert_source: expertSource,
        eval_bots: evalBots,
        num_rounds: numRounds,
        eval_games: evalGames,
        learning_rate: ilLearningRate,
      }),
    });
    poll();
  };

  const startSelfPlay = async () => {
    await fetch(`${API}/api/lab/self-play`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        num_sessions: spSessions,
        games_per_session: spGamesPerSession,
        eval_games: evalGames,
        eval_bots: evalBots,
        eval_interval: spEvalInterval,
        learning_rate: spLearningRate,
        fsp_ratio: spFspRatio,
        pbt_enabled: pbtEnabled,
        population_size: pbtPopulationSize,
        exploit_top_ratio: pbtTopRatio,
        exploit_bottom_ratio: pbtBottomRatio,
        mutation_scale: pbtMutationScale,
        time_limit_minutes: pbtEnabled ? timeLimitMinutes : 0,
      }),
    });
    poll();
  };

  const startOvernight = async () => {
    await fetch(`${API}/api/lab/overnight`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    poll();
  };

  const stopTraining = async () => {
    await fetch(`${API}/api/lab/stop`, { method: 'POST' });
    poll();
  };

  const resetLab = async () => {
    await fetch(`${API}/api/lab/reset`, { method: 'POST' });
    poll();
  };

  const isRunning = state?.running ?? false;
  const hasNetwork = state?.has_network ?? false;
  const phase = state?.phase ?? 'idle';
  const phaseInfo = PHASE_LABELS[phase] ?? PHASE_LABELS.idle;
  const persona = state?.persona;

  // Chart data
  const evalData = useMemo(() => {
    if (!state?.eval_history?.length) return [];
    return state.eval_history.map(e => {
      const row: Record<string, unknown> = { ...e };
      for (const v of ALL_BOT_VERSIONS) {
        const wr = (e as Record<string, unknown>)[`vs_${v}`];
        const sk = sanitizeKey(v);
        row[`vs_${sk}_pct`] = wr != null ? +((wr as number) * 100).toFixed(1) : undefined;
        const sc = (e as Record<string, unknown>)[`avg_score_${v}`];
        if (sc != null) row[`avg_score_${sk}`] = sc;
      }
      return row;
    });
  }, [state?.eval_history]);

  const lossData = useMemo(() => {
    if (!state?.eval_history?.length) return [];
    return state.eval_history.filter(e => e.loss > 0).map(e => ({
      step: e.step,
      loss: +e.loss.toFixed(4),
      label: e.label,
    }));
  }, [state?.eval_history]);

  const progressPct = state && state.total_training_sessions > 0
    ? (state.training_sessions_done / state.total_training_sessions) * 100
    : 0;

  const generationData = useMemo(() => state?.generation_history ?? [], [state?.generation_history]);
  const populationData = useMemo(() => {
    return [...(state?.population ?? [])].sort((a, b) => b.fitness - a.fitness);
  }, [state?.population]);

  const latestGeneration = generationData.length
    ? generationData[generationData.length - 1]
    : null;

  // Per-session self-play history for the Live Metrics tab
  const applySmoothing = useCallback((data: any[], keys: string[]) => {
    if (data.length === 0 || smoothing === 0) return data;
    const result = [{ ...data[0] }];
    for (let i = 1; i < data.length; i++) {
      const prev = result[i - 1];
      const curr = { ...data[i] };
      for (const key of keys) {
        if (typeof curr[key] === 'number' && typeof prev[key] === 'number') {
          curr[key] = prev[key] * smoothing + curr[key] * (1 - smoothing);
        }
      }
      result.push(curr);
    }
    return result;
  }, [smoothing]);

  const sessionHistoryData = useMemo(() => {
    const hist = state?.sp_win_rate_history ?? [];
    if (!hist.length) return [];
    const raw = hist.map((_, i) => ({
      session: i + 1,
      winRate: +((state?.sp_win_rate_history?.[i] ?? 0) * 100).toFixed(1),
      avgReward: +(state?.sp_avg_reward_history?.[i] ?? 0).toFixed(3),
      avgScore: +(state?.sp_avg_score_history?.[i] ?? 0).toFixed(1),
      loss: +(state?.sp_loss_history?.[i] ?? 0).toFixed(4),
      bidRate: +((state?.sp_bid_rate_history?.[i] ?? 0) * 100).toFixed(1),
      klopRate: +((state?.sp_klop_rate_history?.[i] ?? 0) * 100).toFixed(1),
      soloRate: +((state?.sp_solo_rate_history?.[i] ?? 0) * 100).toFixed(1),
      minScore: +(state?.sp_min_score_history?.[i] ?? 0),
      maxScore: +(state?.sp_max_score_history?.[i] ?? 0),
    }));
    return applySmoothing(raw, ['winRate', 'avgReward', 'avgScore', 'loss', 'bidRate', 'klopRate', 'soloRate']);
  }, [state?.sp_win_rate_history, state?.sp_avg_reward_history, state?.sp_avg_score_history,
      state?.sp_loss_history, state?.sp_bid_rate_history, state?.sp_klop_rate_history, state?.sp_solo_rate_history,
      state?.sp_min_score_history, state?.sp_max_score_history, applySmoothing]);

  const hasSelfPlayData = sessionHistoryData.length > 0;

  const formatPercent = (value: number) => `${(value * 100).toFixed(1)}%`;
  const formatMaybe = (value?: number | null, digits = 3) => value === undefined || value === null ? '—' : value.toFixed(digits);
  const formatHparams = (member: PopulationMember) => {
    const hp = member.hparams;
    return `lr ${hp.lr?.toExponential?.(1) ?? hp.lr} · batch ${hp.batch_size ?? '—'} · ent ${formatMaybe(hp.entropy_coef, 3)} · eps ${formatMaybe(hp.clip_epsilon, 2)}`;
  };

  const latest = state?.eval_history?.length
    ? state.eval_history[state.eval_history.length - 1]
    : null;

  const checkpointLabel = (checkpoint: CheckpointOption) => {
    if (checkpoint.model_name) return checkpoint.model_name;
    const parts: string[] = [];
    if (checkpoint.session) parts.push(`S${checkpoint.session}`);
    if (checkpoint.episode) parts.push(`Ep ${checkpoint.episode}`);
    if (checkpoint.win_rate) parts.push(`${(checkpoint.win_rate * 100).toFixed(0)}% win`);
    return parts.length ? `${checkpoint.filename} (${parts.join(', ')})` : checkpoint.filename;
  };

  return (
    <div className="training-lab">
      {/* Header */}
      <div className="lab-header">
        <button className="btn-secondary btn-sm" onClick={onBack}>← Back</button>
        <h2>Neural Network Training Lab</h2>
        <span className="lab-phase" style={{ color: phaseInfo.color }}>
          {phaseInfo.icon} {phaseInfo.text}
        </span>
      </div>

      {/* Persona Identity Card */}
      {hasNetwork && persona?.first_name && (
        <div className="lab-persona">
          <div className="lab-persona-avatar">🤖</div>
          <div className="lab-persona-info">
            <div className="lab-persona-name">
              {persona.first_name} {persona.last_name}
            </div>
            <div className="lab-persona-meta">
              Age {persona.age ?? 0} · #{state?.model_hash}
              {state?.active_program === 'imitation' && <span className="lab-persona-badge lab-badge-il">📚 Imitation</span>}
              {state?.active_program === 'self_play' && <span className="lab-persona-badge lab-badge-sp">🎮 Self-Play</span>}
              {state?.active_program === 'self_play_pbt' && <span className="lab-persona-badge lab-badge-sp">🧬 PBT</span>}
            </div>
          </div>
          <div className="lab-persona-snapshots">
            {(state?.snapshots?.length ?? 0) > 0 && (
              <span className="lab-persona-snap-count">📸 {state!.snapshots.length} snapshots saved</span>
            )}
          </div>
        </div>
      )}

      {/* Welcome Banner */}
      {!hasNetwork && !isRunning && (
        <div className="lab-banner lab-banner-intro">
          <div className="lab-banner-icon">🧠</div>
          <div>
            <h3>Welcome to the Training Lab</h3>
            <p>Create a fresh neural network with a unique identity. Train it with imitation learning
            from expert bots, then further strengthen it with self-play. Watch its win rates climb
            and save snapshots to the Hall of Fame for use in Play vs AI and Spectate modes.</p>
          </div>
        </div>
      )}

      {state?.error && (
        <div className="lab-banner lab-banner-error">
          <div className="lab-banner-icon">⚠️</div>
          <div><strong>Error:</strong> {state.error}</div>
        </div>
      )}

      {!isRunning && (
        <div className="lab-controls">
          <div className="lab-controls-row">
            <label className="lab-field lab-field-wide">
              <span>Resume From Checkpoint</span>
              <select
                value={selectedCheckpoint}
                onChange={e => setSelectedCheckpoint(e.target.value)}
                disabled={checkpoints.length === 0}
              >
                <option value="">{checkpoints.length === 0 ? 'No checkpoints available' : 'Select checkpoint'}</option>
                {checkpoints.map(checkpoint => (
                  <option key={checkpoint.filename} value={checkpoint.filename}>
                    {checkpointLabel(checkpoint)}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="lab-controls-actions">
            <button
              className="btn-secondary btn-lab"
              onClick={loadCheckpoint}
              disabled={!selectedCheckpoint}
            >
              <span className="btn-icon">📥</span>
              {hasNetwork ? 'Replace Lab Model With Checkpoint' : 'Load Checkpoint Into Lab'}
            </button>
          </div>
        </div>
      )}

      {/* Network Creation */}
      {!hasNetwork && !isRunning && (
        <div className="lab-controls">
          <div className="lab-controls-row">
            <label className="lab-field">
              <span>Hidden Size</span>
              <input type="number" value={hiddenSize} onChange={e => setHiddenSize(Number(e.target.value))}
                min={64} step={64} />
            </label>
          </div>
          <div className="lab-controls-actions">
            <button className="btn-gold btn-lab" onClick={createNetwork}>
              <span className="btn-icon">🧠</span> Create Fresh Network
            </button>
          </div>
        </div>
      )}

      {/* Training Programs — two distinct cards */}
      {hasNetwork && !isRunning && (
        <div className="lab-programs">
          {/* Imitation Learning */}
          <div className="lab-program-card">
            <div className="lab-program-header">
              <span className="lab-program-icon">📚</span>
              <div>
                <h3>Imitation Learning</h3>
                <p>Learn by watching expert bots play</p>
              </div>
            </div>
            <div className="lab-program-fields">
              <label className="lab-field">
                <span>Expert Source</span>
                <select value={expertSource} onChange={e => setExpertSource(e.target.value)}>
                  <option value="v2v3">v2 & v3 Mix</option>
                  <option value="v2v3v5">v2, v3 & v5 Mix</option>
                </select>
              </label>
              <label className="lab-field">
                <span>Eval Opponents</span>
                <div className="lab-checkbox-group">
                  {ALL_BOT_VERSIONS.map(v => (
                    <label key={v} className="lab-checkbox">
                      <input type="checkbox" checked={evalBots.includes(v)}
                        onChange={e => {
                          if (e.target.checked) setEvalBots(prev => [...prev, v].sort());
                          else setEvalBots(prev => prev.filter(b => b !== v));
                        }} />
                      <span style={{ color: BOT_COLORS[v] }}>{BOT_LABELS[v]}</span>
                    </label>
                  ))}
                </div>
              </label>
              <label className="lab-field">
                <span>Expert Games</span>
                <input type="number" value={expertGames} onChange={e => setExpertGames(Number(e.target.value))}
                  min={10000} step={50000} />
              </label>
              <label className="lab-field">
                <span>Rounds</span>
                <input type="number" value={numRounds} onChange={e => setNumRounds(Number(e.target.value))}
                  min={1} max={50} />
              </label>
              <label className="lab-field">
                <span>Eval Games</span>
                <input type="number" value={evalGames} onChange={e => setEvalGames(Number(e.target.value))}
                  min={10} step={10} />
              </label>
              <label className="lab-field">
                <span>Learning Rate</span>
                <input type="number" value={ilLearningRate} onChange={e => setIlLearningRate(Number(e.target.value))}
                  min={0.0001} max={0.01} step={0.0001} />
              </label>
            </div>
            <button className="btn-gold btn-lab" onClick={startImitation}>
              <span className="btn-icon">📚</span> Start Imitation Learning
            </button>
          </div>

          {/* Self-Play */}
          <div className="lab-program-card lab-program-sp">
            <div className="lab-program-header">
              <span className="lab-program-icon">🎮</span>
              <div>
                <h3>Self-Play (PPO)</h3>
                <p>Run single-policy PPO or Population Based Training with periodic exploit / explore generations</p>
              </div>
            </div>
            <div className="lab-program-fields">
              <label className="lab-field">
                <span>Mode</span>
                <select value={pbtEnabled ? 'pbt' : 'single'} onChange={e => setPbtEnabled(e.target.value === 'pbt')}>
                  <option value="pbt">Island Model (parallel)</option>
                  <option value="single">Single Agent PPO</option>
                </select>
              </label>
              <label className="lab-field">
                <span>Sessions</span>
                <input type="number" value={spSessions} onChange={e => setSpSessions(Number(e.target.value))}
                  min={5} max={500} />
              </label>
              <label className="lab-field">
                <span>Games/Session</span>
                <input type="number" value={spGamesPerSession} onChange={e => setSpGamesPerSession(Number(e.target.value))}
                  min={5} max={100} />
              </label>
              <label className="lab-field">
                <span>Eval Interval</span>
                <input type="number" value={spEvalInterval} onChange={e => setSpEvalInterval(Number(e.target.value))}
                  min={1} max={50} />
              </label>
              <label className="lab-field">
                <span>Learning Rate</span>
                <input type="number" value={spLearningRate} onChange={e => setSpLearningRate(Number(e.target.value))}
                  min={0.00001} max={0.01} step={0.00001} />
              </label>
              <label className="lab-field">
                <span>FSP %</span>
                <input type="number" value={spFspRatio} onChange={e => setSpFspRatio(Number(e.target.value))}
                  min={0} max={1} step={0.1} />
              </label>
              {pbtEnabled && (
                <>
                  <label className="lab-field">
                    <span>Islands</span>
                    <input type="number" value={pbtPopulationSize} onChange={e => setPbtPopulationSize(Number(e.target.value))}
                      min={2} max={8} />
                  </label>
                  <label className="lab-field">
                    <span>Time (min)</span>
                    <input type="number" value={timeLimitMinutes} onChange={e => setTimeLimitMinutes(Number(e.target.value))}
                      min={1} max={480} />
                  </label>
                  <label className="lab-field">
                    <span>Mutation</span>
                    <input type="number" value={pbtMutationScale} onChange={e => setPbtMutationScale(Number(e.target.value))}
                      min={0.1} max={2} step={0.1} />
                  </label>
                </>
              )}
            </div>
            <button className="btn-gold btn-lab" onClick={startSelfPlay}>
              <span className="btn-icon">🎮</span> Start {pbtEnabled ? 'Island Training' : 'Self-Play Training'}
            </button>
            <button className="btn-gold btn-lab" onClick={startOvernight} style={{ marginLeft: 8, background: 'linear-gradient(135deg, #1a237e, #4a148c)' }}>
              <span className="btn-icon">🌙</span> Run Overnight
            </button>
          </div>
        </div>
      )}

      {/* Running controls */}
      {isRunning && (
        <div className="lab-controls">
          <div className="lab-controls-actions">
            <button className="btn-danger btn-lab" onClick={stopTraining}>
              <span className="btn-icon">⏹</span> Stop
            </button>
            <label className="lab-field" style={{ marginLeft: 16, display: 'inline-flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 12, color: '#aaa' }}>Smoothing {Math.round(smoothing * 100)}%</span>
              <input type="range" min={0} max={0.99} step={0.01} value={smoothing}
                onChange={e => setSmoothing(Number(e.target.value))} style={{ width: 120 }} />
            </label>
          </div>
        </div>
      )}

      {hasNetwork && !isRunning && phase === 'done' && (
        <div className="lab-controls">
          <div className="lab-controls-actions">
            <button className="btn-secondary btn-lab" onClick={resetLab}>
              <span className="btn-icon">🔄</span> Reset Lab (New Agent)
            </button>
            <label className="lab-field" style={{ marginLeft: 16, display: 'inline-flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 12, color: '#aaa' }}>Smoothing {Math.round(smoothing * 100)}%</span>
              <input type="range" min={0} max={0.99} step={0.01} value={smoothing}
                onChange={e => setSmoothing(Number(e.target.value))} style={{ width: 120 }} />
            </label>
          </div>
        </div>
      )}

      {/* Progress bar */}
      {isRunning && (
        <div className="lab-progress">
          <div className="lab-progress-bar">
            <div className="lab-progress-fill" style={{ width: `${progressPct}%` }}>
              <div className="lab-progress-glow" />
            </div>
          </div>
          <span className="lab-progress-text">
            {state?.active_program === 'self_play_pbt' ? 'Time' : state?.active_program === 'self_play' ? 'Session' : 'Round'}{' '}
            {state?.active_program === 'self_play_pbt'
              ? `${state?.training_sessions_done ?? 0}/${state?.total_training_sessions ?? 0} min`
              : `${state?.training_sessions_done ?? 0}/${state?.total_training_sessions ?? 0}`}
            {state?.active_program === 'imitation' && state?.expert_games_generated
              ? ` · ${(state.expert_games_generated / 1000).toFixed(0)}K expert games` : ''}
            {state?.active_program === 'self_play' && state?.self_play_games
              ? ` · ${state.self_play_games} self-play games` : ''}
            {state?.active_program === 'self_play_pbt' && state?.self_play_games
              ? ` · ${state.pbt_population_size} islands · ${state.self_play_games.toLocaleString()} games · ${(state.sp_games_per_second ?? 0).toFixed(0)} gps` : ''}
          </span>
        </div>
      )}

      {/* Stat cards */}
      {hasNetwork && (
        <div className="lab-stats">
          {ALL_BOT_VERSIONS.map(v => {
            const wr = latest ? (latest as Record<string, unknown>)[`vs_${v}`] : undefined;
            return (
              <StatCard
                key={v}
                label={`vs ${BOT_LABELS[v]} Bots`}
                value={wr != null ? `${((wr as number) * 100).toFixed(1)}%` : '—'}
                sublabel="Win Rate"
                color={BOT_COLORS[v]}
              />
            );
          })}
          <StatCard
            label={state?.pbt_enabled ? 'Best Fitness' : 'Loss'}
            value={state?.pbt_enabled ? (latestGeneration ? latestGeneration.max_fitness.toFixed(4) : '—') : (state?.current_loss ? state.current_loss.toFixed(4) : '—')}
            sublabel={state?.pbt_enabled ? 'Generation Leader' : 'Policy Loss'}
            color="#ff9800"
          />
          <StatCard
            label={state?.pbt_enabled ? 'Islands' : 'Age'}
            value={state?.pbt_enabled ? `${state?.pbt_population_size ?? 0}` : `${persona?.age ?? 0}`}
            sublabel={state?.pbt_enabled ? 'Parallel Trainers' : 'Training Rounds'}
            color="#d4a843"
          />
          <StatCard
            label="Snapshots"
            value={`${state?.snapshots?.length ?? 0}`}
            sublabel="In Hall of Fame"
            color="#aaa"
          />
        </div>
      )}

      {/* Self-play live stat cards */}
      {hasNetwork && hasSelfPlayData && (
        <div className="lab-stats">
          <StatCard
            label="SP Win Rate"
            value={`${((state?.sp_win_rate ?? 0) * 100).toFixed(1)}%`}
            sublabel="Self-Play"
            color="#4caf50"
          />
          <StatCard
            label="Avg Reward"
            value={(state?.sp_avg_reward ?? 0).toFixed(3)}
            sublabel="Per Game"
            color="#2196f3"
          />
          <StatCard
            label="Avg Score"
            value={(state?.sp_avg_score ?? 0).toFixed(1)}
            sublabel="Raw Points"
            color="#e040fb"
          />
          <StatCard
            label="Bid Rate"
            value={`${((state?.sp_bid_rate ?? 0) * 100).toFixed(0)}%`}
            sublabel="Games Bid On"
            color="#ff9800"
          />
          <StatCard
            label="Klop Rate"
            value={`${((state?.sp_klop_rate ?? 0) * 100).toFixed(0)}%`}
            sublabel="All Pass"
            color="#f44336"
          />
          <StatCard
            label="Games/sec"
            value={(state?.sp_games_per_second ?? 0).toFixed(1)}
            sublabel="Throughput"
            color="#aaa"
          />
        </div>
      )}

      {/* Tabs */}
      {(evalData.length > 0 || hasSelfPlayData) && (
        <>
          <div className="lab-tabs">
            {(['progress', ...(state?.pbt_enabled ? ['population'] as const : []), ...(hasSelfPlayData ? ['live'] as const : []), 'winrate', 'scores', 'hof'] as const).map(t => (
              <button key={t} className={`lab-tab ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>
                {t === 'progress' ? '📈 Progress' : t === 'population' ? '🏝️ Islands' : t === 'live' ? '🎮 Live Metrics' : t === 'winrate' ? '🎯 Win Rates' : t === 'scores' ? '📊 Scores' : '🏆 Hall of Fame'}
              </button>
            ))}
          </div>

          <div className="lab-tab-content">
            {tab === 'progress' && (
              <div className="lab-charts">
                <ChartCard title="Win Rate vs All Bot Versions" wide>
                  <ResponsiveContainer width="100%" height={320}>
                    <AreaChart data={evalData}>
                      <defs>
                        {ALL_BOT_VERSIONS.map(v => (
                          <linearGradient key={v} id={`grad${BOT_LABELS[v]}`} x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor={BOT_COLORS[v]} stopOpacity={0.3} />
                            <stop offset="95%" stopColor={BOT_COLORS[v]} stopOpacity={0} />
                          </linearGradient>
                        ))}
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                      <XAxis dataKey="label" stroke="#666" fontSize={11} />
                      <YAxis stroke="#666" fontSize={11} domain={[0, 100]} unit="%" />
                      <Tooltip {...tooltipStyle} />
                      <Legend />
                      <ReferenceLine y={25} stroke="rgba(255,255,255,0.15)" strokeDasharray="5 5" label={{ value: "Random (25%)", fill: '#555', fontSize: 10 }} />
                      {ALL_BOT_VERSIONS.map(v => (
                        <Area key={v} type="monotone" dataKey={`vs_${sanitizeKey(v)}_pct`} stroke={BOT_COLORS[v]} fill={`url(#grad${BOT_LABELS[v]})`} strokeWidth={2} name={`vs ${BOT_LABELS[v]}`} />
                      ))}
                    </AreaChart>
                  </ResponsiveContainer>
                </ChartCard>

                {lossData.length > 0 && (
                  <ChartCard title="Training Loss">
                    <ResponsiveContainer width="100%" height={260}>
                      <LineChart data={lossData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                        <XAxis dataKey="label" stroke="#666" fontSize={11} />
                        <YAxis stroke="#666" fontSize={11} />
                        <Tooltip {...tooltipStyle} />
                        <Line type="monotone" dataKey="loss" stroke="#ff9800" strokeWidth={2} dot={{ r: 4 }} name="Policy Loss" />
                      </LineChart>
                    </ResponsiveContainer>
                  </ChartCard>
                )}

                {state?.pbt_enabled && generationData.length > 0 && (
                  <ChartCard title="Population Fitness by Generation" wide>
                    <ResponsiveContainer width="100%" height={280}>
                      <LineChart data={generationData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                        <XAxis dataKey="generation" stroke="#666" fontSize={11} />
                        <YAxis stroke="#666" fontSize={11} domain={[0, 1]} />
                        <Tooltip {...tooltipStyle} />
                        <Legend />
                        <Line type="monotone" dataKey="max_fitness" stroke="#d4a843" strokeWidth={3} dot={{ r: 3 }} name="Best Fitness" />
                        <Line type="monotone" dataKey="avg_fitness" stroke="#4a9eff" strokeWidth={2} dot={false} name="Average Fitness" />
                        <Line type="monotone" dataKey="min_fitness" stroke="#ff6b6b" strokeWidth={2} dot={false} name="Worst Fitness" />
                        <Line type="monotone" dataKey="avg_v3" stroke="#7fd17f" strokeWidth={2} dot={false} name="Avg V3 Win Rate" />
                      </LineChart>
                    </ResponsiveContainer>
                  </ChartCard>
                )}
              </div>
            )}

            {tab === 'population' && state?.pbt_enabled && (
              <div className="lab-charts">
                <ChartCard title="Island Training Loops" wide>
                  {populationData.length === 0 ? (
                    <p style={{ color: '#666', textAlign: 'center', padding: 20 }}>
                      Island status will appear once training processes start.
                    </p>
                  ) : (
                    <div style={{ overflowX: 'auto' }}>
                      <table className="lab-table">
                        <thead>
                          <tr>
                            <th>Island</th>
                            <th>Status</th>
                            <th>Gen</th>
                            <th>Games</th>
                            <th>Games/sec</th>
                            <th>Fitness</th>
                            <th>vs V1</th>
                            <th>vs V3</th>
                            <th>vs V3.2</th>
                            <th>Loss</th>
                            <th>Hyperparameters</th>
                          </tr>
                        </thead>
                        <tbody>
                          {populationData.map((member, idx) => (
                            <tr key={member.index} className={idx === 0 ? 'lab-row-baseline' : ''}>
                              <td style={{ fontWeight: 'bold' }}>🏝️ {member.index}</td>
                              <td>{member.status}</td>
                              <td>{(member as any).generation ?? 0}</td>
                              <td>{(member.games ?? 0).toLocaleString()}</td>
                              <td>{((member as any).games_per_sec ?? 0).toFixed(1)}</td>
                              <td style={{ color: '#d4a843' }}>{member.fitness.toFixed(4)}</td>
                              <td>{formatPercent(member.vs_v1 ?? 0)}</td>
                              <td>{formatPercent(member.vs_v3 ?? 0)}</td>
                              <td>{member.loss.toFixed(4)}</td>
                              <td style={{ maxWidth: 320 }}>{formatHparams(member)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </ChartCard>
              </div>
            )}

            {tab === 'winrate' && (
              <div className="lab-charts">
                {ALL_BOT_VERSIONS.map((v, vi) => {
                  const sk = sanitizeKey(v);
                  const dataKey = `vs_${sk}_pct`;
                  const hasData = evalData.some(e => e[dataKey] != null);
                  if (!hasData) return null;
                  const hueBase = [120, 200, 270, 340, 30, 0][vi] ?? 0;
                  return (
                    <ChartCard key={v} title={`Win Rate vs ${BOT_LABELS[v]}`} wide={vi === 0}>
                      <ResponsiveContainer width="100%" height={280}>
                        <BarChart data={evalData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                          <XAxis dataKey="label" stroke="#666" fontSize={11} />
                          <YAxis stroke="#666" fontSize={11} domain={[0, 100]} unit="%" />
                          <Tooltip {...tooltipStyle} />
                          <ReferenceLine y={25} stroke="rgba(255,255,255,0.2)" strokeDasharray="5 5" />
                          <Bar dataKey={dataKey} name={`Win % vs ${BOT_LABELS[v]}`} radius={[4, 4, 0, 0]}>
                            {evalData.map((e, i) => (
                              <Cell key={i} fill={(e as Record<string, unknown>).program === 'self_play' ? '#e040fb' : i === 0 ? '#555' : `hsl(${hueBase + 30 * (((e[dataKey] as number) ?? 0) / 50)}, 70%, 45%)`} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </ChartCard>
                  );
                })}
              </div>
            )}

            {tab === 'scores' && (
              <div className="lab-charts">
                <ChartCard title="Average Score vs Each Bot Version" wide>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={evalData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                      <XAxis dataKey="label" stroke="#666" fontSize={11} />
                      <YAxis stroke="#666" fontSize={11} />
                      <Tooltip {...tooltipStyle} />
                      <Legend />
                      <ReferenceLine y={0} stroke="rgba(255,255,255,0.2)" />
                      {ALL_BOT_VERSIONS.map(v => (
                        <Line key={v} type="monotone" dataKey={`avg_score_${sanitizeKey(v)}`} stroke={BOT_COLORS[v]} strokeWidth={2} dot={{ r: 4 }} name={`vs ${BOT_LABELS[v]} Score`} />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </ChartCard>

                <ChartCard title="Evaluation History" wide>
                  <div style={{ overflowX: 'auto' }}>
                    <table className="lab-table">
                      <thead>
                        <tr>
                          <th>Round</th>
                          <th>Program</th>
                          {ALL_BOT_VERSIONS.map(v => (
                            <th key={`wr-${v}`}>vs {BOT_LABELS[v]} WR</th>
                          ))}
                          {ALL_BOT_VERSIONS.map(v => (
                            <th key={`sc-${v}`}>Score {BOT_LABELS[v]}</th>
                          ))}
                          <th>Loss</th>
                        </tr>
                      </thead>
                      <tbody>
                        {evalData.map((e, i) => (
                          <tr key={i} className={i === 0 ? 'lab-row-baseline' : ''}>
                            <td>{e.label as string}</td>
                            <td>
                              {e.program === 'imitation' && <span className="lab-badge-il">📚 IL</span>}
                              {e.program === 'self_play' && <span className="lab-badge-sp">🎮 SP</span>}
                              {e.program === 'init' && <span style={{ color: '#666' }}>—</span>}
                            </td>
                            {ALL_BOT_VERSIONS.map(v => {
                              const pct = e[`vs_${sanitizeKey(v)}_pct`];
                              return <td key={`wr-${v}`} style={{ color: BOT_COLORS[v] }}>{pct != null ? `${pct}%` : '—'}</td>;
                            })}
                            {ALL_BOT_VERSIONS.map(v => {
                              const sc = e[`avg_score_${sanitizeKey(v)}`];
                              return <td key={`sc-${v}`}>{sc != null ? (sc as number).toFixed(1) : '—'}</td>;
                            })}
                            <td>{(e.loss as number) ? (e.loss as number).toFixed(4) : '—'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </ChartCard>
              </div>
            )}

            {tab === 'live' && hasSelfPlayData && (
              <div className="lab-charts">
                <ChartCard title="Self-Play Win Rate" wide>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={sessionHistoryData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                      <XAxis dataKey="session" stroke="#666" fontSize={11} />
                      <YAxis stroke="#666" fontSize={11} domain={[0, 100]} tickFormatter={(v: number) => `${v}%`} />
                      <Tooltip {...tooltipStyle} />
                      <Line type="monotone" dataKey="winRate" stroke="#4caf50" strokeWidth={2} dot={false} name="Win Rate %" />
                    </LineChart>
                  </ResponsiveContainer>
                </ChartCard>

                <ChartCard title="Average Reward">
                  <ResponsiveContainer width="100%" height={260}>
                    <LineChart data={sessionHistoryData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                      <XAxis dataKey="session" stroke="#666" fontSize={11} />
                      <YAxis stroke="#666" fontSize={11} />
                      <Tooltip {...tooltipStyle} />
                      <Line type="monotone" dataKey="avgReward" stroke="#2196f3" strokeWidth={2} dot={false} name="Avg Reward" />
                    </LineChart>
                  </ResponsiveContainer>
                </ChartCard>

                <ChartCard title="Average Score">
                  <ResponsiveContainer width="100%" height={260}>
                    <LineChart data={sessionHistoryData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                      <XAxis dataKey="session" stroke="#666" fontSize={11} />
                      <YAxis stroke="#666" fontSize={11} />
                      <Tooltip {...tooltipStyle} />
                      <Line type="monotone" dataKey="avgScore" stroke="#e040fb" strokeWidth={2} dot={false} name="Avg Score" />
                    </LineChart>
                  </ResponsiveContainer>
                </ChartCard>

                <ChartCard title="Game Type Rates" wide>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={sessionHistoryData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                      <XAxis dataKey="session" stroke="#666" fontSize={11} />
                      <YAxis stroke="#666" fontSize={11} domain={[0, 100]} tickFormatter={(v: number) => `${v}%`} />
                      <Tooltip {...tooltipStyle} />
                      <Legend />
                      <Line type="monotone" dataKey="bidRate" stroke="#ff9800" strokeWidth={2} dot={false} name="Bid Rate %" />
                      <Line type="monotone" dataKey="klopRate" stroke="#f44336" strokeWidth={2} dot={false} name="Klop Rate %" />
                      <Line type="monotone" dataKey="soloRate" stroke="#9c27b0" strokeWidth={2} dot={false} name="Solo Rate %" />
                    </LineChart>
                  </ResponsiveContainer>
                </ChartCard>

                <ChartCard title="Training Loss">
                  <ResponsiveContainer width="100%" height={260}>
                    <LineChart data={sessionHistoryData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                      <XAxis dataKey="session" stroke="#666" fontSize={11} />
                      <YAxis stroke="#666" fontSize={11} />
                      <Tooltip {...tooltipStyle} />
                      <Line type="monotone" dataKey="loss" stroke="#ff9800" strokeWidth={2} dot={false} name="Loss" />
                    </LineChart>
                  </ResponsiveContainer>
                </ChartCard>

                <ChartCard title="Score Range (Min / Max per Session)" wide>
                  <ResponsiveContainer width="100%" height={300}>
                    <AreaChart data={sessionHistoryData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                      <XAxis dataKey="session" stroke="#666" fontSize={11} />
                      <YAxis stroke="#666" fontSize={11} />
                      <Tooltip {...tooltipStyle} />
                      <Legend />
                      <Area type="monotone" dataKey="maxScore" stroke="#4caf50" fill="rgba(76,175,80,0.15)" strokeWidth={1.5} dot={false} name="Max Score" />
                      <Area type="monotone" dataKey="avgScore" stroke="#2196f3" fill="rgba(33,150,243,0.1)" strokeWidth={2} dot={false} name="Avg Score" />
                      <Area type="monotone" dataKey="minScore" stroke="#f44336" fill="rgba(244,67,54,0.15)" strokeWidth={1.5} dot={false} name="Min Score" />
                      <ReferenceLine y={0} stroke="rgba(255,255,255,0.3)" strokeDasharray="3 3" />
                    </AreaChart>
                  </ResponsiveContainer>
                </ChartCard>
              </div>
            )}

            {tab === 'hof' && (
              <div className="lab-charts">
                <ChartCard title="Hall of Fame — Saved Snapshots" wide>
                  {(state?.snapshots?.length ?? 0) === 0 ? (
                    <p style={{ color: '#666', textAlign: 'center', padding: 20 }}>
                      No snapshots yet. Snapshots are saved automatically after each evaluation round.
                    </p>
                  ) : (
                    <div style={{ overflowX: 'auto' }}>
                      <table className="lab-table">
                        <thead>
                          <tr>
                            <th>#</th>
                            <th>Name</th>
                            <th>Hash</th>
                            <th>File</th>
                          </tr>
                        </thead>
                        <tbody>
                          {state!.snapshots.map((s, i) => (
                            <tr key={i}>
                              <td>{i + 1}</td>
                              <td style={{ color: '#d4a843' }}>{s.display_name}</td>
                              <td style={{ fontFamily: 'monospace', color: '#888' }}>{s.model_hash}</td>
                              <td style={{ fontFamily: 'monospace', fontSize: 11, color: '#666' }}>{s.filename}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </ChartCard>
                <ChartCard title="How to Use These Models">
                  <div style={{ color: '#aaa', fontSize: 13, padding: '8px 0', lineHeight: 1.6 }}>
                    <p>Hall of Fame models appear in the model selection dropdowns for:</p>
                    <ul style={{ paddingLeft: 20, marginTop: 8 }}>
                      <li><strong>Play vs AI</strong> — choose any snapshot as an opponent</li>
                      <li><strong>Spectate AI vs AI</strong> — pit different life stages against each other</li>
                      <li><strong>Fictitious Self-Play</strong> — used as historical opponents for training diversity</li>
                    </ul>
                  </div>
                </ChartCard>
              </div>
            )}
          </div>
        </>
      )}

      {/* Empty state */}
      {hasNetwork && evalData.length === 0 && !isRunning && (
        <div className="lab-empty">
          <div className="lab-empty-icon">🧪</div>
          <p><strong>{state?.display_name}</strong> is ready!</p>
          <p className="lab-empty-sub">
            Choose a training program above. Start with <strong>Imitation Learning</strong> to teach the basics,
            then switch to <strong>Self-Play</strong> to develop original strategies.
          </p>
        </div>
      )}
    </div>
  );
}

function StatCard({ label, value, sublabel, color }: { label: string; value: string; sublabel: string; color: string }) {
  return (
    <div className="lab-stat" style={{ borderColor: `${color}33` }}>
      <div className="lab-stat-val" style={{ color }}>{value}</div>
      <div className="lab-stat-lbl">{label}</div>
      <div className="lab-stat-sub">{sublabel}</div>
    </div>
  );
}

function ChartCard({ title, children, wide }: { title: string; children: React.ReactNode; wide?: boolean }) {
  return (
    <div className={`lab-chart ${wide ? 'lab-chart-wide' : ''}`}>
      <h3>{title}</h3>
      {children}
    </div>
  );
}
