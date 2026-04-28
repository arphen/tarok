import React, { useState, useEffect, useCallback, useRef } from 'react';
import './BotArena.css';

interface BotArenaProps {
  onBack: () => void;
  checkpoints: { filename: string; episode: number; win_rate: number; model_name?: string; is_hof?: boolean; variant?: string }[];
  onReplayGame?: (gameId: string) => void;
}

interface AgentSetup {
  name: string;
  type: string;
  checkpoint: string;
}

interface PlayerAnalytics {
  name: string;
  type: string;
  games_played: number;
  total_score: number;
  avg_score: number;
  placements: Record<number, number>;
  avg_placement: number;
  win_rate: number;
  positive_rate: number;
  bids_made: Record<string, number>;
  declared_count: number;
  declared_won: number;
  bid_won_count: number;
  declared_win_rate: number;
  avg_declared_win_score: number;
  avg_declared_loss_score: number;
  defended_count: number;
  defended_won: number;
  defended_win_rate: number;
  announcements_made: Record<string, number>;
  kontra_count: number;
  times_called: number;
  avg_taroks_in_hand: number;
  best_game: { score: number | null; game_idx: number | null; hands: number[][] | null; talon: number[] | null; trace: Record<string, unknown> | null };
  worst_game: { score: number | null; game_idx: number | null; hands: number[][] | null; talon: number[] | null; trace: Record<string, unknown> | null };
  avg_win_score: number;
  avg_loss_score: number;
  score_history: number[];
  taroks_per_contract?: Record<string, number>;
  contract_stats?: Record<string, { declared: number; won: number; win_rate: number; avg_score: number }>;
}

interface ContractAnalytics {
  played: number;
  decl_win_rate: number;
  avg_decl_score: number;
  avg_def_score: number;
}

interface NotableGame {
  score: number;
  game_idx: number;
  player_idx: number;
  player_name: string;
  contract: string;
  hands: number[][] | null;
  talon: number[] | null;
  trace: Record<string, unknown> | null;
}

interface NotableGames {
  best_non_valat: NotableGame | null;
  worst_non_valat: NotableGame | null;
  best_valat: NotableGame | null;
  worst_valat: NotableGame | null;
  by_contract: Record<string, NotableGame>;
}

interface ArenaAnalytics {
  games_done: number;
  total_games: number;
  players: PlayerAnalytics[];
  contracts: Record<string, ContractAnalytics>;
  taroks_per_contract?: Record<string, number>;
  notable_games?: NotableGames | null;
}

interface ArenaProgress {
  status: string;
  games_done: number;
  total_games: number;
  analytics: ArenaAnalytics | null;
}

interface ArenaHistoryRun {
  run_id: string;
  created_at: string;
  status: string;
  games_done: number;
  total_games: number;
  session_size: number;
  checkpoints: string[];
  agents: { name: string; type: string; checkpoint: string }[];
  analytics: ArenaAnalytics | null;
}

const DEFAULT_AGENTS_4P: AgentSetup[] = [
  { name: 'Bot-A', type: 'stockskis', checkpoint: '' },
  { name: 'Bot-B', type: 'stockskis', checkpoint: '' },
  { name: 'Bot-C', type: 'stockskis', checkpoint: '' },
  { name: 'Bot-D', type: 'stockskis', checkpoint: '' },
];

const DEFAULT_AGENTS_3P: AgentSetup[] = [
  { name: 'Bot-A', type: 'stockskis_m6_3p', checkpoint: '' },
  { name: 'Bot-B', type: 'stockskis_m6_3p', checkpoint: '' },
  { name: 'Bot-C', type: 'stockskis_m6_3p', checkpoint: '' },
];

const PLAYER_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12'];

type Tab = 'overview' | 'bidding' | 'contracts' | 'announcements' | 'best_worst' | 'scores' | 'history';

export default function BotArena({ onBack, checkpoints, onReplayGame }: BotArenaProps) {
  const [variant, setVariant] = useState<'four_player' | 'three_player'>(() => {
    const stored = typeof window !== 'undefined' ? window.localStorage.getItem('tarok.arena.variant') : null;
    return stored === 'three_player' ? 'three_player' : 'four_player';
  });
  const [agents, setAgents] = useState<AgentSetup[]>(
    () => (variant === 'three_player' ? DEFAULT_AGENTS_3P : DEFAULT_AGENTS_4P).map(a => ({ ...a })),
  );
  const [totalGames, setTotalGames] = useState(100000);
  const [sessionSize, setSessionSize] = useState(50);
  const [lapajneMcWorlds, setLapajneMcWorlds] = useState(8);
  const [lapajneMcSims, setLapajneMcSims] = useState(8);
  const [centaurHandoffTrick, setCentaurHandoffTrick] = useState(8);
  const [centaurPimcWorlds, setCentaurPimcWorlds] = useState(100);
  const [progress, setProgress] = useState<ArenaProgress | null>(null);
  const [running, setRunning] = useState(false);
  const [tab, setTab] = useState<Tab>('overview');
  const [stockskisTypes, setStockskisTypes] = useState<string[]>([]);
  const [historyRuns, setHistoryRuns] = useState<ArenaHistoryRun[]>([]);
  const [startError, setStartError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Fetch available stockskis versions
  useEffect(() => {
    fetch('/api/agents/stockskis')
      .then(r => r.json())
      .then(data => setStockskisTypes(data.types ?? []))
      .catch(() => {});
  }, []);

  const updateAgent = (idx: number, patch: Partial<AgentSetup>) => {
    setAgents(prev => prev.map((a, i) => i === idx ? { ...a, ...patch } : a));
  };

  const updateVariant = (v: 'four_player' | 'three_player') => {
    setVariant(v);
    try { window.localStorage.setItem('tarok.arena.variant', v); } catch { /* ignore */ }
    const slots = v === 'three_player' ? 3 : 4;
    const filler = v === 'three_player' ? 'stockskis_m6_3p' : 'stockskis';
    setAgents(prev => {
      const next: AgentSetup[] = [];
      for (let i = 0; i < slots; i++) {
        const cur = prev[i];
        if (!cur) {
          next.push({ name: `Bot-${String.fromCharCode(65 + i)}`, type: filler, checkpoint: '' });
          continue;
        }
        // Drop checkpoint if it was selected for the wrong variant.
        let { type, checkpoint } = cur;
        if (checkpoint) {
          const cp = checkpoints.find(c => c.filename === checkpoint);
          if (cp && cp.variant && cp.variant !== v) checkpoint = '';
        }
        // Replace 4p-only filler with 3p filler (and vice versa) for fresh slots.
        if (type === 'stockskis' && v === 'three_player') type = filler;
        if ((type === 'stockskis_v3_3p' || type === 'stockskis_m6_3p') && v === 'four_player') type = 'stockskis';
        next.push({ ...cur, type, checkpoint });
      }
      return next;
    });
  };

  const isCheckpointDisabled = (cp: { variant?: string }) =>
    Boolean(cp.variant && cp.variant !== variant);

  const pollProgress = useCallback(() => {
    fetch('/api/arena/progress')
      .then(r => r.json())
      .then((data: ArenaProgress) => {
        setProgress(data);
        setRunning(data.status === 'running');
        if (data.status === 'done' || data.status === 'error' || data.status === 'cancelled') {
          setRunning(false);
          if (pollRef.current) {
            clearInterval(pollRef.current);
            pollRef.current = null;
          }
          fetch('/api/arena/history')
            .then(r => r.json())
            .then(d => setHistoryRuns(d.runs ?? []))
            .catch(() => {});
        }
      })
      .catch(() => {});
  }, []);

  const loadHistory = useCallback(() => {
    fetch('/api/arena/history')
      .then(r => r.json())
      .then(d => setHistoryRuns(d.runs ?? []))
      .catch(() => {});
  }, []);

  const startArena = async () => {
    setStartError(null);
    // Reset UI state
    setTab('overview');
    setRunning(false);
    setProgress({ status: 'running', games_done: 0, total_games: totalGames, analytics: null });

    try {
      const resp = await fetch('/api/arena/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          agents: agents.map(a => ({
            name: a.name,
            type: a.type,
            checkpoint: a.checkpoint || undefined,
          })),
          total_games: totalGames,
          session_size: sessionSize,
          variant,
          lapajne_mc_worlds: lapajneMcWorlds,
          lapajne_mc_sims: lapajneMcSims,
          centaur_handoff_trick: centaurHandoffTrick,
          centaur_pimc_worlds: centaurPimcWorlds,
        }),
      });
      const data = await resp.json();
      if (data.status === 'started') {
        setRunning(true);
        setProgress({ status: 'running', games_done: 0, total_games: totalGames, analytics: null });
        pollRef.current = setInterval(pollProgress, 2000);
        return;
      }
      if (data.status === 'already_running') {
        setRunning(true);
        pollProgress();
        if (!pollRef.current) pollRef.current = setInterval(pollProgress, 2000);
        setStartError('Arena is already running. Showing live progress.');
        return;
      }
      setRunning(false);
      setProgress(null);
      setStartError(data?.message ?? `Arena start failed (${data?.status ?? 'unknown'}).`);
    } catch (e) {
      setRunning(false);
      setProgress(null);
      setStartError(`Arena start failed: ${e instanceof Error ? e.message : 'network error'}`);
    }
  };

  const stopArena = async () => {
    await fetch('/api/arena/stop', { method: 'POST' });
    setRunning(false);
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    // One final poll to get latest state
    pollProgress();
  };

  useEffect(() => {
    pollProgress();
    loadHistory();
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [loadHistory, pollProgress]);

  const analytics = progress?.analytics;
  const latestHistoryAnalytics = [...historyRuns].reverse().find(r => r.analytics)?.analytics ?? null;
  const displayAnalytics = analytics ?? latestHistoryAnalytics;

  useEffect(() => {
    if (!displayAnalytics && historyRuns.length > 0 && tab !== 'history') {
      setTab('history');
    }
  }, [displayAnalytics, historyRuns.length, tab]);

  const pct = progress ? Math.round((progress.games_done / Math.max(progress.total_games, 1)) * 100) : 0;

  return (
    <div className="arena">
      <div className="arena-header">
        <button className="btn-secondary btn-sm" onClick={onBack}>← Back</button>
        <h1>Bot Arena</h1>
        <span className="arena-subtitle">Mass-simulate games and analyse bot behaviour</span>
      </div>

      {/* Configuration */}
      <div className="arena-config">
        {startError && <div className="arena-error-banner">{startError}</div>}
        <div className="arena-variant" style={{ display: 'flex', gap: 16, marginBottom: 12 }}>
          <strong style={{ alignSelf: 'center' }}>Variant:</strong>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
            <input
              type="radio"
              name="arena-variant"
              value="four_player"
              checked={variant === 'four_player'}
              onChange={() => updateVariant('four_player')}
              disabled={running}
            />
            <span>4-player</span>
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
            <input
              type="radio"
              name="arena-variant"
              value="three_player"
              checked={variant === 'three_player'}
              onChange={() => updateVariant('three_player')}
              disabled={running}
            />
            <span>3-player</span>
          </label>
        </div>
        <div className="arena-agents">
          <h3>Agents</h3>
          <div className="arena-agent-grid">
            {agents.map((agent, idx) => (
              <div key={idx} className="arena-agent-card" style={{ borderColor: PLAYER_COLORS[idx] }}>
                <div className="arena-agent-seat" style={{ background: PLAYER_COLORS[idx] }}>Seat {idx}</div>
                <input
                  className="arena-input"
                  value={agent.name}
                  onChange={e => updateAgent(idx, { name: e.target.value })}
                  placeholder="Name"
                />
                <select
                  className="arena-select"
                  value={agent.type}
                  onChange={e => updateAgent(idx, { type: e.target.value, checkpoint: '' })}
                >
                  <option value="random">Random</option>
                  {variant === 'four_player' && (
                    <option value="stockskis">StockŠkis (latest)</option>
                  )}
                  {variant === 'three_player' && (
                    <option value="stockskis_m6_3p">StockSkis m6 (3-player)</option>
                  )}
                  {variant === 'three_player' && (
                    <option value="stockskis_v3_3p">StockŠkis v3 (3-player)</option>
                  )}
                  {variant === 'four_player' && stockskisTypes.map(t => (
                    <option key={t} value={t}>{t.replace('_', ' ')}</option>
                  ))}
                  {variant === 'four_player' && (
                    <option value="centaur">Centaur (NN + PIMC endgame)</option>
                  )}
                  <option value="rl">Neural Network (checkpoint)</option>
                </select>
                {(agent.type === 'rl' || agent.type === 'centaur') && (
                  <select
                    className="arena-select"
                    value={agent.checkpoint}
                    onChange={e => updateAgent(idx, { checkpoint: e.target.value })}
                  >
                    <option value="">Latest checkpoint</option>
                    {checkpoints.map(cp => {
                      const disabled = isCheckpointDisabled(cp);
                      const trainedFor = cp.variant === 'three_player' ? '3-player' : '4-player';
                      return (
                        <option
                          key={cp.filename}
                          value={cp.filename}
                          disabled={disabled}
                          title={disabled ? `Trained for ${trainedFor}; switch variant to use` : undefined}
                        >
                          {cp.model_name || cp.filename} (WR: {(cp.win_rate * 100).toFixed(0)}%){disabled ? ` — ${trainedFor} only` : ''}
                        </option>
                      );
                    })}
                  </select>
                )}
              </div>
            ))}
          </div>
        </div>

        <div className="arena-params">
          <label>
            Total Games
            <input type="number" min={100} max={500000} step={1000} value={totalGames}
              onChange={e => setTotalGames(Number(e.target.value))} />
          </label>
          <label>
            Session Size
            <input type="number" min={1} max={1000} step={10} value={sessionSize}
              onChange={e => setSessionSize(Number(e.target.value))} />
          </label>
          <label>
            Lapajne MC Worlds
            <input type="number" min={1} max={512} step={1} value={lapajneMcWorlds}
              onChange={e => setLapajneMcWorlds(Number(e.target.value))} />
          </label>
          <label>
            Lapajne MC Sims
            <input type="number" min={1} max={4096} step={1} value={lapajneMcSims}
              onChange={e => setLapajneMcSims(Number(e.target.value))} />
          </label>
          <label>
            Centaur Handoff Trick
            <input type="number" min={0} max={12} step={1} value={centaurHandoffTrick}
              onChange={e => setCentaurHandoffTrick(Number(e.target.value))} />
          </label>
          <label>
            Centaur PIMC Worlds
            <input type="number" min={1} max={2000} step={10} value={centaurPimcWorlds}
              onChange={e => setCentaurPimcWorlds(Number(e.target.value))} />
          </label>
          <div className="arena-actions">
            {!running ? (
              <button className="btn-gold btn-large" onClick={startArena}>
                Run {totalGames.toLocaleString()} Games
              </button>
            ) : (
              <button className="btn-secondary btn-large" onClick={stopArena}>
                Stop Arena
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Progress */}
      {progress && progress.status !== 'idle' && (
        <div className="arena-progress-bar-container">
          <div className="arena-progress-bar">
            <div className="arena-progress-fill" style={{ width: `${pct}%` }} />
          </div>
          <span className="arena-progress-label">
            {progress.games_done.toLocaleString()} / {progress.total_games.toLocaleString()} games ({pct}%)
            {progress.status === 'done' && ' — Complete'}
            {progress.status === 'cancelled' && ' — Cancelled'}
            {progress.status === 'error' && ' — Error'}
          </span>
        </div>
      )}

      {/* Analytics */}
      {(displayAnalytics || historyRuns.length > 0) && (
        <div className="arena-analytics">
          <div className="arena-tabs">
            {(['overview', 'bidding', 'contracts', 'announcements', 'best_worst', 'scores', 'history'] as Tab[]).map(t => (
              <button
                key={t}
                className={`arena-tab ${tab === t ? 'active' : ''}`}
                onClick={() => setTab(t)}
              >
                {t === 'overview' ? 'Overview' :
                 t === 'bidding' ? 'Bidding' :
                 t === 'contracts' ? 'Contracts' :
                 t === 'announcements' ? 'Announcements' :
                 t === 'best_worst' ? 'Notable Games' :
                 t === 'scores' ? 'Score Trends' :
                 'History'}
              </button>
            ))}
          </div>

          <div className="arena-tab-content">
            {tab !== 'history' && !displayAnalytics && (
              <p className="arena-empty">No analytics available yet. Run a new arena session to populate stats.</p>
            )}
            {tab === 'overview' && displayAnalytics && <OverviewTab players={displayAnalytics.players} />}
            {tab === 'bidding' && displayAnalytics && <BiddingTab players={displayAnalytics.players} taroksPerContract={displayAnalytics.taroks_per_contract} />}
            {tab === 'contracts' && displayAnalytics && <ContractsTab contracts={displayAnalytics.contracts} players={displayAnalytics.players} />}
            {tab === 'announcements' && displayAnalytics && <AnnouncementsTab players={displayAnalytics.players} />}
            {tab === 'best_worst' && displayAnalytics && <BestWorstTab players={displayAnalytics.players} notableGames={displayAnalytics.notable_games} onReplayGame={onReplayGame} />}
            {tab === 'scores' && displayAnalytics && <ScoresTab players={displayAnalytics.players} sessionSize={sessionSize} />}
            {tab === 'history' && <HistoryTab runs={historyRuns} />}
          </div>
        </div>
      )}

      <DuplicateMatchPanel checkpoints={checkpoints} />
    </div>
  );
}

/* ============ Overview Tab ============ */
function OverviewTab({ players }: { players: PlayerAnalytics[] }) {
  return (
    <div className="arena-overview">
      <div className="arena-table-wrapper">
        <table className="arena-table">
          <thead>
            <tr>
              <th>Player</th>
              <th>Games</th>
              <th>Avg Score / Sess</th>
              <th>Avg Place</th>
              <th>1st %</th>
              <th>Positive %</th>
              <th>Bid Won</th>
              <th>Avg Taroks</th>
              <th>Decl WR</th>
              <th>Decl Avg Win</th>
              <th>Decl Avg Loss</th>
              <th>Called</th>
            </tr>
          </thead>
          <tbody>
            {players.map((p, i) => (
              <tr key={i}>
                <td><span className="arena-dot" style={{ background: PLAYER_COLORS[i] }} />{p.name}</td>
                <td>{p.games_played.toLocaleString()}</td>
                <td className={p.avg_score >= 0 ? 'positive' : 'negative'}>{p.avg_score.toFixed(1)}</td>
                <td>{p.avg_placement.toFixed(2)}</td>
                <td>{p.win_rate.toFixed(1)}%</td>
                <td>{p.positive_rate.toFixed(1)}%</td>
                <td>{p.bid_won_count.toLocaleString()}</td>
                <td>{p.avg_taroks_in_hand.toFixed(2)}</td>
                <td>{p.declared_win_rate.toFixed(1)}%</td>
                <td className="positive">{p.avg_declared_win_score.toFixed(1)}</td>
                <td className="negative">{p.avg_declared_loss_score.toFixed(1)}</td>
                <td>{p.times_called.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Placement Distribution */}
      <h3>Placement Distribution</h3>
      <div className="arena-placement-grid">
        {players.map((p, i) => {
          const total = Object.values(p.placements).reduce((a, b) => a + b, 0);
          const maxCount = Math.max(...Object.values(p.placements), 1);
          return (
            <div key={i} className="arena-placement-card" style={{ borderColor: PLAYER_COLORS[i] }}>
              <h4 style={{ color: PLAYER_COLORS[i] }}>{p.name} <span className="arena-placement-total">({total.toLocaleString()} games)</span></h4>
              <div className="arena-placement-bars">
                {[1, 2, 3, 4].map(place => {
                  const count = p.placements[place] || 0;
                  const pct = total > 0 ? (count / total * 100) : 0;
                  const barPct = (count / maxCount) * 100;
                  return (
                    <div key={place} className="arena-placement-row">
                      <span className="arena-placement-label">{place === 1 ? '1st' : place === 2 ? '2nd' : place === 3 ? '3rd' : '4th'}</span>
                      <div className="arena-bar-track">
                        <div className="arena-bar-fill" style={{ width: `${barPct}%`, background: PLAYER_COLORS[i], opacity: 1 - (place - 1) * 0.2 }} />
                      </div>
                      <span className="arena-placement-value">{count.toLocaleString()} ({pct.toFixed(1)}%)</span>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ============ Bidding Tab ============ */
function BiddingTab({ players, taroksPerContract }: { players: PlayerAnalytics[]; taroksPerContract?: Record<string, number> }) {
  // Collect all bid types
  const allBids = new Set<string>();
  players.forEach(p => Object.keys(p.bids_made).forEach(b => allBids.add(b)));
  const bidNames = Array.from(allBids).sort();

  return (
    <div className="arena-bidding">
      <h3>Bids Made per Player</h3>
      <div className="arena-table-wrapper">
        <table className="arena-table">
          <thead>
            <tr>
              <th>Player</th>
              {bidNames.map(b => <th key={b}>{formatContractName(b)}</th>)}
              <th>Total Bids</th>
              <th>Bid Wins</th>
              <th>Declared</th>
            </tr>
          </thead>
          <tbody>
            {players.map((p, i) => {
              const totalBids = Object.values(p.bids_made).reduce((a, b) => a + b, 0);
              return (
                <tr key={i}>
                  <td><span className="arena-dot" style={{ background: PLAYER_COLORS[i] }} />{p.name}</td>
                  {bidNames.map(b => <td key={b}>{(p.bids_made[b] || 0).toLocaleString()}</td>)}
                  <td><strong>{totalBids.toLocaleString()}</strong></td>
                  <td>{p.bid_won_count.toLocaleString()}</td>
                  <td>{p.declared_count.toLocaleString()}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Bid distribution visual */}
      <h3>Bid Frequency (% of games)</h3>
      <div className="arena-bid-bars">
        {bidNames.map(bid => (
          <div key={bid} className="arena-bid-group">
            <span className="arena-bid-label">{formatContractName(bid)}</span>
            <div className="arena-bid-group-bars">
              {players.map((p, i) => {
                const cnt = p.bids_made[bid] || 0;
                const pct = cnt / Math.max(p.games_played, 1) * 100;
                return (
                  <div key={i} className="arena-bid-row">
                    <div className="arena-bar-track narrow">
                      <div className="arena-bar-fill" style={{ width: `${Math.min(pct, 100)}%`, background: PLAYER_COLORS[i] }} />
                    </div>
                    <span className="arena-bar-value">{pct.toFixed(1)}%</span>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      {/* Taroks per Contract chart — per player */}
      {players.some(p => p.taroks_per_contract && Object.keys(p.taroks_per_contract).length > 0) && (
        <>
          <h3>Avg Taroks in Declarer&apos;s Hand by Contract</h3>
          <div className="arena-taroks-xy">
            {(() => {
              const CONTRACT_ORDER = ['THREE', 'TWO', 'ONE', 'SOLO_THREE', 'SOLO_TWO', 'SOLO_ONE', 'SOLO', 'BERAC', 'BARVNI_VALAT'];
              // Collect all contracts across all players
              const allContracts = new Set<string>();
              players.forEach(p => {
                if (p.taroks_per_contract) Object.keys(p.taroks_per_contract).forEach(c => allContracts.add(c));
              });
              if (taroksPerContract) Object.keys(taroksPerContract).forEach(c => allContracts.add(c));
              const contractNames = Array.from(allContracts)
                .filter(c => c !== 'KLOP')
                .sort((a, b) => {
                  const ai = CONTRACT_ORDER.indexOf(a);
                  const bi = CONTRACT_ORDER.indexOf(b);
                  return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi);
                });
              if (contractNames.length === 0) return null;

              // SVG chart dimensions
              const W = 600, H = 300, PAD_L = 50, PAD_R = 20, PAD_T = 20, PAD_B = 60;
              const plotW = W - PAD_L - PAD_R;
              const plotH = H - PAD_T - PAD_B;

              // Compute max Y
              let maxY = 0;
              players.forEach(p => {
                if (p.taroks_per_contract) {
                  Object.values(p.taroks_per_contract).forEach(v => { if (v > maxY) maxY = v; });
                }
              });
              if (taroksPerContract) {
                Object.values(taroksPerContract).forEach(v => { if (v > maxY) maxY = v; });
              }
              maxY = Math.ceil(maxY + 1);

              const xScale = (i: number) => PAD_L + (i / (contractNames.length - 1 || 1)) * plotW;
              const yScale = (v: number) => PAD_T + plotH - (v / maxY) * plotH;

              return (
                <svg viewBox={`0 0 ${W} ${H}`} className="arena-taroks-svg">
                  {/* Y axis gridlines */}
                  {Array.from({ length: Math.min(maxY + 1, 13) }, (_, i) => i).map(tick => (
                    <g key={tick}>
                      <line x1={PAD_L} y1={yScale(tick)} x2={W - PAD_R} y2={yScale(tick)} stroke="#333" strokeWidth={0.5} strokeDasharray={tick === 0 ? '' : '3,3'} />
                      <text x={PAD_L - 8} y={yScale(tick) + 4} textAnchor="end" fontSize={10} fill="#aaa">{tick}</text>
                    </g>
                  ))}
                  {/* X axis labels */}
                  {contractNames.map((c, i) => (
                    <text key={c} x={xScale(i)} y={H - PAD_B + 20} textAnchor="middle" fontSize={9} fill="#aaa" transform={`rotate(-30, ${xScale(i)}, ${H - PAD_B + 20})`}>
                      {formatContractName(c)}
                    </text>
                  ))}
                  {/* Pooled average (dashed) */}
                  {taroksPerContract && (() => {
                    const pts = contractNames
                      .map((c, i) => taroksPerContract[c] != null ? `${xScale(i)},${yScale(taroksPerContract[c])}` : null)
                      .filter(Boolean);
                    return pts.length > 1 ? (
                      <polyline points={pts.join(' ')} fill="none" stroke="#888" strokeWidth={1.5} strokeDasharray="5,4" />
                    ) : null;
                  })()}
                  {/* Per-player lines */}
                  {players.map((p, pi) => {
                    if (!p.taroks_per_contract) return null;
                    const pts = contractNames.map((c, i) => {
                      const v = p.taroks_per_contract?.[c];
                      return v != null ? { x: xScale(i), y: yScale(v), v } : null;
                    });
                    const validPts = pts.filter(Boolean) as { x: number; y: number; v: number }[];
                    if (validPts.length === 0) return null;
                    return (
                      <g key={pi}>
                        <polyline
                          points={validPts.map(p => `${p.x},${p.y}`).join(' ')}
                          fill="none"
                          stroke={PLAYER_COLORS[pi]}
                          strokeWidth={2}
                        />
                        {validPts.map((pt, j) => (
                          <circle key={j} cx={pt.x} cy={pt.y} r={3.5} fill={PLAYER_COLORS[pi]}>
                            <title>{p.name}: {pt.v.toFixed(2)}</title>
                          </circle>
                        ))}
                      </g>
                    );
                  })}
                </svg>
              );
            })()}
            {/* Legend */}
            <div className="arena-taroks-legend">
              {players.map((p, i) => (
                <span key={i} className="arena-taroks-legend-item">
                  <span className="arena-dot" style={{ background: PLAYER_COLORS[i] }} />{p.name}
                </span>
              ))}
              <span className="arena-taroks-legend-item">
                <span className="arena-dot" style={{ background: '#888', borderStyle: 'dashed' }} />Pooled avg
              </span>
            </div>
          </div>
        </>
      )}
      {/* Fallback: pooled bar chart if no per-player data */}
      {!players.some(p => p.taroks_per_contract && Object.keys(p.taroks_per_contract).length > 0) &&
       taroksPerContract && Object.keys(taroksPerContract).length > 0 && (
        <>
          <h3>Avg Taroks in Declarer&apos;s Hand by Contract</h3>
          <div className="arena-taroks-chart">
            {(() => {
              const CONTRACT_ORDER = ['KLOP', 'THREE', 'TWO', 'ONE', 'SOLO_THREE', 'SOLO_TWO', 'SOLO_ONE', 'SOLO', 'BERAC', 'BARVNI_VALAT'];
              const entries = Object.entries(taroksPerContract)
                .sort((a, b) => {
                  const ai = CONTRACT_ORDER.indexOf(a[0]);
                  const bi = CONTRACT_ORDER.indexOf(b[0]);
                  return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi);
                });
              const maxTaroks = Math.max(...entries.map(([, v]) => v), 1);
              return entries.map(([name, avg]) => (
                <div key={name} className="arena-taroks-row">
                  <span className="arena-taroks-label">{formatContractName(name)}</span>
                  <div className="arena-bar-track">
                    <div className="arena-bar-fill taroks-bar" style={{ width: `${(avg / maxTaroks) * 100}%` }} />
                  </div>
                  <span className="arena-taroks-value">{avg.toFixed(2)}</span>
                </div>
              ));
            })()}
          </div>
        </>
      )}
    </div>
  );
}

/* ============ Contracts Tab ============ */
function ContractsTab({ contracts, players }: { contracts: Record<string, ContractAnalytics>; players: PlayerAnalytics[] }) {
  const entries = Object.entries(contracts).sort((a, b) => b[1].played - a[1].played);
  const maxPlayed = Math.max(...entries.map(([, c]) => c.played), 1);
  // Collect all contract names across all players
  const allContracts = new Set<string>();
  entries.forEach(([name]) => allContracts.add(name));
  players.forEach(p => {
    if (p.contract_stats) Object.keys(p.contract_stats).forEach(n => allContracts.add(n));
  });
  const sortedContracts = Array.from(allContracts).sort((a, b) => {
    const ca = contracts[a]?.played ?? 0;
    const cb = contracts[b]?.played ?? 0;
    return cb - ca;
  });

  return (
    <div className="arena-contracts">
      <h3>Contract Statistics (Global)</h3>
      <div className="arena-table-wrapper">
        <table className="arena-table">
          <thead>
            <tr>
              <th>Contract</th>
              <th>Times Played</th>
              <th>Decl Win Rate</th>
              <th>Avg Decl Score</th>
              <th>Avg Def Score</th>
              <th>Frequency</th>
            </tr>
          </thead>
          <tbody>
            {entries.map(([name, c]) => (
              <tr key={name}>
                <td><strong>{formatContractName(name)}</strong></td>
                <td>{c.played.toLocaleString()}</td>
                <td>{c.decl_win_rate.toFixed(1)}%</td>
                <td className={c.avg_decl_score >= 0 ? 'positive' : 'negative'}>{c.avg_decl_score.toFixed(1)}</td>
                <td className={c.avg_def_score >= 0 ? 'positive' : 'negative'}>{c.avg_def_score.toFixed(1)}</td>
                <td>
                  <div className="arena-bar-track">
                    <div className="arena-bar-fill contract-bar" style={{ width: `${(c.played / maxPlayed) * 100}%` }} />
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h3 style={{ marginTop: '2rem' }}>Contract Statistics by Player</h3>
      <div className="arena-table-wrapper">
        <table className="arena-table">
          <thead>
            <tr>
              <th>Contract</th>
              {players.map((p, i) => (
                <th key={i} colSpan={3}>
                  <span className="arena-dot" style={{ background: PLAYER_COLORS[i] }} />{p.name}
                </th>
              ))}
            </tr>
            <tr>
              <th></th>
              {players.map((_, i) => (
                <React.Fragment key={i}>
                  <th>Declared</th>
                  <th>Win %</th>
                  <th>Avg Score</th>
                </React.Fragment>
              ))}
            </tr>
          </thead>
          <tbody>
            {sortedContracts.map(name => (
              <tr key={name}>
                <td><strong>{formatContractName(name)}</strong></td>
                {players.map((p, i) => {
                  const cs = p.contract_stats?.[name];
                  if (!cs || cs.declared === 0) {
                    return (
                      <React.Fragment key={i}>
                        <td>0</td>
                        <td>—</td>
                        <td>—</td>
                      </React.Fragment>
                    );
                  }
                  return (
                    <React.Fragment key={i}>
                      <td>{cs.declared.toLocaleString()}</td>
                      <td>{cs.win_rate.toFixed(1)}%</td>
                      <td className={cs.avg_score >= 0 ? 'positive' : 'negative'}>{cs.avg_score.toFixed(1)}</td>
                    </React.Fragment>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ============ Announcements Tab ============ */
function AnnouncementsTab({ players }: { players: PlayerAnalytics[] }) {
  const allAnns = new Set<string>();
  players.forEach(p => Object.keys(p.announcements_made).forEach(a => allAnns.add(a)));
  const annNames = Array.from(allAnns).sort();

  const hasAny = annNames.length > 0 || players.some(p => p.kontra_count > 0);

  return (
    <div className="arena-announcements">
      <h3>Announcement & Kontra Stats</h3>
      {!hasAny ? (
        <p className="arena-empty">No announcements or kontras were made in these games.</p>
      ) : (
        <div className="arena-table-wrapper">
          <table className="arena-table">
            <thead>
              <tr>
                <th>Player</th>
                {annNames.map(a => <th key={a}>{formatAnnouncement(a)}</th>)}
                <th>Kontras</th>
              </tr>
            </thead>
            <tbody>
              {players.map((p, i) => (
                <tr key={i}>
                  <td><span className="arena-dot" style={{ background: PLAYER_COLORS[i] }} />{p.name}</td>
                  {annNames.map(a => <td key={a}>{(p.announcements_made[a] || 0).toLocaleString()}</td>)}
                  <td>{p.kontra_count.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

/* ============ Best / Worst Tab ============ */
function BestWorstTab({ players, notableGames, onReplayGame }: { players: PlayerAnalytics[]; notableGames?: NotableGames | null; onReplayGame?: (gameId: string) => void }) {
  const [replaying, setReplaying] = useState<string | null>(null);

  const handleReplayNotable = async (game: NotableGame) => {
    if (!game.hands || !game.talon || !onReplayGame) return;
    setReplaying(`${game.contract}-${game.game_idx}`);
    try {
      const agents = players.map(p => ({ name: p.name, type: p.type }));
      const res = await fetch('/api/arena/replay', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          hands: game.hands,
          talon: game.talon,
          agents,
          dealer: game.trace?.dealer ?? game.game_idx % 4,
          delay: 0,
          trace: game.trace ?? undefined,
        }),
      });
      const data = await res.json();
      onReplayGame(data.game_id);
    } catch (e) {
      console.error('Failed to start arena replay', e);
    } finally {
      setReplaying(null);
    }
  };

  const renderNotableCard = (label: string, game: NotableGame | null | undefined, colorClass: 'positive' | 'negative') => (
    <div className="arena-notable-card">
      <span className="arena-bw-label">{label}</span>
      {game ? (
        <>
          {game.hands && onReplayGame ? (
            <button
              className={`arena-bw-score ${colorClass} arena-bw-replay-btn`}
              onClick={() => handleReplayNotable(game)}
              disabled={replaying !== null}
              title="Click to replay this game"
            >
              {game.score > 0 ? '+' : ''}{game.score} ▶
            </button>
          ) : (
            <span className={`arena-bw-score ${colorClass}`}>{game.score > 0 ? '+' : ''}{game.score}</span>
          )}
          <span className="arena-notable-contract">{game.contract.replace(/_/g, ' ')}</span>
          <span className="arena-bw-game">{game.player_name} · Game #{game.game_idx}</span>
        </>
      ) : (
        <span className="arena-bw-score" style={{ opacity: 0.3 }}>—</span>
      )}
    </div>
  );

  const CONTRACT_ORDER = ['THREE', 'TWO', 'ONE', 'SOLO_THREE', 'SOLO_TWO', 'SOLO_ONE', 'SOLO', 'BERAC', 'KLOP', 'BARVNI_VALAT'];
  const byContract = notableGames?.by_contract ?? {};
  const sortedContracts = Object.keys(byContract).sort((a, b) => {
    const ai = CONTRACT_ORDER.indexOf(a), bi = CONTRACT_ORDER.indexOf(b);
    return (ai < 0 ? 99 : ai) - (bi < 0 ? 99 : bi);
  });

  return (
    <div className="arena-best-worst">
      <h3>Notable Games</h3>

      <div className="arena-notable-grid">
        <div className="arena-notable-section">
          <h4>Non-Valat</h4>
          <div className="arena-notable-pair">
            {renderNotableCard('Best', notableGames?.best_non_valat, 'positive')}
            {renderNotableCard('Worst', notableGames?.worst_non_valat, 'negative')}
          </div>
        </div>
        <div className="arena-notable-section">
          <h4>Valat</h4>
          <div className="arena-notable-pair">
            {renderNotableCard('Best', notableGames?.best_valat, 'positive')}
            {renderNotableCard('Worst', notableGames?.worst_valat, 'negative')}
          </div>
        </div>
      </div>

      {sortedContracts.length > 0 && (
        <>
          <h3 style={{ marginTop: '24px' }}>Sample Games by Contract</h3>
          <div className="arena-table-wrapper">
            <table className="arena-table arena-contract-samples">
              <thead>
                <tr>
                  <th>Contract</th>
                  <th>Best Score</th>
                  <th>Player</th>
                  <th>Game</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {sortedContracts.map(cname => {
                  const g = byContract[cname];
                  return (
                    <tr key={cname}>
                      <td><strong>{cname.replace(/_/g, ' ')}</strong></td>
                      <td className={g.score > 0 ? 'positive' : g.score < 0 ? 'negative' : ''}>{g.score > 0 ? '+' : ''}{g.score}</td>
                      <td><span className="arena-dot" style={{ background: PLAYER_COLORS[g.player_idx] ?? '#888' }} />{g.player_name}</td>
                      <td className="arena-bw-game">#{g.game_idx}</td>
                      <td>
                        {g.hands && onReplayGame ? (
                          <button
                            className="arena-bw-replay-btn arena-bw-score"
                            onClick={() => handleReplayNotable(g)}
                            disabled={replaying !== null}
                            title="Replay this game"
                            style={{ fontSize: '0.85rem' }}
                          >
                            ▶
                          </button>
                        ) : null}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}

/* ============ Score Trends Tab ============ */
function ScoresTab({ players, sessionSize }: { players: PlayerAnalytics[]; sessionSize: number }) {
  // Simple SVG line chart of cumulative score per session
  const maxLen = Math.max(...players.map(p => p.score_history.length), 1);

  // Convert to cumulative
  const cumulative = players.map(p => {
    let sum = 0;
    return p.score_history.map(s => { sum += s; return sum; });
  });

  if (maxLen <= 1) {
    return <div className="arena-scores"><p className="arena-empty">Waiting for session data...</p></div>;
  }

  const allVals = cumulative.flat();
  const minVal = Math.min(...allVals, 0);
  const maxVal = Math.max(...allVals, 1);
  const range = maxVal - minVal || 1;

  const W = 800;
  const H = 300;
  const pad = 40;

  const toX = (i: number) => pad + (i / (maxLen - 1)) * (W - 2 * pad);
  const toY = (v: number) => H - pad - ((v - minVal) / range) * (H - 2 * pad);

  return (
    <div className="arena-scores">
      <h3>Cumulative Score per Session ({sessionSize} games each)</h3>
      <svg viewBox={`0 0 ${W} ${H}`} className="arena-chart">
        {/* Grid */}
        <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="#444" strokeWidth={1} />
        <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="#444" strokeWidth={1} />
        {/* Zero line */}
        {minVal < 0 && maxVal > 0 && (
          <line x1={pad} y1={toY(0)} x2={W - pad} y2={toY(0)} stroke="#666" strokeWidth={1} strokeDasharray="4" />
        )}
        {/* Lines */}
        {cumulative.map((data, pi) => {
          if (data.length < 2) return null;
          const d = data.map((v, i) => `${i === 0 ? 'M' : 'L'}${toX(i).toFixed(1)},${toY(v).toFixed(1)}`).join(' ');
          return <path key={pi} d={d} fill="none" stroke={PLAYER_COLORS[pi]} strokeWidth={2} />;
        })}
        {/* Y-axis labels */}
        <text x={pad - 5} y={pad} textAnchor="end" fill="#aaa" fontSize={10}>{maxVal.toLocaleString()}</text>
        <text x={pad - 5} y={H - pad} textAnchor="end" fill="#aaa" fontSize={10}>{minVal.toLocaleString()}</text>
        {/* X-axis label */}
        <text x={W / 2} y={H - 5} textAnchor="middle" fill="#aaa" fontSize={11}>Sessions</text>
      </svg>
      <div className="arena-legend">
        {players.map((p, i) => (
          <span key={i} className="arena-legend-item">
            <span className="arena-dot" style={{ background: PLAYER_COLORS[i] }} />
            {p.name}
          </span>
        ))}
      </div>
    </div>
  );
}

/* ============ Helpers ============ */

function formatContractName(name: string): string {
  const map: Record<string, string> = {
    KLOP: 'Klop', THREE: 'Three', TWO: 'Two', ONE: 'One',
    SOLO_THREE: 'Solo Three', SOLO_TWO: 'Solo Two', SOLO_ONE: 'Solo One',
    SOLO: 'Solo', BERAC: 'Berač', BARVNI_VALAT: 'Barvni Valat',
  };
  return map[name] || name;
}

function formatAnnouncement(name: string): string {
  const map: Record<string, string> = {
    TRULA: 'Trula', KINGS: 'Kings', PAGAT_ULTIMO: 'Pagat Ultimo',
    VALAT: 'Valat', BARVNI_VALAT: 'Barvni Valat',
  };
  return map[name] || name;
}

function HistoryTab({ runs }: { runs: ArenaHistoryRun[] }) {
  if (runs.length === 0) {
    return <p className="arena-empty">No persisted arena runs yet.</p>;
  }

  const ordered = [...runs].reverse();
  return (
    <div className="arena-history">
      <h3>Saved Arena Runs</h3>
      <div className="arena-table-wrapper">
        <table className="arena-table">
          <thead>
            <tr>
              <th>Started</th>
              <th>Status</th>
              <th>Games</th>
              <th>Checkpoints</th>
              <th>Top Bot</th>
            </tr>
          </thead>
          <tbody>
            {ordered.map((run) => {
              const players = run.analytics?.players ?? [];
              const top = [...players].sort((a, b) => a.avg_placement - b.avg_placement)[0];
              return (
                <tr key={run.run_id}>
                  <td>{new Date(run.created_at).toLocaleString()}</td>
                  <td>{run.status}</td>
                  <td>{run.games_done.toLocaleString()} / {run.total_games.toLocaleString()}</td>
                  <td>{run.checkpoints.length ? run.checkpoints.join(', ') : 'none'}</td>
                  <td>{top ? `${top.name} (${top.avg_placement.toFixed(2)})` : '—'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ============ Duplicate Match Panel ============ */
interface DuplicateResult {
  boards_played: number;
  challenger_mean_score: number;
  defender_mean_score: number;
  mean_duplicate_advantage: number;
  duplicate_advantage_std: number;
  ci_low_95: number;
  ci_high_95: number;
  imps_per_board: number;
}

interface DuplicateRun {
  run_id: string;
  created_at: string;
  status: string;
  challenger: string;
  defender: string;
  boards: number;
  seed: number;
  pairing: string;
  result: DuplicateResult | null;
  error: string | null;
}

interface DuplicateProgress {
  status: string;
  challenger?: string;
  defender?: string;
  boards?: number;
  seed?: number;
  pairing?: string;
  result: DuplicateResult | null;
  error?: string | null;
}

function DuplicateMatchPanel({ checkpoints }: { checkpoints: BotArenaProps['checkpoints'] }) {
  const [challenger, setChallenger] = useState<string>('');
  const [defender, setDefender] = useState<string>('');
  const [boards, setBoards] = useState<number>(1000);
  const [seed, setSeed] = useState<number>(0);
  const [pairing, setPairing] = useState<string>('rotation_8game');
  const [progress, setProgress] = useState<DuplicateProgress | null>(null);
  const [runs, setRuns] = useState<DuplicateRun[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState<boolean>(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const loadHistory = useCallback(() => {
    fetch('/api/arena/duplicate/history')
      .then(r => r.json())
      .then(d => setRuns(d.runs ?? []))
      .catch(() => {});
  }, []);

  const pollProgress = useCallback(() => {
    fetch('/api/arena/duplicate/progress')
      .then(r => r.json())
      .then((data: DuplicateProgress) => {
        setProgress(data);
        if (data.status !== 'running') {
          setRunning(false);
          if (pollRef.current) {
            clearInterval(pollRef.current);
            pollRef.current = null;
          }
          loadHistory();
        } else {
          setRunning(true);
        }
      })
      .catch(() => {});
  }, [loadHistory]);

  useEffect(() => {
    pollProgress();
    loadHistory();
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [pollProgress, loadHistory]);

  const startMatch = async () => {
    setError(null);
    if (!challenger || !defender) {
      setError('Pick both challenger and defender checkpoints.');
      return;
    }
    if (boards <= 0) {
      setError('Boards must be > 0.');
      return;
    }
    try {
      const resp = await fetch('/api/arena/duplicate/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          challenger,
          defender,
          boards,
          seed,
          pairing,
        }),
      });
      const data = await resp.json();
      if (data.status === 'started' || data.status === 'already_running') {
        setRunning(true);
        pollProgress();
        if (!pollRef.current) pollRef.current = setInterval(pollProgress, 2000);
        return;
      }
      setError(data?.message ?? `Failed to start (${data?.status ?? 'unknown'}).`);
    } catch (e) {
      setError(`Network error: ${e instanceof Error ? e.message : 'unknown'}`);
    }
  };

  const stopMatch = async () => {
    await fetch('/api/arena/duplicate/stop', { method: 'POST' });
    setRunning(false);
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    pollProgress();
  };

  const currentResult = progress?.result ?? null;

  return (
    <div className="arena-duplicate" style={{ marginTop: 32 }}>
      <h2>Duplicate Match</h2>
      <p className="arena-subtitle">
        Head-to-head duplicate match: both checkpoints play the same deals in rotated seats.
        Reports mean duplicate advantage with 95% bootstrap CI and IMPs/board.
      </p>

      <div className="arena-duplicate-config" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12, margin: '12px 0' }}>
        <label>
          <div>Challenger</div>
          <select value={challenger} onChange={e => setChallenger(e.target.value)}>
            <option value="">— pick a checkpoint —</option>
            {checkpoints.map(c => (
              <option key={c.filename} value={c.filename}>{c.model_name ? `${c.model_name} · ${c.filename}` : c.filename}</option>
            ))}
          </select>
        </label>
        <label>
          <div>Defender</div>
          <select value={defender} onChange={e => setDefender(e.target.value)}>
            <option value="">— pick a checkpoint —</option>
            {checkpoints.map(c => (
              <option key={c.filename} value={c.filename}>{c.model_name ? `${c.model_name} · ${c.filename}` : c.filename}</option>
            ))}
          </select>
        </label>
        <label>
          <div>Boards</div>
          <input type="number" min={1} max={50000} value={boards} onChange={e => setBoards(parseInt(e.target.value || '0', 10))} />
        </label>
        <label>
          <div>Seed</div>
          <input type="number" value={seed} onChange={e => setSeed(parseInt(e.target.value || '0', 10))} />
        </label>
        <label>
          <div>Pairing</div>
          <select value={pairing} onChange={e => setPairing(e.target.value)}>
            <option value="rotation_8game">rotation_8game</option>
            <option value="rotation_4game">rotation_4game</option>
            <option value="single_seat_2game">single_seat_2game</option>
          </select>
        </label>
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
        <button className="btn-primary" onClick={startMatch} disabled={running}>
          {running ? 'Running…' : 'Run Duplicate Match'}
        </button>
        {running && <button className="btn-secondary" onClick={stopMatch}>Stop</button>}
      </div>

      {error && <div className="arena-error-banner">{error}</div>}

      {progress && progress.status !== 'idle' && (
        <div className="arena-duplicate-status" style={{ margin: '8px 0' }}>
          <strong>Status:</strong> {progress.status}
          {progress.challenger && progress.defender && (
            <span> — {progress.challenger} vs {progress.defender} ({progress.boards} boards, seed={progress.seed}, {progress.pairing})</span>
          )}
          {progress.error && <div className="negative">Error: {progress.error}</div>}
        </div>
      )}

      {currentResult && (
        <div className="arena-duplicate-result" style={{ marginTop: 12, padding: 12, border: '1px solid #ccc', borderRadius: 6 }}>
          <h3>Latest Result</h3>
          <table className="arena-table">
            <tbody>
              <tr><td>Boards played</td><td>{currentResult.boards_played.toLocaleString()}</td></tr>
              <tr><td>Challenger mean score</td><td className={currentResult.challenger_mean_score >= 0 ? 'positive' : 'negative'}>{currentResult.challenger_mean_score.toFixed(3)}</td></tr>
              <tr><td>Defender mean score</td><td className={currentResult.defender_mean_score >= 0 ? 'positive' : 'negative'}>{currentResult.defender_mean_score.toFixed(3)}</td></tr>
              <tr><td>Mean duplicate advantage</td><td className={currentResult.mean_duplicate_advantage >= 0 ? 'positive' : 'negative'}>{currentResult.mean_duplicate_advantage.toFixed(3)} ± {currentResult.duplicate_advantage_std.toFixed(3)}</td></tr>
              <tr><td>95% CI</td><td>[{currentResult.ci_low_95.toFixed(3)}, {currentResult.ci_high_95.toFixed(3)}]</td></tr>
              <tr><td>IMPs / board</td><td>{currentResult.imps_per_board.toFixed(4)}</td></tr>
            </tbody>
          </table>
        </div>
      )}

      <h3 style={{ marginTop: 24 }}>History</h3>
      <div className="arena-table-wrapper">
        <table className="arena-table">
          <thead>
            <tr>
              <th>When</th>
              <th>Status</th>
              <th>Challenger</th>
              <th>Defender</th>
              <th>Boards</th>
              <th>Seed</th>
              <th>Pairing</th>
              <th>Adv (mean ± std)</th>
              <th>95% CI</th>
              <th>IMPs/board</th>
            </tr>
          </thead>
          <tbody>
            {[...runs].reverse().map(run => (
              <tr key={run.run_id}>
                <td>{new Date(run.created_at).toLocaleString()}</td>
                <td>{run.status}</td>
                <td>{run.challenger}</td>
                <td>{run.defender}</td>
                <td>{run.boards.toLocaleString()}</td>
                <td>{run.seed}</td>
                <td>{run.pairing}</td>
                <td className={run.result && run.result.mean_duplicate_advantage >= 0 ? 'positive' : run.result ? 'negative' : ''}>
                  {run.result ? `${run.result.mean_duplicate_advantage.toFixed(3)} ± ${run.result.duplicate_advantage_std.toFixed(3)}` : (run.error ?? '—')}
                </td>
                <td>{run.result ? `[${run.result.ci_low_95.toFixed(3)}, ${run.result.ci_high_95.toFixed(3)}]` : '—'}</td>
                <td>{run.result ? run.result.imps_per_board.toFixed(4) : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
