import React, { useState, useEffect, useCallback, useRef } from 'react';
import './BotArena.css';

interface BotArenaProps {
  onBack: () => void;
  checkpoints: { filename: string; episode: number; win_rate: number; model_name?: string; is_hof?: boolean }[];
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
  best_game: { score: number | null; game_idx: number | null };
  worst_game: { score: number | null; game_idx: number | null };
  avg_win_score: number;
  avg_loss_score: number;
  score_history: number[];
}

interface ContractAnalytics {
  played: number;
  decl_win_rate: number;
  avg_decl_score: number;
  avg_def_score: number;
}

interface ArenaAnalytics {
  games_done: number;
  total_games: number;
  players: PlayerAnalytics[];
  contracts: Record<string, ContractAnalytics>;
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

const DEFAULT_AGENTS: AgentSetup[] = [
  { name: 'Bot-A', type: 'stockskis', checkpoint: '' },
  { name: 'Bot-B', type: 'stockskis', checkpoint: '' },
  { name: 'Bot-C', type: 'stockskis', checkpoint: '' },
  { name: 'Bot-D', type: 'stockskis', checkpoint: '' },
];

const PLAYER_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12'];

type Tab = 'overview' | 'bidding' | 'contracts' | 'announcements' | 'best_worst' | 'scores' | 'history';

export default function BotArena({ onBack, checkpoints }: BotArenaProps) {
  const [agents, setAgents] = useState<AgentSetup[]>(DEFAULT_AGENTS.map(a => ({ ...a })));
  const [totalGames, setTotalGames] = useState(100000);
  const [sessionSize, setSessionSize] = useState(50);
  const [progress, setProgress] = useState<ArenaProgress | null>(null);
  const [running, setRunning] = useState(false);
  const [tab, setTab] = useState<Tab>('overview');
  const [stockskisTypes, setStockskisTypes] = useState<string[]>([]);
  const [historyRuns, setHistoryRuns] = useState<ArenaHistoryRun[]>([]);
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

  const pollProgress = useCallback(() => {
    fetch('/api/arena/progress')
      .then(r => r.json())
      .then((data: ArenaProgress) => {
        setProgress(data);
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
      }),
    });
    const data = await resp.json();
    if (data.status === 'started') {
      setRunning(true);
      setProgress({ status: 'running', games_done: 0, total_games: totalGames, analytics: null });
      pollRef.current = setInterval(pollProgress, 2000);
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
                  <option value="stockskis">StockŠkis (latest)</option>
                  {stockskisTypes.map(t => (
                    <option key={t} value={t}>{t.replace('_', ' ')}</option>
                  ))}
                  <option value="rl">Neural Network (checkpoint)</option>
                  <option value="lookahead">Lookahead</option>
                </select>
                {agent.type === 'rl' && (
                  <select
                    className="arena-select"
                    value={agent.checkpoint}
                    onChange={e => updateAgent(idx, { checkpoint: e.target.value })}
                  >
                    <option value="">Latest checkpoint</option>
                    {checkpoints.map(cp => (
                      <option key={cp.filename} value={cp.filename}>
                        {cp.model_name || cp.filename} (WR: {(cp.win_rate * 100).toFixed(0)}%)
                      </option>
                    ))}
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
                 t === 'best_worst' ? 'Best / Worst' :
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
            {tab === 'bidding' && displayAnalytics && <BiddingTab players={displayAnalytics.players} />}
            {tab === 'contracts' && displayAnalytics && <ContractsTab contracts={displayAnalytics.contracts} />}
            {tab === 'announcements' && displayAnalytics && <AnnouncementsTab players={displayAnalytics.players} />}
            {tab === 'best_worst' && displayAnalytics && <BestWorstTab players={displayAnalytics.players} />}
            {tab === 'scores' && displayAnalytics && <ScoresTab players={displayAnalytics.players} sessionSize={sessionSize} />}
            {tab === 'history' && <HistoryTab runs={historyRuns} />}
          </div>
        </div>
      )}
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
              <th>Avg Score</th>
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

    </div>
  );
}

/* ============ Bidding Tab ============ */
function BiddingTab({ players }: { players: PlayerAnalytics[] }) {
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
    </div>
  );
}

/* ============ Contracts Tab ============ */
function ContractsTab({ contracts }: { contracts: Record<string, ContractAnalytics> }) {
  const entries = Object.entries(contracts).sort((a, b) => b[1].played - a[1].played);
  const maxPlayed = Math.max(...entries.map(([, c]) => c.played), 1);

  return (
    <div className="arena-contracts">
      <h3>Contract Statistics</h3>
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
function BestWorstTab({ players }: { players: PlayerAnalytics[] }) {
  return (
    <div className="arena-best-worst">
      <h3>Best & Worst Single-Game Scores</h3>
      <div className="arena-bw-grid">
        {players.map((p, i) => (
          <div key={i} className="arena-bw-card" style={{ borderColor: PLAYER_COLORS[i] }}>
            <h4 style={{ color: PLAYER_COLORS[i] }}>{p.name}</h4>
            <div className="arena-bw-row">
              <div className="arena-bw-stat best">
                <span className="arena-bw-label">Best</span>
                <span className="arena-bw-score positive">{p.best_game.score ?? '—'}</span>
                {p.best_game.game_idx != null && <span className="arena-bw-game">Game #{p.best_game.game_idx}</span>}
              </div>
              <div className="arena-bw-stat worst">
                <span className="arena-bw-label">Worst</span>
                <span className="arena-bw-score negative">{p.worst_game.score ?? '—'}</span>
                {p.worst_game.game_idx != null && <span className="arena-bw-game">Game #{p.worst_game.game_idx}</span>}
              </div>
            </div>
            <div className="arena-bw-summary">
              <div><strong>Total Score:</strong> {p.total_score.toLocaleString()}</div>
              <div><strong>Avg Score:</strong> {p.avg_score.toFixed(1)}</div>
              <div><strong>Avg Win Score:</strong> <span className="positive">{p.avg_win_score.toFixed(1)}</span></div>
              <div><strong>Avg Loss Score:</strong> <span className="negative">{p.avg_loss_score.toFixed(1)}</span></div>
              <div><strong>Score Spread:</strong> {((p.best_game.score ?? 0) - (p.worst_game.score ?? 0)).toLocaleString()}</div>
            </div>
          </div>
        ))}
      </div>
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
