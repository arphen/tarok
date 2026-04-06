import React, { useState, useEffect, useCallback } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, BarChart, Bar, Cell,
  ScatterChart, Scatter, ZAxis, Area, AreaChart,
} from 'recharts';
import type { EvoProgress, EvoGenStats, EvoIndividual } from '../types/game';
import './EvoDashboard.css';

interface Props { onBack: () => void }

const API = '';

const HPARAM_LABELS: Record<string, string> = {
  lr: 'Learning Rate',
  gamma: 'Discount (γ)',
  gae_lambda: 'GAE Lambda',
  clip_epsilon: 'Clip ε',
  value_coef: 'Value Coef',
  entropy_coef: 'Entropy Coef',
  epochs_per_update: 'PPO Epochs',
  batch_size: 'Batch Size',
  hidden_size: 'Hidden Size',
  explore_rate: 'Explore Rate',
  fsp_ratio: 'FSP Ratio',
};

const HPARAM_COLORS = [
  '#4caf50', '#2196f3', '#ff9800', '#e91e63', '#9c27b0',
  '#00bcd4', '#d4a843', '#f44336', '#8bc34a', '#3f51b5', '#795548',
];

export default function EvoDashboard({ onBack }: Props) {
  const [progress, setProgress] = useState<EvoProgress | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [popSize, setPopSize] = useState(12);
  const [numGens, setNumGens] = useState(10);
  const [evalSessions, setEvalSessions] = useState(20);
  const [evalGames, setEvalGames] = useState(10);
  const [oracle, setOracle] = useState(false);
  const [tab, setTab] = useState<'fitness' | 'population' | 'hparams' | 'halloffame'>('fitness');

  const poll = useCallback(async () => {
    try {
      const [pRes, sRes] = await Promise.all([
        fetch(`${API}/api/evo/progress`),
        fetch(`${API}/api/evo/status`),
      ]);
      const pData = await pRes.json();
      const sData = await sRes.json();
      setProgress(pData);
      setIsRunning(sData.running);
    } catch { /* server not up */ }
  }, []);

  useEffect(() => {
    poll();
    const id = setInterval(poll, 1500);
    return () => clearInterval(id);
  }, [poll]);

  const startEvo = async () => {
    await fetch(`${API}/api/evo/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        population_size: popSize,
        num_generations: numGens,
        eval_sessions: evalSessions,
        games_per_session: evalGames,
        oracle,
      }),
    });
    setIsRunning(true);
  };

  const stopEvo = async () => {
    await fetch(`${API}/api/evo/stop`, { method: 'POST' });
    setIsRunning(false);
  };

  // Fitness over generations (line + area for std)
  const fitnessData = (progress?.gen_stats ?? []).map((gs: EvoGenStats) => ({
    gen: gs.gen,
    avg: +gs.avg.toFixed(4),
    min: +gs.min.toFixed(4),
    max: +gs.max.toFixed(4),
    std_low: +(gs.avg - gs.std).toFixed(4),
    std_high: +(gs.avg + gs.std).toFixed(4),
  }));

  // Population scatter: fitness vs individual index
  const popScatter = (progress?.population ?? []).map((ind: EvoIndividual) => ({
    index: ind.index,
    fitness: ind.fitness,
    win_rate: +(ind.win_rate * 100).toFixed(1),
    trend: +(ind.reward_trend * 100).toFixed(1),
  }));

  // Per-hparam evolution: track best individual's hparams per gen
  // We'll use hall_of_fame across gen_stats
  const hparamKeys = Object.keys(progress?.best_hparams ?? {});

  const progressPct = progress && progress.total_generations > 0
    ? ((progress.generation + (progress.evaluating_index / Math.max(progress.evaluating_total, 1)))
       / (progress.total_generations + 1)) * 100
    : 0;

  const formatTime = (secs: number) => {
    if (secs < 60) return `${Math.round(secs)}s`;
    if (secs < 3600) return `${Math.floor(secs / 60)}m ${Math.round(secs % 60)}s`;
    return `${Math.floor(secs / 3600)}h ${Math.floor((secs % 3600) / 60)}m`;
  };

  return (
    <div className="evo-dashboard">
      {/* Header */}
      <div className="evo-header">
        <button className="btn-secondary btn-sm" onClick={onBack}>← Back</button>
        <h2>🧬 Evolution Dashboard</h2>
      </div>

      {/* Controls */}
      <div className="evo-controls-bar">
        <label className="evo-field">
          <span>Population</span>
          <input type="number" value={popSize} onChange={e => setPopSize(Number(e.target.value))}
            disabled={isRunning} min={4} step={2} />
        </label>
        <label className="evo-field">
          <span>Generations</span>
          <input type="number" value={numGens} onChange={e => setNumGens(Number(e.target.value))}
            disabled={isRunning} min={2} step={1} />
        </label>
        <label className="evo-field">
          <span>Eval Sessions</span>
          <input type="number" value={evalSessions} onChange={e => setEvalSessions(Number(e.target.value))}
            disabled={isRunning} min={5} step={5} />
        </label>
        <label className="evo-field">
          <span>Eval Games</span>
          <input type="number" value={evalGames} onChange={e => setEvalGames(Number(e.target.value))}
            disabled={isRunning} min={5} step={5} />
        </label>
        <label className="evo-check">
          <input type="checkbox" checked={oracle} onChange={e => setOracle(e.target.checked)} disabled={isRunning} />
          <span>Oracle</span>
        </label>
        {isRunning ? (
          <button className="btn-danger" onClick={stopEvo}>Stop</button>
        ) : (
          <button className="btn-gold" onClick={startEvo}>Start Evolution</button>
        )}
      </div>

      {/* Progress bar */}
      {progress && progress.phase !== 'idle' && (
        <div className="evo-progress">
          <div className="evo-progress-bar">
            <div className="evo-progress-fill" style={{ width: `${Math.min(progressPct, 100)}%` }} />
          </div>
          <span className="evo-progress-text">
            Gen {progress.generation}/{progress.total_generations}
            {progress.phase === 'evaluating' && ` · Evaluating ${progress.evaluating_index}/${progress.evaluating_total}`}
            {progress.phase === 'selecting' && ' · Selecting'}
            {progress.phase === 'done' && ' · Complete'}
            {` · ${formatTime(progress.elapsed_seconds)}`}
          </span>
        </div>
      )}

      {/* Stat cards */}
      <div className="evo-stats">
        <EvoStat label="Generation" value={`${progress?.generation ?? 0} / ${progress?.total_generations ?? 0}`} />
        <EvoStat label="Best Fitness" value={(progress?.best_fitness ?? 0).toFixed(4)} highlight />
        <EvoStat label="Population" value={`${progress?.population?.length ?? 0}`} />
        <EvoStat label="Phase" value={progress?.phase ?? 'idle'} />
        <EvoStat label="Elapsed" value={formatTime(progress?.elapsed_seconds ?? 0)} />
        {progress?.gen_stats?.length ? (
          <EvoStat label="Pop Avg Fitness"
            value={progress.gen_stats[progress.gen_stats.length - 1].avg.toFixed(4)} />
        ) : (
          <EvoStat label="Pop Avg Fitness" value="—" />
        )}
      </div>

      {/* Best hparams summary */}
      {progress?.best_hparams && Object.keys(progress.best_hparams).length > 0 && (
        <div className="evo-best-banner">
          <strong>Best:</strong>
          {Object.entries(progress.best_hparams).map(([k, v]) => (
            <span key={k} className="evo-best-tag">
              {HPARAM_LABELS[k] || k}: <b>{typeof v === 'number' ? (v < 0.01 ? v.toExponential(2) : Number(v.toFixed(4))) : v}</b>
            </span>
          ))}
        </div>
      )}

      {/* Tabs */}
      <div className="evo-tabs">
        {(['fitness', 'population', 'hparams', 'halloffame'] as const).map(t => (
          <button key={t} className={`evo-tab ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>
            {t === 'fitness' ? '📈 Fitness' : t === 'population' ? '👥 Population' : t === 'hparams' ? '⚙️ Hyperparams' : '🏆 Hall of Fame'}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="evo-tab-content">
        {tab === 'fitness' && (
          <div className="evo-chart-grid">
            <ChartCard title="Fitness Over Generations" wide>
              <ResponsiveContainer width="100%" height={320}>
                <AreaChart data={fitnessData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                  <XAxis dataKey="gen" stroke="#666" fontSize={11} label={{ value: 'Generation', position: 'insideBottom', offset: -5, fill: '#888' }} />
                  <YAxis stroke="#666" fontSize={11} domain={['auto', 'auto']} />
                  <Tooltip {...tooltipStyle} />
                  <Legend />
                  <Area type="monotone" dataKey="std_low" stroke="none" fill="rgba(74,158,255,0.1)" name="σ band (low)" />
                  <Area type="monotone" dataKey="std_high" stroke="none" fill="rgba(74,158,255,0.15)" name="σ band (high)" />
                  <Line type="monotone" dataKey="max" stroke="#4caf50" strokeWidth={2} dot={{ r: 3 }} name="Best" />
                  <Line type="monotone" dataKey="avg" stroke="#d4a843" strokeWidth={2} dot={{ r: 3 }} name="Average" />
                  <Line type="monotone" dataKey="min" stroke="#e94560" strokeWidth={1.5} dot={{ r: 2 }} name="Worst" strokeDasharray="4 4" />
                </AreaChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Evaluations per Generation">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={progress?.gen_stats ?? []}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                  <XAxis dataKey="gen" stroke="#666" fontSize={11} />
                  <YAxis stroke="#666" fontSize={11} />
                  <Tooltip {...tooltipStyle} />
                  <Bar dataKey="nevals" fill="#4a9eff" name="Evaluations" />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Fitness Std Dev">
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={fitnessData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                  <XAxis dataKey="gen" stroke="#666" fontSize={11} />
                  <YAxis stroke="#666" fontSize={11} />
                  <Tooltip {...tooltipStyle} />
                  <Line type="monotone" dataKey="std_high" stroke="#9c27b0" strokeWidth={2} dot={{ r: 3 }} name="Diversity (σ)" />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>
          </div>
        )}

        {tab === 'population' && (
          <div className="evo-chart-grid">
            <ChartCard title="Current Population: Fitness Distribution" wide>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={popScatter.sort((a, b) => b.fitness - a.fitness)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                  <XAxis dataKey="index" stroke="#666" fontSize={11} label={{ value: 'Individual', position: 'insideBottom', offset: -5, fill: '#888' }} />
                  <YAxis stroke="#666" fontSize={11} domain={['auto', 'auto']} />
                  <Tooltip
                    {...tooltipStyle}
                    formatter={(value: number, name: string) => [
                      name === 'fitness' ? value.toFixed(4) : `${value}%`,
                      name === 'fitness' ? 'Fitness' : name === 'win_rate' ? 'Win Rate' : 'Trend',
                    ]}
                  />
                  <Legend />
                  <Bar dataKey="fitness" name="Fitness">
                    {popScatter.sort((a, b) => b.fitness - a.fitness).map((_, i) => (
                      <Cell key={i} fill={i === 0 ? '#d4a843' : i < 3 ? '#4caf50' : '#4a9eff'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Win Rate vs Reward Trend">
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                  <XAxis dataKey="win_rate" stroke="#666" fontSize={11} name="Win Rate %" unit="%" />
                  <YAxis dataKey="trend" stroke="#666" fontSize={11} name="Trend %" unit="%" />
                  <ZAxis dataKey="fitness" range={[40, 200]} name="Fitness" />
                  <Tooltip
                    {...tooltipStyle}
                    formatter={(value: number, name: string) => [`${value}`, name]}
                  />
                  <Scatter data={popScatter} fill="#d4a843" />
                </ScatterChart>
              </ResponsiveContainer>
            </ChartCard>

            {/* Population table */}
            <ChartCard title="Population Details" wide>
              <div className="evo-table-wrap">
                <table className="evo-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Fitness</th>
                      <th>Win Rate</th>
                      <th>Trend</th>
                      {hparamKeys.slice(0, 6).map(k => (
                        <th key={k}>{HPARAM_LABELS[k] || k}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {[...(progress?.population ?? [])]
                      .sort((a, b) => b.fitness - a.fitness)
                      .map((ind, i) => (
                        <tr key={ind.index} className={i === 0 ? 'evo-row-best' : ''}>
                          <td>{ind.index}</td>
                          <td><b>{ind.fitness.toFixed(4)}</b></td>
                          <td>{(ind.win_rate * 100).toFixed(1)}%</td>
                          <td>{(ind.reward_trend * 100).toFixed(1)}%</td>
                          {hparamKeys.slice(0, 6).map(k => (
                            <td key={k}>{formatHparam(k, ind.hparams[k])}</td>
                          ))}
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </ChartCard>
          </div>
        )}

        {tab === 'hparams' && (
          <div className="evo-chart-grid">
            {/* Show each hparam as a bar across the current population */}
            {hparamKeys.map((key, ki) => (
              <ChartCard key={key} title={HPARAM_LABELS[key] || key}>
                <ResponsiveContainer width="100%" height={200}>
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                    <XAxis
                      dataKey="value"
                      stroke="#666" fontSize={10}
                      name={HPARAM_LABELS[key] || key}
                      type="number"
                      domain={['auto', 'auto']}
                      tickFormatter={(v: number) => key === 'lr' || key === 'entropy_coef' ? v.toExponential(1) : v.toFixed(2)}
                    />
                    <YAxis dataKey="fitness" stroke="#666" fontSize={10} name="Fitness" domain={['auto', 'auto']} />
                    <Tooltip
                      {...tooltipStyle}
                      formatter={(value: number, name: string) => [
                        name === 'Fitness' ? value.toFixed(4) : formatHparam(key, value),
                        name,
                      ]}
                    />
                    <Scatter
                      data={(progress?.population ?? []).map(ind => ({
                        value: ind.hparams[key] ?? 0,
                        fitness: ind.fitness,
                      }))}
                      fill={HPARAM_COLORS[ki % HPARAM_COLORS.length]}
                      name={HPARAM_LABELS[key] || key}
                    />
                  </ScatterChart>
                </ResponsiveContainer>
              </ChartCard>
            ))}
          </div>
        )}

        {tab === 'halloffame' && (
          <div className="evo-chart-grid">
            <ChartCard title="Top 5 Individuals (Hall of Fame)" wide>
              <div className="evo-table-wrap">
                <table className="evo-table">
                  <thead>
                    <tr>
                      <th>Rank</th>
                      <th>Fitness</th>
                      <th>Win Rate</th>
                      <th>Trend</th>
                      {hparamKeys.map(k => (
                        <th key={k}>{HPARAM_LABELS[k] || k}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(progress?.hall_of_fame ?? []).map((ind, i) => (
                      <tr key={i} className={i === 0 ? 'evo-row-best' : ''}>
                        <td>🏅 {i + 1}</td>
                        <td><b>{ind.fitness.toFixed(4)}</b></td>
                        <td>{(ind.win_rate * 100).toFixed(1)}%</td>
                        <td>{(ind.reward_trend * 100).toFixed(1)}%</td>
                        {hparamKeys.map(k => (
                          <td key={k}>{formatHparam(k, ind.hparams[k])}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </ChartCard>

            {/* Best vs generation comparison */}
            {(progress?.gen_stats?.length ?? 0) > 0 && (
              <ChartCard title="Best Fitness Progression" wide>
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={fitnessData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                    <XAxis dataKey="gen" stroke="#666" fontSize={11} />
                    <YAxis stroke="#666" fontSize={11} domain={['auto', 'auto']} />
                    <Tooltip {...tooltipStyle} />
                    <Line type="monotone" dataKey="max" stroke="#d4a843" strokeWidth={3} dot={{ r: 4, fill: '#d4a843' }} name="Best Fitness" />
                  </LineChart>
                </ResponsiveContainer>
              </ChartCard>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function formatHparam(key: string, value: number | undefined): string {
  if (value === undefined) return '—';
  if (key === 'lr' || key === 'entropy_coef') return value.toExponential(2);
  if (key === 'hidden_size' || key === 'batch_size' || key === 'epochs_per_update') return String(Math.round(value));
  return value.toFixed(3);
}

function EvoStat({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className={`evo-stat ${highlight ? 'evo-stat-hl' : ''}`}>
      <div className="evo-stat-val">{value}</div>
      <div className="evo-stat-lbl">{label}</div>
    </div>
  );
}

function ChartCard({ title, children, wide }: { title: string; children: React.ReactNode; wide?: boolean }) {
  return (
    <div className={`evo-chart ${wide ? 'evo-chart-wide' : ''}`}>
      <h3>{title}</h3>
      {children}
    </div>
  );
}

const tooltipStyle = {
  contentStyle: { background: '#1a1a2e', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', fontSize: 13 },
  labelStyle: { color: '#aaa' },
};
