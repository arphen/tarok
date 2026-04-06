import React, { useState, useEffect, useCallback } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, BarChart, Bar, Cell,
  AreaChart, Area, RadarChart, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, Radar,
} from 'recharts';
import type { BreedProgress, BreedIndividual, BreedGenStats, BreedCycleSummary } from '../types/game';
import './BreedingDashboard.css';

interface Props { onBack: () => void }

const API = '';

const TRAIT_LABELS: Record<string, string> = {
  bid_aggression: 'Bid Aggression',
  solo_propensity: 'Solo Propensity',
  trump_eagerness: 'Trump Eagerness',
  defensive_caution: 'Defensive Caution',
  announce_boldness: 'Announce Boldness',
  kontra_aggression: 'Kontra Aggression',
  temperature: 'Temperature',
  explore_decay: 'Explore Decay',
  explore_floor: 'Explore Floor',
};

const TRAIT_COLORS = [
  '#4caf50', '#2196f3', '#ff9800', '#e91e63', '#9c27b0',
  '#00bcd4', '#d4a843', '#f44336', '#8bc34a',
];

const tooltipStyle = {
  contentStyle: { background: 'rgba(30,30,30,0.95)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, fontSize: 12 },
  labelStyle: { color: '#aaa' },
};

function StatCard({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className={`breed-stat ${highlight ? 'breed-stat-highlight' : ''}`}>
      <span className="breed-stat-value">{value}</span>
      <span className="breed-stat-label">{label}</span>
    </div>
  );
}

function ChartCard({ title, children, wide }: { title: string; children: React.ReactNode; wide?: boolean }) {
  return (
    <div className={`breed-chart-card ${wide ? 'breed-chart-wide' : ''}`}>
      <h4 className="breed-chart-title">{title}</h4>
      {children}
    </div>
  );
}

export default function BreedingDashboard({ onBack }: Props) {
  const [progress, setProgress] = useState<BreedProgress | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [warmup, setWarmup] = useState(50);
  const [popSize, setPopSize] = useState(12);
  const [numGens, setNumGens] = useState(5);
  const [numCycles, setNumCycles] = useState(3);
  const [evalGames, setEvalGames] = useState(100);
  const [refine, setRefine] = useState(30);
  const [oracle, setOracle] = useState(false);
  const [stockskisEval, setStockskisEval] = useState(false);
  const [stockskisStrength, setStockskisStrength] = useState(1.0);
  const [resume, setResume] = useState(false);
  const [resumeFrom, setResumeFrom] = useState('');
  const [modelName, setModelName] = useState('');
  const [checkpoints, setCheckpoints] = useState<{filename: string; model_name?: string; is_bred: boolean}[]>([]);
  const [tab, setTab] = useState<'fitness' | 'population' | 'traits' | 'cycles'>('fitness');

  const poll = useCallback(async () => {
    try {
      const [pRes, sRes] = await Promise.all([
        fetch(`${API}/api/breed/progress`),
        fetch(`${API}/api/breed/status`),
      ]);
      const pData = await pRes.json();
      const sData = await sRes.json();
      setProgress(pData);
      setIsRunning(sData.running);
    } catch { /* server not up */ }
  }, []);

  useEffect(() => {
    poll();
    fetch(`${API}/api/checkpoints`)
      .then(res => res.json())
      .then(d => {
        setCheckpoints(d.checkpoints || []);
        if (d.checkpoints && d.checkpoints.length) setResumeFrom(d.checkpoints[0].filename);
      })
      .catch(() => {});
    
    const id = setInterval(poll, 1500);
    return () => clearInterval(id);
  }, [poll]);

  const startBreed = async () => {
    await fetch(`${API}/api/breed/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        warmup_sessions: warmup,
        population_size: popSize,
        num_generations: numGens,
        num_cycles: numCycles,
        eval_games: evalGames,
        refine_sessions: refine,
        oracle,
        resume,
        resume_from: resumeFrom,
        model_name: modelName || undefined,
        stockskis_eval: stockskisEval,
        stockskis_strength: stockskisStrength,
      }),
    });
    setIsRunning(true);
  };

  const stopBreed = async () => {
    await fetch(`${API}/api/breed/stop`, { method: 'POST' });
    setIsRunning(false);
  };

  // Fitness over generations
  const fitnessData = (progress?.gen_stats ?? []).map((gs: BreedGenStats) => ({
    label: `C${gs.cycle}G${gs.gen}`,
    avg: +gs.avg.toFixed(4),
    min: +gs.min.toFixed(4),
    max: +gs.max.toFixed(4),
    std_low: +(gs.avg - gs.std).toFixed(4),
    std_high: +(gs.avg + gs.std).toFixed(4),
  }));

  // Radar chart data for best profile
  const bestProfile = progress?.best_profile ?? {};
  const traitKeys = Object.keys(bestProfile).filter(k => k !== 'temperature' && k !== 'explore_decay' && k !== 'explore_floor');
  const radarData = traitKeys.map(k => ({
    trait: TRAIT_LABELS[k] ?? k,
    value: +((bestProfile[k] + 1) / 2 * 100).toFixed(0), // normalize -1..1 to 0..100
  }));

  // Cycle summary chart
  const cycleData = (progress?.cycle_summaries ?? []).map((cs: BreedCycleSummary) => ({
    cycle: `Cycle ${cs.cycle}`,
    best_fitness: +cs.best_fitness.toFixed(4),
    refine_wr: +(cs.refine_win_rate * 100).toFixed(1),
    refine_reward: +cs.refine_avg_reward.toFixed(3),
  }));

  // Population bar chart
  const popData = [...(progress?.population ?? [])]
    .sort((a, b) => b.fitness - a.fitness)
    .map((ind, i) => ({
      index: ind.index,
      fitness: +ind.fitness.toFixed(4),
      bid_agg: +(ind.profile.bid_aggression ?? 0).toFixed(2),
      solo: +(ind.profile.solo_propensity ?? 0).toFixed(2),
      trump: +(ind.profile.trump_eagerness ?? 0).toFixed(2),
      def: +(ind.profile.defensive_caution ?? 0).toFixed(2),
      rank: i + 1,
    }));

  // Compute progress percentage
  let progressPct = 0;
  if (progress) {
    const p = progress;
    if (p.phase === 'warmup') {
      progressPct = (p.warmup_session / Math.max(p.warmup_total_sessions, 1)) * 15;
    } else if (p.phase === 'evaluating' || p.phase === 'breeding') {
      const cycleBase = 15 + ((p.cycle - 1) / Math.max(p.total_cycles, 1)) * 85;
      const cycleRange = 85 / Math.max(p.total_cycles, 1);
      const genPct = (p.generation + (p.evaluating_index / Math.max(p.evaluating_total, 1))) / Math.max(p.total_generations + 1, 1);
      progressPct = cycleBase + genPct * cycleRange * 0.6;
    } else if (p.phase === 'refining') {
      const cycleBase = 15 + ((p.cycle - 1) / Math.max(p.total_cycles, 1)) * 85;
      const cycleRange = 85 / Math.max(p.total_cycles, 1);
      progressPct = cycleBase + cycleRange * 0.6 + (p.refine_session / Math.max(p.refine_total_sessions, 1)) * cycleRange * 0.4;
    } else if (p.phase === 'done') {
      progressPct = 100;
    }
  }

  const formatTime = (secs: number) => {
    if (secs < 60) return `${Math.round(secs)}s`;
    if (secs < 3600) return `${Math.floor(secs / 60)}m ${Math.round(secs % 60)}s`;
    return `${Math.floor(secs / 3600)}h ${Math.floor((secs % 3600) / 60)}m`;
  };

  const phaseLabel = (phase: string) => {
    switch (phase) {
      case 'warmup': return 'Warming Up (Self-Play)';
      case 'breeding': return 'Breeding Population';
      case 'evaluating': return 'Evaluating Variants';
      case 'refining': return 'Refining Top 2';
      case 'done': return 'Complete';
      default: return phase;
    }
  };

  return (
    <div className="breed-dashboard">
      <div className="breed-header">
        <button className="btn-secondary btn-sm" onClick={onBack}>&larr; Back</button>
        <h2>🧬 Behavioral Breeding{progress?.model_name ? ` — ${progress.model_name}` : ''}</h2>
      </div>

      {/* Controls */}
      <div className="breed-controls">
        <label className="breed-field">
          <span>Warmup</span>
          <input type="number" value={warmup} onChange={e => setWarmup(Number(e.target.value))} disabled={isRunning} min={10} step={10} />
        </label>
        <label className="breed-field">
          <span>Population</span>
          <input type="number" value={popSize} onChange={e => setPopSize(Number(e.target.value))} disabled={isRunning} min={4} step={2} />
        </label>
        <label className="breed-field">
          <span>Gens/Cycle</span>
          <input type="number" value={numGens} onChange={e => setNumGens(Number(e.target.value))} disabled={isRunning} min={1} step={1} />
        </label>
        <label className="breed-field">
          <span>Cycles</span>
          <input type="number" value={numCycles} onChange={e => setNumCycles(Number(e.target.value))} disabled={isRunning} min={1} step={1} />
        </label>
        <label className="breed-field">
          <span>Eval Games</span>
          <input type="number" value={evalGames} onChange={e => setEvalGames(Number(e.target.value))} disabled={isRunning} min={10} step={10} />
        </label>
        <label className="breed-field">
          <span>Refine Sess</span>
          <input type="number" value={refine} onChange={e => setRefine(Number(e.target.value))} disabled={isRunning} min={5} step={5} />
        </label>
        <label className="breed-check">
          <input type="checkbox" checked={oracle} onChange={e => setOracle(e.target.checked)} disabled={isRunning} />
          <span>Oracle</span>
        </label>
        <label className="breed-check">
          <input type="checkbox" checked={stockskisEval} onChange={e => setStockskisEval(e.target.checked)} disabled={isRunning} />
          <span>Vs StockŠkis</span>
        </label>
        {stockskisEval && (
          <label className="breed-field" style={{ minWidth: 80 }}>
            <span>StockŠkis STR</span>
            <input type="range" value={stockskisStrength} onChange={e => setStockskisStrength(Number(e.target.value))}
              disabled={isRunning} min={0.1} max={1.0} step={0.1} />
          </label>
        )}
        
        <label className="breed-field">
            <span>Model Name</span>
            <input type="text" placeholder="Auto-generated" value={modelName} style={{width: 120}} onChange={e => setModelName(e.target.value)} disabled={isRunning} />
        </label>
        
        <label className="breed-field" style={{flexDirection: 'row', alignItems: 'center', gap: 6}}>
            <input type="checkbox" checked={resume} onChange={e => setResume(e.target.checked)} disabled={isRunning} />
            <span>Resume Base Model From:</span>
        </label>
        
        <select value={resumeFrom} onChange={e => setResumeFrom(e.target.value)} disabled={!resume || isRunning} style={{
                background: 'rgba(255,255,255,0.06)', color: '#fff', border: '1px solid rgba(255,255,255,0.15)', borderRadius: 6, padding: '4px 8px', fontSize: 13
        }}>
            {checkpoints.map(c => (
                <option key={c.filename} value={c.filename}>{c.model_name ? `[${c.model_name}] ` : ''}{c.filename}</option>
            ))}
        </select>

        <div style={{ flex: 1 }}></div>

        {isRunning ? (
          <button className="btn-danger" onClick={stopBreed}>Stop</button>
        ) : (
          <button className="btn-gold" onClick={startBreed}>Start Breeding</button>
        )}
      </div>

      {/* Progress bar */}
      {progress && progress.phase !== 'idle' && (
        <div className="breed-progress">
          <div className="breed-progress-bar">
            <div className="breed-progress-fill" style={{ width: `${Math.min(progressPct, 100)}%` }} />
          </div>
          <span className="breed-progress-text">
            {phaseLabel(progress.phase)}
            {progress.phase === 'warmup' && ` · Session ${progress.warmup_session}/${progress.warmup_total_sessions}`}
            {progress.phase === 'evaluating' && ` · C${progress.cycle} G${progress.generation} · ${progress.evaluating_index}/${progress.evaluating_total}`}
            {progress.phase === 'refining' && ` · C${progress.cycle} · Session ${progress.refine_session}/${progress.refine_total_sessions}`}
            {` · ${formatTime(progress.elapsed_seconds)}`}
          </span>
        </div>
      )}

      {/* Stat cards */}
      <div className="breed-stats">
        <StatCard label="Phase" value={progress?.phase ?? 'idle'} />
        <StatCard label="Cycle" value={`${progress?.cycle ?? 0} / ${progress?.total_cycles ?? 0}`} />
        <StatCard label="Generation" value={`${progress?.generation ?? 0} / ${progress?.total_generations ?? 0}`} />
        <StatCard label="Best Fitness" value={(progress?.best_fitness ?? 0).toFixed(4)} highlight />
        <StatCard label="Population" value={`${progress?.population?.length ?? 0}`} />
        <StatCard label="Elapsed" value={formatTime(progress?.elapsed_seconds ?? 0)} />
      </div>

      {/* Best profile banner */}
      {Object.keys(bestProfile).length > 0 && (
        <div className="breed-best-banner">
          <strong>Best Profile:</strong>
          {Object.entries(bestProfile).map(([k, v]) => (
            <span key={k} className="breed-best-tag">
              {TRAIT_LABELS[k] ?? k}: <b>{typeof v === 'number' ? v.toFixed(3) : v}</b>
            </span>
          ))}
        </div>
      )}

      {/* Tabs */}
      <div className="breed-tabs">
        {(['fitness', 'population', 'traits', 'cycles'] as const).map(t => (
          <button key={t} className={`breed-tab ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>
            {t === 'fitness' ? '📈 Fitness' : t === 'population' ? '👥 Population' : t === 'traits' ? '🧠 Traits' : '🔄 Cycles'}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="breed-tab-content">
        {tab === 'fitness' && (
          <div className="breed-chart-grid">
            <ChartCard title="Fitness Across Generations" wide>
              <ResponsiveContainer width="100%" height={320}>
                <AreaChart data={fitnessData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                  <XAxis dataKey="label" stroke="#666" fontSize={10} />
                  <YAxis stroke="#666" fontSize={11} domain={['auto', 'auto']} />
                  <Tooltip {...tooltipStyle} />
                  <Legend />
                  <Area type="monotone" dataKey="std_low" stroke="none" fill="rgba(74,158,255,0.1)" name="sigma band (low)" />
                  <Area type="monotone" dataKey="std_high" stroke="none" fill="rgba(74,158,255,0.15)" name="sigma band (high)" />
                  <Line type="monotone" dataKey="max" stroke="#4caf50" strokeWidth={2} dot={{ r: 3 }} name="Best" />
                  <Line type="monotone" dataKey="avg" stroke="#d4a843" strokeWidth={2} dot={{ r: 3 }} name="Average" />
                  <Line type="monotone" dataKey="min" stroke="#e94560" strokeWidth={1.5} dot={{ r: 2 }} name="Worst" strokeDasharray="4 4" />
                </AreaChart>
              </ResponsiveContainer>
            </ChartCard>
          </div>
        )}

        {tab === 'population' && (
          <div className="breed-chart-grid">
            <ChartCard title="Population Fitness" wide>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={popData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                  <XAxis dataKey="index" stroke="#666" fontSize={11} label={{ value: 'Individual', position: 'insideBottom', offset: -5, fill: '#888' }} />
                  <YAxis stroke="#666" fontSize={11} domain={['auto', 'auto']} />
                  <Tooltip {...tooltipStyle} />
                  <Bar dataKey="fitness" name="Fitness">
                    {popData.map((_, i) => (
                      <Cell key={i} fill={i === 0 ? '#d4a843' : i < 3 ? '#4caf50' : '#4a9eff'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>

            {/* Population table */}
            <ChartCard title="Population Details" wide>
              <div className="breed-table-wrap">
                <table className="breed-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Fitness</th>
                      <th>Bid Agg</th>
                      <th>Solo</th>
                      <th>Trump</th>
                      <th>Defense</th>
                      <th>Announce</th>
                      <th>Kontra</th>
                      <th>Temp</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...(progress?.population ?? [])]
                      .sort((a, b) => b.fitness - a.fitness)
                      .map((ind, i) => (
                        <tr key={ind.index} className={i === 0 ? 'breed-row-best' : ''}>
                          <td>{ind.index}</td>
                          <td><b>{ind.fitness.toFixed(4)}</b></td>
                          <td className={traitColor(ind.profile.bid_aggression)}>{fmtTrait(ind.profile.bid_aggression)}</td>
                          <td className={traitColor(ind.profile.solo_propensity)}>{fmtTrait(ind.profile.solo_propensity)}</td>
                          <td className={traitColor(ind.profile.trump_eagerness)}>{fmtTrait(ind.profile.trump_eagerness)}</td>
                          <td className={traitColor(ind.profile.defensive_caution)}>{fmtTrait(ind.profile.defensive_caution)}</td>
                          <td className={traitColor(ind.profile.announce_boldness)}>{fmtTrait(ind.profile.announce_boldness)}</td>
                          <td className={traitColor(ind.profile.kontra_aggression)}>{fmtTrait(ind.profile.kontra_aggression)}</td>
                          <td>{(ind.profile.temperature ?? 1).toFixed(2)}</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </ChartCard>
          </div>
        )}

        {tab === 'traits' && (
          <div className="breed-chart-grid">
            {/* Radar chart of best profile */}
            <ChartCard title="Best Profile — Trait Radar">
              <ResponsiveContainer width="100%" height={320}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="rgba(255,255,255,0.1)" />
                  <PolarAngleAxis dataKey="trait" stroke="#888" fontSize={11} />
                  <PolarRadiusAxis domain={[0, 100]} tick={false} />
                  <Radar name="Best" dataKey="value" stroke="#d4a843" fill="#d4a843" fillOpacity={0.3} strokeWidth={2} />
                </RadarChart>
              </ResponsiveContainer>
            </ChartCard>

            {/* Hall of fame */}
            <ChartCard title="Hall of Fame" wide>
              <div className="breed-table-wrap">
                <table className="breed-table">
                  <thead>
                    <tr>
                      <th>Rank</th>
                      <th>Fitness</th>
                      <th>Win Rate</th>
                      <th>Avg Reward</th>
                      <th>Bid Rate</th>
                      <th>Solo Rate</th>
                      <th>Bid Agg</th>
                      <th>Solo Prop</th>
                      <th>Trump</th>
                      <th>Temp</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(progress?.hall_of_fame ?? []).map((ind: BreedIndividual, i: number) => (
                      <tr key={i} className={i === 0 ? 'breed-row-best' : ''}>
                        <td>{i + 1}</td>
                        <td><b>{ind.fitness.toFixed(4)}</b></td>
                        <td>{(ind.win_rate * 100).toFixed(1)}%</td>
                        <td>{ind.avg_reward.toFixed(3)}</td>
                        <td>{(ind.bid_rate * 100).toFixed(0)}%</td>
                        <td>{(ind.solo_rate * 100).toFixed(0)}%</td>
                        <td className={traitColor(ind.profile.bid_aggression)}>{fmtTrait(ind.profile.bid_aggression)}</td>
                        <td className={traitColor(ind.profile.solo_propensity)}>{fmtTrait(ind.profile.solo_propensity)}</td>
                        <td className={traitColor(ind.profile.trump_eagerness)}>{fmtTrait(ind.profile.trump_eagerness)}</td>
                        <td>{(ind.profile.temperature ?? 1).toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </ChartCard>
          </div>
        )}

        {tab === 'cycles' && (
          <div className="breed-chart-grid">
            <ChartCard title="Cycle Results" wide>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={cycleData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                  <XAxis dataKey="cycle" stroke="#666" fontSize={11} />
                  <YAxis stroke="#666" fontSize={11} />
                  <Tooltip {...tooltipStyle} />
                  <Legend />
                  <Bar dataKey="best_fitness" fill="#d4a843" name="Best Fitness" />
                  <Bar dataKey="refine_wr" fill="#4caf50" name="Refine Win%" />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Refinement Reward Trend">
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={cycleData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                  <XAxis dataKey="cycle" stroke="#666" fontSize={11} />
                  <YAxis stroke="#666" fontSize={11} />
                  <Tooltip {...tooltipStyle} />
                  <Line type="monotone" dataKey="refine_reward" stroke="#2196f3" strokeWidth={2} dot={{ r: 4 }} name="Refine Avg Reward" />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>

            {/* Cycle summaries table */}
            <ChartCard title="Cycle Summary Table" wide>
              <div className="breed-table-wrap">
                <table className="breed-table">
                  <thead>
                    <tr>
                      <th>Cycle</th>
                      <th>Best Fitness</th>
                      <th>Refine Win%</th>
                      <th>Refine Reward</th>
                      <th>Top Trait: Bid Agg</th>
                      <th>Top Trait: Solo</th>
                      <th>Top Trait: Temp</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(progress?.cycle_summaries ?? []).map((cs: BreedCycleSummary) => (
                      <tr key={cs.cycle}>
                        <td>{cs.cycle}</td>
                        <td><b>{cs.best_fitness.toFixed(4)}</b></td>
                        <td>{(cs.refine_win_rate * 100).toFixed(1)}%</td>
                        <td>{cs.refine_avg_reward.toFixed(3)}</td>
                        <td className={traitColor(cs.best_profile.bid_aggression)}>{fmtTrait(cs.best_profile.bid_aggression)}</td>
                        <td className={traitColor(cs.best_profile.solo_propensity)}>{fmtTrait(cs.best_profile.solo_propensity)}</td>
                        <td>{(cs.best_profile.temperature ?? 1).toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </ChartCard>
          </div>
        )}
      </div>
    </div>
  );
}

function fmtTrait(v: number | undefined): string {
  if (v === undefined) return '—';
  return (v >= 0 ? '+' : '') + v.toFixed(2);
}

function traitColor(v: number | undefined): string {
  if (v === undefined) return '';
  if (v > 0.3) return 'trait-positive';
  if (v < -0.3) return 'trait-negative';
  return 'trait-neutral';
}
