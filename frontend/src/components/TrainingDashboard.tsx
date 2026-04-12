import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, BarChart, Bar, Cell,
} from 'recharts';
import type { TrainingMetrics, ContractStat } from '../types/game';
import './TrainingDashboard.css';

interface Props { onBack: () => void }

const CONTRACT_LABELS: Record<string, string> = {
  klop: 'Klop', three: 'Tri', two: 'Dve', one: 'Ena',
  solo_three: 'Solo 3', solo_two: 'Solo 2', solo_one: 'Solo 1', solo: 'Solo', berac: 'Berač',
};
const CONTRACT_COLORS: Record<string, string> = {
  klop: '#888', three: '#4caf50', two: '#2196f3', one: '#ff9800',
  solo_three: '#9c27b0', solo_two: '#e91e63', solo_one: '#f44336', solo: '#d4a843', berac: '#00bcd4',
};

const API = '';

export default function TrainingDashboard({ onBack }: Props) {
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [sessions, setSessions] = useState(100);
  const [gamesPerSession, setGamesPerSession] = useState(100);
  const [resume, setResume] = useState(false);
  const [resumeFrom, setResumeFrom] = useState('tarok_agent_latest.pt');
  const [stockskisRatio, setStockskisRatio] = useState(0.0);
  const [stockskisStrength, setStockskisStrength] = useState(1.0);
  const [lookaheadRatio, setLookaheadRatio] = useState(0.0);
  const [lookaheadSims, setLookaheadSims] = useState(20);
  const [lookaheadPerfectInfo, setLookaheadPerfectInfo] = useState(true);
  const [availableSnapshots, setAvailableSnapshots] = useState<{filename: string, session: number, episode: number}[]>([]);
  const [tab, setTab] = useState<'overview' | 'contracts' | 'loss' | 'snapshots' | 'tarok_bids'>('overview');
  const [smoothing, setSmoothing] = useState(0.8);
  const [useRustEngine, setUseRustEngine] = useState(true);
  const [warmupGames, setWarmupGames] = useState(0);
  const [labMode, setLabMode] = useState(false);
  const [concurrency, setConcurrency] = useState(128);
  const [trainingMode, setTrainingMode] = useState<'idle' | 'legacy' | 'lab'>('idle');

  const poll = useCallback(async () => {
    try {
      const [mRes, sRes, cRes] = await Promise.all([
        fetch(`${API}/api/training/metrics`),
        fetch(`${API}/api/training/status`),
        fetch(`${API}/api/checkpoints`),
      ]);
      const mData = await mRes.json();
      const sData = await sRes.json();
      const cData = await cRes.json();
      setMetrics(mData);
      setIsTraining(sData.running);
      setTrainingMode(sData.mode ?? 'idle');
      setAvailableSnapshots(cData.checkpoints || []);
    } catch { /* server not up */ }
  }, []);

  useEffect(() => {
    poll();
    const id = setInterval(poll, 2000);
    return () => clearInterval(id);
  }, [poll]);

  const startTraining = async () => {
    const isLatest = resumeFrom === '' || resumeFrom === 'tarok_agent_latest.pt';

    if (labMode) {
      await fetch(`${API}/api/training/lab/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          num_sessions: sessions,
          games_per_session: gamesPerSession,
          resume_from: resume ? resumeFrom : undefined,
          concurrency,
          device: 'auto',
        }),
      });
    } else {
      await fetch(`${API}/api/training/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          num_sessions: sessions, 
          games_per_session: gamesPerSession, 
          resume,
          resume_from: resume && !isLatest ? resumeFrom : undefined,
          stockskis_ratio: stockskisRatio,
          stockskis_strength: stockskisStrength,
          lookahead_ratio: lookaheadRatio,
          lookahead_sims: lookaheadSims,
          lookahead_perfect_info: lookaheadPerfectInfo,
          use_rust_engine: useRustEngine,
          warmup_games: warmupGames,
        }),
      });
    }
    setIsTraining(true);
  };

  const stopTraining = async () => {
    await fetch(`${API}/api/training/stop`, { method: 'POST' });
    setIsTraining(false);
  };

  // Chart data — memoised to avoid re-deriving every render
  // Downsample large datasets to MAX_CHART_POINTS to keep SVG rendering fast
  const MAX_CHART_POINTS = 500;

  const downsample = useCallback(<T,>(data: T[]): T[] => {
    if (data.length <= MAX_CHART_POINTS) return data;
    const step = data.length / MAX_CHART_POINTS;
    const result: T[] = [];
    for (let i = 0; i < MAX_CHART_POINTS - 1; i++) {
      result.push(data[Math.floor(i * step)]);
    }
    result.push(data[data.length - 1]); // always include latest point
    return result;
  }, []);

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

  const smoothAndSample = useCallback((data: any[], keys: string[]) => {
    return downsample(applySmoothing(data, keys));
  }, [applySmoothing, downsample]);

  const historyOffset = metrics?.history_offset ?? 0;

  const rewardData = useMemo(() => smoothAndSample(metrics?.reward_history.map((v, i) => ({ s: historyOffset + i + 1, reward: v })) ?? [], ['reward']), [metrics?.reward_history, historyOffset, smoothAndSample]);
  const placementData = useMemo(() => smoothAndSample(metrics?.avg_placement_history.map((v, i) => ({ s: historyOffset + i + 1, placement: +(v).toFixed(2) })) ?? [], ['placement']), [metrics?.avg_placement_history, historyOffset, smoothAndSample]);
  const placementByOpponentData = useMemo(() => {
    if (!metrics) return [];
    const s = metrics.placement_selfplay_history ?? [];
    const h = metrics.placement_hof_history ?? [];
    const v5 = metrics.placement_v5_history ?? [];
    const n = Math.max(s.length, h.length, v5.length);
    const rows = Array.from({ length: n }, (_, i) => ({
      s: historyOffset + i + 1,
      selfplay: s[i],
      hof: h[i],
      v5: v5[i],
    }));
    return smoothAndSample(rows, ['selfplay', 'hof', 'v5']);
  }, [metrics, historyOffset, smoothAndSample]);
  const lossData = useMemo(() => smoothAndSample(metrics?.loss_history.map((v, i) => ({ s: historyOffset + i + 1, loss: v })) ?? [], ['loss']), [metrics?.loss_history, historyOffset, smoothAndSample]);
  const sessionScoreData = useMemo(() => smoothAndSample(metrics?.session_avg_score_history?.map((v, i) => ({ s: historyOffset + i + 1, avgScore: v })) ?? [], ['avgScore']), [metrics?.session_avg_score_history, historyOffset, smoothAndSample]);
  const stockskisPlaceData = useMemo(() => smoothAndSample(metrics?.stockskis_place_history?.map((v, i) => ({ s: historyOffset + i + 1, place: v })) ?? [], ['place']), [metrics?.stockskis_place_history, historyOffset, smoothAndSample]);
  const lookaheadScoreData = useMemo(() => smoothAndSample(metrics?.lookahead_score_history?.map((v, i) => ({ s: historyOffset + i + 1, score: v })) ?? [], ['score']), [metrics?.lookahead_score_history, historyOffset, smoothAndSample]);
  const lookaheadBidData = useMemo(() => smoothAndSample(metrics?.lookahead_bid_rate_history?.map((v, i) => ({ s: historyOffset + i + 1, bidRate: +(v * 100).toFixed(1) })) ?? [], ['bidRate']), [metrics?.lookahead_bid_rate_history, historyOffset, smoothAndSample]);
  const bidKlopData = useMemo(() => smoothAndSample(metrics?.bid_rate_history?.map((v, i) => ({
    s: historyOffset + i + 1,
    bid: +((metrics.bid_rate_history[i] ?? 0) * 100).toFixed(1),
    klop: +((metrics.klop_rate_history?.[i] ?? 0) * 100).toFixed(1),
    solo: +((metrics.solo_rate_history?.[i] ?? 0) * 100).toFixed(1),
  })) ?? [], ['bid', 'klop', 'solo']), [metrics?.bid_rate_history, metrics?.klop_rate_history, metrics?.solo_rate_history, historyOffset, smoothAndSample]);

  // Contract bar data (declarer stats only — the meaningful metric)
  const contractBarData = metrics?.contract_stats ? Object.entries(metrics.contract_stats)
    .filter(([, cs]) => cs.decl_played > 0)
    .map(([name, cs]) => ({
      name: CONTRACT_LABELS[name] || name,
      key: name,
      played: cs.decl_played,
      winRate: +(cs.decl_win_rate * 100).toFixed(1),
      avgScore: +cs.decl_avg_score.toFixed(1),
    })) : [];

  // Contract win-rate over time
  const contractWinData = useMemo(() => smoothAndSample(metrics?.contract_win_rate_history
    ? (metrics.avg_placement_history || []).map((_, i) => {
        const row: any = { s: historyOffset + i + 1 };
        for (const [cname, arr] of Object.entries(metrics.contract_win_rate_history)) {
          if (arr[i] !== undefined) row[cname] = +(arr[i] * 100).toFixed(1);
        }
        return row;
      })
    : [], Object.keys(metrics?.contract_win_rate_history || {})), [metrics?.contract_win_rate_history, metrics?.avg_placement_history, historyOffset, smoothAndSample]);

  const sessionPct = metrics && metrics.total_sessions > 0
    ? (metrics.session / metrics.total_sessions) * 100
    : 0;

  return (
    <div className="training-dashboard">
      {/* Header */}
      <div className="dashboard-header">
        <button className="btn-secondary btn-sm" onClick={onBack}>← Back</button>
        <h2>AI Training Dashboard</h2>
      </div>

      {/* Controls */}
      <div className="training-controls-bar">
        <label className="td-field min-width-80">
          <span>Sessions</span>
          <input type="number" value={sessions} onChange={e => setSessions(Number(e.target.value))}
            disabled={isTraining} min={10} step={10} />
        </label>
        <label className="td-field min-width-80">
          <span>Games/Session</span>
          <input type="number" value={gamesPerSession} onChange={e => setGamesPerSession(Number(e.target.value))}
            disabled={isTraining} min={10} step={10} />
        </label>
        {!labMode && (
          <label className="td-field min-width-80">
            <span>StockŠkis Ratio {Math.round(stockskisRatio * 100)}%</span>
            <input type="range" value={stockskisRatio} onChange={e => setStockskisRatio(Number(e.target.value))}
              disabled={isTraining} min={0} max={1.0} step={0.1} />
          </label>
        )}
        {!labMode && stockskisRatio > 0 && (
          <label className="td-field min-width-80">
            <span>StockŠkis STR {Math.round(stockskisStrength * 100)}%</span>
            <input type="range" value={stockskisStrength} onChange={e => setStockskisStrength(Number(e.target.value))}
              disabled={isTraining} min={0.1} max={1.0} step={0.1} />
          </label>
        )}
        {!labMode && (
          <label className="td-field min-width-80">
            <span>Lookahead Ratio {Math.round(lookaheadRatio * 100)}%</span>
            <input type="range" value={lookaheadRatio} onChange={e => setLookaheadRatio(Number(e.target.value))}
              disabled={isTraining} min={0} max={1.0} step={0.1} />
          </label>
        )}
        {!labMode && lookaheadRatio > 0 && (
          <label className="td-field min-width-80">
            <span>Lookahead Sims</span>
            <input type="number" value={lookaheadSims} onChange={e => setLookaheadSims(Number(e.target.value))}
              disabled={isTraining} min={1} max={200} step={5} />
          </label>
        )}
        {!labMode && lookaheadRatio > 0 && (
          <label className="td-check" style={{ margin: 0 }}>
            <input type="checkbox" checked={lookaheadPerfectInfo} onChange={e => setLookaheadPerfectInfo(e.target.checked)} disabled={isTraining} />
            <span>Perfect Info</span>
          </label>
        )}
        <label className="td-field min-width-80">
          <span>Smoothing {Math.round(smoothing * 100)}%</span>
          <input type="range" value={smoothing} onChange={e => setSmoothing(Number(e.target.value))}
            min={0} max={0.99} step={0.01} />
        </label>
        <label className="td-check" style={{ margin: 0 }}>
          <input type="checkbox" checked={labMode} onChange={e => setLabMode(e.target.checked)} disabled={isTraining} />
          <span style={{ color: labMode ? '#d4a843' : undefined, fontWeight: labMode ? 600 : undefined }}>GPU Lab</span>
        </label>
        {labMode && (
          <label className="td-field min-width-80">
            <span>Concurrency</span>
            <input type="number" value={concurrency} onChange={e => setConcurrency(Number(e.target.value))}
              disabled={isTraining} min={1} max={512} step={16} />
          </label>
        )}
        {!labMode && (
          <>
            <label className="td-check" style={{ margin: 0 }}>
              <input type="checkbox" checked={useRustEngine} onChange={e => setUseRustEngine(e.target.checked)} disabled={isTraining} />
              <span>Rust Engine</span>
            </label>
            <label className="td-field min-width-80">
              <span>Warmup Games</span>
              <input type="number" value={warmupGames} onChange={e => setWarmupGames(Number(e.target.value))}
                disabled={isTraining} min={0} step={100000} placeholder="0" />
            </label>
          </>
        )}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <label className="td-check" style={{ margin: 0 }}>
            <input type="checkbox" checked={resume} onChange={e => setResume(e.target.checked)} disabled={isTraining} />
            <span>Resume</span>
          </label>
          {resume && (
            <select 
              value={resumeFrom} 
              onChange={e => setResumeFrom(e.target.value)} 
              disabled={isTraining}
              style={{
                background: '#232529',
                color: '#fff',
                border: '1px solid #444',
                padding: '4px 8px',
                borderRadius: '4px',
                fontSize: '12px'
              }}
            >
              <option value="tarok_agent_latest.pt">Latest (tarok_agent_latest.pt)</option>
              {availableSnapshots.map(s => (
                <option key={s.filename} value={s.filename}>{s.filename} (Session {s.session}, Ep {s.episode})</option>
              ))}
            </select>
          )}
        </div>
        {isTraining ? (
          <button className="btn-danger" onClick={stopTraining}>Stop</button>
        ) : (
          <button className="btn-gold" onClick={startTraining}>Start Training</button>
        )}
      </div>

      {/* Progress */}
      {metrics && metrics.total_episodes > 0 && (
        <div className="td-progress">
          <div className="td-progress-bar">
            <div className="td-progress-fill" style={{ width: `${sessionPct}%` }} />
          </div>
          <span className="td-progress-text">
            {trainingMode === 'lab' && <span style={{ color: '#d4a843', fontWeight: 600, marginRight: 6 }}>GPU Lab</span>}
            {metrics.run_id && <span className="td-run-id" title="Training run ID">#{metrics.run_id} · </span>}
            Session {metrics.session}/{metrics.total_sessions} · {metrics.episode.toLocaleString()} games · {metrics.games_per_second.toFixed(1)} g/s
            {(metrics as any).buffer_size > 0 && <> · buf {(metrics as any).buffer_size.toLocaleString()}</>}
            {(metrics as any).policy_version > 0 && <> · v{(metrics as any).policy_version}</>}
          </span>
        </div>
      )}

      {/* Stat cards */}
      <div className="td-stats">
        <StatCard label="Avg Placement" value={metrics?.avg_placement?.toFixed(2) ?? "0.00"} highlight />
        <StatCard label="Avg Reward" value={(metrics?.avg_reward ?? 0).toFixed(2)} />
        <StatCard label="Sess. Avg Score"
          value={metrics?.session_avg_score_history?.length
            ? metrics.session_avg_score_history[metrics.session_avg_score_history.length - 1].toFixed(1)
            : '—'}
        />
        <StatCard label="Bid Rate" value={`${((metrics?.bid_rate ?? 0) * 100).toFixed(0)}%`} />
        <StatCard label="Klop Rate" value={`${((metrics?.klop_rate ?? 0) * 100).toFixed(0)}%`} />
        <StatCard label="Solo Rate" value={`${((metrics?.solo_rate ?? 0) * 100).toFixed(0)}%`} />
        <StatCard label="Games/sec" value={(metrics?.games_per_second ?? 0).toFixed(1)} />
        <StatCard label="Entropy" value={(metrics?.entropy ?? 0).toFixed(4)} />
        <StatCard label="Policy Loss" value={(metrics?.policy_loss ?? 0).toFixed(4)} />
      </div>

      {/* Tabs */}
      <div className="td-tabs">
        {(['overview', 'contracts', 'loss', 'tarok_bids', 'snapshots'] as const).map(t => (
          <button key={t} className={`td-tab ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>
            {t === 'overview' ? '📈 Overview' : t === 'contracts' ? '🃏 Contracts' : t === 'loss' ? '📉 Loss & Entropy' : t === 'tarok_bids' ? '🎯 Taroks vs Bid' : '💾 Snapshots'}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="td-tab-content">
        {tab === 'overview' && (
          <div className="chart-grid">
            <ChartCard title="Avg Placement Over Time">
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={placementByOpponentData.length > 0 ? placementByOpponentData : placementData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                  <XAxis dataKey="s" stroke="#666" fontSize={11} type="number" domain={['dataMin', 'dataMax']} />
                  <YAxis stroke="#666" fontSize={11} domain={[1.0, 4.0]} reversed tickCount={4} />
                  <Tooltip {...tooltipStyle} />
                  <Legend />
                  <Line type="monotone" dataKey="selfplay" stroke="#4caf50" strokeWidth={2} dot={false} name="Self-Play" connectNulls />
                  <Line type="monotone" dataKey="hof" stroke="#ff9800" strokeWidth={2} dot={false} name="HoF" connectNulls />
                  <Line type="monotone" dataKey="v5" stroke="#2196f3" strokeWidth={2} dot={false} name="StockSkis v5" connectNulls />
                  <Line type="monotone" dataKey="placement" stroke="#9e9e9e" strokeWidth={1.5} dot={false} name="Overall" connectNulls />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Average Reward">
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={rewardData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                  <XAxis dataKey="s" stroke="#666" fontSize={11} type="number" domain={['dataMin', 'dataMax']} />
                  <YAxis stroke="#666" fontSize={11} />
                  <Tooltip {...tooltipStyle} />
                  <Line type="monotone" dataKey="reward" stroke="#d4a843" strokeWidth={2} dot={false} name="Reward" />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Session Avg Score (P0)">
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={sessionScoreData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                  <XAxis dataKey="s" stroke="#666" fontSize={11} type="number" domain={['dataMin', 'dataMax']} />
                  <YAxis stroke="#666" fontSize={11} />
                  <Tooltip {...tooltipStyle} />
                  <Line type="monotone" dataKey="avgScore" stroke="#ff9800" strokeWidth={2} dot={false} name="Avg Score" />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Bid / Klop / Solo Rates">
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={bidKlopData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                  <XAxis dataKey="s" stroke="#666" fontSize={11} type="number" domain={['dataMin', 'dataMax']} />
                  <YAxis stroke="#666" fontSize={11} domain={[0, 100]} unit="%" />
                  <Tooltip {...tooltipStyle} />
                  <Legend />
                  <Line type="monotone" dataKey="bid" stroke="#2196f3" strokeWidth={2} dot={false} name="Bid %" />
                  <Line type="monotone" dataKey="klop" stroke="#888" strokeWidth={2} dot={false} name="Klop %" />
                  <Line type="monotone" dataKey="solo" stroke="#e91e63" strokeWidth={2} dot={false} name="Solo %" />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>

            {stockskisPlaceData.length > 0 && (
              <ChartCard title="StockŠkis Avg Finishing Place">
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={stockskisPlaceData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                    <XAxis dataKey="s" stroke="#666" fontSize={11} type="number" domain={['dataMin', 'dataMax']} />
                    <YAxis stroke="#666" fontSize={11} domain={[1, 4]} reversed />
                    <Tooltip {...tooltipStyle} />
                    <Line type="monotone" dataKey="place" stroke="#9c27b0" strokeWidth={2} dot={false} name="Avg Place" />
                  </LineChart>
                </ResponsiveContainer>
              </ChartCard>
            )}

            {lookaheadScoreData.length > 0 && (
              <ChartCard title="Avg Score vs Lookahead Opponents">
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={lookaheadScoreData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                    <XAxis dataKey="s" stroke="#666" fontSize={11} type="number" domain={['dataMin', 'dataMax']} />
                    <YAxis stroke="#666" fontSize={11} />
                    <Tooltip {...tooltipStyle} />
                    <Line type="monotone" dataKey="score" stroke="#00bcd4" strokeWidth={2} dot={false} name="Avg Score" />
                  </LineChart>
                </ResponsiveContainer>
              </ChartCard>
            )}

            {lookaheadBidData.length > 0 && (
              <ChartCard title="Bid Rate vs Lookahead Opponents">
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={lookaheadBidData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                    <XAxis dataKey="s" stroke="#666" fontSize={11} type="number" domain={['dataMin', 'dataMax']} />
                    <YAxis stroke="#666" fontSize={11} domain={[0, 100]} unit="%" />
                    <Tooltip {...tooltipStyle} />
                    <Line type="monotone" dataKey="bidRate" stroke="#00bcd4" strokeWidth={2} dot={false} name="Bid %" />
                  </LineChart>
                </ResponsiveContainer>
              </ChartCard>
            )}
          </div>
        )}

        {tab === 'contracts' && (
          <div className="chart-grid">
            <ChartCard title="Declarer: Play Count & Win Rate">
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={contractBarData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                  <XAxis xAxisId="count" type="number" stroke="#666" fontSize={11} orientation="bottom" />
                  <XAxis xAxisId="pct" type="number" stroke="#4caf50" fontSize={11} orientation="top" domain={[0, 100]} unit="%" />
                  <YAxis type="category" dataKey="name" stroke="#666" fontSize={12} width={70} />
                  <Tooltip {...tooltipStyle} />
                  <Legend />
                  <Bar xAxisId="count" dataKey="played" name="Declared" fill="#4a9eff" />
                  <Bar xAxisId="pct" dataKey="winRate" name="Win %" fill="#4caf50" />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Declarer: Average Score by Contract">
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={contractBarData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                  <XAxis type="number" stroke="#666" fontSize={11} />
                  <YAxis type="category" dataKey="name" stroke="#666" fontSize={12} width={70} />
                  <Tooltip {...tooltipStyle} />
                  <Bar dataKey="avgScore" name="Avg Score">
                    {contractBarData.map((entry) => (
                      <Cell key={entry.key} fill={CONTRACT_COLORS[entry.key] || '#888'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Declarer Win Rates Over Time" wide>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={contractWinData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                  <XAxis dataKey="s" stroke="#666" fontSize={11} type="number" domain={['dataMin', 'dataMax']} />
                  <YAxis stroke="#666" fontSize={11} domain={[0, 100]} unit="%" />
                  <Tooltip {...tooltipStyle} />
                  <Legend />
                  {Object.keys(CONTRACT_LABELS).map(cname => (
                    <Line
                      key={cname}
                      type="monotone"
                      dataKey={cname}
                      stroke={CONTRACT_COLORS[cname]}
                      strokeWidth={1.5}
                      dot={false}
                      name={CONTRACT_LABELS[cname]}
                      connectNulls
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Contract Breakdown (Declarer vs Defender)" wide>
              <table className="td-table">
                <thead>
                  <tr>
                    <th rowSpan={2}>Contract</th>
                    <th colSpan={3} className="th-group">As Declarer</th>
                    <th colSpan={3} className="th-group">As Defender</th>
                  </tr>
                  <tr>
                    <th>Played</th>
                    <th>Win %</th>
                    <th>Avg Score</th>
                    <th>Played</th>
                    <th>Win %</th>
                    <th>Avg Score</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics?.contract_stats && Object.entries(metrics.contract_stats).map(([name, cs]) => (
                    <tr key={name} className={cs.played === 0 ? 'dimmed' : ''}>
                      <td>
                        <span className="td-dot" style={{ background: CONTRACT_COLORS[name] }} />
                        {CONTRACT_LABELS[name] || name}
                      </td>
                      <td>{cs.decl_played || '—'}</td>
                      <td>{cs.decl_played > 0 ? `${(cs.decl_win_rate * 100).toFixed(1)}%` : '—'}</td>
                      <td>{cs.decl_played > 0 ? cs.decl_avg_score.toFixed(1) : '—'}</td>
                      <td>{cs.def_played || '—'}</td>
                      <td>{cs.def_played > 0 ? `${(cs.def_win_rate * 100).toFixed(1)}%` : '—'}</td>
                      <td>{cs.def_played > 0 ? cs.def_avg_score.toFixed(1) : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </ChartCard>
          </div>
        )}

        {tab === 'loss' && (
          <div className="chart-grid">
            <ChartCard title="Total Loss">
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={lossData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                  <XAxis dataKey="s" stroke="#666" fontSize={11} type="number" domain={['dataMin', 'dataMax']} />
                  <YAxis stroke="#666" fontSize={11} />
                  <Tooltip {...tooltipStyle} />
                  <Line type="monotone" dataKey="loss" stroke="#e94560" strokeWidth={2} dot={false} name="Loss" />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>
          </div>
        )}

        {tab === 'tarok_bids' && (() => {
          const tcb = metrics?.tarok_count_bids;
          const chartData: Record<string, unknown>[] = [];
          if (tcb) {
            for (let t = 0; t <= 12; t++) {
              const row: Record<string, unknown> = { taroks: t };
              const bucket = tcb[String(t)] || {};
              let total = 0;
              for (const cnt of Object.values(bucket)) total += cnt;
              for (const [cname, cnt] of Object.entries(bucket)) {
                row[cname] = total > 0 ? Math.round((cnt / total) * 100) : 0;
              }
              row._total = total;
              chartData.push(row);
            }
          }
          const allContracts = new Set<string>();
          chartData.forEach(r => Object.keys(r).forEach(k => { if (k !== 'taroks' && k !== '_total') allContracts.add(k); }));

          return (
            <div className="chart-grid">
              <ChartCard title="Contract Distribution by Tarok Count at Deal (P0)">
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart data={chartData} stackOffset="expand">
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                    <XAxis dataKey="taroks" stroke="#666" fontSize={11} label={{ value: 'Taroks in hand', position: 'insideBottom', offset: -5, fill: '#888' }} />
                    <YAxis stroke="#666" fontSize={11} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} />
                    <Tooltip
                      contentStyle={{ background: '#1e1f23', border: '1px solid #333', borderRadius: 8 }}
                      labelStyle={{ color: '#d4a843' }}
                      formatter={(value: number, name: string) => [`${value}%`, CONTRACT_LABELS[name] || name]}
                      labelFormatter={(label) => {
                        const row = chartData.find(r => r.taroks === label);
                        return `${label} taroks (${row?._total ?? 0} games)`;
                      }}
                    />
                    <Legend formatter={(v: string) => CONTRACT_LABELS[v] || v} />
                    {Array.from(allContracts).map(cname => (
                      <Bar key={cname} dataKey={cname} stackId="a" fill={CONTRACT_COLORS[cname] || '#666'} />
                    ))}
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>
              <ChartCard title="Games per Tarok Count (P0)">
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                    <XAxis dataKey="taroks" stroke="#666" fontSize={11} />
                    <YAxis stroke="#666" fontSize={11} />
                    <Tooltip
                      contentStyle={{ background: '#1e1f23', border: '1px solid #333', borderRadius: 8 }}
                      labelStyle={{ color: '#d4a843' }}
                    />
                    <Bar dataKey="_total" fill="#d4a843" name="Total games" />
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>
            </div>
          );
        })()}

        {tab === 'snapshots' && (
          <div className="td-snapshots">
            <p className="td-snap-info">
              Snapshots are saved periodically during training. Use them to resume training or play against a specific version.
            </p>
            {metrics?.snapshots && metrics.snapshots.length > 0 ? (
              <table className="td-table">
                <thead>
                  <tr>
                    <th>Checkpoint</th>
                    <th>Session</th>
                    <th>Games</th>
                    <th>Win %</th>
                    <th>Avg Reward</th>
                    <th>Speed</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics.snapshots.map((snap, i) => (
                    <tr key={i}>
                      <td className="td-snap-file">{snap.filename}</td>
                      <td>{snap.session}</td>
                      <td>{snap.episode.toLocaleString()}</td>
                      <td>{(snap.win_rate * 100).toFixed(1)}%</td>
                      <td>{snap.avg_reward.toFixed(2)}</td>
                      <td>{snap.games_per_second.toFixed(1)} g/s</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="td-empty">No snapshots yet. Start training to generate checkpoints.</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function StatCard({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className={`td-stat ${highlight ? 'td-stat-hl' : ''}`}>
      <div className="td-stat-val">{value}</div>
      <div className="td-stat-lbl">{label}</div>
    </div>
  );
}

function ChartCard({ title, children, wide }: { title: string; children: React.ReactNode; wide?: boolean }) {
  return (
    <div className={`td-chart ${wide ? 'td-chart-wide' : ''}`}>
      <h3>{title}</h3>
      {children}
    </div>
  );
}

const tooltipStyle = {
  contentStyle: { background: '#1a1a2e', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', fontSize: 13 },
  labelStyle: { color: '#aaa' },
};
