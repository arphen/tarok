import React, { useEffect, useMemo, useState } from 'react';
import './ArenaCheckpointLeaderboard.css';

interface ArenaCheckpointLeaderboardProps {
  onBack: () => void;
  checkpoints: { filename: string; episode: number; win_rate: number; model_name?: string; is_hof?: boolean }[];
}

interface LeaderboardRow {
  checkpoint: string;
  appearances: number;
  runs: number;
  games: number;
  avg_placement: number;
  bid_wins: number;
  bid_win_rate_per_game: number;
  avg_taroks_in_hand: number;
  declared_games: number;
  declared_win_rate: number;
  avg_declared_win_score: number;
  avg_declared_loss_score: number;
  times_called: number;
  latest_run_at: string;
}

interface ArenaRun {
  run_id: string;
  created_at: string;
  status: string;
  games_done: number;
  total_games: number;
  checkpoints: string[];
}

export default function ArenaCheckpointLeaderboard({ onBack, checkpoints }: ArenaCheckpointLeaderboardProps) {
  const [rows, setRows] = useState<LeaderboardRow[]>([]);
  const [runs, setRuns] = useState<ArenaRun[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let active = true;
    setLoading(true);
    Promise.all([
      fetch('/api/arena/leaderboard/checkpoints').then(r => r.json()),
      fetch('/api/arena/history').then(r => r.json()),
    ])
      .then(([lbData, historyData]) => {
        if (!active) return;
        setRows(lbData.leaderboard ?? []);
        setRuns(historyData.runs ?? []);
      })
      .catch(() => {
        if (!active) return;
        setRows([]);
        setRuns([]);
      })
      .finally(() => {
        if (!active) return;
        setLoading(false);
      });

    return () => {
      active = false;
    };
  }, []);

  const checkpointMeta = useMemo(() => {
    const meta = new Map<string, { model_name?: string; episode: number; win_rate: number; is_hof?: boolean }>();
    checkpoints.forEach(cp => {
      meta.set(cp.filename, cp);
    });
    return meta;
  }, [checkpoints]);

  const checkpointLabel = (checkpoint: string) => {
    if (checkpoint === 'latest') return 'Latest checkpoint';
    const meta = checkpointMeta.get(checkpoint);
    if (!meta) return checkpoint;
    const name = meta.model_name || checkpoint;
    const wr = `${(meta.win_rate * 100).toFixed(0)}%`;
    const hof = meta.is_hof ? ' HoF' : '';
    return `${name} (ep ${meta.episode}, WR ${wr})${hof}`;
  };

  const recentRuns = [...runs].reverse().slice(0, 8);

  return (
    <div className="arena-checkpoint-leaderboard-view">
      <div className="arena-checkpoint-leaderboard-header">
        <button className="btn-secondary btn-sm" onClick={onBack}>← Back</button>
        <h1>Arena Checkpoint Leaderboard</h1>
        <span className="arena-checkpoint-subtitle">Persistent results aggregated by checkpoint from Bot Arena runs</span>
      </div>

      {loading && <p className="arena-checkpoint-loading">Loading leaderboard...</p>}

      {!loading && rows.length === 0 && (
        <p className="arena-checkpoint-empty">No checkpoint data yet. Run Bot Arena with RL checkpoints first.</p>
      )}

      {!loading && rows.length > 0 && (
        <div className="arena-checkpoint-section">
          <div className="arena-checkpoint-table-wrapper">
            <table className="arena-checkpoint-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Checkpoint</th>
                  <th>Avg Place</th>
                  <th>Runs</th>
                  <th>Games</th>
                  <th>Bid Wins</th>
                  <th>Bid Win %</th>
                  <th>Avg Taroks</th>
                  <th>Decl WR</th>
                  <th>Decl Avg Win</th>
                  <th>Decl Avg Loss</th>
                  <th>Called</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row, idx) => (
                  <tr key={row.checkpoint} className={idx === 0 ? 'arena-checkpoint-top' : ''}>
                    <td>{idx + 1}</td>
                    <td title={row.checkpoint}>{checkpointLabel(row.checkpoint)}</td>
                    <td><strong>{row.avg_placement.toFixed(3)}</strong></td>
                    <td>{row.runs}</td>
                    <td>{row.games.toLocaleString()}</td>
                    <td>{row.bid_wins.toLocaleString()}</td>
                    <td>{row.bid_win_rate_per_game.toFixed(2)}%</td>
                    <td>{row.avg_taroks_in_hand.toFixed(3)}</td>
                    <td>{row.declared_win_rate.toFixed(2)}%</td>
                    <td className="positive">{row.avg_declared_win_score.toFixed(2)}</td>
                    <td className="negative">{row.avg_declared_loss_score.toFixed(2)}</td>
                    <td>{row.times_called.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {!loading && recentRuns.length > 0 && (
        <div className="arena-checkpoint-section">
          <h3>Recent Persisted Arena Runs</h3>
          <div className="arena-checkpoint-table-wrapper">
            <table className="arena-checkpoint-table compact">
              <thead>
                <tr>
                  <th>Started</th>
                  <th>Status</th>
                  <th>Games</th>
                  <th>Checkpoints</th>
                </tr>
              </thead>
              <tbody>
                {recentRuns.map(run => (
                  <tr key={run.run_id}>
                    <td>{new Date(run.created_at).toLocaleString()}</td>
                    <td>{run.status}</td>
                    <td>{run.games_done.toLocaleString()} / {run.total_games.toLocaleString()}</td>
                    <td>{run.checkpoints.length ? run.checkpoints.map(checkpointLabel).join(', ') : 'none'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
