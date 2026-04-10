import React, { useState, useEffect } from 'react';
import './ModelLeaderboard.css';

interface Standing {
  rank: number;
  model: string;
  type: string;
  avg_placement: number;
  first_places: number;
  top2: number;
  top_half: number;
  rounds_played: number;
}

interface Tournament {
  id: number;
  date: string;
  num_rounds: number;
  num_players: number;
  note?: string;
  standings: Standing[];
}

interface ResultsData {
  tournaments: Tournament[];
  top_models: string[];
}

export default function ModelLeaderboard() {
  const [open, setOpen] = useState(false);
  const [data, setData] = useState<ResultsData | null>(null);
  const [activeTournament, setActiveTournament] = useState(0);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (open && !data) {
      setLoading(true);
      fetch('/api/tournament/results')
        .then(r => r.json())
        .then(d => { setData(d); setActiveTournament(d.tournaments.length - 1); })
        .catch(() => {})
        .finally(() => setLoading(false));
    }
  }, [open, data]);

  const t = data?.tournaments[activeTournament];

  return (
    <div className={`model-leaderboard ${open ? 'leaderboard-open' : ''}`}>
      <button className="leaderboard-tab" onClick={() => setOpen(o => !o)} title="Model Leaderboard">
        <span className="leaderboard-tab-icon">🏆</span>
        <span className="leaderboard-tab-label">Rankings</span>
      </button>

      <div className="leaderboard-panel">
        <div className="leaderboard-header">
          <h3>Model Leaderboard</h3>
          <button className="leaderboard-close" onClick={() => setOpen(false)}>✕</button>
        </div>

        {loading && <p className="leaderboard-loading">Loading…</p>}

        {data && data.tournaments.length === 0 && (
          <p className="leaderboard-empty">No tournament results yet.</p>
        )}

        {data && data.tournaments.length > 0 && (
          <>
            {/* Top models banner */}
            {data.top_models.length > 0 && (
              <div className="top-models">
                <span className="top-label">Best</span>
                {data.top_models.slice(0, 3).map((m, i) => (
                  <span key={m} className={`top-model top-model-${i}`}>
                    {['🥇', '🥈', '🥉'][i]} {m}
                  </span>
                ))}
              </div>
            )}

            {/* Tournament tabs */}
            <div className="tournament-tabs">
              {data.tournaments.map((tour, i) => (
                <button
                  key={tour.id}
                  className={`tournament-tab-btn ${i === activeTournament ? 'active' : ''}`}
                  onClick={() => setActiveTournament(i)}
                >
                  T{tour.id}
                </button>
              ))}
            </div>

            {/* Standings table */}
            {t && (
              <div className="leaderboard-table-wrap">
                <table className="leaderboard-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Model</th>
                      <th>Avg</th>
                      <th>1st</th>
                      <th>Top 2</th>
                      <th>Top½</th>
                    </tr>
                  </thead>
                  <tbody>
                    {t.standings.map(s => (
                      <tr key={s.rank} className={s.type === 'random' ? 'row-random' : s.rank <= 3 ? 'row-top' : ''}>
                        <td className="col-rank">{s.rank}</td>
                        <td className="col-model" title={s.model}>{s.model}</td>
                        <td className="col-avg">{s.avg_placement.toFixed(2)}</td>
                        <td>{s.first_places}</td>
                        <td>{s.top2}</td>
                        <td>{s.top_half}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="table-meta">
                  {t.num_rounds} rounds · {t.num_players} players · {t.date}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
