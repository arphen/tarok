import React from 'react';
import type { MatchInfo } from '../types/game';
import { CONTRACT_NAMES } from '../types/game';
import './Scoreboard.css';

interface ScoreboardProps {
  matchInfo: MatchInfo;
  playerNames: string[];
}

const Scoreboard = React.memo(function Scoreboard({ matchInfo, playerNames }: ScoreboardProps) {
  const { cumulative_scores, caller_counts, called_counts, round_history } = matchInfo;
  const name = (idx: number) => playerNames[idx] ?? `P${idx}`;

  // Sort players by cumulative score descending
  const ranked = [0, 1, 2, 3].sort(
    (a, b) => (cumulative_scores[String(b)] ?? 0) - (cumulative_scores[String(a)] ?? 0)
  );

  return (
    <div className="scoreboard" data-testid="scoreboard">
      <h3 className="scoreboard-title">
        Scoreboard
        <span className="scoreboard-round">
          Rd {matchInfo.round_num}/{matchInfo.total_rounds}
        </span>
      </h3>

      <table className="scoreboard-table">
        <thead>
          <tr>
            <th></th>
            <th>Player</th>
            <th>Score</th>
            <th title="Times as declarer">Decl</th>
            <th title="Times as partner">Part</th>
          </tr>
        </thead>
        <tbody>
          {ranked.map((p, rank) => {
            const score = cumulative_scores[String(p)] ?? 0;
            const callerN = caller_counts[String(p)] ?? 0;
            const calledN = called_counts[String(p)] ?? 0;
            return (
              <tr key={p} className={p === 0 ? 'scoreboard-human' : ''}>
                <td className="scoreboard-rank">{rank + 1}.</td>
                <td className="scoreboard-name">{name(p)}</td>
                <td className={`scoreboard-score ${score > 0 ? 'score-positive' : score < 0 ? 'score-negative' : ''}`}>
                  {score > 0 ? '+' : ''}{score}
                </td>
                <td className="scoreboard-stat">{callerN}</td>
                <td className="scoreboard-stat">{calledN}</td>
              </tr>
            );
          })}
        </tbody>
      </table>

      {round_history.length > 0 && (
        <div className="scoreboard-history">
          <h4>Round History</h4>
          <div className="history-scroll">
            {round_history.map(r => (
              <div key={r.round} className="history-entry">
                <span className="history-round">R{r.round}</span>
                <span className="history-contract">
                  {r.contract !== null ? CONTRACT_NAMES[r.contract] ?? r.contract : '—'}
                </span>
                <span className="history-scores">
                  {[0, 1, 2, 3].map(p => {
                    const s = r.scores[String(p)] ?? 0;
                    return (
                      <span key={p} className={`history-score ${s > 0 ? 'score-positive' : s < 0 ? 'score-negative' : ''}`}>
                        {s > 0 ? '+' : ''}{s}
                      </span>
                    );
                  })}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
});

export default Scoreboard;
