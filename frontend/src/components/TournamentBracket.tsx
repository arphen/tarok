import React, { useState, useEffect, useCallback } from 'react';
import type {
  BracketMatch,
  TournamentEntry,
} from '../hooks/useTournament';
import { useTournament } from '../hooks/useTournament';
import './TournamentBracket.css';

interface TournamentBracketProps {
  checkpoints: { filename: string; episode: number; win_rate: number }[];
  onBack: () => void;
}

type AgentType = 'rl' | 'random';

let entryCounter = 0;
function makeEntry(name: string, type: AgentType, checkpoint: string): TournamentEntry {
  return { id: `entry-${++entryCounter}`, name, type, checkpoint };
}

export default function TournamentBracket({ checkpoints: initialCheckpoints, onBack }: TournamentBracketProps) {
  const tournament = useTournament();
  const [running, setRunning] = useState(false);
  const [mode, setMode] = useState<'single' | 'multi'>('single');
  const [checkpoints, setCheckpoints] = useState(initialCheckpoints);

  // ---- Setup state for adding entries ----
  const [newName, setNewName] = useState('');
  const [newType, setNewType] = useState<AgentType>('rl');
  const [newCheckpoint, setNewCheckpoint] = useState('');

  // Refresh checkpoints periodically
  const refreshCheckpoints = useCallback(async () => {
    try {
      const res = await fetch('/api/checkpoints');
      const data = await res.json();
      if (data.checkpoints) setCheckpoints(data.checkpoints);
    } catch { /* server not up */ }
  }, []);

  useEffect(() => {
    refreshCheckpoints();
    const id = setInterval(refreshCheckpoints, 5000);
    return () => clearInterval(id);
  }, [refreshCheckpoints]);

  const addEntry = () => {
    const name = newName.trim() || `${newType === 'rl' ? 'RL' : 'Rnd'}-${tournament.entries.length}`;
    tournament.setEntries([...tournament.entries, makeEntry(name, newType, newCheckpoint)]);
    setNewName('');
  };

  const removeEntry = (id: string) => {
    tournament.setEntries(tournament.entries.filter(e => e.id !== id));
  };

  const handleRunMatch = async () => {
    setRunning(true);
    try {
      await tournament.runCurrentMatch();
    } finally {
      setRunning(false);
    }
  };

  const handleAutoRun = async () => {
    setRunning(true);
    try {
      let result: 'running' | 'finished' = 'running';
      for (let i = 0; i < 6 && result === 'running'; i++) {
        result = await tournament.runCurrentMatch();
      }
    } finally {
      setRunning(false);
    }
  };

  // ---- Setup screen ----
  if (tournament.phase === 'setup') {
    return (
      <div className="tournament-view">
        <div className="app-bar">
          <button className="btn-secondary btn-sm" onClick={onBack}>← Back</button>
          <span className="spectator-title">Tournament Mode</span>
        </div>

        <div className="tournament-setup">
          <h2>Double Elimination Tournament</h2>
          <p className="setup-subtitle">
            Add 4–8 models. They will compete in a double-elimination bracket (losers get a second chance).
          </p>

          <div className="tournament-entries">
            <h3>Models ({tournament.entries.length}/8)</h3>
            {tournament.entries.map(e => (
              <div key={e.id} className="tournament-entry-row">
                <span className="entry-name">{e.name}</span>
                <span className="entry-type">{e.type}</span>
                {e.checkpoint && <span className="entry-ckpt">{e.checkpoint}</span>}
                <button className="btn-sm btn-danger" onClick={() => removeEntry(e.id)}>×</button>
              </div>
            ))}
          </div>

          {tournament.entries.length < 8 && (
            <div className="add-entry-form">
              <input
                type="text"
                placeholder="Name (optional)"
                value={newName}
                onChange={e => setNewName(e.target.value)}
                maxLength={20}
              />
              <select value={newType} onChange={e => setNewType(e.target.value as AgentType)}>
                <option value="rl">RL Agent</option>
                <option value="random">Random</option>
              </select>
              {newType === 'rl' && (
                <select value={newCheckpoint} onChange={e => setNewCheckpoint(e.target.value)}>
                  <option value="">Latest</option>
                  {checkpoints.map(c => (
                    <option key={c.filename} value={c.filename}>
                      {c.filename} (ep{c.episode}, {(c.win_rate * 100).toFixed(0)}%)
                    </option>
                  ))}
                </select>
              )}
              <button className="btn-secondary btn-sm" onClick={addEntry}>+ Add</button>
            </div>
          )}

          <label className="config-field config-field-inline" style={{ marginTop: 16 }}>
            <span>Games per round</span>
            <input
              type="number"
              min={1}
              max={100}
              value={tournament.gamesPerRound}
              onChange={e => tournament.setGamesPerRound(Number(e.target.value))}
            />
          </label>

          {/* Mode selection */}
          <div className="tournament-mode-toggle" style={{ marginTop: 16 }}>
            <label>
              <input type="radio" name="mode" value="single" checked={mode === 'single'} onChange={() => setMode('single')} />
              {' '}Single Tournament (bracket view)
            </label>
            <label style={{ marginLeft: 16 }}>
              <input type="radio" name="mode" value="multi" checked={mode === 'multi'} onChange={() => setMode('multi')} />
              {' '}Multi-Tournament Simulation
            </label>
          </div>

          {mode === 'multi' && (
            <label className="config-field config-field-inline" style={{ marginTop: 8 }}>
              <span>Number of tournaments</span>
              <input
                type="number"
                min={1}
                max={100}
                value={tournament.numTournaments}
                onChange={e => tournament.setNumTournaments(Number(e.target.value))}
              />
            </label>
          )}

          {mode === 'single' ? (
            <button
              className="btn-gold btn-large"
              style={{ marginTop: 20 }}
              onClick={tournament.startTournament}
              disabled={tournament.entries.length < 4}
            >
              Start Tournament ({tournament.entries.length < 4 ? `need ${4 - tournament.entries.length} more` : `${tournament.entries.length} models`})
            </button>
          ) : (
            <button
              className="btn-gold btn-large"
              style={{ marginTop: 20 }}
              onClick={tournament.startMultiTournament}
              disabled={tournament.entries.length < 4}
            >
              Simulate {tournament.numTournaments} Tournaments ({tournament.entries.length < 4 ? `need ${4 - tournament.entries.length} more` : `${tournament.entries.length} models`})
            </button>
          )}
        </div>
      </div>
    );
  }

  // ---- Multi-tournament simulation view ----
  if (tournament.multiProgress) {
    const mp = tournament.multiProgress;
    const standings = Object.values(mp.standings)
      .filter(s => s.tournaments_played > 0)
      .sort((a, b) => a.avg_placement - b.avg_placement);

    return (
      <div className="tournament-view">
        <div className="app-bar">
          <button className="btn-secondary btn-sm" onClick={() => { if (mp.status === 'running') tournament.stopMultiTournament(); tournament.reset(); }}>← Setup</button>
          <span className="spectator-title">Multi-Tournament Simulation</span>
        </div>

        <div className="multi-tournament-progress" data-testid="multi-tournament-progress">
          <div className="multi-tournament-header">
            <span className="multi-tournament-counter" data-testid="multi-tournament-counter">
              {mp.status === 'done' ? 'Completed' : mp.status === 'running' ? 'Running' : mp.status}:{' '}
              {mp.current} / {mp.total} tournaments
            </span>
            {mp.status === 'running' && (
              <button className="btn-danger btn-sm" onClick={tournament.stopMultiTournament}>Stop</button>
            )}
            {mp.status === 'done' && (
              <button className="btn-gold btn-sm" onClick={tournament.reset}>New Simulation</button>
            )}
          </div>

          {/* Progress bar */}
          <div className="multi-tournament-bar">
            <div className="multi-tournament-bar-fill" style={{ width: `${mp.total ? (mp.current / mp.total) * 100 : 0}%` }} />
          </div>

          {/* Standings table */}
          {standings.length > 0 && (
            <table className="multi-tournament-standings" data-testid="multi-tournament-standings">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Model</th>
                  <th>Type</th>
                  <th>Avg Place</th>
                  <th>Wins</th>
                  <th>Top 2</th>
                  <th>Top 4</th>
                  <th>Played</th>
                  <th>Placements</th>
                </tr>
              </thead>
              <tbody>
                {standings.map((s, i) => (
                  <tr key={s.name} className={i === 0 && mp.status === 'done' ? 'standing-champion' : ''}>
                    <td>{i + 1}</td>
                    <td className="standing-name">{s.name}</td>
                    <td>{s.type}</td>
                    <td className="standing-avg"><strong>{s.avg_placement.toFixed(2)}</strong></td>
                    <td>{s.wins}</td>
                    <td>{s.top2}</td>
                    <td>{s.top4}</td>
                    <td>{s.tournaments_played}</td>
                    <td className="standing-placements">{s.placements.join(', ')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    );
  }

  // ---- Bracket view ----
  const winnersMatches = tournament.matches.filter(m => m.side === 'winners');
  const losersMatches = tournament.matches.filter(m => m.side === 'losers');

  return (
    <div className="tournament-view">
      <div className="app-bar">
        <button className="btn-secondary btn-sm" onClick={() => { tournament.reset(); }}>← Setup</button>
        <span className="spectator-title">Tournament</span>
        {tournament.phase === 'finished' && tournament.champion && (
          <span className="champion-badge">Champion: {tournament.champion.name}</span>
        )}
      </div>

      <div className="bracket-layout">
        {/* Winners bracket */}
        <div className="bracket-column">
          <h3 className="bracket-heading bracket-winners">Winners Bracket</h3>
          {winnersMatches.map(match => (
            <MatchCard
              key={match.id}
              match={match}
              isCurrent={tournament.currentMatch?.id === match.id}
              isGrandFinal={match.id === 'grand-final'}
            />
          ))}
        </div>

        {/* Losers bracket */}
        <div className="bracket-column">
          <h3 className="bracket-heading bracket-losers">Losers Bracket</h3>
          {losersMatches.map(match => (
            <MatchCard
              key={match.id}
              match={match}
              isCurrent={tournament.currentMatch?.id === match.id}
            />
          ))}
        </div>
      </div>

      {/* Controls */}
      <div className="tournament-controls-bar">
        {tournament.phase === 'running' && !tournament.multiProgress && (
          <>
            <button
              className="btn-gold btn-sm"
              onClick={handleRunMatch}
              disabled={running || !tournament.currentMatch || tournament.currentMatch.entries.length < 4}
            >
              {running ? 'Playing…' : `Play: ${tournament.currentMatch?.label ?? '—'}`}
            </button>
            <button
              className="btn-secondary btn-sm"
              onClick={handleAutoRun}
              disabled={running}
            >
              {running ? 'Running…' : 'Auto-Run All'}
            </button>
          </>
        )}
        {tournament.phase === 'finished' && !tournament.multiProgress && (
          <div className="tournament-champion-banner">
            <span className="champion-trophy">🏆</span>
            <span className="champion-text">
              Champion: <strong>{tournament.champion?.name}</strong>
            </span>
            <button className="btn-gold btn-sm" onClick={tournament.reset}>New Tournament</button>
          </div>
        )}
      </div>
    </div>
  );
}

// ---- Match card sub-component ----

function MatchCard({ match, isCurrent, isGrandFinal }: { match: BracketMatch; isCurrent: boolean; isGrandFinal?: boolean }) {
  const done = match.result !== null;
  const waiting = match.entries.length < 4;

  return (
    <div className={`match-card ${done ? 'match-done' : ''} ${isCurrent ? 'match-current' : ''} ${isGrandFinal ? 'match-grand-final' : ''} ${waiting ? 'match-waiting' : ''}`}>
      <div className="match-header">
        <span className="match-label">{match.label}</span>
        {done && <span className="match-check">✓</span>}
        {isCurrent && !done && <span className="match-live">NEXT</span>}
      </div>
      <div className="match-entries">
        {match.entries.length === 0 && (
          <div className="match-tbd">TBD — waiting for earlier rounds</div>
        )}
        {match.entries.map((entry, i) => {
          const score = match.result?.cumulative[String(i)] ?? null;
          const rank = match.result?.ranked.findIndex(r => r.seat === i);
          const advanced = match.advancedIdx.includes(i);
          const eliminated = match.eliminatedIdx.includes(i);
          return (
            <div
              key={entry.id}
              className={`match-entry ${advanced ? 'entry-advanced' : ''} ${eliminated ? 'entry-eliminated' : ''} ${rank === 0 ? 'entry-winner' : ''}`}
            >
              <span className="entry-rank">
                {rank !== undefined && rank !== -1 && done ? `#${rank + 1}` : ''}
              </span>
              <span className="entry-info">
                <span className="entry-name">{entry.name}</span>
                <span className="entry-type-badge">{entry.type}</span>
              </span>
              <span className="entry-score">
                {score !== null ? (score > 0 ? `+${score}` : `${score}`) : '—'}
              </span>
              <span className="entry-fate">
                {advanced && '▲'}
                {eliminated && '✗'}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
