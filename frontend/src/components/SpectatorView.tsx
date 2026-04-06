import React, { useState } from 'react';
import { useSpectator } from '../hooks/useSpectator';
import type { AgentConfig } from '../hooks/useSpectator';
import type { CardData, CompletedTrick, ScoreBreakdown, TrickSummaryEntry } from '../types/game';
import { CONTRACT_NAMES, SUIT_SYMBOLS } from '../types/game';
import Hand from './Hand';
import TrickArea from './TrickArea';
import Card from './Card';
import './SpectatorView.css';

interface SpectatorViewProps {
  onBack: () => void;
  checkpoints: { filename: string; episode: number; win_rate: number }[];
}

type AgentType = 'rl' | 'random';

interface AgentSetup {
  name: string;
  type: AgentType;
  checkpoint: string;
}

const DEFAULT_AGENTS: AgentSetup[] = [
  { name: 'Agent-0', type: 'rl', checkpoint: '' },
  { name: 'Agent-1', type: 'rl', checkpoint: '' },
  { name: 'Agent-2', type: 'rl', checkpoint: '' },
  { name: 'Agent-3', type: 'rl', checkpoint: '' },
];

export default function SpectatorView({ onBack, checkpoints }: SpectatorViewProps) {
  const spectator = useSpectator();
  const [agents, setAgents] = useState<AgentSetup[]>(DEFAULT_AGENTS);
  const [delay, setDelay] = useState(1.5);
  const [selectedTrick, setSelectedTrick] = useState<number | null>(null);

  const isSetup = !spectator.gameId;
  const { state } = spectator;
  const names = state.player_names.length > 0 ? state.player_names : agents.map(a => a.name);

  const handleStart = () => {
    const configs: AgentConfig[] = agents.map(a => ({
      name: a.name,
      type: a.type,
      checkpoint: a.checkpoint || undefined,
    }));
    spectator.startGame(configs);
  };

  const updateAgent = (idx: number, patch: Partial<AgentSetup>) => {
    setAgents(prev => prev.map((a, i) => i === idx ? { ...a, ...patch } : a));
  };

  // Setup screen — agent selection
  if (isSetup) {
    return (
      <div className="spectator-view">
        <div className="app-bar">
          <button className="btn-secondary btn-sm" onClick={onBack}>← Menu</button>
          <span className="spectator-title">Spectator Mode</span>
        </div>

        <div className="spectator-setup">
          <h2>Configure Agents</h2>
          <p className="setup-subtitle">Select 4 AI agents to watch them play against each other</p>

          <div className="agent-config-grid">
            {agents.map((agent, i) => (
              <div key={i} className="agent-config-card">
                <div className="agent-config-header">
                  <span className="agent-seat">Seat {i}</span>
                </div>

                <label className="config-field">
                  <span>Name</span>
                  <input
                    type="text"
                    value={agent.name}
                    onChange={e => updateAgent(i, { name: e.target.value })}
                    maxLength={20}
                  />
                </label>

                <label className="config-field">
                  <span>Type</span>
                  <select
                    value={agent.type}
                    onChange={e => updateAgent(i, { type: e.target.value as AgentType })}
                  >
                    <option value="rl">RL Agent (trained)</option>
                    <option value="random">Random Agent</option>
                  </select>
                </label>

                {agent.type === 'rl' && (
                  <label className="config-field">
                    <span>Checkpoint</span>
                    <select
                      value={agent.checkpoint}
                      onChange={e => updateAgent(i, { checkpoint: e.target.value })}
                    >
                      <option value="">Latest</option>
                      {checkpoints.map(c => (
                        <option key={c.filename} value={c.filename}>
                          {c.filename} (ep{c.episode}, {(c.win_rate * 100).toFixed(0)}% WR)
                        </option>
                      ))}
                    </select>
                  </label>
                )}
              </div>
            ))}
          </div>

          <div className="delay-config">
            <label className="config-field config-field-inline">
              <span>Playback Speed (ms per step)</span>
              <input
                type="number"
                min={10}
                max={5000}
                step={50}
                value={spectator.playbackSpeed}
                onChange={e => spectator.setPlaybackSpeed(Number(e.target.value))}
              />
            </label>
          </div>

          <button className="btn-gold btn-large" onClick={handleStart} disabled={spectator.loading}>
            {spectator.loading ? 'Starting…' : 'Start Game'}
          </button>
        </div>
      </div>
    );
  }

  // Active spectator view
  const viewingTrick = selectedTrick !== null && selectedTrick < state.completed_tricks.length
    ? state.completed_tricks[selectedTrick]
    : null;

  return (
    <div className="spectator-view">
      <div className="app-bar">
        <button className="btn-secondary btn-sm" onClick={() => { spectator.disconnect(); }}>← Setup</button>
        <span className="spectator-title">Spectating</span>
        <span className="connection-status">
          {spectator.connected ? '🟢 Connected' : '🔴 Disconnected'}
        </span>
      </div>

      <div className="spectator-layout">
        {/* Main board area */}
        <div className="spectator-board">
          {/* Game info bar */}
          <div className="game-info-bar">
            <div className="info-item">
              <span className="info-label">Tricks</span>
              <span className="info-value">{state.tricks_played}/12</span>
            </div>
            {state.contract !== null && (
              <div className="info-item">
                <span className="info-label">Contract</span>
                <span className="info-value">{CONTRACT_NAMES[state.contract] ?? state.contract}</span>
              </div>
            )}
            {state.declarer !== null && (
              <div className="info-item">
                <span className="info-label">Declarer</span>
                <span className="info-value">{names[state.declarer]}</span>
              </div>
            )}
            {state.called_king && (
              <div className="info-item">
                <span className="info-label">Called</span>
                <span className="info-value">{state.called_king.label}</span>
              </div>
            )}
            {state.partner !== null && (
              <div className="info-item">
                <span className="info-label">Partner</span>
                <span className="info-value">{names[state.partner]}</span>
              </div>
            )}
            <div className="info-item">
              <span className="info-label">Phase</span>
              <span className="info-value phase-badge">{state.phase.replace(/_/g, ' ')}</span>
            </div>
          </div>

          {/* 4-player table: all hands visible */}
          <div className="table spectator-table">
            {/* Top player (P2) */}
            <div className="table-top">
              <Hand
                cards={state.hands[2] ?? []}
                position="top"
                label={`${names[2]}${roleLabel(state.roles['2'])}`}
              />
            </div>

            {/* Left player (P1) */}
            <div className="table-left">
              <Hand
                cards={state.hands[1] ?? []}
                position="left"
                label={`${names[1]}${roleLabel(state.roles['1'])}`}
              />
            </div>

            {/* Center — current trick or completed trick replay */}
            <div className="table-center">
              {viewingTrick ? (
                <div className="trick-replay">
                  <div className="trick-replay-header">
                    <span>Trick {selectedTrick! + 1}</span>
                    <button className="btn-sm btn-secondary" onClick={() => setSelectedTrick(null)}>
                      Back to live
                    </button>
                  </div>
                  <TrickArea
                    trickCards={viewingTrick.cards}
                    playerNames={names}
                    playerIndex={0}
                  />
                  <div className="trick-replay-winner">
                    Winner: {names[viewingTrick.winner]}
                  </div>
                </div>
              ) : (
                <>
                  {state.phase === 'trick_play' && (
                    <TrickArea
                      trickCards={state.current_trick}
                      playerNames={names}
                      playerIndex={0}
                    />
                  )}

                  {state.phase === 'bidding' && (
                    <div className="spectator-bidding">
                      <h3>Bidding</h3>
                      <div className="bid-history">
                        {state.bids.map((b, i) => (
                          <div key={i} className="bid-entry">
                            <span className="bid-player">{names[b.player]}</span>
                            <span className="bid-value">
                              {b.contract !== null ? (CONTRACT_NAMES[b.contract] ?? b.contract) : 'Pass'}
                            </span>
                          </div>
                        ))}
                      </div>
                      <div className="spectator-waiting">
                        Waiting for {names[state.current_player]}…
                      </div>
                    </div>
                  )}

                  {state.phase === 'king_calling' && (
                    <div className="spectator-waiting-center">
                      <h3>King Calling</h3>
                      <p>Waiting for {names[state.declarer ?? 0]} to call a king…</p>
                    </div>
                  )}

                  {state.phase === 'talon_exchange' && state.talon_groups && (
                    <div className="spectator-talon">
                      <h3>Talon Exchange</h3>
                      <div className="talon-groups">
                        {state.talon_groups.map((group, i) => (
                          <div key={i} className="talon-group">
                            {group.map((card, j) => (
                              <Card key={j} card={card} small />
                            ))}
                          </div>
                        ))}
                      </div>
                      {state.put_down.length > 0 && (
                        <div className="talon-put-down">
                          <h4>Put Down</h4>
                          <div className="put-down-cards">
                            {state.put_down.map((card, j) => (
                              <Card key={j} card={card} small />
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Show talon and put-down info during announcements/trick play */}
                  {(state.phase === 'announcements' || state.phase === 'trick_play') && state.put_down.length > 0 && state.talon_groups && (
                    <div className="spectator-talon-info">
                      <div className="talon-groups talon-groups-mini">
                        {state.talon_groups.map((group, i) => (
                          <div key={i} className="talon-group">
                            {group.map((card, j) => (
                              <Card key={j} card={card} small />
                            ))}
                          </div>
                        ))}
                      </div>
                      <div className="talon-put-down">
                        <span className="put-down-label">Put down:</span>
                        {state.put_down.map((card, j) => (
                          <Card key={j} card={card} small />
                        ))}
                      </div>
                    </div>
                  )}

                  {state.phase === 'finished' && state.scores && (
                    <div className="score-display score-display-full">
                      <h3>Game Over!</h3>

                      {/* Final scores table */}
                      <table className="results-table">
                        <thead>
                          <tr>
                            <th>Player</th>
                            <th>Role</th>
                            <th>Score</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(state.scores).map(([pid, score]) => (
                            <tr key={pid} className={Number(score) > 0 ? 'score-positive' : 'score-negative'}>
                              <td>{names[Number(pid)]}</td>
                              <td>{roleLabel(state.roles[pid]).replace(/[() ]/g, '') || '—'}</td>
                              <td className="score-value">{Number(score) > 0 ? '+' : ''}{score}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>

                      {/* Score justification breakdown */}
                      {state.score_breakdown && (
                        <div className="score-breakdown">
                          <h4>Score Breakdown</h4>
                          {state.score_breakdown.explanation && (
                            <p className="breakdown-explanation">{state.score_breakdown.explanation}</p>
                          )}
                          <table className="breakdown-table">
                            <tbody>
                              {state.score_breakdown.lines.map((line, i) => (
                                <tr key={i}>
                                  <td className="breakdown-label">{line.label}</td>
                                  <td className="breakdown-value">
                                    {line.value !== undefined
                                      ? <span className={line.value > 0 ? 'val-pos' : line.value < 0 ? 'val-neg' : ''}>{line.value > 0 ? '+' : ''}{line.value}</span>
                                      : <span className="val-detail">{line.detail}</span>
                                    }
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      )}

                      {/* Trick-by-trick results table */}
                      {state.trick_summary && state.trick_summary.length > 0 && (
                        <div className="trick-results">
                          <h4>Trick Results</h4>
                          <table className="trick-results-table">
                            <thead>
                              <tr>
                                <th>#</th>
                                <th>Cards</th>
                                <th>Winner</th>
                                <th>Pts</th>
                              </tr>
                            </thead>
                            <tbody>
                              {state.trick_summary.map(trick => (
                                <tr key={trick.trick_num} className={
                                  state.roles[String(trick.winner)] === 'declarer' || state.roles[String(trick.winner)] === 'partner'
                                    ? 'trick-decl' : 'trick-opp'
                                }>
                                  <td>{trick.trick_num}</td>
                                  <td className="trick-cards-cell">
                                    {trick.cards.map((c, j) => (
                                      <span key={j} className={c.player === trick.winner ? 'winning-card' : ''}>
                                        {names[c.player]}: {c.label}{j < trick.cards.length - 1 ? ' · ' : ''}
                                      </span>
                                    ))}
                                  </td>
                                  <td className="trick-winner-cell">{names[trick.winner]}</td>
                                  <td>{trick.card_points}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      )}
                    </div>
                  )}
                </>
              )}
            </div>

            {/* Right player (P3) */}
            <div className="table-right">
              <Hand
                cards={state.hands[3] ?? []}
                position="right"
                label={`${names[3]}${roleLabel(state.roles['3'])}`}
              />
            </div>

            {/* Bottom player (P0) */}
            <div className="table-bottom">
              <Hand
                cards={state.hands[0] ?? []}
                position="bottom"
                label={`${names[0]}${roleLabel(state.roles['0'])}`}
              />
            </div>
          </div>

          {/* Playback Controls */}
          {spectator.totalSteps > 0 && (
            <div className="spectator-playback">
              <div className="playback-controls">
                <button className="btn-secondary btn-sm" onClick={spectator.prevTrick} disabled={spectator.currentIndex === 0}>|◀</button>
                <button className="btn-secondary btn-sm" onClick={spectator.stepBack} disabled={spectator.currentIndex === 0}>◀</button>
                <button className={spectator.isPlaying ? "btn-secondary btn-sm" : "btn-primary btn-sm"} onClick={spectator.togglePlay}>
                  {spectator.isPlaying ? '⏸ Pause' : '▶ Play'}
                </button>
                <button className="btn-secondary btn-sm" onClick={spectator.stepForward} disabled={spectator.currentIndex >= spectator.totalSteps - 1}>▶</button>
                <button className="btn-secondary btn-sm" onClick={spectator.nextTrick} disabled={spectator.currentIndex >= spectator.totalSteps - 1}>▶|</button>
              </div>
              <div className="playback-timeline">
                <input 
                  type="range" 
                  min={0} 
                  max={Math.max(0, spectator.totalSteps - 1)} 
                  value={spectator.currentIndex}
                  onChange={e => spectator.jumpToIndex(Number(e.target.value))}
                  className="timeline-slider"
                />
                <span className="timeline-text">{spectator.currentIndex} / {spectator.totalSteps - 1}</span>
              </div>
            </div>
          )}

          {/* Turn indicator */}
          {state.phase === 'trick_play' && !viewingTrick && (
            <div className="turn-indicator spectator-turn">
              <span>{names[state.current_player]}'s turn</span>
            </div>
          )}
        </div>

        {/* Right panel: log + trick history */}
        <div className="spectator-sidebar">
          {/* Trick history */}
          {state.completed_tricks.length > 0 && (
            <div className="trick-history">
              <div className="trick-history-header">
                <h4>Trick History</h4>
              </div>
              <div className="trick-history-list">
                {state.completed_tricks.map((trick, i) => (
                  <button
                    key={i}
                    className={`trick-history-item ${selectedTrick === i ? 'active' : ''}`}
                    onClick={() => setSelectedTrick(selectedTrick === i ? null : i)}
                  >
                    <span className="trick-num">#{i + 1}</span>
                    <span className="trick-cards-mini">
                      {trick.cards.map(([, c]) => c.label).join(', ')}
                    </span>
                    <span className="trick-winner-badge">{names[trick.winner]}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Game log */}
          <div className="spectator-log">
            <div className="spectator-log-header">
              <h4>Game Log</h4>
            </div>
            <div className="spectator-log-entries">
              {spectator.logEntries.length === 0 && (
                <div className="spectator-log-empty">Waiting for game to start…</div>
              )}
              {spectator.logEntries.map(entry => (
                <div
                  key={entry.id}
                  className={`spectator-log-entry log-${entry.category}`}
                >
                  <span className="log-icon">{CATEGORY_ICONS[entry.category] ?? '•'}</span>
                  <span className="log-message">{entry.message}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function roleLabel(role?: string): string {
  if (!role) return '';
  switch (role) {
    case 'declarer': return ' (D)';
    case 'partner': return ' (P)';
    case 'opponent': return ' (Opp)';
    default: return '';
  }
}

const CATEGORY_ICONS: Record<string, string> = {
  system: '⚙️',
  bid: '🗣️',
  king: '👑',
  talon: '📦',
  play: '🃏',
  trick: '✅',
  score: '🏆',
  announce: '📢',
};
