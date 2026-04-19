import React, { useEffect, useState } from 'react';
import { useSpectator } from '../hooks/useSpectator';
import type { AgentConfig } from '../hooks/useSpectator';
import type { RoundResult, TrickCard } from '../types/game';
import { CONTRACT_NAMES, SUIT_SYMBOLS } from '../types/game';
import Hand from './Hand';
import TrickArea from './TrickArea';
import Card from './Card';
import TalonDrawer from './TalonDrawer';
import Scoreboard from './Scoreboard';
import './SpectatorView.css';
import ModelLeaderboard from './ModelLeaderboard';
import { buildCountingExam } from '../utils/cardCounting';

interface SpectatorViewProps {
  onBack: () => void;
  checkpoints: { filename: string; episode: number; win_rate: number; model_name?: string; is_hof?: boolean }[];
  arenaReplayGameId?: string | null;
}

type AgentType = string;

interface AgentSetup {
  name: string;
  type: AgentType;
  checkpoint: string;
}

interface ReplayOption {
  filename: string;
  created_at: number;
  source: string;
  label: string;
  player_names: string[];
  events: number;
}

const DEFAULT_AGENTS: AgentSetup[] = [
  { name: 'Agent-0', type: 'rl', checkpoint: '' },
  { name: 'Agent-1', type: 'rl', checkpoint: '' },
  { name: 'Agent-2', type: 'rl', checkpoint: '' },
  { name: 'Agent-3', type: 'rl', checkpoint: '' },
];

const SOLO_CONTRACTS = new Set([-3, -2, -1, 0, -100, -101]);

type TeamRole = 'declarer' | 'defender' | null;

export default function SpectatorView({ onBack, checkpoints, arenaReplayGameId }: SpectatorViewProps) {
  const spectator = useSpectator();
  const [agents, setAgents] = useState<AgentSetup[]>(DEFAULT_AGENTS);
  const [stockskisTypes, setStockskisTypes] = useState<string[]>(['stockskis_v2', 'stockskis_v3', 'stockskis_v4', 'stockskis_v5', 'stockskis_m6']);
  const [selectedTrick, setSelectedTrick] = useState<number | null>(null);
  const [replays, setReplays] = useState<ReplayOption[]>([]);
  const [selectedReplay, setSelectedReplay] = useState('');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [numGames, setNumGames] = useState(1);
  const [callerCounts, setCallerCounts] = useState<Record<string, number>>({});
  const [calledCounts, setCalledCounts] = useState<Record<string, number>>({});
  const [roundHistory, setRoundHistory] = useState<RoundResult[]>([]);
  const [showCountingCards, setShowCountingCards] = useState(false);
  const [showMatchDetail, setShowMatchDetail] = useState(false);
  const [showCardBreakdown, setShowCardBreakdown] = useState(false);

  useEffect(() => {
    fetch('/api/replays')
      .then(r => r.json())
      .then(data => {
        const items: ReplayOption[] = data.replays ?? [];
        setReplays(items);
        if (!selectedReplay && items.length > 0) {
          setSelectedReplay(items[0].filename);
        }
      })
      .catch(() => {});
  }, [selectedReplay]);

  useEffect(() => {
    fetch('/api/agents/stockskis')
      .then(r => r.json())
      .then(data => {
        const types = (data?.types ?? []) as unknown;
        if (Array.isArray(types) && types.every(t => typeof t === 'string') && types.length > 0) {
          setStockskisTypes(types);
        }
      })
      .catch(() => {});
  }, []);

  // Auto-connect to arena replay if game_id passed from BotArena
  useEffect(() => {
    if (arenaReplayGameId && spectator.connectToGame) {
      spectator.connectToGame(arenaReplayGameId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [arenaReplayGameId]);

  const isSetup = !spectator.gameId;
  const { state } = spectator;
  const names = state.player_names.length > 0 ? state.player_names : agents.map(a => a.name);

  const handleStart = () => {
    const configs: AgentConfig[] = agents.map(a => ({
      name: a.name,
      type: a.type,
      checkpoint: a.checkpoint || undefined,
    }));
    setCallerCounts({});
    setCalledCounts({});
    setRoundHistory([]);
    spectator.startGame(configs, numGames);
  };

  const handleCountingExamStart = () => {
    const preferredType = stockskisTypes.includes('stockskis_v5')
      ? 'stockskis_v5'
      : stockskisTypes[0] ?? 'stockskis_v5';

    const configs: AgentConfig[] = [0, 1, 2, 3].map(i => ({
      name: `Stockskis-${i}`,
      type: preferredType,
    }));

    setAgents(configs.map(c => ({ name: c.name, type: c.type, checkpoint: '' })));
    setNumGames(1);
    setCallerCounts({});
    setCalledCounts({});
    setRoundHistory([]);
    setShowCountingCards(false);
    spectator.startGame(configs, 1, { autoJumpToEndOnFinish: true });
  };

  // Auto-advance to next game in tournament mode
  const handleNextGame = () => {
    if (spectator.currentGameNum >= spectator.totalGames) return;
    // Accumulate declarer/partner counts and round history from finished game
    if (state.scores) {
      if (state.declarer !== null) {
        setCallerCounts(prev => ({ ...prev, [String(state.declarer)]: (prev[String(state.declarer!)] ?? 0) + 1 }));
      }
      if (state.partner !== null) {
        setCalledCounts(prev => ({ ...prev, [String(state.partner)]: (prev[String(state.partner!)] ?? 0) + 1 }));
      }
      setRoundHistory(prev => [...prev, {
        round: spectator.currentGameNum,
        scores: Object.fromEntries(Object.entries(state.scores!).map(([k, v]) => [k, Number(v)])),
        contract: state.contract,
        declarer: state.declarer,
        partner: state.partner,
      }]);
    }
    const configs: AgentConfig[] = agents.map(a => ({
      name: a.name,
      type: a.type,
      checkpoint: a.checkpoint || undefined,
    }));
    spectator.continueNextGame(configs);
  };

  // Compute live cumulative totals (past games + current game if finished)
  const liveCumulative: Record<string, number> = { ...spectator.cumulativeScores };
  if (state.scores) {
    Object.entries(state.scores).forEach(([pid, score]) => {
      liveCumulative[pid] = (liveCumulative[pid] ?? 0) + Number(score);
    });
  }
  const maxCumulative = Math.max(...Object.values(liveCumulative), -Infinity);

  const updateAgent = (idx: number, patch: Partial<AgentSetup>) => {
    setAgents(prev => prev.map((a, i) => i === idx ? { ...a, ...patch } : a));
  };

  const handleLoadReplay = () => {
    if (!selectedReplay) return;
    spectator.loadReplay(selectedReplay);
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
                    {stockskisTypes.map(t => (
                      <option key={t} value={t}>
                        StockŠkis (heuristic) {t.replace('stockskis_', '').toUpperCase()}
                      </option>
                    ))}
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
                          {c.model_name || `${c.filename} (ep${c.episode}, ${(c.win_rate * 100).toFixed(0)}% WR)`}
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
            <label className="config-field config-field-inline">
              <span>Number of Games</span>
              <input
                type="number"
                min={1}
                max={100}
                value={numGames}
                onChange={e => setNumGames(Math.max(1, Math.min(100, Number(e.target.value))))}
                data-testid="num-games-input"
              />
            </label>
          </div>

          <div className="replay-config">
            <h3>Saved Replays</h3>
            <p className="setup-subtitle">Load a saved spectate game or a sample replay exported from PBT generations.</p>
            <div className="replay-actions">
              <label className="config-field replay-select-field">
                <span>Replay File</span>
                <select value={selectedReplay} onChange={e => setSelectedReplay(e.target.value)} disabled={replays.length === 0}>
                  <option value="">{replays.length === 0 ? 'No replays available yet' : 'Select replay'}</option>
                  {replays.map(replay => (
                    <option key={replay.filename} value={replay.filename}>
                      {replay.label} ({replay.events} events)
                    </option>
                  ))}
                </select>
              </label>
              <button className="btn-secondary btn-large" onClick={handleLoadReplay} disabled={!selectedReplay || spectator.loading}>
                {spectator.loading ? 'Loading…' : 'Load Replay'}
              </button>
            </div>
          </div>

          <button className="btn-gold btn-large" onClick={handleStart} disabled={spectator.loading}>
            {spectator.loading ? 'Starting…' : 'Start Game'}
          </button>

          <button className="btn-secondary btn-large" onClick={handleCountingExamStart} disabled={spectator.loading}>
            {spectator.loading ? 'Starting…' : 'Run Counting Exam (Stockskis)'}
          </button>
        </div>
        <ModelLeaderboard />
      </div>
    );
  }

  // Team role computation (same logic as GameBoard)
  const isSolo = state.contract !== null && SOLO_CONTRACTS.has(state.contract);
  const getTeamRole = (playerIdx: number): TeamRole => {
    if (state.contract === null || state.contract === -99) return null;
    if (state.declarer === null) return null;
    if (playerIdx === state.declarer) return 'declarer';
    if (!isSolo && state.partner_revealed && state.partner === playerIdx) return 'declarer';
    return 'defender';
  };

  const countingTeams = buildCountingExam(state.trick_summary, state.roles, names, state.contract, state.put_down, state.talon_groups);

  // Trick-won animation: when timeline is at a trick_won event, show sweep animation
  const isTrickWon = spectator.currentEventName === 'trick_won';
  const lastCompletedTrick = state.completed_tricks.length > 0
    ? state.completed_tricks[state.completed_tricks.length - 1]
    : null;
  const trickWinner = isTrickWon && lastCompletedTrick ? lastCompletedTrick.winner : null;
  const trickWinCards: TrickCard[] = isTrickWon && lastCompletedTrick ? lastCompletedTrick.cards : [];
  const showTrickAnimation = trickWinner != null && trickWinCards.length > 0;

  // Active spectator view
  const viewingTrick = selectedTrick !== null && selectedTrick < state.completed_tricks.length
    ? state.completed_tricks[selectedTrick]
    : null;

  return (
    <div className="spectator-view">
      <div className="app-bar">
        <button className="btn-secondary btn-sm" onClick={() => { spectator.disconnect(); }}>← Setup</button>
        <span className="spectator-title">{spectator.mode === 'replay' ? 'Replay Viewer' : 'Spectating'}</span>
        {spectator.totalGames > 1 && (
          <span className="game-counter" data-testid="game-counter">
            Game {spectator.currentGameNum}/{spectator.totalGames}
          </span>
        )}
        <span className="connection-status">
          {spectator.mode === 'replay' ? `📼 ${spectator.replayName}` : spectator.connected ? '🟢 Connected' : '🔴 Disconnected'}
        </span>
      </div>

      <div className="spectator-layout">
        {/* Talon drawer — only show after talon is revealed */}
        {state.talon_groups && state.talon_groups.length > 0 && state.put_down.length > 0 && (
          <TalonDrawer
            talonGroups={state.talon_groups}
            putDown={state.put_down}
            side="left"
          />
        )}

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
                teamRole={getTeamRole(2)}
                isSolo={isSolo}
              />
            </div>

            {/* Left player (P1) */}
            <div className="table-left">
              <Hand
                cards={state.hands[1] ?? []}
                position="left"
                label={`${names[1]}${roleLabel(state.roles['1'])}`}
                teamRole={getTeamRole(1)}
                isSolo={isSolo}
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
                    getTeamRole={(idx) => getTeamRole(idx)}
                    isSolo={isSolo}
                  />
                  <div className="trick-replay-winner">
                    Winner: {names[viewingTrick.winner]}
                  </div>
                </div>
              ) : (
                <>
                  {(state.phase === 'trick_play' || showTrickAnimation) && (
                    <TrickArea
                      trickCards={showTrickAnimation ? trickWinCards : state.current_trick}
                      playerNames={names}
                      playerIndex={0}
                      getTeamRole={(idx) => getTeamRole(idx)}
                      isSolo={isSolo}
                      trickWinner={trickWinner}
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

                      {/* Match Data & Metadata */}
                      <div className="match-metadata">
                        <div className="match-metadata-header">
                          <h4>Match Data</h4>
                          <button
                            className="btn-secondary btn-sm"
                            onClick={() => setShowMatchDetail(v => !v)}
                            type="button"
                          >
                            {showMatchDetail ? 'Hide Details' : 'Show Details'}
                          </button>
                        </div>

                        <div className="match-metadata-summary">
                          <div className="metadata-row">
                            <span className="metadata-label">Date &amp; Time</span>
                            <span className="metadata-value">
                              {spectator.logEntries.length > 0
                                ? new Date(spectator.logEntries[0].timestamp).toLocaleString()
                                : '—'}
                            </span>
                          </div>
                          <div className="metadata-row">
                            <span className="metadata-label">Players</span>
                            <span className="metadata-value">{names.join(', ')}</span>
                          </div>
                          <div className="metadata-row">
                            <span className="metadata-label">Dealer</span>
                            <span className="metadata-value">
                              {state.dealer !== null ? names[state.dealer] : '—'}
                            </span>
                          </div>
                          <div className="metadata-row">
                            <span className="metadata-label">Contract</span>
                            <span className="metadata-value">
                              {state.contract !== null ? (CONTRACT_NAMES[state.contract] ?? state.contract) : '—'}
                              {state.declarer !== null && ` by ${names[state.declarer]}`}
                            </span>
                          </div>
                          {state.called_king && (
                            <div className="metadata-row">
                              <span className="metadata-label">Called King</span>
                              <span className="metadata-value">
                                {SUIT_SYMBOLS[state.called_king.suit!] ?? ''} {state.called_king.label}
                                {state.partner !== null && ` → Partner: ${names[state.partner]}`}
                              </span>
                            </div>
                          )}
                        </div>

                        {showMatchDetail && (
                          <div className="match-metadata-detail">
                            {/* Bidding sequence */}
                            {state.bids.length > 0 && (
                              <div className="metadata-section">
                                <h5>Bidding Sequence</h5>
                                <div className="bidding-sequence">
                                  {state.bids.map((b, i) => (
                                    <span key={i} className={`bid-chip ${b.contract !== null ? 'bid-chip-active' : 'bid-chip-pass'}`}>
                                      {names[b.player]}: {b.contract !== null ? (CONTRACT_NAMES[b.contract] ?? b.contract) : 'Pass'}
                                    </span>
                                  ))}
                                </div>
                              </div>
                            )}

                            {/* Talon */}
                            {state.talon_groups && state.talon_groups.length > 0 && (
                              <div className="metadata-section">
                                <h5>Talon</h5>
                                <div className="metadata-talon-groups">
                                  {state.talon_groups.map((group, i) => (
                                    <div key={i} className="metadata-talon-group">
                                      <span className="metadata-talon-group-label">Group {i + 1}:</span>
                                      {group.map((card, j) => (
                                        <span key={j} className="metadata-card-chip">{card.label}</span>
                                      ))}
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}

                            {/* Put down */}
                            {state.put_down.length > 0 && (
                              <div className="metadata-section">
                                <h5>Put Down</h5>
                                <div className="metadata-put-down">
                                  {state.put_down.map((card, j) => (
                                    <span key={j} className="metadata-card-chip">{card.label}</span>
                                  ))}
                                </div>
                              </div>
                            )}

                            {/* Card points breakdown by team */}
                            {state.trick_summary && state.trick_summary.length > 0 && (
                              <div className="metadata-section">
                                <div className="metadata-section-header">
                                  <h5>Card Points Breakdown</h5>
                                  <button
                                    className="btn-secondary btn-xs"
                                    onClick={() => setShowCardBreakdown(v => !v)}
                                    type="button"
                                  >
                                    {showCardBreakdown ? 'Hide' : 'Show Cards'}
                                  </button>
                                </div>
                                {(() => {
                                  const declCards: { label: string; points: number }[] = [];
                                  const oppCards: { label: string; points: number }[] = [];
                                  let declPts = 0;
                                  let oppPts = 0;
                                  for (const trick of state.trick_summary!) {
                                    const winnerRole = state.roles[String(trick.winner)];
                                    const isDeclTeam = winnerRole === 'declarer' || winnerRole === 'partner';
                                    for (const c of trick.cards) {
                                      if (isDeclTeam) {
                                        declCards.push({ label: c.label, points: c.points });
                                        declPts += c.points;
                                      } else {
                                        oppCards.push({ label: c.label, points: c.points });
                                        oppPts += c.points;
                                      }
                                    }
                                  }
                                  // Add put-down cards to declarer
                                  for (const c of state.put_down) {
                                    declCards.push({ label: c.label, points: c.points });
                                    declPts += c.points;
                                  }
                                  return (
                                    <div className="card-points-breakdown">
                                      <div className="cpb-team cpb-decl">
                                        <span className="cpb-team-label">Declarer team</span>
                                        <span className="cpb-team-pts">{declPts} pts ({declCards.length} cards)</span>
                                        {showCardBreakdown && (
                                          <div className="cpb-cards">
                                            {declCards.map((c, i) => (
                                              <span key={i} className="cpb-card" title={`${c.points} pts`}>
                                                {c.label}<sup>{c.points}</sup>
                                              </span>
                                            ))}
                                          </div>
                                        )}
                                      </div>
                                      <div className="cpb-team cpb-opp">
                                        <span className="cpb-team-label">Opponents</span>
                                        <span className="cpb-team-pts">{oppPts} pts ({oppCards.length} cards)</span>
                                        {showCardBreakdown && (
                                          <div className="cpb-cards">
                                            {oppCards.map((c, i) => (
                                              <span key={i} className="cpb-card" title={`${c.points} pts`}>
                                                {c.label}<sup>{c.points}</sup>
                                              </span>
                                            ))}
                                          </div>
                                        )}
                                      </div>
                                    </div>
                                  );
                                })()}
                              </div>
                            )}
                          </div>
                        )}
                      </div>

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
                                      <span key={j} className={`${c.player === trick.winner ? 'winning-card' : ''}${c.player === trick.lead_player ? ' lead-card' : ''}`}>
                                        {c.player === trick.lead_player && <span className="lead-arrow" title="Lead">➤ </span>}
                                        {names[c.player]}: {c.label}
                                        {c.player === trick.winner && <span className="winner-arrow" title="Takes trick"> ✓</span>}
                                        {j < trick.cards.length - 1 ? ' · ' : ''}
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

                      {countingTeams.length > 0 && (
                        <div className="counting-exam" data-testid="counting-exam-panel">
                          <div className="counting-exam-header-row">
                            <h4>Card Counting Examination</h4>
                            <button
                              className="btn-secondary btn-sm"
                              onClick={() => setShowCountingCards(v => !v)}
                              type="button"
                            >
                              {showCountingCards ? 'Hide Cards' : 'Show Cards'}
                            </button>
                          </div>
                          <p className="breakdown-explanation">
                            Cards are grouped in threes. Full groups: sum of card points &minus; 2.
                            Incomplete last group: 2 cards use sum &minus; 1, 1 card uses sum &minus; 0.
                            {' '}Groups marked with * are incomplete final groups.
                            {' '}Grand total: {countingTeams.reduce((s, t) => s + t.total, 0)} (should be ~70).
                          </p>
                          <div className="counting-exam-grid">
                            {countingTeams.map(team => (
                              <div key={team.key} className="counting-team-card">
                                <div className="counting-team-header">
                                  <span className="counting-team-title">{team.label}</span>
                                  <span className="counting-team-players">
                                    {team.key === 'talon' ? 'Talon' : team.players.map(pid => names[pid]).join(' + ')}
                                  </span>
                                </div>

                                {showCountingCards && team.allCards.length > 0 && (
                                  <div className="counting-all-cards">
                                    <span className="counting-all-cards-label">All {team.key === 'talon' ? 'talon' : 'won'} cards ({team.allCards.length}):</span>
                                    <div className="counting-all-cards-list">
                                      {team.allCards.map((c, i) => (
                                        <span key={i} className="counting-card-chip" title={`${c.points} pts`}>{c.label}</span>
                                      ))}
                                    </div>
                                  </div>
                                )}

                                <table className="counting-team-table">
                                  <thead>
                                    <tr>
                                      <th>Group</th>
                                      <th>Cards</th>
                                      <th>Calculation</th>
                                      <th>Value</th>
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {team.groups.map((g, gi) => (
                                      <tr key={gi} className={!g.isComplete ? 'counting-incomplete-group' : ''}>
                                        <td>#{gi + 1}{!g.isComplete && ' *'}</td>
                                        <td className="counting-cards-cell">
                                          {g.cards.map(c => c.label).join(', ')}
                                        </td>
                                        <td className="counting-calc-cell">
                                          {g.isComplete
                                            ? <>({g.cards.map(c => c.points).join(' + ')}) &minus; 2</>
                                            : <>({g.cards.map(c => c.points).join(' + ')}) &minus; {g.cards.length === 2 ? 1 : 0}</>
                                          }
                                        </td>
                                        <td className="counting-value-cell">{g.value}</td>
                                      </tr>
                                    ))}
                                    {team.groups.length === 0 && (
                                      <tr>
                                        <td colSpan={4} className="counting-empty">No cards{team.key === 'talon' ? '' : ' won'}</td>
                                      </tr>
                                    )}
                                  </tbody>
                                  <tfoot>
                                    <tr>
                                      <td colSpan={3}>Total ({team.allCards.length} cards, {team.groups.length} groups)</td>
                                      <td className="counting-value-cell">{team.total}</td>
                                    </tr>
                                  </tfoot>
                                </table>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Tournament: cumulative scores + next game button */}
                  {state.phase === 'finished' && spectator.totalGames > 1 && (
                    <div className="tournament-controls" data-testid="tournament-controls">
                      <div className="cumulative-scores">
                        <h4>Tournament Standings ({spectator.gamesPlayed + 1} game{spectator.gamesPlayed + 1 !== 1 ? 's' : ''} played)</h4>
                        <table className="results-table">
                          <thead><tr><th>Player</th><th>Total</th></tr></thead>
                          <tbody>
                            {Object.entries(liveCumulative)
                              .sort(([, a], [, b]) => b - a)
                              .map(([pid, score]) => (
                              <tr key={pid} className={`${score > 0 ? 'score-positive' : 'score-negative'} ${score === maxCumulative ? 'scoreboard-leader-row' : ''}`}>
                                <td>{score === maxCumulative && '👑 '}{names[Number(pid)]}</td>
                                <td className="score-value">{score > 0 ? '+' : ''}{score}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      {spectator.currentGameNum < spectator.totalGames && (
                        <button className="btn-gold btn-sm" onClick={handleNextGame} data-testid="next-game-btn">
                          Next Game ({spectator.currentGameNum + 1}/{spectator.totalGames})
                        </button>
                      )}
                      {spectator.currentGameNum >= spectator.totalGames && (
                        <div className="tournament-finished">
                          🏆 Tournament Complete! Winner: {names[Number(Object.entries(liveCumulative).sort(([, a], [, b]) => b - a)[0]?.[0] ?? 0)]}
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
                teamRole={getTeamRole(3)}
                isSolo={isSolo}
              />
            </div>

            {/* Bottom player (P0) */}
            <div className="table-bottom">
              <Hand
                cards={state.hands[0] ?? []}
                position="bottom"
                label={`${names[0]}${roleLabel(state.roles['0'])}`}
                teamRole={getTeamRole(0)}
                isSolo={isSolo}
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

        {/* Right drawer: log + trick history */}
        <div className={`spectator-drawer ${sidebarOpen ? 'drawer-open' : ''}`}>
          <button className="drawer-tab drawer-tab-right" onClick={() => setSidebarOpen(o => !o)}>
            {sidebarOpen ? '▶' : '◀'} Log
          </button>
          <div className="drawer-panel">
            {/* Scoreboard (same as play vs AI) */}
            <Scoreboard
              matchInfo={{
                round_num: spectator.currentGameNum,
                total_rounds: spectator.totalGames,
                cumulative_scores: liveCumulative,
                caller_counts: callerCounts,
                called_counts: calledCounts,
                round_history: roundHistory,
              }}
              playerNames={names}
            />

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
      <ModelLeaderboard />
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

// Contracts where no talon exchange happens (talon sits aside) — used by buildCountingExam in ../utils/cardCounting
