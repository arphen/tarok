import React, { useState, useEffect, Suspense, lazy } from 'react';
import GameBoard from './components/GameBoard';
import GameLog from './components/GameLog';
import Scoreboard from './components/Scoreboard';
import GameInfoDrawer from './components/GameInfoDrawer';
import { useGame } from './hooks/useGame';
import type { CardData } from './types/game';
import './App.css';

// Lazy-load heavy dashboard components — they pull in Recharts and other large deps
const TrainingLab = lazy(() => import('./components/TrainingLab'));
const CameraAgent = lazy(() => import('./components/CameraAgent'));
const SpectatorView = lazy(() => import('./components/SpectatorView'));
const TournamentBracket = lazy(() => import('./components/TournamentBracket'));
const BotArena = lazy(() => import('./components/BotArena'));
const ArenaCheckpointLeaderboard = lazy(() => import('./components/ArenaCheckpointLeaderboard'));

type Page = 'home' | 'training' | 'lab' | 'play' | 'lobby' | 'camera' | 'spectate' | 'tournament' | 'arena' | 'arenaLeaderboard';

export default function App() {
  const [page, setPage] = useState<Page>('home');
  const [checkpoints, setCheckpoints] = useState<{ filename: string; episode: number; win_rate: number; session?: number; model_name?: string; is_hof?: boolean }[]>([]);
  const [agents, setAgents] = useState<{ id: string; name: string; description: string; category: string }[]>([]);
  const [opponents, setOpponents] = useState<[string, string, string]>(['latest', 'latest', 'latest']);
  const [numRounds, setNumRounds] = useState(1);
  const [playSidebarOpen, setPlaySidebarOpen] = useState(false);
  const [showAllHands, setShowAllHands] = useState(false);
  const game = useGame();

  useEffect(() => {
    fetch('/api/checkpoints')
      .then(r => r.json())
      .then(data => setCheckpoints(data.checkpoints ?? []))
      .catch(() => {});
    fetch('/api/agents')
      .then(r => r.json())
      .then(data => setAgents(data.agents ?? []))
      .catch(() => {});
  }, [page]);

  const handleStartGame = async () => {
    await game.startNewGame(opponents, numRounds);
    setPage('play');
  };

  if (page === 'training') {
    return <Suspense fallback={<div className="app"><p>Loading…</p></div>}><TrainingLab onBack={() => setPage('home')} /></Suspense>;
  }

  if (page === 'lab') {
    return <Suspense fallback={<div className="app"><p>Loading…</p></div>}><TrainingLab onBack={() => setPage('home')} /></Suspense>;
  }

  if (page === 'camera') {
    return <Suspense fallback={<div className="app"><p>Loading…</p></div>}><CameraAgent onBack={() => setPage('home')} /></Suspense>;
  }

  if (page === 'spectate') {
    return <Suspense fallback={<div className="app"><p>Loading…</p></div>}><SpectatorView onBack={() => setPage('home')} checkpoints={checkpoints} /></Suspense>;
  }

  if (page === 'tournament') {
    return <Suspense fallback={<div className="app"><p>Loading…</p></div>}><TournamentBracket onBack={() => setPage('home')} checkpoints={checkpoints} /></Suspense>;
  }

  if (page === 'arena') {
    return <Suspense fallback={<div className="app"><p>Loading…</p></div>}><BotArena onBack={() => setPage('home')} checkpoints={checkpoints} /></Suspense>;
  }

  if (page === 'arenaLeaderboard') {
    return <Suspense fallback={<div className="app"><p>Loading…</p></div>}><ArenaCheckpointLeaderboard onBack={() => setPage('home')} checkpoints={checkpoints} /></Suspense>;
  }

  if (page === 'lobby') {
    const setOpponent = (idx: number, value: string) => {
      setOpponents(prev => {
        const next = [...prev] as [string, string, string];
        next[idx] = value;
        return next;
      });
    };

    const ckptLabel = (filename: string) => {
      const cp = checkpoints.find(c => c.filename === filename);
      if (!cp) return filename;
      if (cp.model_name) return cp.model_name;
      const parts: string[] = [];
      if (cp.session) parts.push(`S${cp.session}`);
      if (cp.episode) parts.push(`Ep ${cp.episode}`);
      if (cp.win_rate) parts.push(`${(cp.win_rate * 100).toFixed(0)}% win`);
      return parts.length ? `${filename} (${parts.join(', ')})` : filename;
    };

    return (
      <div className="app">
        <div className="lobby-page">
          <button className="btn-secondary btn-sm lobby-back" onClick={() => setPage('home')}>← Back</button>
          <h2 className="lobby-title">Game Setup</h2>
          <p className="lobby-subtitle">Choose a model for each AI opponent</p>

          <div className="lobby-opponents">
            {[0, 1, 2].map(i => (
              <div key={i} className="lobby-opponent">
                <label className="lobby-label">AI-{i + 1}</label>
                <select
                  className="lobby-select"
                  value={opponents[i]}
                  onChange={e => setOpponent(i, e.target.value)}
                >
                  {/* Neural network models */}
                  <optgroup label="Neural Network">
                    <option value="latest">Latest trained model</option>
                    {checkpoints.map(cp => (
                      <option key={cp.filename} value={cp.filename}>
                        {ckptLabel(cp.filename)}
                      </option>
                    ))}
                  </optgroup>

                  {/* Heuristic bots (StockŠkis v1-v5+ from registry) */}
                  {agents.filter(a => a.category === 'heuristic').length > 0 && (
                    <optgroup label="Heuristic Bots (StockŠkis)">
                      {agents.filter(a => a.category === 'heuristic').map(a => (
                        <option key={a.id} value={a.id} title={a.description}>
                          {a.name}
                        </option>
                      ))}
                    </optgroup>
                  )}

                  {/* Search-based agents */}
                  {agents.filter(a => a.category === 'search').length > 0 && (
                    <optgroup label="Search">
                      {agents.filter(a => a.category === 'search').map(a => (
                        <option key={a.id} value={a.id} title={a.description}>
                          {a.name}
                        </option>
                      ))}
                    </optgroup>
                  )}

                  {/* Baseline agents */}
                  <optgroup label="Baseline">
                    {agents.filter(a => a.category === 'baseline').map(a => (
                      <option key={a.id} value={a.id} title={a.description}>
                        {a.name}
                      </option>
                    ))}
                    {/* Fallback if agents haven't loaded yet */}
                    {agents.filter(a => a.category === 'baseline').length === 0 && (
                      <option value="random">Random (untrained)</option>
                    )}
                  </optgroup>
                </select>
              </div>
            ))}
          </div>

          <div className="lobby-rounds">
            <label className="lobby-label">Number of Rounds</label>
            <div className="rounds-selector">
              {[1, 3, 5, 10, 20].map(n => (
                <button
                  key={n}
                  className={`rounds-btn ${numRounds === n ? 'rounds-btn-active' : ''}`}
                  onClick={() => setNumRounds(n)}
                >
                  {n}
                </button>
              ))}
              <input
                type="number"
                className="rounds-input"
                min={1}
                max={100}
                value={numRounds}
                onChange={e => setNumRounds(Math.max(1, Math.min(100, Number(e.target.value))))}
              />
            </div>
          </div>

          <button className="btn-gold btn-large lobby-start" onClick={handleStartGame}>
            <span className="btn-icon">🃏</span>
            <span><strong>Start Game</strong></span>
          </button>
        </div>
      </div>
    );
  }

  if (page === 'play') {
    return (
      <div className="app">
        <div className="app-bar">
          <button className="btn-secondary btn-sm" onClick={() => setPage('home')}>← Menu</button>
          {game.gameState.match_info && game.gameState.match_info.total_rounds > 1 && (
            <span className="round-indicator">
              Round {game.gameState.match_info.round_num}/{game.gameState.match_info.total_rounds}
            </span>
          )}
          <label className="speed-control">
            AI Speed
            <input
              type="range"
              min="0"
              max="3"
              step="0.25"
              defaultValue="1"
              onChange={(e) => game.setDelay(Number(e.target.value))}
            />
          </label>
          <label className="reveal-toggle">
            <input
              type="checkbox"
              checked={showAllHands}
              onChange={(e) => {
                setShowAllHands(e.target.checked);
                game.revealHands(e.target.checked);
              }}
            />
            Show hands
          </label>
          <span className="connection-status">
            {game.connected ? '🟢 Connected' : '🔴 Disconnected'}
          </span>
        </div>
        <div className="play-layout">
          <GameInfoDrawer state={game.gameState} />
          <div className="play-main">
            <GameBoard
              state={game.gameState}
              onPlayCard={(card: CardData) => game.playCard(card)}
              onBid={(contract) => game.bid(contract)}
              onCallKing={(suit) => game.callKing(suit)}
              onChooseTalon={(idx) => game.chooseTalon(idx)}
              onDiscard={(cards: CardData[]) => game.discard(cards)}
              trickWinner={game.trickWinner}
              trickWinCards={game.trickWinCards}
            />
          </div>
          <div className={`play-drawer ${playSidebarOpen ? 'drawer-open' : ''}`}>
            <button className="drawer-tab drawer-tab-right" onClick={() => setPlaySidebarOpen(o => !o)}>
              {playSidebarOpen ? '▶' : '◀'} Log
            </button>
            <div className="drawer-panel">
              {game.gameState.match_info && game.gameState.match_info.total_rounds > 1 && (
                <Scoreboard
                  matchInfo={game.gameState.match_info}
                  playerNames={game.gameState.player_names.length > 0 ? game.gameState.player_names : ['You', 'AI-1', 'AI-2', 'AI-3']}
                />
              )}
              <GameLog entries={game.logEntries} />
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Home page
  return (
    <div className="app">
      <div className="home-page">
        <div className="hero">
          <h1 className="title">Slovenian Tarok</h1>
          <p className="subtitle">A 4-player trick-taking card game with AI agents that learn by self-play</p>

          <div className="hero-cards">
            <div className="hero-card hero-card-1">★</div>
            <div className="hero-card hero-card-2">♠</div>
            <div className="hero-card hero-card-3">XXI</div>
          </div>
        </div>

        <div className="home-actions">
          <button className="btn-gold btn-large" onClick={() => setPage('lab')}>
            <span className="btn-icon">🧪</span>
            <span>
              <strong>Training Lab</strong>
              <small>Imitation learning + self-play PPO pipeline</small>
            </span>
          </button>

          <button className="btn-primary btn-large" onClick={() => setPage('lobby')}>
            <span className="btn-icon">🃏</span>
            <span>
              <strong>Play vs AI</strong>
              <small>Choose opponents and play</small>
            </span>
          </button>

          <button className="btn-secondary btn-large" onClick={() => setPage('camera')}>
            <span className="btn-icon">📸</span>
            <span>
              <strong>Camera Agent</strong>
              <small>Get AI advice for a real-world hand</small>
            </span>
          </button>

          <button className="btn-secondary btn-large" onClick={() => setPage('spectate')}>
            <span className="btn-icon">👁️</span>
            <span>
              <strong>Spectate AI vs AI</strong>
              <small>Watch 4 agents play and verify every move</small>
            </span>
          </button>

          <button className="btn-secondary btn-large" onClick={() => setPage('tournament')}>
            <span className="btn-icon">🏆</span>
            <span>
              <strong>Tournament</strong>
              <small>Double-elimination bracket between AI models</small>
            </span>
          </button>

          <button className="btn-secondary btn-large" onClick={() => setPage('arena')}>
            <span className="btn-icon">📊</span>
            <span>
              <strong>Bot Arena</strong>
              <small>Mass-simulate 100K+ games with detailed analytics</small>
            </span>
          </button>

          <button className="btn-secondary btn-large" onClick={() => setPage('arenaLeaderboard')}>
            <span className="btn-icon">📈</span>
            <span>
              <strong>Arena Leaderboard</strong>
              <small>Checkpoint-focused ranking from persisted arena runs</small>
            </span>
          </button>
        </div>

        <div className="rules-summary">
          <h3>How it works</h3>
          <div className="rules-grid">
            <div className="rule-card">
              <div className="rule-icon">📦</div>
              <h4>54 Cards</h4>
              <p>22 Taroks (trumps) + 32 suit cards across 4 suits</p>
            </div>
            <div className="rule-card">
              <div className="rule-icon">👥</div>
              <h4>2v2 Teams</h4>
              <p>Declarer calls a king — the holder becomes their secret partner</p>
            </div>
            <div className="rule-card">
              <div className="rule-icon">🎯</div>
              <h4>12 Tricks</h4>
              <p>Win tricks to collect card points. Team with 36+ points wins</p>
            </div>
            <div className="rule-card">
              <div className="rule-icon">🤖</div>
              <h4>Self-Play RL</h4>
              <p>AI agents learn optimal play through PPO deep reinforcement learning</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
