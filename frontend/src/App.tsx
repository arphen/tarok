import React, { useState, useEffect, Suspense, lazy } from 'react';
import GameBoard from './components/GameBoard';
import GameLog from './components/GameLog';
import Scoreboard from './components/Scoreboard';
import GameInfoDrawer from './components/GameInfoDrawer';
import { useGame } from './hooks/useGame';
import type { CardData } from './types/game';
import './App.css';

// Lazy-load heavy dashboard components — they pull in Recharts and other large deps
const SpectatorView = lazy(() => import('./components/SpectatorView'));
const TournamentBracket = lazy(() => import('./components/TournamentBracket'));
const BotArena = lazy(() => import('./components/BotArena'));

type Page = 'home' | 'training' | 'lab' | 'play' | 'lobby' | 'spectate' | 'tournament' | 'arena';

export default function App() {
  const [page, setPage] = useState<Page>('home');
  const [checkpoints, setCheckpoints] = useState<{ filename: string; episode: number; win_rate: number; session?: number; model_name?: string; is_hof?: boolean; model_arch?: string; variant?: string }[]>([]);
  const [agents, setAgents] = useState<{ id: string; name: string; description: string; category: string }[]>([]);
  const [variant, setVariant] = useState<'four_player' | 'three_player'>(() => {
    const stored = typeof window !== 'undefined' ? window.localStorage.getItem('tarok.variant') : null;
    return stored === 'three_player' ? 'three_player' : 'four_player';
  });
  const [opponents, setOpponents] = useState<string[]>(['latest', 'latest', 'latest']);
  const [numRounds, setNumRounds] = useState(1);
  const [playSidebarOpen, setPlaySidebarOpen] = useState(false);
  const [showAllHands, setShowAllHands] = useState(false);
  const [arenaReplayGameId, setArenaReplayGameId] = useState<string | null>(null);
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
    const nOpp = variant === 'three_player' ? 2 : 3;
    const sized = opponents.slice(0, nOpp);
    while (sized.length < nOpp) sized.push('latest');
    await game.startNewGame(sized, numRounds, undefined, variant);
    setPage('play');
  };

  if (page === 'spectate') {
    const replayId = arenaReplayGameId;
    return <Suspense fallback={<div className="app"><p>Loading…</p></div>}><SpectatorView onBack={() => { setArenaReplayGameId(null); setPage('home'); }} checkpoints={checkpoints} arenaReplayGameId={replayId} /></Suspense>;
  }

  if (page === 'tournament') {
    return <Suspense fallback={<div className="app"><p>Loading…</p></div>}><TournamentBracket onBack={() => setPage('home')} checkpoints={checkpoints} /></Suspense>;
  }

  if (page === 'arena') {
    return <Suspense fallback={<div className="app"><p>Loading…</p></div>}><BotArena onBack={() => setPage('home')} checkpoints={checkpoints} onReplayGame={(gameId) => { setArenaReplayGameId(gameId); setPage('spectate'); }} /></Suspense>;
  }

  if (page === 'lobby') {
    const nOpp = variant === 'three_player' ? 2 : 3;
    const setOpponent = (idx: number, value: string) => {
      setOpponents(prev => {
        const next = [...prev];
        while (next.length < nOpp) next.push('latest');
        next[idx] = value;
        return next;
      });
    };

    const updateVariant = (v: 'four_player' | 'three_player') => {
      setVariant(v);
      try { window.localStorage.setItem('tarok.variant', v); } catch { /* ignore */ }
      // Reset any opponent slots that point to checkpoints from the wrong variant.
      setOpponents(prev => {
        const next: string[] = [];
        const slots = v === 'three_player' ? 2 : 3;
        for (let i = 0; i < slots; i++) {
          const cur = prev[i] ?? 'latest';
          const cp = checkpoints.find(c => c.filename === cur);
          if (cp && cp.variant && cp.variant !== v) {
            next.push('latest');
          } else {
            next.push(cur);
          }
        }
        return next;
      });
    };

    const isCheckpointDisabled = (cp: { variant?: string }) =>
      Boolean(cp.variant && cp.variant !== variant);

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
          <p className="lobby-subtitle">Choose a variant and a model for each AI opponent</p>

          <div className="lobby-variant" style={{ display: 'flex', gap: 16, justifyContent: 'center', marginBottom: 16 }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
              <input
                type="radio"
                name="variant"
                value="four_player"
                checked={variant === 'four_player'}
                onChange={() => updateVariant('four_player')}
              />
              <span>4-player Tarok</span>
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
              <input
                type="radio"
                name="variant"
                value="three_player"
                checked={variant === 'three_player'}
                onChange={() => updateVariant('three_player')}
              />
              <span>3-player Tarok</span>
            </label>
          </div>

          <div className="lobby-opponents">
            {Array.from({ length: nOpp }).map((_, i) => (
              <div key={i} className="lobby-opponent">
                <label className="lobby-label">AI-{i + 1}</label>
                <select
                  className="lobby-select"
                  value={opponents[i] ?? 'latest'}
                  onChange={e => setOpponent(i, e.target.value)}
                >
                  {/* Neural network models */}
                  <optgroup label="Neural Network">
                    <option value="latest">Latest trained model</option>
                    {checkpoints.map(cp => {
                      const disabled = isCheckpointDisabled(cp);
                      const trainedFor = cp.variant === 'three_player' ? '3-player' : '4-player';
                      return (
                        <option
                          key={cp.filename}
                          value={cp.filename}
                          disabled={disabled}
                          title={disabled ? `Trained for ${trainedFor}; switch variant to use` : undefined}
                        >
                          {ckptLabel(cp.filename)}{disabled ? ` — ${trainedFor} only` : ''}
                        </option>
                      );
                    })}
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
              onPlayAgain={() => game.startNewGame(opponents, numRounds)}
              trickWinner={game.trickWinner}
              trickWinCards={game.trickWinCards}
            />
          </div>
          <div className={`play-drawer ${playSidebarOpen ? 'drawer-open' : ''}`}>
            <button className="drawer-tab drawer-tab-right" onClick={() => setPlaySidebarOpen(o => !o)}>
              {playSidebarOpen ? '▶' : '◀'} Log
            </button>
            <div className="drawer-panel">
              {game.gameState.match_info && (
                <Scoreboard
                  matchInfo={game.gameState.match_info}
                  playerNames={game.gameState.player_names.length > 0 ? game.gameState.player_names : ['You', 'AI-1', 'AI-2', 'AI-3']}
                />
              )}
              {game.completedTricks.length > 0 && (
                <div className="trick-history">
                  <div className="trick-history-header">
                    <h4>Trick History</h4>
                  </div>
                  <div className="trick-history-list">
                    {game.completedTricks.map((trick, i) => {
                      const playerNames = game.gameState.player_names.length > 0 ? game.gameState.player_names : ['You', 'AI-1', 'AI-2', 'AI-3'];
                      return (
                        <div key={i} className="trick-history-item">
                          <span className="trick-num">#{i + 1}</span>
                          <span className="trick-cards-mini">
                            {trick.cards.map(([, c]) => c.label).join(', ')}
                          </span>
                          <span className="trick-winner-badge">{playerNames[trick.winner]}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
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
          <button className="btn-primary btn-large" onClick={() => setPage('lobby')}>
            <span className="btn-icon">🃏</span>
            <span>
              <strong>Play vs AI</strong>
              <small>Choose opponents and play</small>
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
