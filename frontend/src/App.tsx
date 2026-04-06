import React, { useState, useEffect } from 'react';
import GameBoard from './components/GameBoard';
import GameLog from './components/GameLog';
import TrainingDashboard from './components/TrainingDashboard';
import EvoDashboard from './components/EvoDashboard';
import BreedingDashboard from './components/BreedingDashboard';
import CameraAgent from './components/CameraAgent';
import SpectatorView from './components/SpectatorView';
import { useGame } from './hooks/useGame';
import type { CardData } from './types/game';
import './App.css';

type Page = 'home' | 'training' | 'play' | 'lobby' | 'camera' | 'spectate' | 'evolve' | 'breed';

export default function App() {
  const [page, setPage] = useState<Page>('home');
  const [checkpoints, setCheckpoints] = useState<{ filename: string; episode: number; win_rate: number; session?: number }[]>([]);
  const [opponents, setOpponents] = useState<[string, string, string]>(['latest', 'latest', 'latest']);
  const game = useGame();

  useEffect(() => {
    fetch('/api/checkpoints')
      .then(r => r.json())
      .then(data => setCheckpoints(data.checkpoints ?? []))
      .catch(() => {});
  }, []);

  const handleStartGame = async () => {
    await game.startNewGame(opponents);
    setPage('play');
  };

  if (page === 'training') {
    return <TrainingDashboard onBack={() => setPage('home')} />;
  }

  if (page === 'evolve') {
    return <EvoDashboard onBack={() => setPage('home')} />;
  }

  if (page === 'breed') {
    return <BreedingDashboard onBack={() => setPage('home')} />;
  }

  if (page === 'camera') {
    return <CameraAgent onBack={() => setPage('home')} />;
  }

  if (page === 'spectate') {
    return <SpectatorView onBack={() => setPage('home')} checkpoints={checkpoints} />;
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
                  <option value="latest">Latest trained model</option>
                  <option value="random">Random (untrained)</option>
                  {checkpoints.map(cp => (
                    <option key={cp.filename} value={cp.filename}>
                      {ckptLabel(cp.filename)}
                    </option>
                  ))}
                </select>
              </div>
            ))}
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
          <span className="connection-status">
            {game.connected ? '🟢 Connected' : '🔴 Disconnected'}
          </span>
        </div>
        <div className="play-layout">
          <div className="play-main">
            <GameBoard
              state={game.gameState}
              onPlayCard={(card: CardData) => game.playCard(card)}
              onBid={(contract) => game.bid(contract)}
              onCallKing={(suit) => game.callKing(suit)}
              onChooseTalon={(idx) => game.chooseTalon(idx)}
              onDiscard={(cards: CardData[]) => game.discard(cards)}
            />
          </div>
          <GameLog entries={game.logEntries} />
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
          <button className="btn-gold btn-large" onClick={() => setPage('training')}>
            <span className="btn-icon">🧠</span>
            <span>
              <strong>Train AI Agents</strong>
              <small>Watch agents learn through self-play</small>
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

          <button className="btn-secondary btn-large" onClick={() => setPage('evolve')}>
            <span className="btn-icon">🧬</span>
            <span>
              <strong>Evolve Hyperparams</strong>
              <small>Evolutionary optimization of training parameters</small>
            </span>
          </button>

          <button className="btn-secondary btn-large" onClick={() => setPage('breed')}>
            <span className="btn-icon">🧫</span>
            <span>
              <strong>Breed Behaviors</strong>
              <small>Evolve agent personality traits via behavioral breeding</small>
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
