import React, { useState } from 'react';
import type { GameState } from '../types/game';
import { SUIT_SYMBOLS } from '../types/game';
import './GameInfoDrawer.css';

interface GameInfoDrawerProps {
  state: GameState;
}

export default function GameInfoDrawer({ state }: GameInfoDrawerProps) {
  const [pinned, setPinned] = useState(false);
  const tracker = state.card_tracker;
  const names = state.player_names.length > 0 ? state.player_names : ['You', 'AI-1', 'AI-2', 'AI-3'];

  if (!tracker) {
    return (
      <div className={`game-info-drawer ${pinned ? 'drawer-pinned' : ''}`}>
        <div className="drawer-tab" onClick={() => setPinned(!pinned)}>
          <span className="drawer-tab-icon">📋</span>
        </div>
        <div className="drawer-content">
          <div className="drawer-header">
            <h3>Game Info</h3>
            <button className="drawer-pin" onClick={() => setPinned(!pinned)} title={pinned ? 'Unpin' : 'Pin open'}>
              {pinned ? '📌' : '📍'}
            </button>
          </div>
          <p className="drawer-empty">Card tracking available during trick play</p>
        </div>
      </div>
    );
  }

  const { remaining_by_group, remaining_count, player_info } = tracker;

  const suitGroups: [string, string][] = [
    ['taroks', '🃏'],
    ['hearts', SUIT_SYMBOLS.hearts],
    ['diamonds', SUIT_SYMBOLS.diamonds],
    ['clubs', SUIT_SYMBOLS.clubs],
    ['spades', SUIT_SYMBOLS.spades],
  ];

  return (
    <div className={`game-info-drawer ${pinned ? 'drawer-pinned' : ''}`} data-testid="game-info-drawer">
      <div className="drawer-tab" onClick={() => setPinned(!pinned)}>
        <span className="drawer-tab-icon">📋</span>
        <span className="drawer-tab-count">{remaining_count}</span>
      </div>

      <div className="drawer-content">
        <div className="drawer-header">
          <h3>Game Info</h3>
          <button className="drawer-pin" onClick={() => setPinned(!pinned)} title={pinned ? 'Unpin' : 'Pin open'}>
            {pinned ? '📌' : '📍'}
          </button>
        </div>

        <div className="drawer-scroll">
          {/* Remaining cards */}
          <section className="drawer-section">
            <h4>Remaining Cards ({remaining_count})</h4>
            {suitGroups.map(([key, symbol]) => {
              const cards = remaining_by_group[key] ?? [];
              if (cards.length === 0) return null;
              return (
                <div key={key} className="remaining-group">
                  <span className="remaining-suit">{symbol}</span>
                  <span className="remaining-cards">
                    {cards.map((c, i) => (
                      <span key={i} className="remaining-card" title={c.label}>
                        {c.card_type === 'tarok' ? c.value : c.label.split(' ')[0]}
                      </span>
                    ))}
                  </span>
                  <span className="remaining-count">({cards.length})</span>
                </div>
              );
            })}
          </section>

          {/* Player info */}
          <section className="drawer-section">
            <h4>Player Intel</h4>
            {[1, 2, 3].map(p => {
              const info = player_info[String(p)];
              if (!info) return null;
              return (
                <div key={p} className="player-intel">
                  <div className="intel-name">{names[p]}</div>
                  {info.void_suits.length > 0 && (
                    <div className="intel-row">
                      <span className="intel-label">Void:</span>
                      <span className="intel-voids">
                        {info.void_suits.map(s => SUIT_SYMBOLS[s] ?? s).join(' ')}
                      </span>
                    </div>
                  )}
                  {info.taroks_played_count > 0 && (
                    <div className="intel-row">
                      <span className="intel-label">Taroks:</span>
                      <span className="intel-value">
                        {info.taroks_played_count} played
                        {info.highest_tarok !== null && ` (high: ${info.highest_tarok}`}
                        {info.lowest_tarok !== null && `, low: ${info.lowest_tarok})`}
                      </span>
                    </div>
                  )}
                  {info.void_suits.length === 0 && info.taroks_played_count === 0 && (
                    <div className="intel-row intel-none">No intel yet</div>
                  )}
                </div>
              );
            })}
          </section>
        </div>
      </div>
    </div>
  );
}
