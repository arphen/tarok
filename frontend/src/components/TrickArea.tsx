import React from 'react';
import Card from './Card';
import type { CardData, TrickCard } from '../types/game';
import './TrickArea.css';

interface TrickAreaProps {
  trickCards: TrickCard[];
  playerNames: string[];
  playerIndex: number;
}

const POSITIONS = ['bottom', 'left', 'top', 'right'] as const;

export default function TrickArea({ trickCards, playerNames, playerIndex }: TrickAreaProps) {
  // Map absolute player indices to relative positions
  // Player 0 (human) is always at bottom
  function relativePosition(absIdx: number): typeof POSITIONS[number] {
    const rel = (absIdx - playerIndex + 4) % 4;
    return POSITIONS[rel];
  }

  return (
    <div className="trick-area">
      {trickCards.length === 0 && (
        <div className="trick-empty">Play a card</div>
      )}
      {trickCards.map(([playerIdx, card]) => (
        <div key={`${playerIdx}-${card.label}`} className={`trick-card trick-card-${relativePosition(playerIdx)}`}>
          <Card card={card} small />
          <span className="trick-player-name">{playerNames[playerIdx] || `P${playerIdx}`}</span>
        </div>
      ))}
    </div>
  );
}
