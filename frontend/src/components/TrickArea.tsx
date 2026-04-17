import React from 'react';
import Card from './Card';
import type { CardData, TrickCard } from '../types/game';
import './TrickArea.css';

interface TrickAreaProps {
  trickCards: TrickCard[];
  playerNames: string[];
  playerIndex: number;
  getTeamRole?: (idx: number) => 'declarer' | 'defender' | null;
  isSolo?: boolean;
  /** When set, all cards animate toward the winner's position then fade out */
  trickWinner?: number | null;
}

const POSITIONS = ['bottom', 'left', 'top', 'right'] as const;

const TrickArea = React.memo(function TrickArea({ trickCards, playerNames, playerIndex, getTeamRole, isSolo, trickWinner }: TrickAreaProps) {
  // Map absolute player indices to relative positions
  // Player 0 (human) is always at bottom
  function relativePosition(absIdx: number): typeof POSITIONS[number] {
    const rel = (absIdx - playerIndex + 4) % 4;
    return POSITIONS[rel];
  }

  function teamClass(idx: number): string {
    if (!getTeamRole) return '';
    const role = getTeamRole(idx);
    if (role === 'declarer') return isSolo ? 'team-solo' : 'team-declarer';
    if (role === 'defender') return 'team-defender';
    return '';
  }

  const winnerPos = trickWinner != null ? relativePosition(trickWinner) : null;
  const isAnimating = winnerPos != null;

  return (
    <div className="trick-area">
      {trickCards.length === 0 && !isAnimating && (
        <div className="trick-empty">Play a card</div>
      )}
      {trickCards.map(([playerIdx, card], idx) => (
        <div
          key={`${idx}-${playerIdx}-${card.label}`}
          className={[
            'trick-card',
            `trick-card-${relativePosition(playerIdx)}`,
            teamClass(playerIdx),
            isAnimating ? `trick-sweep-to-${winnerPos}` : '',
          ].filter(Boolean).join(' ')}
        >
          <Card card={card} small />
          <span className={`trick-player-name ${teamClass(playerIdx)}`}>{playerNames[playerIdx] || `P${playerIdx}`}</span>
        </div>
      ))}
      {isAnimating && (
        <div className={`trick-winner-flash trick-winner-flash-${winnerPos}`}>
          {playerNames[trickWinner!] ?? `P${trickWinner}`} wins!
        </div>
      )}
    </div>
  );
});

export default TrickArea;
