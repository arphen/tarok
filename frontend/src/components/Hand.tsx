import React from 'react';
import Card from './Card';
import type { CardData } from '../types/game';
import './Hand.css';

interface HandProps {
  cards: CardData[];
  legalPlays?: CardData[];
  onCardClick?: (card: CardData) => void;
  faceDown?: boolean;
  position?: 'bottom' | 'top' | 'left' | 'right';
  label?: string;
  cardCount?: number;
}

function cardKey(c: CardData): string {
  return `${c.card_type}-${c.value}-${c.suit ?? 'none'}`;
}

function isLegal(card: CardData, legalPlays?: CardData[]): boolean {
  if (!legalPlays) return true;
  return legalPlays.some(
    lp => lp.card_type === card.card_type && lp.value === card.value && lp.suit === card.suit
  );
}

export default function Hand({ cards, legalPlays, onCardClick, faceDown, position = 'bottom', label, cardCount }: HandProps) {
  const isHorizontal = position === 'bottom' || position === 'top';
  const count = cardCount ?? cards.length;

  return (
    <div className={`hand hand-${position}`}>
      {label && <div className="hand-label">{label}</div>}
      <div className={`hand-cards ${isHorizontal ? 'hand-horizontal' : 'hand-vertical'}`}>
        {faceDown ? (
          Array.from({ length: count }).map((_, i) => (
            <Card key={i} card={{ card_type: 'tarok', value: 0, suit: null, label: '', points: 0 }} faceDown small />
          ))
        ) : (
          cards.map(card => (
            <Card
              key={cardKey(card)}
              card={card}
              onClick={onCardClick && isLegal(card, legalPlays) ? () => onCardClick(card) : undefined}
              disabled={legalPlays !== undefined && !isLegal(card, legalPlays)}
              highlighted={legalPlays !== undefined && isLegal(card, legalPlays)}
            />
          ))
        )}
      </div>
    </div>
  );
}
