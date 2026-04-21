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
  teamRole?: 'declarer' | 'defender' | null;
  isSolo?: boolean;
    shadowCard?: CardData | null;
}

  function isShadowCard(card: CardData, shadowCard?: CardData | null): boolean {
    if (!shadowCard) return false;
    return card.card_type === shadowCard.card_type && card.value === shadowCard.value && card.suit === shadowCard.suit;
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

const Hand = React.memo(function Hand({ cards, legalPlays, onCardClick, faceDown, position = 'bottom', label, cardCount, teamRole, isSolo, shadowCard }: HandProps) {
  const count = cardCount ?? cards.length;
  const teamClass = teamRole === 'declarer' ? (isSolo ? 'team-solo' : 'team-declarer') : teamRole === 'defender' ? 'team-defender' : '';

  return (
    <div className={`hand hand-${position} ${teamClass}`}>
      {label && <div className={`hand-label ${teamClass}`}>{label}</div>}
      <div className={`hand-cards hand-horizontal`}>
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
                shadowHint={isShadowCard(card, shadowCard)}
            />
          ))
        )}
      </div>
    </div>
  );
});

export default Hand;
