import React from 'react';
import type { CardData } from '../types/game';
import { getCardImageUrl } from '../utils/cardImages';
import './Card.css';

interface CardProps {
  card: CardData;
  onClick?: () => void;
  disabled?: boolean;
  highlighted?: boolean;
  faceDown?: boolean;
  small?: boolean;
}

const SUIT_SYMBOLS: Record<string, string> = {
  hearts: '♥',
  diamonds: '♦',
  clubs: '♣',
  spades: '♠',
};

const TAROK_NUMERALS: Record<number, string> = {
  1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V',
  6: 'VI', 7: 'VII', 8: 'VIII', 9: 'IX', 10: 'X',
  11: 'XI', 12: 'XII', 13: 'XIII', 14: 'XIV', 15: 'XV',
  16: 'XVI', 17: 'XVII', 18: 'XVIII', 19: 'XIX', 20: 'XX',
  21: 'XXI', 22: 'Škis',
};

function isTrula(card: CardData): boolean {
  return card.card_type === 'tarok' && [1, 21, 22].includes(card.value);
}

export default function Card({ card, onClick, disabled, highlighted, faceDown, small }: CardProps) {
  if (faceDown) {
    return (
      <div className={`card card-back ${small ? 'card-small' : ''}`}>
        <div className="card-back-pattern">
          <div className="card-back-inner">✦</div>
        </div>
      </div>
    );
  }

  const isTarok = card.card_type === 'tarok';
  const isRed = card.suit === 'hearts' || card.suit === 'diamonds';
  const trula = isTrula(card);

  const imageUrl = getCardImageUrl(card);
  const hasFallback = !imageUrl;

  const classes = [
    'card',
    isTarok ? 'card-tarok' : 'card-suit',
    isRed ? 'card-red' : 'card-black',
    trula ? 'card-trula' : '',
    hasFallback ? 'card-fallback' : '',
    highlighted ? 'card-highlighted' : '',
    disabled ? 'card-disabled' : '',
    onClick && !disabled ? 'card-clickable' : '',
    small ? 'card-small' : '',
  ].filter(Boolean).join(' ');

  return (
    <div className={classes} onClick={!disabled && onClick ? onClick : undefined} data-testid={`card-${card.card_type}-${card.value}-${card.suit ?? 'none'}`}>
      {imageUrl ? (
        <img src={imageUrl} alt={card.label} className="card-image" />
      ) : isTarok ? (
        <div className="card-content card-content-tarok">
          <div className="card-corner card-corner-top">
            <span className="card-numeral">{TAROK_NUMERALS[card.value]}</span>
          </div>
          <div className="card-center">
            <div className="tarok-emblem">
              {trula ? (
                <span className="tarok-star">★</span>
              ) : (
                <span className="tarok-number">{TAROK_NUMERALS[card.value]}</span>
              )}
            </div>
          </div>
          <div className="card-corner card-corner-bottom">
            <span className="card-numeral">{TAROK_NUMERALS[card.value]}</span>
          </div>
          <div className="card-points">{card.points}pt</div>
        </div>
      ) : (
        <div className="card-content card-content-suit">
          <div className="card-corner card-corner-top">
            <span className="card-rank">{card.label.replace(/[♥♦♣♠]/g, '')}</span>
            <span className="card-suit-symbol">{SUIT_SYMBOLS[card.suit!]}</span>
          </div>
          <div className="card-center">
            <span className="suit-large">{SUIT_SYMBOLS[card.suit!]}</span>
          </div>
          <div className="card-corner card-corner-bottom">
            <span className="card-rank">{card.label.replace(/[♥♦♣♠]/g, '')}</span>
            <span className="card-suit-symbol">{SUIT_SYMBOLS[card.suit!]}</span>
          </div>
          <div className="card-points">{card.points}pt</div>
        </div>
      )}
    </div>
  );
}
