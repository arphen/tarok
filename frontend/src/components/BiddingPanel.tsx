import React from 'react';
import type { CardData } from '../types/game';
import { CONTRACT_NAMES, SUIT_SYMBOLS } from '../types/game';
import './BiddingPanel.css';

interface BiddingPanelProps {
  phase: string;
  bids: { player: number; contract: number | null }[];
  legalBids?: (number | null)[];
  onBid: (contract: number | null) => void;
  playerNames: string[];
  callableKings?: CardData[];
  onCallKing?: (suit: string) => void;
}

const BID_OPTIONS = [
  { value: 3, label: 'Three', description: 'Pick 3 talon cards (2v2)' },
  { value: 2, label: 'Two', description: 'Pick 2 talon cards (2v2)' },
  { value: 1, label: 'One', description: 'Pick 1 talon card (2v2)' },
  { value: -3, label: 'Solo Three', description: 'Solo, pick 3 talon cards' },
  { value: -2, label: 'Solo Two', description: 'Solo, pick 2 talon cards' },
  { value: -1, label: 'Solo One', description: 'Solo, pick 1 talon card' },
  { value: 0, label: 'Solo', description: 'No talon, play alone (1v3)' },
  { value: -100, label: 'Berač', description: 'Win 0 tricks, solo' },
];

export default function BiddingPanel({
  phase, bids, legalBids, onBid, playerNames, callableKings, onCallKing,
}: BiddingPanelProps) {
  if (phase === 'king_calling' && callableKings && onCallKing) {
    return (
      <div className="bidding-panel">
        <h3>Call a King</h3>
        <p className="bidding-subtitle">Choose which king to call — the holder becomes your partner</p>
        <div className="king-options">
          {callableKings.map(king => (
            <button
              key={king.suit}
              className="btn-gold king-btn"
              onClick={() => onCallKing(king.suit!)}
            >
              {SUIT_SYMBOLS[king.suit!]} King of {king.suit}
            </button>
          ))}
        </div>
      </div>
    );
  }

  if (phase !== 'bidding') return null;

  return (
    <div className="bidding-panel">
      <h3>Bidding</h3>

      {bids.length > 0 && (
        <div className="bid-history">
          {bids.map((bid, i) => (
            <div key={i} className="bid-entry">
              <span className="bid-player">{playerNames[bid.player] || `P${bid.player}`}</span>
              <span className="bid-value">
                {bid.contract !== null ? CONTRACT_NAMES[bid.contract] || `${bid.contract}` : 'Pass'}
              </span>
            </div>
          ))}
        </div>
      )}

      {legalBids && (
        <div className="bid-actions">
          <button className="btn-secondary" data-testid="bid-pass" onClick={() => onBid(null)}>
            Pass
          </button>
          {BID_OPTIONS.filter(opt => legalBids.includes(opt.value)).map(opt => (
            <button key={opt.value} className="btn-primary bid-btn" onClick={() => onBid(opt.value)}>
              <span className="bid-btn-label">{opt.label}</span>
              <span className="bid-btn-desc">{opt.description}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
