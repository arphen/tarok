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
  { value: 1, label: 'Three', description: 'Pick 3 talon cards (2v2)' },
  { value: 2, label: 'Two', description: 'Pick 2 talon cards (2v2)' },
  { value: 3, label: 'One', description: 'Pick 1 talon card (2v2)' },
  { value: 4, label: 'Solo Three', description: 'Solo, pick 3 talon cards' },
  { value: 5, label: 'Solo Two', description: 'Solo, pick 2 talon cards' },
  { value: 6, label: 'Solo One', description: 'Solo, pick 1 talon card' },
  { value: 7, label: 'Solo', description: 'No talon, play alone (1v3)' },
  { value: 8, label: 'Berač', description: 'Win 0 tricks, solo' },
];

function toRustBidId(v: number | null): number | null {
  if (v === null) return null;
  const pyToRust: Record<number, number> = {
    3: 1,
    2: 2,
    1: 3,
    [-3]: 4,
    [-2]: 5,
    [-1]: 6,
    0: 7,
    [-100]: 8,
  };
  return pyToRust[v] ?? v;
}

const BiddingPanel = React.memo(function BiddingPanel({
  phase, bids, legalBids, onBid, playerNames, callableKings, onCallKing,
}: BiddingPanelProps) {
  const usesLegacyPyIds = !!legalBids?.some(v => typeof v === 'number' && (v < 0 || v > 8));
  const normalizedLegal = legalBids
    ? (usesLegacyPyIds ? legalBids.map(toRustBidId) : legalBids)
    : [];
  const visibleBidOptions = legalBids
    ? BID_OPTIONS.filter(opt => normalizedLegal.includes(opt.value))
    : [];
  const compactMode = visibleBidOptions.length >= 5;

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
        <div className={`bid-actions ${compactMode ? 'bid-actions-compact' : ''}`}>
          {visibleBidOptions.map(opt => (
            <button key={opt.value} className={`btn-primary bid-btn ${compactMode ? 'bid-btn-compact' : ''}`} onClick={() => onBid(opt.value)}>
              <span className="bid-btn-label">{opt.label}</span>
              {!compactMode && <span className="bid-btn-desc">{opt.description}</span>}
            </button>
          ))}
          <button className="btn-secondary bid-pass-btn" data-testid="bid-pass" onClick={() => onBid(null)}>
            Pass
          </button>
        </div>
      )}
    </div>
  );
}
);

export default BiddingPanel;
