import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import BiddingPanel from '../components/BiddingPanel';

const playerNames = ['You', 'AI-1', 'AI-2', 'AI-3'];

describe('BiddingPanel', () => {
  it('renders nothing when phase is not bidding or king_calling', () => {
    const { container } = render(
      <BiddingPanel phase="trick_play" bids={[]} onBid={vi.fn()} playerNames={playerNames} />
    );
    expect(container.innerHTML).toBe('');
  });

  it('renders bidding title during bidding phase', () => {
    render(
      <BiddingPanel phase="bidding" bids={[]} onBid={vi.fn()} playerNames={playerNames} />
    );
    expect(screen.getByText('Bidding')).toBeInTheDocument();
  });

  it('shows bid history', () => {
    const bids = [
      { player: 1, contract: 3 },
      { player: 2, contract: null },
    ];
    render(
      <BiddingPanel phase="bidding" bids={bids} onBid={vi.fn()} playerNames={playerNames} />
    );
    expect(screen.getByText('AI-1')).toBeInTheDocument();
    expect(screen.getByText('Pass')).toBeInTheDocument();
  });

  it('shows legal bid buttons when it is my turn', () => {
    const onBid = vi.fn();
    render(
      <BiddingPanel
        phase="bidding"
        bids={[]}
        legalBids={[null, 1, 2]}
        onBid={onBid}
        playerNames={playerNames}
      />
    );
    expect(screen.getByTestId('bid-pass')).toBeInTheDocument();
    expect(screen.getByText('Three')).toBeInTheDocument();
    expect(screen.getByText('Two')).toBeInTheDocument();
  });

  it('calls onBid with null when pass is clicked', () => {
    const onBid = vi.fn();
    render(
      <BiddingPanel phase="bidding" bids={[]} legalBids={[null, 1]} onBid={onBid} playerNames={playerNames} />
    );
    fireEvent.click(screen.getByTestId('bid-pass'));
    expect(onBid).toHaveBeenCalledWith(null);
  });

  it('calls onBid with contract value when bid button clicked', () => {
    const onBid = vi.fn();
    render(
      <BiddingPanel phase="bidding" bids={[]} legalBids={[null, 1]} onBid={onBid} playerNames={playerNames} />
    );
    fireEvent.click(screen.getByText('Three'));
    expect(onBid).toHaveBeenCalledWith(1);
  });

  it('shows king calling UI during king_calling phase', () => {
    const onCallKing = vi.fn();
    const kings = [
      { card_type: 'suit' as const, value: 8, suit: 'hearts' as const, label: '♥K', points: 5 },
      { card_type: 'suit' as const, value: 8, suit: 'spades' as const, label: '♠K', points: 5 },
    ];
    render(
      <BiddingPanel
        phase="king_calling"
        bids={[]}
        onBid={vi.fn()}
        playerNames={playerNames}
        callableKings={kings}
        onCallKing={onCallKing}
      />
    );
    expect(screen.getByText('Call a King')).toBeInTheDocument();
    const heartBtn = screen.getByText(/♥ King of hearts/);
    fireEvent.click(heartBtn);
    expect(onCallKing).toHaveBeenCalledWith('hearts');
  });
});
