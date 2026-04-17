import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import TrickArea from '../components/TrickArea';
import type { CardData, TrickCard } from '../types/game';

const playerNames = ['You', 'AI-1', 'AI-2', 'AI-3'];

const makeCard = (type: 'tarok' | 'suit', value: number, suit: CardData['suit'] = null): CardData => ({
  card_type: type,
  value,
  suit,
  label: type === 'tarok' ? `T${value}` : `${suit}${value}`,
  points: 1,
});

describe('TrickArea', () => {
  it('shows empty prompt when no cards played', () => {
    render(<TrickArea trickCards={[]} playerNames={playerNames} playerIndex={0} />);
    expect(screen.getByText('Play a card')).toBeInTheDocument();
  });

  it('renders played cards with player names', () => {
    const trickCards: TrickCard[] = [
      [0, makeCard('tarok', 14)],
      [1, makeCard('suit', 6, 'hearts')],
    ];
    render(<TrickArea trickCards={trickCards} playerNames={playerNames} playerIndex={0} />);
    expect(screen.getByText('You')).toBeInTheDocument();
    expect(screen.getByText('AI-1')).toBeInTheDocument();
  });

  it('shows winner flash during animation', () => {
    const trickCards: TrickCard[] = [
      [0, makeCard('tarok', 14)],
      [1, makeCard('suit', 6, 'hearts')],
    ];
    render(
      <TrickArea
        trickCards={trickCards}
        playerNames={playerNames}
        playerIndex={0}
        trickWinner={0}
      />
    );
    expect(screen.getByText('You wins!')).toBeInTheDocument();
  });

  it('positions cards relative to player index', () => {
    const trickCards: TrickCard[] = [
      [2, makeCard('tarok', 10)],
    ];
    const { container } = render(
      <TrickArea trickCards={trickCards} playerNames={playerNames} playerIndex={0} />
    );
    // Player 2 is the top position (2 seats from player 0)
    expect(container.querySelector('.trick-card-top')).toBeInTheDocument();
  });
});
