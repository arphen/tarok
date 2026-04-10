import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import Hand from '../components/Hand';
import type { CardData } from '../types/game';

const makeCard = (type: 'tarok' | 'suit', value: number, suit: string | null = null): CardData => ({
  card_type: type,
  value,
  suit: suit as any,
  label: type === 'tarok' ? `T${value}` : `${suit}${value}`,
  points: value >= 5 ? 5 : 1,
});

const hand: CardData[] = [
  makeCard('tarok', 1),
  makeCard('tarok', 14),
  makeCard('suit', 8, 'hearts'),
  makeCard('suit', 4, 'clubs'),
];

describe('Hand', () => {
  it('renders all cards in a hand', () => {
    render(<Hand cards={hand} />);
    expect(screen.getByTestId('card-tarok-1-none')).toBeInTheDocument();
    expect(screen.getByTestId('card-tarok-14-none')).toBeInTheDocument();
    expect(screen.getByTestId('card-suit-8-hearts')).toBeInTheDocument();
    expect(screen.getByTestId('card-suit-4-clubs')).toBeInTheDocument();
  });

  it('renders face-down cards with correct count', () => {
    render(<Hand cards={[]} faceDown cardCount={5} />);
    const backs = document.querySelectorAll('.card-back');
    expect(backs.length).toBe(5);
  });

  it('shows player label', () => {
    render(<Hand cards={hand} label="You" />);
    expect(screen.getByText('You')).toBeInTheDocument();
  });

  it('highlights legal plays', () => {
    const legalPlays = [hand[0], hand[2]]; // tarok 1 and hearts king
    render(<Hand cards={hand} legalPlays={legalPlays} />);
    expect(screen.getByTestId('card-tarok-1-none')).toHaveClass('card-highlighted');
    expect(screen.getByTestId('card-tarok-14-none')).toHaveClass('card-disabled');
  });

  it('applies team role classes', () => {
    const { container } = render(<Hand cards={hand} teamRole="declarer" />);
    expect(container.querySelector('.hand')).toHaveClass('team-declarer');
  });

  it('applies solo styling for declarer', () => {
    const { container } = render(<Hand cards={hand} teamRole="declarer" isSolo />);
    expect(container.querySelector('.hand')).toHaveClass('team-solo');
  });
});
