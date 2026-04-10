import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import Card from '../components/Card';
import type { CardData } from '../types/game';

const tarokCard: CardData = {
  card_type: 'tarok',
  value: 14,
  suit: null,
  label: 'XIV',
  points: 1,
};

const suitCard: CardData = {
  card_type: 'suit',
  value: 8,
  suit: 'hearts',
  label: '♥K',
  points: 5,
};

const trulaCard: CardData = {
  card_type: 'tarok',
  value: 21,
  suit: null,
  label: 'XXI',
  points: 5,
};

describe('Card', () => {
  it('renders a face-down card with back pattern', () => {
    render(<Card card={tarokCard} faceDown />);
    expect(screen.getByText('✦')).toBeInTheDocument();
  });

  it('renders a tarok card with correct numeral', () => {
    render(<Card card={tarokCard} />);
    const el = screen.getByTestId('card-tarok-14-none');
    expect(el).toBeInTheDocument();
    expect(el).toHaveClass('card-tarok');
  });

  it('renders a suit card with suit symbol', () => {
    render(<Card card={suitCard} />);
    const el = screen.getByTestId('card-suit-8-hearts');
    expect(el).toBeInTheDocument();
    expect(el).toHaveClass('card-suit');
    expect(el).toHaveClass('card-red');
  });

  it('highlights trula cards', () => {
    render(<Card card={trulaCard} />);
    const el = screen.getByTestId('card-tarok-21-none');
    expect(el).toHaveClass('card-trula');
  });

  it('applies highlighted and clickable classes', () => {
    const onClick = vi.fn();
    render(<Card card={suitCard} highlighted onClick={onClick} />);
    const el = screen.getByTestId('card-suit-8-hearts');
    expect(el).toHaveClass('card-highlighted');
    expect(el).toHaveClass('card-clickable');
  });

  it('fires onClick when clicked', () => {
    const onClick = vi.fn();
    render(<Card card={suitCard} onClick={onClick} />);
    fireEvent.click(screen.getByTestId('card-suit-8-hearts'));
    expect(onClick).toHaveBeenCalledOnce();
  });

  it('does not fire onClick when disabled', () => {
    const onClick = vi.fn();
    render(<Card card={suitCard} onClick={onClick} disabled />);
    fireEvent.click(screen.getByTestId('card-suit-8-hearts'));
    expect(onClick).not.toHaveBeenCalled();
  });

  it('applies small class', () => {
    render(<Card card={tarokCard} small />);
    const el = screen.getByTestId('card-tarok-14-none');
    expect(el).toHaveClass('card-small');
  });

  it('renders black suit cards correctly', () => {
    const spadesCard: CardData = { card_type: 'suit', value: 8, suit: 'spades', label: '♠K', points: 5 };
    render(<Card card={spadesCard} />);
    expect(screen.getByTestId('card-suit-8-spades')).toHaveClass('card-black');
  });
});
