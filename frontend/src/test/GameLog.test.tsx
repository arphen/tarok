import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import GameLog from '../components/GameLog';
import type { LogEntry } from '../hooks/useGame';

const entries: LogEntry[] = [
  { id: 1, message: 'Game started. Dealing cards...', category: 'system' },
  { id: 2, message: 'AI-1 bids: Three', category: 'bid', player: 1 },
  { id: 3, message: 'You plays ♥K', category: 'play', player: 0, isHuman: true },
  { id: 4, message: 'AI-2 wins the trick!', category: 'trick', player: 2 },
];

describe('GameLog', () => {
  it('renders all log entries', () => {
    render(<GameLog entries={entries} />);
    expect(screen.getByText('Game started. Dealing cards...')).toBeInTheDocument();
    expect(screen.getByText('AI-1 bids: Three')).toBeInTheDocument();
    expect(screen.getByText('You plays ♥K')).toBeInTheDocument();
    expect(screen.getByText('AI-2 wins the trick!')).toBeInTheDocument();
  });

  it('shows empty state when no entries', () => {
    render(<GameLog entries={[]} />);
    expect(screen.getByText('Waiting for game to start…')).toBeInTheDocument();
  });

  it('renders category icons', () => {
    render(<GameLog entries={entries} />);
    // System icon is ⚙️
    const icons = document.querySelectorAll('.log-icon');
    expect(icons.length).toBe(4);
  });

  it('applies human class to human entries', () => {
    render(<GameLog entries={entries} />);
    const humanEntry = document.querySelector('.log-human');
    expect(humanEntry).toBeInTheDocument();
    expect(humanEntry).toHaveTextContent('You plays ♥K');
  });

  it('renders the header', () => {
    render(<GameLog entries={entries} />);
    expect(screen.getByText('Game Log')).toBeInTheDocument();
  });
});
