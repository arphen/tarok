import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import Scoreboard from '../components/Scoreboard';
import type { MatchInfo } from '../types/game';

const playerNames = ['You', 'AI-1', 'AI-2', 'AI-3'];

describe('Scoreboard', () => {
  it('renders cumulative scores ranked by score', () => {
    const matchInfo: MatchInfo = {
      round_num: 2,
      total_rounds: 5,
      cumulative_scores: { '0': 50, '1': -20, '2': 80, '3': -10 },
      caller_counts: { '0': 1, '1': 0, '2': 1, '3': 0 },
      called_counts: { '0': 0, '1': 1, '2': 0, '3': 1 },
      round_history: [
        { round: 1, scores: { '0': 30, '1': -10, '2': 40, '3': -10 }, contract: 3, declarer: 2, partner: 0 },
      ],
    };

    render(<Scoreboard matchInfo={matchInfo} playerNames={playerNames} />);

    expect(screen.getByText('Scoreboard')).toBeInTheDocument();
    expect(screen.getByText('Rd 2/5')).toBeInTheDocument();
    expect(screen.getByText('AI-2')).toBeInTheDocument();
    expect(screen.getByText('+80')).toBeInTheDocument();
    expect(screen.getByText('-20')).toBeInTheDocument();
  });

  it('shows round history', () => {
    const matchInfo: MatchInfo = {
      round_num: 2,
      total_rounds: 3,
      cumulative_scores: { '0': 10, '1': 0, '2': 0, '3': 0 },
      caller_counts: { '0': 1, '1': 0, '2': 0, '3': 0 },
      called_counts: { '0': 0, '1': 0, '2': 0, '3': 0 },
      round_history: [
        { round: 1, scores: { '0': 10, '1': -5, '2': -5, '3': 0 }, contract: 3, declarer: 0, partner: 1 },
      ],
    };

    render(<Scoreboard matchInfo={matchInfo} playerNames={playerNames} />);
    expect(screen.getByText('Round History')).toBeInTheDocument();
    expect(screen.getByText('R1')).toBeInTheDocument();
  });
});
