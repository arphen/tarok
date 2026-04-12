import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import ArenaCheckpointLeaderboard from '../components/ArenaCheckpointLeaderboard';

const checkpoints = [
  { filename: 'cp_a.pt', episode: 120, win_rate: 0.67, model_name: 'Alpha' },
];

describe('ArenaCheckpointLeaderboard', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn((url: string) => {
      if (url.includes('/api/arena/leaderboard/checkpoints')) {
        return Promise.resolve({
          json: () => Promise.resolve({
            leaderboard: [
              {
                checkpoint: 'cp_a.pt',
                appearances: 2,
                runs: 2,
                games: 300,
                avg_placement: 1.83,
                bid_wins: 64,
                bid_win_rate_per_game: 21.33,
                avg_taroks_in_hand: 5.2,
                declared_games: 46,
                declared_win_rate: 63.04,
                avg_declared_win_score: 43.12,
                avg_declared_loss_score: -18.65,
                times_called: 15,
                latest_run_at: '2026-04-12T10:00:00+00:00',
              },
            ],
          }),
        }) as unknown as Response;
      }

      return Promise.resolve({
        json: () => Promise.resolve({ runs: [] }),
      }) as unknown as Response;
    }));
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('renders leaderboard title and checkpoint row', async () => {
    render(<ArenaCheckpointLeaderboard onBack={() => {}} checkpoints={checkpoints} />);

    expect(screen.getByText('Arena Checkpoint Leaderboard')).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText(/Alpha/)).toBeInTheDocument();
      expect(screen.getByText('300')).toBeInTheDocument();
    });
  });
});
