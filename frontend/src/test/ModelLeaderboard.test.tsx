import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import ModelLeaderboard from '../components/ModelLeaderboard';

describe('ModelLeaderboard', () => {
  it('renders the rankings tab button', () => {
    render(<ModelLeaderboard />);
    expect(screen.getByTitle('Model Leaderboard')).toBeInTheDocument();
  });

  it('shows Rankings label on the tab', () => {
    render(<ModelLeaderboard />);
    expect(screen.getByText('Rankings')).toBeInTheDocument();
  });

  it('shows trophy icon on the tab', () => {
    render(<ModelLeaderboard />);
    expect(screen.getByText('🏆')).toBeInTheDocument();
  });

  it('starts in collapsed state', () => {
    render(<ModelLeaderboard />);
    const panel = document.querySelector('.leaderboard-panel');
    expect(panel).toBeInTheDocument();
    // Panel should be hidden (display: none) when not open
    const drawer = document.querySelector('.model-leaderboard');
    expect(drawer).not.toHaveClass('leaderboard-open');
  });

  it('toggles open on tab click', () => {
    render(<ModelLeaderboard />);
    const tab = screen.getByTitle('Model Leaderboard');
    fireEvent.click(tab);
    const drawer = document.querySelector('.model-leaderboard');
    expect(drawer).toHaveClass('leaderboard-open');
  });

  it('closes via close button', () => {
    render(<ModelLeaderboard />);
    // Open first
    fireEvent.click(screen.getByTitle('Model Leaderboard'));
    expect(document.querySelector('.model-leaderboard')).toHaveClass('leaderboard-open');
    // Close
    fireEvent.click(screen.getByText('✕'));
    expect(document.querySelector('.model-leaderboard')).not.toHaveClass('leaderboard-open');
  });
});
