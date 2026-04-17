import { describe, it, expect } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useGame } from '../hooks/useGame';
import type { LogEntry } from '../hooks/useGame';

describe('useGame log deduplication', () => {
  it('initializes with empty log entries', () => {
    const { result } = renderHook(() => useGame());
    expect(result.current.logEntries).toEqual([]);
  });

  it('initializes with default game state', () => {
    const { result } = renderHook(() => useGame());
    expect(result.current.gameState.phase).toBe('waiting');
    expect(result.current.gameState.hand).toEqual([]);
    expect(result.current.connected).toBe(false);
  });

  it('initializes trick animation state', () => {
    const { result } = renderHook(() => useGame());
    expect(result.current.trickWinner).toBeNull();
    expect(result.current.trickWinCards).toEqual([]);
  });
});
