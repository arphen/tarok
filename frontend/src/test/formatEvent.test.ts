import { describe, it, expect, beforeEach } from 'vitest';
import { formatEvent } from '../hooks/useGame';
import type { LogEntry } from '../hooks/useGame';

const NAMES = ['You', 'AI-1', 'AI-2', 'AI-3'];

describe('formatEvent', () => {
  it('returns null for unknown events', () => {
    expect(formatEvent('unknown_event', {}, NAMES)).toBeNull();
  });

  it('returns null for trick_start (no log for trick starts)', () => {
    expect(formatEvent('trick_start', {}, NAMES)).toBeNull();
  });

  it('returns null for rule_verified (no log for rule checks)', () => {
    expect(formatEvent('rule_verified', { player: 1, rule: 'some rule' }, NAMES)).toBeNull();
  });

  it('formats game_start event', () => {
    const entry = formatEvent('game_start', {}, NAMES);
    expect(entry).not.toBeNull();
    expect(entry!.category).toBe('system');
    expect(entry!.message).toBe('Game started. Dealing cards...');
  });

  it('formats deal event', () => {
    const entry = formatEvent('deal', {}, NAMES);
    expect(entry).not.toBeNull();
    expect(entry!.category).toBe('system');
    expect(entry!.message).toBe('Cards dealt to all players.');
  });

  it('formats bid event with contract', () => {
    const entry = formatEvent('bid', { player: 1, contract: 3 }, NAMES);
    expect(entry).not.toBeNull();
    expect(entry!.category).toBe('bid');
    expect(entry!.player).toBe(1);
    expect(entry!.isHuman).toBe(false);
    expect(entry!.message).toContain('AI-1');
  });

  it('formats bid event as pass', () => {
    const entry = formatEvent('bid', { player: 0, contract: null }, NAMES);
    expect(entry).not.toBeNull();
    expect(entry!.message).toContain('Pass');
    expect(entry!.isHuman).toBe(true);
  });

  it('formats contract_won event', () => {
    const entry = formatEvent('contract_won', { player: 2, contract: 2 }, NAMES);
    expect(entry).not.toBeNull();
    expect(entry!.category).toBe('bid');
    expect(entry!.message).toContain('AI-2');
    expect(entry!.message).toContain('wins the contract');
  });

  it('formats king_called event', () => {
    const king = { card_type: 'suit', value: 8, suit: 'hearts', label: 'King of Hearts', points: 5 };
    const entry = formatEvent('king_called', { player: 0, king }, NAMES);
    expect(entry).not.toBeNull();
    expect(entry!.category).toBe('king');
    expect(entry!.message).toContain('♥');
    expect(entry!.message).toContain('King');
  });

  it('formats talon_revealed event', () => {
    const entry = formatEvent('talon_revealed', {}, NAMES);
    expect(entry).not.toBeNull();
    expect(entry!.category).toBe('talon');
    expect(entry!.message).toBe('Talon revealed.');
  });

  it('formats talon_group_picked event', () => {
    const entry = formatEvent('talon_group_picked', {}, NAMES);
    expect(entry).not.toBeNull();
    expect(entry!.category).toBe('talon');
    expect(entry!.message).toContain('discard');
  });

  it('formats talon_exchanged event', () => {
    const entry = formatEvent('talon_exchanged', {}, NAMES);
    expect(entry).not.toBeNull();
    expect(entry!.message).toBe('Talon exchange complete.');
  });

  it('formats card_played event', () => {
    const card = { card_type: 'tarok', value: 21, suit: null, label: 'Mond', points: 5 };
    const entry = formatEvent('card_played', { player: 3, card }, NAMES);
    expect(entry).not.toBeNull();
    expect(entry!.category).toBe('play');
    expect(entry!.message).toContain('AI-3');
    expect(entry!.message).toContain('Mond');
    expect(entry!.isHuman).toBe(false);
  });

  it('formats card_played for human player', () => {
    const card = { card_type: 'suit', value: 8, suit: 'hearts', label: 'King of Hearts', points: 5 };
    const entry = formatEvent('card_played', { player: 0, card }, NAMES);
    expect(entry).not.toBeNull();
    expect(entry!.isHuman).toBe(true);
    expect(entry!.message).toContain('You');
  });

  it('formats trick_won event', () => {
    const entry = formatEvent('trick_won', { winner: 1 }, NAMES);
    expect(entry).not.toBeNull();
    expect(entry!.category).toBe('trick');
    expect(entry!.message).toContain('AI-1');
    expect(entry!.message).toContain('wins the trick');
  });

  it('formats game_end event', () => {
    const entry = formatEvent('game_end', { scores: { '0': 35, '1': -10, '2': -10, '3': -15 } }, NAMES);
    expect(entry).not.toBeNull();
    expect(entry!.category).toBe('score');
    expect(entry!.message).toContain('+35');
    expect(entry!.message).toContain('-10');
  });

  it('formats match_update event', () => {
    const entry = formatEvent('match_update', { round_num: 2, total_rounds: 5 }, NAMES);
    expect(entry).not.toBeNull();
    expect(entry!.message).toContain('2/5');
  });

  it('formats match_end event', () => {
    const entry = formatEvent('match_end', { cumulative_scores: { '0': 100, '1': -30 } }, NAMES);
    expect(entry).not.toBeNull();
    expect(entry!.category).toBe('score');
    expect(entry!.message).toContain('Match over');
  });

  it('generates unique IDs for consecutive calls', () => {
    const e1 = formatEvent('game_start', {}, NAMES);
    const e2 = formatEvent('deal', {}, NAMES);
    const e3 = formatEvent('talon_revealed', {}, NAMES);
    expect(e1!.id).not.toBe(e2!.id);
    expect(e2!.id).not.toBe(e3!.id);
  });

  it('falls back to P{idx} for missing player names', () => {
    const entry = formatEvent('card_played', {
      player: 5,
      card: { card_type: 'tarok', value: 1, suit: null, label: 'Pagat', points: 5 },
    }, []);
    expect(entry).not.toBeNull();
    expect(entry!.message).toContain('P5');
  });
});
