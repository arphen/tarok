import { describe, it, expect } from 'vitest';
import { groupCards, buildCountingExam } from '../utils/cardCounting';
import type { CardInGroup } from '../utils/cardCounting';
import type { TrickSummaryEntry, CardData } from '../types/game';

// Helper to make a CardInGroup quickly
function card(label: string, points: number, player = 0): CardInGroup {
  return { label, points, player };
}

// Helper to make a TrickSummaryEntry
function trick(num: number, winner: number, cards: { player: number; label: string; points: number }[]): TrickSummaryEntry {
  return { trick_num: num, lead_player: cards[0]?.player ?? 0, cards, winner, card_points: 0 };
}

function cardData(label: string, points: number): CardData {
  return { card_type: 'suit', value: 0, suit: 'hearts', label, points };
}

// ----- groupCards -----

describe('groupCards', () => {
  it('returns empty for no cards', () => {
    const result = groupCards([]);
    expect(result.groups).toHaveLength(0);
    expect(result.total).toBe(0);
  });

  it('groups exactly 3 cards: sum - 2', () => {
    const cards = [card('K♥', 5), card('Q♥', 4), card('J♥', 3)];
    const result = groupCards(cards);
    expect(result.groups).toHaveLength(1);
    expect(result.groups[0].isComplete).toBe(true);
    expect(result.groups[0].rawSum).toBe(12);
    expect(result.groups[0].deduction).toBe(2);
    expect(result.groups[0].value).toBe(10);
    expect(result.total).toBe(10);
  });

  it('groups 6 cards into 2 full groups', () => {
    const cards = [
      card('K♥', 5), card('Q♥', 4), card('J♥', 3),
      card('K♠', 5), card('7♠', 1), card('8♠', 1),
    ];
    const result = groupCards(cards);
    expect(result.groups).toHaveLength(2);
    expect(result.groups[0].value).toBe(10); // 5+4+3 - 2
    expect(result.groups[1].value).toBe(5);  // 5+1+1 - 2
    expect(result.total).toBe(15);
  });

  it('handles incomplete group of 2 cards with deduction of 1', () => {
    // 2 leftover cards: sum − 1
    const cards = [card('K♥', 5), card('Q♥', 4)];
    const result = groupCards(cards);
    expect(result.groups).toHaveLength(1);
    expect(result.groups[0].isComplete).toBe(false);
    expect(result.groups[0].deduction).toBe(1);
    expect(result.groups[0].value).toBe(8);
    expect(result.total).toBe(8);
  });

  it('handles incomplete group of 1 card with no deduction', () => {
    const cards = [card('K♥', 5)];
    const result = groupCards(cards);
    expect(result.groups).toHaveLength(1);
    expect(result.groups[0].isComplete).toBe(false);
    expect(result.groups[0].deduction).toBe(0);
    expect(result.groups[0].value).toBe(5);
    expect(result.total).toBe(5);
  });

  it('handles 1-point cards in incomplete group', () => {
    // 1-point card in a single-card remainder: deduction is 0
    const cards = [card('7♥', 1)];
    const result = groupCards(cards);
    expect(result.groups[0].value).toBe(1);
  });

  it('handles mix of full + incomplete groups (4 cards)', () => {
    const cards = [
      card('K♥', 5), card('Q♥', 4), card('J♥', 3),
      card('10♥', 2),
    ];
    const result = groupCards(cards);
    expect(result.groups).toHaveLength(2);
    expect(result.groups[0].isComplete).toBe(true);
    expect(result.groups[0].value).toBe(10); // 5+4+3 - 2
    expect(result.groups[1].isComplete).toBe(false);
    // Last 1 card: sum - 0
    expect(result.groups[1].value).toBe(2);
    expect(result.total).toBe(12);
  });

  it('handles mix of full + incomplete groups (5 cards)', () => {
    const cards = [
      card('K♥', 5), card('Q♥', 4), card('J♥', 3),
      card('Škis', 5), card('Mond', 5),
    ];
    const result = groupCards(cards);
    expect(result.groups).toHaveLength(2);
    expect(result.groups[0].value).toBe(10); // 5+4+3 - 2
    // Last 2 cards: sum - 1
    expect(result.groups[1].value).toBe(9);
    expect(result.total).toBe(19);
  });

  it('three 1-point cards → sum 3 minus 2 = 1', () => {
    const cards = [card('7♥', 1), card('8♥', 1), card('9♥', 1)];
    const result = groupCards(cards);
    expect(result.groups[0].value).toBe(1);
    expect(result.total).toBe(1);
  });

  it('all 54 tarok cards sum to ~70 (simulated with known values)', () => {
    // Simulate 18 groups of 3 cards (54 total) with known point makeup
    // In actual tarok: total raw points before deduction = 70 + 18*2 = 106
    // After grouping: 106 - 36 = 70
    // We'll use a simpler test: 18 groups, each having raw sum 6 → 6-2=4 each → 72
    const cards: CardInGroup[] = [];
    for (let i = 0; i < 54; i++) {
      cards.push(card(`Card${i}`, 2)); // 54 cards × 2 points = 108 raw
    }
    const result = groupCards(cards);
    expect(result.groups).toHaveLength(18);
    // Each group: 6 - 2 = 4, total = 72
    expect(result.total).toBe(72);
    // All groups are complete
    expect(result.groups.every(g => g.isComplete)).toBe(true);
  });
});

// ----- buildCountingExam -----

describe('buildCountingExam', () => {
  const NAMES = ['P0', 'P1', 'P2', 'P3'];

  it('returns empty for null trick summary', () => {
    expect(buildCountingExam(null, {}, NAMES, 3, [], null)).toEqual([]);
  });

  it('returns empty for empty trick summary', () => {
    expect(buildCountingExam([], {}, NAMES, 3, [], null)).toEqual([]);
  });

  it('returns empty for klop contract (-99)', () => {
    const tricks = [trick(1, 0, [{ player: 0, label: 'K♥', points: 5 }])];
    expect(buildCountingExam(tricks, { '0': 'declarer' }, NAMES, -99, [], null)).toEqual([]);
  });

  it('returns empty when no declarer role assigned', () => {
    const tricks = [
      trick(1, 0, [
        { player: 0, label: 'K♥', points: 5 },
        { player: 1, label: 'Q♥', points: 4 },
        { player: 2, label: 'J♥', points: 3 },
        { player: 3, label: '7♥', points: 1 },
      ]),
    ];
    expect(buildCountingExam(tricks, {}, NAMES, 3, [], null)).toEqual([]);
  });

  it('splits cards between declarer and defenders (solo)', () => {
    const tricks = [
      trick(1, 0, [
        { player: 0, label: 'K♥', points: 5 },
        { player: 1, label: 'Q♥', points: 4 },
        { player: 2, label: 'J♥', points: 3 },
        { player: 3, label: '7♥', points: 1 },
      ]),
      trick(2, 1, [
        { player: 1, label: 'K♠', points: 5 },
        { player: 2, label: 'Q♠', points: 4 },
        { player: 3, label: 'J♠', points: 3 },
        { player: 0, label: '7♠', points: 1 },
      ]),
    ];
    const roles = { '0': 'declarer', '1': 'opponent', '2': 'opponent', '3': 'opponent' };
    const result = buildCountingExam(tricks, roles, NAMES, -1, [], null);

    expect(result).toHaveLength(2);
    // Declarer won trick 1 (4 cards), Defenders won trick 2 (4 cards)
    expect(result[0].key).toBe('decl');
    expect(result[0].allCards).toHaveLength(4);
    expect(result[1].key).toBe('def');
    expect(result[1].allCards).toHaveLength(4);
  });

  it('includes partner cards in declarer team (2v2)', () => {
    const tricks = [
      trick(1, 0, [
        { player: 0, label: 'K♥', points: 5 },
        { player: 1, label: 'Q♥', points: 4 },
        { player: 2, label: 'J♥', points: 3 },
        { player: 3, label: '7♥', points: 1 },
      ]),
      trick(2, 2, [
        { player: 2, label: 'K♠', points: 5 },
        { player: 3, label: 'Q♠', points: 4 },
        { player: 0, label: 'J♠', points: 3 },
        { player: 1, label: '7♠', points: 1 },
      ]),
    ];
    const roles = { '0': 'declarer', '1': 'opponent', '2': 'partner', '3': 'opponent' };
    const result = buildCountingExam(tricks, roles, NAMES, 3, [], null);

    expect(result[0].key).toBe('decl');
    expect(result[0].label).toContain('2v2');
    expect(result[0].players).toEqual([0, 2]);
    // Both tricks won by declarer team
    expect(result[0].allCards).toHaveLength(8);
    expect(result[1].allCards).toHaveLength(0);
  });

  it('appends put-down cards to declarer pile for talon exchange contracts', () => {
    const tricks = [
      trick(1, 0, [
        { player: 0, label: 'K♥', points: 5 },
        { player: 1, label: '7♥', points: 1 },
        { player: 2, label: '8♥', points: 1 },
        { player: 3, label: '9♥', points: 1 },
      ]),
    ];
    const roles = { '0': 'declarer', '1': 'opponent', '2': 'opponent', '3': 'opponent' };
    const putDown: CardData[] = [cardData('Q♣', 4), cardData('J♣', 3)];
    const result = buildCountingExam(tricks, roles, NAMES, 3, putDown, null);

    // Contract 3 = Three (has talon exchange), so put-down added to declarer
    expect(result[0].allCards).toHaveLength(6); // 4 from trick + 2 put-down
    expect(result[0].allCards[4].label).toBe('Q♣');
    expect(result[0].allCards[5].label).toBe('J♣');
  });

  it('does NOT add put-down cards for solo (no-talon) contracts', () => {
    const tricks = [
      trick(1, 0, [
        { player: 0, label: 'K♥', points: 5 },
        { player: 1, label: '7♥', points: 1 },
        { player: 2, label: '8♥', points: 1 },
        { player: 3, label: '9♥', points: 1 },
      ]),
    ];
    const roles = { '0': 'declarer', '1': 'opponent', '2': 'opponent', '3': 'opponent' };
    const putDown: CardData[] = [cardData('Q♣', 4)];
    // Contract 0 = Solo (no talon exchange)
    const result = buildCountingExam(tricks, roles, NAMES, 0, putDown, null);

    expect(result[0].allCards).toHaveLength(4); // only trick cards, no put-down
  });

  it('adds talon section for solo (no-talon) contracts', () => {
    const tricks = [
      trick(1, 0, [
        { player: 0, label: 'K♥', points: 5 },
        { player: 1, label: '7♥', points: 1 },
        { player: 2, label: '8♥', points: 1 },
        { player: 3, label: '9♥', points: 1 },
      ]),
    ];
    const roles = { '0': 'declarer', '1': 'opponent', '2': 'opponent', '3': 'opponent' };
    const talonGroups: CardData[][] = [
      [cardData('Q♣', 4), cardData('J♣', 3), cardData('10♣', 2)],
      [cardData('Q♠', 4), cardData('J♠', 3), cardData('10♠', 2)],
    ];
    // Contract 0 = Solo
    const result = buildCountingExam(tricks, roles, NAMES, 0, [], talonGroups);

    expect(result).toHaveLength(3);
    expect(result[2].key).toBe('talon');
    expect(result[2].label).toContain('Talon');
    expect(result[2].allCards).toHaveLength(6);
    // 4+3+2 - 2 = 7, 4+3+2 - 2 = 7 → total 14
    expect(result[2].total).toBe(14);
  });

  it('does NOT add talon section for normal contracts (Three, Two, etc.)', () => {
    const tricks = [
      trick(1, 0, [
        { player: 0, label: 'K♥', points: 5 },
        { player: 1, label: '7♥', points: 1 },
        { player: 2, label: '8♥', points: 1 },
        { player: 3, label: '9♥', points: 1 },
      ]),
    ];
    const roles = { '0': 'declarer', '1': 'opponent', '2': 'opponent', '3': 'opponent' };
    const talonGroups: CardData[][] = [
      [cardData('Q♣', 4), cardData('J♣', 3), cardData('10♣', 2)],
    ];
    // Contract 3 = Three (has talon exchange)
    const result = buildCountingExam(tricks, roles, NAMES, 3, [], talonGroups);

    expect(result).toHaveLength(2); // no talon section
  });

  it('both team totals sum correctly with group-of-3 counting', () => {
    // Make 12 tricks × 4 cards = 48 cards (no talon in this normal game simulation)
    const allTricks: TrickSummaryEntry[] = [];
    for (let i = 0; i < 12; i++) {
      // Alternate winners between player 0 (declarer) and player 1 (opponent)
      allTricks.push(trick(i + 1, i % 2 === 0 ? 0 : 1, [
        { player: 0, label: `C${i * 4}`, points: i % 3 === 0 ? 5 : 1 },
        { player: 1, label: `C${i * 4 + 1}`, points: i % 3 === 1 ? 4 : 1 },
        { player: 2, label: `C${i * 4 + 2}`, points: i % 3 === 2 ? 3 : 1 },
        { player: 3, label: `C${i * 4 + 3}`, points: 1 },
      ]));
    }
    const roles = { '0': 'declarer', '1': 'opponent', '2': 'opponent', '3': 'opponent' };
    const result = buildCountingExam(allTricks, roles, NAMES, -1, [], null);

    const declTotal = result[0].total;
    const defTotal = result[1].total;
    // Each team has 24 cards → 8 full groups → both should sum to a consistent value
    expect(declTotal + defTotal).toBeGreaterThan(0);
    // Verify groups are all complete (24 cards = 8 groups of 3)
    expect(result[0].groups.every(g => g.isComplete)).toBe(true);
    expect(result[1].groups.every(g => g.isComplete)).toBe(true);
  });
});
