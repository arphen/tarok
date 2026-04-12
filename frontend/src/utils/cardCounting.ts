/**
 * Card counting logic for Slovenian Tarok.
 *
 * Cards won by a team are laid out in trick order, then split into groups of 3.
 * - Full group (3 cards): sum of card points − 2
 * - Incomplete last group (1–2 cards): each card − round(2/3 × card_points)
 * Total over all 54 cards = ~70.
 */

import type { CardData, TrickSummaryEntry } from '../types/game';

export interface CardInGroup {
  label: string;
  points: number;
  player: number;
}

export interface CountingGroup {
  cards: CardInGroup[];
  rawSum: number;
  deduction: number;
  value: number;
  isComplete: boolean;
  perCardValues: number[];
}

export interface CountingTeamView {
  key: string;
  label: string;
  players: number[];
  allCards: CardInGroup[];
  groups: CountingGroup[];
  total: number;
}

// Contracts where no talon exchange happens (talon sits aside)
const NO_TALON_CONTRACTS = new Set([0, -100, -101]); // Solo, Berač, Barvni Valat

export function groupCards(cards: CardInGroup[]): { groups: CountingGroup[]; total: number } {
  const groups: CountingGroup[] = [];
  let total = 0;
  for (let i = 0; i < cards.length; i += 3) {
    const chunk = cards.slice(i, i + 3);
    const rawSum = chunk.reduce((s, c) => s + c.points, 0);
    const isComplete = chunk.length === 3;

    let deduction: number;
    let perCardValues: number[];
    if (isComplete) {
      // Full group: sum − 2
      deduction = 2;
      perCardValues = chunk.map(c => c.points);
    } else {
      // Incomplete last group: each card − round(2/3 × card_value)
      perCardValues = chunk.map(c => c.points - Math.round(c.points * 2 / 3));
      deduction = chunk.reduce((s, c) => s + Math.round(c.points * 2 / 3), 0);
    }
    const value = rawSum - deduction;
    groups.push({ cards: chunk, rawSum, deduction, value, isComplete, perCardValues });
    total += value;
  }
  return { groups, total };
}

export function buildCountingExam(
  trickSummary: TrickSummaryEntry[] | null,
  roles: Record<string, string>,
  names: string[],
  contract: number | null,
  putDown: CardData[],
  talonGroups: CardData[][] | null,
): CountingTeamView[] {
  if (!trickSummary || trickSummary.length === 0) return [];
  if (contract === -99) return []; // klop — individual scoring

  const declarerPlayers = new Set<number>();
  const defenderPlayers = new Set<number>();
  const maxPlayerIdx = Math.max(3, names.length - 1);

  for (let p = 0; p <= maxPlayerIdx; p += 1) {
    const role = roles[String(p)];
    if (role === 'declarer' || role === 'partner') {
      declarerPlayers.add(p);
    } else {
      defenderPlayers.add(p);
    }
  }

  if (declarerPlayers.size === 0) return [];

  const isNoTalon = contract !== null && NO_TALON_CONTRACTS.has(contract);

  // Collect all cards per team in trick order (as they appeared on the table)
  const declCards: CardInGroup[] = [];
  const defCards: CardInGroup[] = [];

  for (const trick of trickSummary) {
    const isDeclWinner = declarerPlayers.has(trick.winner);
    const target = isDeclWinner ? declCards : defCards;
    for (const c of trick.cards) {
      target.push({ label: c.label, points: c.points, player: c.player });
    }
  }

  // For contracts WITH talon exchange, put-down cards count for the declarer
  if (!isNoTalon && putDown.length > 0) {
    for (const c of putDown) {
      declCards.push({ label: c.label, points: c.points, player: -1 });
    }
  }

  const declResult = groupCards(declCards);
  const defResult = groupCards(defCards);

  const teams: CountingTeamView[] = [
    {
      key: 'decl',
      label: declarerPlayers.size > 1 ? 'Declarer Team (2v2)' : 'Declarer Team (Solo)',
      players: Array.from(declarerPlayers).sort((a, b) => a - b),
      allCards: declCards,
      groups: declResult.groups,
      total: declResult.total,
    },
    {
      key: 'def',
      label: defenderPlayers.size > 1 ? 'Defenders Team' : 'Defender',
      players: Array.from(defenderPlayers).sort((a, b) => a - b),
      allCards: defCards,
      groups: defResult.groups,
      total: defResult.total,
    },
  ];

  // For contracts without talon, show talon counted separately
  if (isNoTalon && talonGroups && talonGroups.length > 0) {
    const talonCards: CardInGroup[] = [];
    for (const group of talonGroups) {
      for (const c of group) {
        talonCards.push({ label: c.label, points: c.points, player: -1 });
      }
    }
    if (talonCards.length > 0) {
      const talonResult = groupCards(talonCards);
      teams.push({
        key: 'talon',
        label: 'Talon (not exchanged)',
        players: [],
        allCards: talonCards,
        groups: talonResult.groups,
        total: talonResult.total,
      });
    }
  }

  return teams;
}
