import { useState, useCallback, useRef, useEffect } from 'react';

// ---- Types ----

export interface TournamentEntry {
  id: string;
  name: string;
  type: 'rl' | 'random' | 'lookahead';
  checkpoint: string;
}

export interface MatchResult {
  /** seat index → cumulative score over all games in the match */
  cumulative: Record<string, number>;
  /** seats ranked best to worst */
  ranked: { seat: number; name: string; score: number }[];
}

export type BracketSide = 'winners' | 'losers';

export interface BracketMatch {
  id: string;
  round: number;
  side: BracketSide;
  label: string;
  entries: TournamentEntry[];
  result: MatchResult | null;
  /** indices into entries: who advanced */
  advancedIdx: number[];
  /** indices into entries: who was eliminated */
  eliminatedIdx: number[];
}

export type TournamentPhase = 'setup' | 'running' | 'finished';

export interface AgentStanding {
  name: string;
  type: string;
  checkpoint: string;
  wins: number;
  top2: number;
  top4: number;
  total_placement: number;
  tournaments_played: number;
  avg_placement: number;
  placements: number[];
}

export interface MultiTournamentProgress {
  status: 'idle' | 'running' | 'done' | 'cancelled' | 'error';
  current: number;
  total: number;
  standings: Record<string, AgentStanding>;
}

export interface TournamentState {
  phase: TournamentPhase;
  entries: TournamentEntry[];
  gamesPerRound: number;
  matches: BracketMatch[];
  currentMatchIdx: number;
  champion: TournamentEntry | null;
  // Multi-tournament simulation
  numTournaments: number;
  multiProgress: MultiTournamentProgress | null;
}

// ---- Bracket generation (8-model double elimination) ----

function buildBracket(entries: TournamentEntry[]): BracketMatch[] {
  // We need exactly 8 entries; pad with randoms if needed
  const padded = [...entries];
  while (padded.length < 8) {
    padded.push({
      id: `fill-${padded.length}`,
      name: `Random-${padded.length}`,
      type: 'random',
      checkpoint: '',
    });
  }

  // Shuffle for fairness
  const shuffled = [...padded].sort(() => Math.random() - 0.5);

  const matches: BracketMatch[] = [
    // WB R1 — two groups of 4
    {
      id: 'wb-r1-a',
      round: 1,
      side: 'winners',
      label: 'Winners R1 — Group A',
      entries: shuffled.slice(0, 4),
      result: null,
      advancedIdx: [],
      eliminatedIdx: [],
    },
    {
      id: 'wb-r1-b',
      round: 1,
      side: 'winners',
      label: 'Winners R1 — Group B',
      entries: shuffled.slice(4, 8),
      result: null,
      advancedIdx: [],
      eliminatedIdx: [],
    },
    // LB R1 — 4 losers from WB R1
    {
      id: 'lb-r1',
      round: 2,
      side: 'losers',
      label: 'Losers R1',
      entries: [], // filled after WB R1
      result: null,
      advancedIdx: [],
      eliminatedIdx: [],
    },
    // WB Final — 4 winners from WB R1
    {
      id: 'wb-final',
      round: 2,
      side: 'winners',
      label: 'Winners Final',
      entries: [], // filled after WB R1
      result: null,
      advancedIdx: [],
      eliminatedIdx: [],
    },
    // LB Final — 2 from LB R1 + 2 drops from WB Final
    {
      id: 'lb-final',
      round: 3,
      side: 'losers',
      label: 'Losers Final',
      entries: [], // filled after LB R1 + WB Final
      result: null,
      advancedIdx: [],
      eliminatedIdx: [],
    },
    // Grand Final — 2 from WB Final + 2 from LB Final
    {
      id: 'grand-final',
      round: 4,
      side: 'winners',
      label: 'Grand Final',
      entries: [], // filled after WB Final + LB Final
      result: null,
      advancedIdx: [],
      eliminatedIdx: [],
    },
  ];

  return matches;
}

function rankEntries(match: BracketMatch): TournamentEntry[] {
  if (!match.result) return match.entries;
  const ranked = match.result.ranked;
  return ranked.map(r => match.entries[r.seat]);
}

/** After a match finishes, populate the next matches' entries. */
function advanceBracket(matches: BracketMatch[], justFinishedIdx: number): BracketMatch[] {
  const updated = matches.map(m => ({ ...m }));
  const finished = updated[justFinishedIdx];
  const ranked = rankEntries(finished);

  // Top 2 advance, bottom 2 lose
  finished.advancedIdx = finished.result!.ranked.slice(0, 2).map(r => r.seat);
  finished.eliminatedIdx = finished.result!.ranked.slice(2).map(r => r.seat);

  const top2 = ranked.slice(0, 2);
  const bot2 = ranked.slice(2, 4);

  switch (finished.id) {
    case 'wb-r1-a': {
      const wbFinal = updated.find(m => m.id === 'wb-final')!;
      wbFinal.entries = [...wbFinal.entries, ...top2];
      const lbR1 = updated.find(m => m.id === 'lb-r1')!;
      lbR1.entries = [...lbR1.entries, ...bot2];
      break;
    }
    case 'wb-r1-b': {
      const wbFinal = updated.find(m => m.id === 'wb-final')!;
      wbFinal.entries = [...wbFinal.entries, ...top2];
      const lbR1 = updated.find(m => m.id === 'lb-r1')!;
      lbR1.entries = [...lbR1.entries, ...bot2];
      break;
    }
    case 'lb-r1': {
      const lbFinal = updated.find(m => m.id === 'lb-final')!;
      lbFinal.entries = [...lbFinal.entries, ...top2];
      // bottom 2 eliminated
      break;
    }
    case 'wb-final': {
      const grandFinal = updated.find(m => m.id === 'grand-final')!;
      grandFinal.entries = [...grandFinal.entries, ...top2];
      const lbFinal = updated.find(m => m.id === 'lb-final')!;
      lbFinal.entries = [...lbFinal.entries, ...bot2];
      break;
    }
    case 'lb-final': {
      const grandFinal = updated.find(m => m.id === 'grand-final')!;
      grandFinal.entries = [...grandFinal.entries, ...top2];
      // bottom 2 eliminated
      break;
    }
    // grand-final: nothing to advance
  }

  return updated;
}

// The order in which matches are played
const MATCH_ORDER = ['wb-r1-a', 'wb-r1-b', 'lb-r1', 'wb-final', 'lb-final', 'grand-final'];

// ---- Hook ----

export function useTournament() {
  const [state, setState] = useState<TournamentState>({
    phase: 'setup',
    entries: [],
    gamesPerRound: 5,
    matches: [],
    currentMatchIdx: 0,
    champion: null,
    numTournaments: 5,
    multiProgress: null,
  });

  // Ref for latest state so async functions always read fresh values
  const stateRef = useRef(state);
  stateRef.current = state;

  const setEntries = useCallback((entries: TournamentEntry[]) => {
    setState(prev => ({ ...prev, entries }));
  }, []);

  const setGamesPerRound = useCallback((n: number) => {
    setState(prev => ({ ...prev, gamesPerRound: Math.max(1, Math.min(100, n)) }));
  }, []);

  const setNumTournaments = useCallback((n: number) => {
    setState(prev => ({ ...prev, numTournaments: Math.max(1, Math.min(100, n)) }));
  }, []);

  const startTournament = useCallback(() => {
    if (state.entries.length < 4) return;
    const matches = buildBracket(state.entries);
    setState(prev => ({
      ...prev,
      phase: 'running',
      matches,
      currentMatchIdx: 0,
      champion: null,
    }));
  }, [state.entries]);

  const runCurrentMatch = useCallback(async (): Promise<'running' | 'finished'> => {
    const s = stateRef.current;
    const matchIdx = s.currentMatchIdx;
    const matchId = MATCH_ORDER[matchIdx];
    const match = s.matches.find(m => m.id === matchId);
    if (!match || match.entries.length < 4) return 'running';

    const body = {
      agents: match.entries.map(e => ({
        name: e.name,
        type: e.type,
        checkpoint: e.checkpoint || undefined,
      })),
      num_games: s.gamesPerRound,
    };

    const res = await fetch('/api/tournament/match', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data: MatchResult = await res.json();

    const isLast = matchId === 'grand-final';

    setState(prev => {
      let matches = prev.matches.map(m =>
        m.id === matchId ? { ...m, result: data } : m
      );
      const finIdx = matches.findIndex(m => m.id === matchId);
      matches = advanceBracket(matches, finIdx);

      const nextIdx = prev.currentMatchIdx + 1;
      const champion = isLast ? rankEntries({ ...matches[finIdx], result: data })[0] : null;

      const newState = {
        ...prev,
        matches,
        currentMatchIdx: isLast ? prev.currentMatchIdx : nextIdx,
        phase: (isLast ? 'finished' : 'running') as TournamentPhase,
        champion,
      };
      stateRef.current = newState;
      return newState;
    });

    return isLast ? 'finished' : 'running';
  }, []);

  const reset = useCallback(() => {
    setState(prev => ({
      ...prev,
      phase: 'setup',
      matches: [],
      currentMatchIdx: 0,
      champion: null,
      multiProgress: null,
    }));
  }, []);

  // ---- Multi-tournament simulation ----

  const startMultiTournament = useCallback(async () => {
    const s = stateRef.current;
    if (s.entries.length < 4) return;

    const body = {
      agents: s.entries.map(e => ({
        name: e.name,
        type: e.type,
        checkpoint: e.checkpoint || undefined,
      })),
      num_tournaments: s.numTournaments,
      games_per_round: s.gamesPerRound,
    };

    await fetch('/api/tournament/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    setState(prev => ({
      ...prev,
      phase: 'running',
      multiProgress: { status: 'running', current: 0, total: s.numTournaments, standings: {} },
    }));
  }, []);

  const stopMultiTournament = useCallback(async () => {
    await fetch('/api/tournament/simulate/stop', { method: 'POST' });
  }, []);

  // Poll multi-tournament progress
  useEffect(() => {
    const mp = state.multiProgress;
    if (!mp || (mp.status !== 'running')) return;

    const id = setInterval(async () => {
      try {
        const res = await fetch('/api/tournament/simulate/progress');
        const data: MultiTournamentProgress = await res.json();
        setState(prev => {
          const done = data.status === 'done' || data.status === 'error' || data.status === 'cancelled';
          const phase = done ? 'finished' as const : prev.phase;
          return { ...prev, multiProgress: data, phase };
        });
      } catch { /* server not up */ }
    }, 1000);

    return () => clearInterval(id);
  }, [state.multiProgress?.status]);

  const currentMatch = state.matches.find(m => m.id === MATCH_ORDER[state.currentMatchIdx]) ?? null;

  return {
    ...state,
    currentMatch,
    setEntries,
    setGamesPerRound,
    setNumTournaments,
    startTournament,
    runCurrentMatch,
    startMultiTournament,
    stopMultiTournament,
    reset,
  };
}
