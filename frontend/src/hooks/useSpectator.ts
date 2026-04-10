import { useState, useEffect, useRef, useCallback } from 'react';
import type { SpectatorState, SpectatorEvent, ScoreBreakdown, TrickSummaryEntry } from '../types/game';
import { CONTRACT_NAMES, SUIT_SYMBOLS } from '../types/game';

export interface SpectatorLogEntry {
  id: number;
  message: string;
  category: 'system' | 'bid' | 'king' | 'talon' | 'play' | 'trick' | 'score' | 'announce';
  player?: number;
  timestamp: number;
}

const INITIAL_STATE: SpectatorState = {
  phase: 'waiting',
  hands: [[], [], [], []],
  hand_sizes: [0, 0, 0, 0],
  talon_groups: null,
  bids: [],
  contract: null,
  declarer: null,
  called_king: null,
  partner_revealed: false,
  partner: null,
  current_trick: [],
  tricks_played: 0,
  current_player: 0,
  scores: null,
  player_names: [],
  completed_tricks: [],
  roles: {},
  announcements: {},
  kontra_levels: {},
  put_down: [],
  score_breakdown: null,
  trick_summary: null,
};

function formatSpectatorEvent(
  event: string,
  data: Record<string, unknown>,
  names: string[],
): SpectatorLogEntry | null {
  const name = (idx: number) => names[idx] ?? `Player ${idx}`;
  const ts = Date.now();

  switch (event) {
    case 'game_start':
      return { id: ts, message: 'Game started.', category: 'system', timestamp: ts };
    case 'deal':
      return { id: ts, message: 'Cards dealt to all players.', category: 'system', timestamp: ts };
    case 'bid': {
      const p = data.player as number;
      const c = data.contract as number | null;
      const bidText = c !== null ? (CONTRACT_NAMES[c] ?? `${c}`) : 'Pass';
      return { id: ts, message: `${name(p)} bids: ${bidText}`, category: 'bid', player: p, timestamp: ts };
    }
    case 'contract_won': {
      const p = data.player as number;
      const c = data.contract as number;
      const who = p === -1 ? 'Nobody' : name(p);
      return { id: ts, message: `${who} wins the contract: ${CONTRACT_NAMES[c] ?? c}`, category: 'bid', player: p, timestamp: ts };
    }
    case 'king_called': {
      const p = data.player as number;
      const king = data.king as { suit?: string; label?: string };
      const suit = king.suit ? (SUIT_SYMBOLS[king.suit] ?? king.suit) : '';
      return { id: ts, message: `${name(p)} calls ${suit} King`, category: 'king', player: p, timestamp: ts };
    }
    case 'talon_revealed':
      return { id: ts, message: 'Talon revealed.', category: 'talon', timestamp: ts };
    case 'talon_exchanged': {
      const picked = data.picked as { label?: string }[] | undefined;
      const discarded = data.discarded as { label?: string }[] | undefined;
      const pickedStr = picked?.map(c => c.label ?? '?').join(', ') ?? '';
      const discardedStr = discarded?.map(c => c.label ?? '?').join(', ') ?? '';
      let msg = 'Talon exchange complete.';
      if (pickedStr) msg += ` Picked: ${pickedStr}.`;
      if (discardedStr) msg += ` Put down: ${discardedStr}.`;
      return { id: ts, message: msg, category: 'talon', timestamp: ts };
    }
    case 'card_played': {
      const p = data.player as number;
      const card = data.card as { label?: string };
      return { id: ts, message: `${name(p)} plays ${card.label ?? '?'}`, category: 'play', player: p, timestamp: ts };
    }
    case 'rule_verified': {
      const p = data.player as number;
      const rule = data.rule as string;
      return { id: ts, message: `[Rule Check] ${name(p)} ${rule}`, category: 'announce', player: p, timestamp: ts };
    }
    case 'trick_won': {
      const w = data.winner as number;
      return { id: ts, message: `${name(w)} wins the trick!`, category: 'trick', player: w, timestamp: ts };
    }
    case 'game_end': {
      const scores = data.scores as Record<string, number>;
      const lines = Object.entries(scores)
        .map(([pid, s]) => `${name(Number(pid))}: ${s > 0 ? '+' : ''}${s}`)
        .join(' | ');
      return { id: ts, message: `Game over — ${lines}`, category: 'score', timestamp: ts };
    }
    default:
      return null;
  }
}

export interface AgentConfig {
  name: string;
  type: string;
  checkpoint?: string;
}

function toTimelineItem(msg: SpectatorEvent) {
  const names = msg.state.player_names.length > 0
    ? msg.state.player_names
    : ['Agent-0', 'Agent-1', 'Agent-2', 'Agent-3'];
  const entry = formatSpectatorEvent(msg.event, msg.data, names);

  const enrichedState: SpectatorState = {
    ...msg.state,
    score_breakdown: msg.event === 'game_end'
      ? (msg.data.breakdown as ScoreBreakdown) ?? null
      : null,
    trick_summary: msg.event === 'game_end'
      ? (msg.data.trick_summary as TrickSummaryEntry[]) ?? null
      : null,
  };

  return { state: enrichedState, eventName: msg.event, logEntry: entry };
}

export function useSpectator() {
  const [timeline, setTimeline] = useState<{ state: SpectatorState; eventName: string; logEntry: SpectatorLogEntry | null }[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1500); // ms per step

  const [gameId, setGameId] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState<'live' | 'replay'>('live');
  const [replayName, setReplayName] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Tournament tracking
  const [totalGames, setTotalGames] = useState(1);
  const [currentGameNum, setCurrentGameNum] = useState(1);
  const [cumulativeScores, setCumulativeScores] = useState<Record<string, number>>({});
  const [gamesPlayed, setGamesPlayed] = useState(0);

  const disconnect = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
    setConnected(false);
    setGameId(null);
    setIsPlaying(false);
    setMode('live');
    setReplayName(null);
  }, []);

  const startGame = useCallback(async (agents: AgentConfig[], numGames?: number) => {
    setLoading(true);
    setTimeline([]);
    setCurrentIndex(0);
    setIsPlaying(false);
    setMode('live');
    setReplayName(null);
    if (numGames !== undefined) {
      setTotalGames(numGames);
      setCurrentGameNum(1);
      setCumulativeScores({});
      setGamesPlayed(0);
    }

    try {
      const res = await fetch('/api/spectate/new', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        // Always send 0 delay so the whole game is instantly available for scrubbing
        body: JSON.stringify({ agents, delay: 0 }),
      });
      const data = await res.json();
      const id = data.game_id;
      setGameId(id);

      const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
      const ws = new WebSocket(`${protocol}://${window.location.host}/ws/spectate/${id}`);

      ws.onopen = () => setConnected(true);

      ws.onmessage = (e) => {
        const msg: SpectatorEvent = JSON.parse(e.data);
        setTimeline(prev => {
          const newState = [...prev, toTimelineItem(msg)];
          if (newState.length === 1) {
            setIsPlaying(true);
          }
          return newState;
        });
      };

      ws.onclose = () => setConnected(false);
      ws.onerror = () => setConnected(false);

      wsRef.current = ws;
    } finally {
      setLoading(false);
    }
  }, []);

  const loadReplay = useCallback(async (filename: string) => {
    setLoading(true);
    disconnect();
    setTimeline([]);
    setCurrentIndex(0);

    try {
      const res = await fetch(`/api/replays/${encodeURIComponent(filename)}`);
      const payload = await res.json();
      const replayTimeline = ((payload.timeline ?? []) as SpectatorEvent[]).map(toTimelineItem);
      setTimeline(replayTimeline);
      setGameId(`replay:${filename}`);
      setMode('replay');
      setReplayName(filename);
      if (replayTimeline.length > 0) {
        setIsPlaying(true);
      }
    } finally {
      setLoading(false);
    }
  }, [disconnect]);

  // Playback control
  useEffect(() => {
    if (isPlaying) {
      timerRef.current = setInterval(() => {
        setCurrentIndex(idx => {
          if (idx < timeline.length - 1) return idx + 1;
          setIsPlaying(false);
          return idx;
        });
      }, playbackSpeed);
    } else if (timerRef.current) {
      clearInterval(timerRef.current);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isPlaying, playbackSpeed, timeline.length]);

  const togglePlay = () => setIsPlaying(p => !p);

  const stepForward = () => {
    setIsPlaying(false);
    setCurrentIndex(idx => Math.min(idx + 1, timeline.length - 1));
  };

  const stepBack = () => {
    setIsPlaying(false);
    setCurrentIndex(idx => Math.max(idx - 1, 0));
  };

  const nextTrick = () => {
    setIsPlaying(false);
    setCurrentIndex(idx => {
      for (let i = idx + 1; i < timeline.length; i++) {
        if (timeline[i].eventName === 'trick_won') return i;
      }
      return timeline.length - 1;
    });
  };

  const prevTrick = () => {
    setIsPlaying(false);
    setCurrentIndex(idx => {
      // Find the start of the previous trick (immediately after a trick_won)
      for (let i = idx - 1; i > 0; i--) {
        if (timeline[i].eventName === 'trick_won') return i;
      }
      return 0;
    });
  };

  const jumpToIndex = (index: number) => {
    setIsPlaying(false);
    setCurrentIndex(Math.max(0, Math.min(index, timeline.length - 1)));
  };

  useEffect(() => {
    return () => disconnect();
  }, [disconnect]);

  const currentItem = timeline[currentIndex];
  const state = currentItem?.state ?? INITIAL_STATE;
  
  // Collect logs up to current index
  const logEntries = timeline.slice(0, currentIndex + 1).map(t => t.logEntry).filter(Boolean) as SpectatorLogEntry[];

  // Accumulate scores when game ends and current index is at the end
  const gameFinished = state.phase === 'finished' && state.scores !== null;

  // Start next game in a tournament, accumulating current game scores first
  const continueNextGame = useCallback(async (agents: AgentConfig[]) => {
    if (!gameFinished || !state.scores) return;
    // Accumulate
    setCumulativeScores(prev => {
      const updated = { ...prev };
      Object.entries(state.scores!).forEach(([pid, score]) => {
        updated[pid] = (updated[pid] ?? 0) + Number(score);
      });
      return updated;
    });
    setGamesPlayed(g => g + 1);
    setCurrentGameNum(n => n + 1);

    // Start fresh game without resetting tournament state
    setLoading(true);
    setTimeline([]);
    setCurrentIndex(0);
    setIsPlaying(false);

    try {
      const res = await fetch('/api/spectate/new', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ agents, delay: 0 }),
      });
      const data = await res.json();
      const id = data.game_id;
      setGameId(id);

      const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
      const ws = new WebSocket(`${protocol}://${window.location.host}/ws/spectate/${id}`);

      ws.onopen = () => setConnected(true);

      ws.onmessage = (e) => {
        const msg: SpectatorEvent = JSON.parse(e.data);
        const evNames = msg.state.player_names.length > 0
          ? msg.state.player_names
          : ['Agent-0', 'Agent-1', 'Agent-2', 'Agent-3'];
        const entry = formatSpectatorEvent(msg.event, msg.data, evNames);
        const enrichedState: SpectatorState = {
          ...msg.state,
          score_breakdown: msg.event === 'game_end' ? (msg.data.breakdown as ScoreBreakdown) ?? null : null,
          trick_summary: msg.event === 'game_end' ? (msg.data.trick_summary as TrickSummaryEntry[]) ?? null : null,
        };
        setTimeline(prev => {
          const newState = [...prev, { state: enrichedState, eventName: msg.event, logEntry: entry }];
          if (newState.length === 1) setIsPlaying(true);
          return newState;
        });
      };

      ws.onclose = () => setConnected(false);
      ws.onerror = () => setConnected(false);
      wsRef.current = ws;
    } finally {
      setLoading(false);
    }
  }, [gameFinished, state.scores]);

  const currentEventName = currentItem?.eventName ?? null;

  return {
    state,
    gameId,
    connected,
    loading,
    mode,
    replayName,
    logEntries,
    currentEventName,
    startGame,
    loadReplay,
    disconnect,
    
    // Playback
    isPlaying,
    currentIndex,
    totalSteps: timeline.length,
    togglePlay,
    stepForward,
    stepBack,
    nextTrick,
    prevTrick,
    jumpToIndex,
    playbackSpeed,
    setPlaybackSpeed,

    // Tournament
    totalGames,
    setTotalGames,
    currentGameNum,
    cumulativeScores,
    gamesPlayed,
    gameFinished,
    continueNextGame,
  };
}
