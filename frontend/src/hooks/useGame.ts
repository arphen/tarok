import { useState, useEffect, useRef, useCallback } from 'react';
import type { GameState, GameEvent, CardData } from '../types/game';
import { CONTRACT_NAMES, SUIT_SYMBOLS } from '../types/game';

export interface LogEntry {
  id: number;
  message: string;
  category: 'system' | 'bid' | 'king' | 'talon' | 'play' | 'trick' | 'score';
  player?: number;
  isHuman?: boolean;
}

const INITIAL_STATE: GameState = {
  phase: 'waiting',
  hand: [],
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
  legal_plays: [],
  legal_bids: null,
  callable_kings: null,
  must_discard: 0,
  player_names: [],
  card_tracker: null,
  match_info: null,
  hands: null,
};

function cardLabel(card: CardData): string {
  if (card.card_type === 'tarok') return card.label;
  return `${card.label}`;
}

function formatEvent(event: string, data: Record<string, unknown>, names: string[]): LogEntry | null {
  const name = (idx: number) => names[idx] ?? `P${idx}`;
  const isHuman = (idx: number) => idx === 0;
  let nextId = Date.now();

  switch (event) {
    case 'game_start':
      return { id: nextId, message: 'Game started. Dealing cards...', category: 'system' };
    case 'deal':
      return { id: nextId, message: 'Cards dealt to all players.', category: 'system' };
    case 'bid': {
      const p = data.player as number;
      const c = data.contract as number | null;
      const bidText = c !== null ? (CONTRACT_NAMES[c] ?? `${c}`) : 'Pass';
      return { id: nextId, message: `${name(p)} bids: ${bidText}`, category: 'bid', player: p, isHuman: isHuman(p) };
    }
    case 'contract_won': {
      const p = data.player as number;
      const c = data.contract as number;
      return { id: nextId, message: `${name(p)} wins the contract: ${CONTRACT_NAMES[c] ?? c}`, category: 'bid', player: p, isHuman: isHuman(p) };
    }
    case 'king_called': {
      const p = data.player as number;
      const king = data.king as CardData;
      const suit = king.suit ? (SUIT_SYMBOLS[king.suit] ?? king.suit) : '';
      return { id: nextId, message: `${name(p)} calls ${suit} King — the holder is the secret partner!`, category: 'king', player: p, isHuman: isHuman(p) };
    }
    case 'talon_revealed':
      return { id: nextId, message: 'Talon revealed.', category: 'talon' };
    case 'talon_exchanged':
      return { id: nextId, message: 'Talon exchange complete.', category: 'talon' };
    case 'card_played': {
      const p = data.player as number;
      const card = data.card as CardData;
      return { id: nextId, message: `${name(p)} plays ${cardLabel(card)}`, category: 'play', player: p, isHuman: isHuman(p) };
    }
    case 'trick_won': {
      const w = data.winner as number;
      return { id: nextId, message: `${name(w)} wins the trick!`, category: 'trick', player: w, isHuman: isHuman(w) };
    }
    case 'game_end': {
      const scores = data.scores as Record<string, number>;
      const lines = Object.entries(scores)
        .map(([pid, s]) => `${name(Number(pid))}: ${s > 0 ? '+' : ''}${s}`)
        .join(' | ');
      return { id: nextId, message: `Round over — ${lines}`, category: 'score' };
    }
    case 'match_update': {
      const rn = data.round_num as number;
      const tr = data.total_rounds as number;
      return { id: nextId, message: `Round ${rn}/${tr} complete. Next round starting...`, category: 'system' };
    }
    case 'match_end': {
      const cs = data.cumulative_scores as Record<string, number>;
      const lines = Object.entries(cs)
        .map(([pid, s]) => `${name(Number(pid))}: ${s > 0 ? '+' : ''}${s}`)
        .join(' | ');
      return { id: nextId, message: `Match over! Final scores — ${lines}`, category: 'score' };
    }
    default:
      return null;
  }
}

export function useGame() {
  const [gameState, setGameState] = useState<GameState>(INITIAL_STATE);
  const [gameId, setGameId] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const [events, setEvents] = useState<string[]>([]);
  const [logEntries, setLogEntries] = useState<LogEntry[]>([]);
  const [trickWinner, setTrickWinner] = useState<number | null>(null);
  const [trickWinCards, setTrickWinCards] = useState<GameState['current_trick']>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const trickTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const addEvent = useCallback((msg: string) => {
    setEvents(prev => [...prev.slice(-50), msg]);
  }, []);

  const addLogEntry = useCallback((entry: LogEntry) => {
    setLogEntries(prev => [...prev.slice(-49), entry]);
  }, []);

  const connect = useCallback((id: string) => {
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${protocol}://${window.location.host}/ws/game/${id}`);

    ws.onopen = () => {
      setConnected(true);
      addEvent('Connected to game');
    };

    ws.onmessage = (e) => {
      const data: GameEvent = JSON.parse(e.data);
      addEvent(`Event: ${data.event}`);
      const names = data.state.player_names.length > 0 ? data.state.player_names : ['You', 'AI-1', 'AI-2', 'AI-3'];
      const entry = formatEvent(data.event, data.data, names);
      if (entry) addLogEntry(entry);

      // Always apply state immediately to avoid stale-state rollback
      setGameState(data.state);

      if (data.event === 'trick_won') {
        // Show sweep animation with the trick cards from the event
        const winner = data.data.winner as number;
        const cards = data.data.cards as GameState['current_trick'];
        setTrickWinCards(cards);
        setTrickWinner(winner);
        // Clear after animation completes
        if (trickTimerRef.current) clearTimeout(trickTimerRef.current);
        trickTimerRef.current = setTimeout(() => {
          setTrickWinner(null);
          setTrickWinCards([]);
        }, 900);
      }
    };

    ws.onclose = () => {
      setConnected(false);
      addEvent('Disconnected');
    };

    ws.onerror = () => {
      addEvent('WebSocket error');
    };

    wsRef.current = ws;
  }, [addEvent, addLogEntry]);

  const startNewGame = useCallback(async (opponents?: string[], numRounds?: number) => {
    try {
      const body: Record<string, unknown> = {};
      if (opponents) body.opponents = opponents;
      if (numRounds && numRounds > 1) body.num_rounds = numRounds;
      const hasBody = Object.keys(body).length > 0;
      const res = await fetch('/api/game/new', {
        method: 'POST',
        headers: hasBody ? { 'Content-Type': 'application/json' } : {},
        body: hasBody ? JSON.stringify(body) : undefined,
      });
      const data = await res.json();
      setGameId(data.game_id);
      connect(data.game_id);
    } catch (e) {
      addEvent('Failed to create game');
    }
  }, [connect, addEvent]);

  const sendAction = useCallback((action: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(action));
    }
  }, []);

  const playCard = useCallback((card: { card_type: string; value: number; suit: string | null }) => {
    sendAction({ action: 'play_card', card });
  }, [sendAction]);

  const bid = useCallback((contract: number | null) => {
    sendAction({ action: 'bid', contract });
  }, [sendAction]);

  const callKing = useCallback((suit: string) => {
    sendAction({ action: 'call_king', suit });
  }, [sendAction]);

  const chooseTalon = useCallback((groupIndex: number) => {
    sendAction({ action: 'choose_talon', group_index: groupIndex });
  }, [sendAction]);

  const discard = useCallback((cards: { card_type: string; value: number; suit: string | null }[]) => {
    sendAction({ action: 'discard', cards });
  }, [sendAction]);

  const setDelay = useCallback((delay: number) => {
    sendAction({ action: 'set_delay', delay });
  }, [sendAction]);

  const revealHands = useCallback((reveal: boolean) => {
    sendAction({ action: 'reveal_hands', reveal });
  }, [sendAction]);

  useEffect(() => {
    return () => {
      wsRef.current?.close();
    };
  }, []);

  return {
    gameState,
    gameId,
    connected,
    events,
    logEntries,
    trickWinner,
    trickWinCards,
    startNewGame,
    playCard,
    bid,
    callKing,
    chooseTalon,
    discard,
    setDelay,
    revealHands,
  };
}
