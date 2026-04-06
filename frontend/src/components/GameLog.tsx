import React, { useEffect, useRef } from 'react';
import type { LogEntry } from '../hooks/useGame';
import './GameLog.css';

interface GameLogProps {
  entries: LogEntry[];
}

const CATEGORY_ICONS: Record<string, string> = {
  system: '⚙️',
  bid: '🗣️',
  king: '👑',
  talon: '📦',
  play: '🃏',
  trick: '✅',
  score: '🏆',
};

export default function GameLog({ entries }: GameLogProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [entries]);

  return (
    <div className="game-log" data-testid="game-log">
      <div className="game-log-header">
        <h3>Game Log</h3>
      </div>
      <div className="game-log-entries">
        {entries.length === 0 && (
          <div className="game-log-empty">Waiting for game to start…</div>
        )}
        {entries.map((entry) => (
          <div
            key={entry.id}
            className={`game-log-entry log-${entry.category}${entry.isHuman ? ' log-human' : ''}`}
          >
            <span className="log-icon">{CATEGORY_ICONS[entry.category] ?? '•'}</span>
            <span className="log-message">{entry.message}</span>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
