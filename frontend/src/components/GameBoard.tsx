import React from 'react';
import type { GameState, CardData } from '../types/game';
import { CONTRACT_NAMES } from '../types/game';
import Hand from './Hand';
import TrickArea from './TrickArea';
import BiddingPanel from './BiddingPanel';
import Card from './Card';
import './GameBoard.css';

interface GameBoardProps {
  state: GameState;
  onPlayCard: (card: CardData) => void;
  onBid: (contract: number | null) => void;
  onCallKing: (suit: string) => void;
  onChooseTalon: (groupIndex: number) => void;
  onDiscard: (cards: CardData[]) => void;
}

export default function GameBoard({
  state, onPlayCard, onBid, onCallKing, onChooseTalon, onDiscard,
}: GameBoardProps) {
  const isMyTurn = state.current_player === 0;
  const names = state.player_names.length > 0 ? state.player_names : ['You', 'AI-1', 'AI-2', 'AI-3'];

  return (
    <div className="game-board" data-testid="game-board" data-phase={state.phase}>      {/* Game info bar */}
      <div className="game-info-bar">
        <div className="info-item">
          <span className="info-label">Tricks</span>
          <span className="info-value">{state.tricks_played}/12</span>
        </div>
        {state.contract !== null && (
          <div className="info-item">
            <span className="info-label">Contract</span>
            <span className="info-value">{CONTRACT_NAMES[state.contract] ?? state.contract}</span>
          </div>
        )}
        {state.declarer !== null && (
          <div className="info-item">
            <span className="info-label">Declarer</span>
            <span className="info-value">{names[state.declarer]}</span>
          </div>
        )}
        {state.called_king && (
          <div className="info-item">
            <span className="info-label">Called</span>
            <span className="info-value">{state.called_king.label}</span>
          </div>
        )}
        <div className="info-item">
          <span className="info-label">Phase</span>
          <span className="info-value phase-badge">{state.phase.replace(/_/g, ' ')}</span>
        </div>
      </div>

      {/* Table layout */}
      <div className="table">
        {/* Top player (P2) */}
        <div className="table-top">
          <Hand cards={[]} faceDown cardCount={state.hand_sizes[2]} position="top" label={names[2]} />
        </div>

        {/* Left player (P1) */}
        <div className="table-left">
          <Hand cards={[]} faceDown cardCount={state.hand_sizes[1]} position="left" label={names[1]} />
        </div>

        {/* Center — trick area */}
        <div className="table-center">
          {state.phase === 'trick_play' && (
            <TrickArea
              trickCards={state.current_trick}
              playerNames={names}
              playerIndex={0}
            />
          )}

          {/* Bidding */}
          {(state.phase === 'bidding' || state.phase === 'king_calling') && (
            <BiddingPanel
              phase={state.phase}
              bids={state.bids}
              legalBids={isMyTurn && state.phase === 'bidding' ? getLegalBidValues(state) : undefined}
              onBid={onBid}
              playerNames={names}
              callableKings={isMyTurn && state.phase === 'king_calling' ? state.legal_plays : undefined}
              onCallKing={onCallKing}
            />
          )}

          {/* Talon selection */}
          {state.phase === 'talon_exchange' && state.talon_groups && isMyTurn && (
            <div className="talon-selection">
              <h3>Choose a talon group</h3>
              <div className="talon-groups">
                {state.talon_groups.map((group, i) => (
                  <div key={i} className="talon-group" onClick={() => onChooseTalon(i)}>
                    {group.map((card, j) => (
                      <Card key={j} card={card} small />
                    ))}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Scores */}
          {state.phase === 'finished' && state.scores && (
            <div className="score-display" data-testid="score-display">
              <h3>Game Over!</h3>
              <div className="score-list">
                {Object.entries(state.scores).map(([pid, score]) => (
                  <div key={pid} className={`score-entry ${Number(score) > 0 ? 'score-positive' : 'score-negative'}`}>
                    <span>{names[Number(pid)]}</span>
                    <span className="score-value">{Number(score) > 0 ? '+' : ''}{score}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right player (P3) */}
        <div className="table-right">
          <Hand cards={[]} faceDown cardCount={state.hand_sizes[3]} position="right" label={names[3]} />
        </div>

        {/* Bottom player (human, P0) */}
        <div className="table-bottom">
          <Hand
            cards={state.hand}
            legalPlays={isMyTurn && state.phase === 'trick_play' ? state.legal_plays : undefined}
            onCardClick={isMyTurn && state.phase === 'trick_play' ? onPlayCard : undefined}
            position="bottom"
            label={names[0]}
          />
        </div>
      </div>

      {/* Turn indicator */}
      {state.phase === 'trick_play' && (
        <div className={`turn-indicator ${isMyTurn ? 'your-turn' : ''}`} data-testid="turn-indicator">
          {isMyTurn ? '🎯 Your turn — play a card' : `Waiting for ${names[state.current_player]}...`}
        </div>
      )}
    </div>
  );
}

function getLegalBidValues(state: GameState): (number | null)[] {
  // Derived from the bids so far
  const highestBid = state.bids.reduce((max, b) => {
    if (b.contract !== null && (max === null || bidStrength(b.contract) > bidStrength(max))) {
      return b.contract;
    }
    return max;
  }, null as number | null);

  const options: (number | null)[] = [null]; // pass
  const allContracts = [3, 2, 1, 0];
  for (const c of allContracts) {
    if (highestBid === null || bidStrength(c) > bidStrength(highestBid)) {
      options.push(c);
    }
  }
  return options;
}

function bidStrength(contract: number): number {
  const strengths: Record<number, number> = { 3: 1, 2: 2, 1: 3, '-3': 4, '-2': 5, '-1': 6, 0: 7 };
  return strengths[contract] ?? 0;
}
