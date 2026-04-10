import React, { useState } from 'react';
import type { GameState, CardData, TrickCard } from '../types/game';
import { CONTRACT_NAMES } from '../types/game';
import Hand from './Hand';
import TrickArea from './TrickArea';
import BiddingPanel from './BiddingPanel';
import Card from './Card';
import './GameBoard.css';

function cardKey(c: CardData): string {
  return `${c.card_type}-${c.value}-${c.suit ?? 'none'}`;
}

const SOLO_CONTRACTS = new Set([-3, -2, -1, 0, -100, -101]);

type TeamRole = 'declarer' | 'defender' | null;

function getTeamRole(state: GameState, playerIdx: number): TeamRole {
  if (state.contract === null || state.contract === -99) return null; // no contract or klop
  if (state.declarer === null) return null;
  const solo = SOLO_CONTRACTS.has(state.contract);
  if (playerIdx === state.declarer) return 'declarer';
  if (!solo && state.partner_revealed && state.partner === playerIdx) return 'declarer';
  return 'defender';
}

interface GameBoardProps {
  state: GameState;
  onPlayCard: (card: CardData) => void;
  onBid: (contract: number | null) => void;
  onCallKing: (suit: string) => void;
  onChooseTalon: (groupIndex: number) => void;
  onDiscard: (cards: CardData[]) => void;
  trickWinner?: number | null;
  trickWinCards?: TrickCard[];
}

export default function GameBoard({
  state, onPlayCard, onBid, onCallKing, onChooseTalon, onDiscard, trickWinner, trickWinCards,
}: GameBoardProps) {
  const isMyTurn = state.current_player === 0;
  const names = state.player_names.length > 0 ? state.player_names : ['You', 'AI-1', 'AI-2', 'AI-3'];
  const [discardSelection, setDiscardSelection] = useState<CardData[]>([]);

  const mustDiscard = state.must_discard;
  const suitDiscardables = state.hand.filter(c => c.card_type !== 'tarok' && !(c.card_type === 'suit' && c.value === 8));
  const canUseTarokDiscards = suitDiscardables.length < mustDiscard;
  const discardCandidates = canUseTarokDiscards
    ? state.hand.filter(c => !(c.card_type === 'suit' && c.value === 8))
    : suitDiscardables;
  const isSolo = state.contract !== null && SOLO_CONTRACTS.has(state.contract);
  const teamOf = (idx: number) => getTeamRole(state, idx);

  const isDiscardSelectionValid = (selection: CardData[]) => {
    if (selection.length !== mustDiscard) return false;
    if (selection.some(c => c.card_type === 'suit' && c.value === 8)) return false;

    const selectedSuitDiscardables = selection.filter(
      c => c.card_type !== 'tarok' && !(c.card_type === 'suit' && c.value === 8),
    );
    const selectedTaroks = selection.filter(c => c.card_type === 'tarok');

    // Backend rule: if any tarok is discarded, all discardable suit cards must also be discarded.
    if (selectedTaroks.length > 0 && selectedSuitDiscardables.length < suitDiscardables.length) {
      return false;
    }

    return true;
  };

  const toggleDiscard = (card: CardData) => {
    setDiscardSelection(prev => {
      const exists = prev.some(c => cardKey(c) === cardKey(card));
      if (exists) return prev.filter(c => cardKey(c) !== cardKey(card));
      if (prev.length >= mustDiscard) return prev;
      return [...prev, card];
    });
  };

  const submitDiscard = () => {
    if (isDiscardSelectionValid(discardSelection)) {
      onDiscard(discardSelection);
      setDiscardSelection([]);
    }
  };

  // Show trick-win animation cards when winner is set
  const showTrickAnimation = trickWinner != null && trickWinCards && trickWinCards.length > 0;
  const displayTrickCards = showTrickAnimation ? trickWinCards! : state.current_trick;

  const revealedHands = state.hands;

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
          <Hand cards={revealedHands?.['2'] ?? []} faceDown={!revealedHands?.['2']} cardCount={state.hand_sizes[2]} position="top" label={names[2]} teamRole={teamOf(2)} isSolo={isSolo} />
        </div>

        {/* Left player (P1) */}
        <div className="table-left">
          <Hand cards={revealedHands?.['1'] ?? []} faceDown={!revealedHands?.['1']} cardCount={state.hand_sizes[1]} position="left" label={names[1]} teamRole={teamOf(1)} isSolo={isSolo} />
        </div>

        {/* Center — trick area */}
        <div className="table-center">
          {(state.phase === 'trick_play' || showTrickAnimation) && (
            <TrickArea
              trickCards={displayTrickCards}
              playerNames={names}
              playerIndex={0}
              getTeamRole={(idx) => teamOf(idx)}
              isSolo={isSolo}
              trickWinner={trickWinner}
            />
          )}

          {/* Bidding */}
          {(state.phase === 'bidding' || state.phase === 'king_calling') && (
            <BiddingPanel
              phase={state.phase}
              bids={state.bids}
              legalBids={isMyTurn && state.phase === 'bidding' && state.legal_bids ? state.legal_bids : undefined}
              onBid={onBid}
              playerNames={names}
              callableKings={isMyTurn && state.phase === 'king_calling' && state.callable_kings ? state.callable_kings : undefined}
              onCallKing={onCallKing}
            />
          )}

          {/* Talon selection / Discard — shown in center only during active choice */}
          {state.phase === 'talon_exchange' && isMyTurn && mustDiscard > 0 && (
            <div className="talon-selection">
              <h3>Discard {mustDiscard} card{mustDiscard > 1 ? 's' : ''}</h3>
              <p className="bidding-subtitle">
                {canUseTarokDiscards
                  ? 'Select cards to put down (no kings). If discarding taroks, include all discardable suit cards.'
                  : 'Select cards from your hand to put down (no kings or taroks)'}
              </p>
              <div className="discard-hand">
                {discardCandidates.map((card, j) => {
                  const selected = discardSelection.some(c => cardKey(c) === cardKey(card));
                  return (
                    <div key={j} className={`discard-card ${selected ? 'discard-selected' : ''}`} onClick={() => toggleDiscard(card)}>
                      <Card card={card} small highlighted={selected} />
                    </div>
                  );
                })}
              </div>
              <button
                className="btn-gold"
                data-testid="discard-confirm"
                disabled={!isDiscardSelectionValid(discardSelection)}
                onClick={submitDiscard}
              >
                Confirm Discard ({discardSelection.length}/{mustDiscard})
              </button>
            </div>
          )}
          {state.phase === 'talon_exchange' && state.talon_groups && isMyTurn && mustDiscard === 0 && (
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
                {Object.entries(state.scores).map(([pid, score]) => {
                  const role = teamOf(Number(pid));
                  const teamCls = role === 'declarer' ? (isSolo ? 'team-solo' : 'team-declarer') : role === 'defender' ? 'team-defender' : '';
                  return (
                  <div key={pid} className={`score-entry ${Number(score) > 0 ? 'score-positive' : 'score-negative'} ${teamCls}`}>
                    <span>{names[Number(pid)]}</span>
                    <span className="score-value">{Number(score) > 0 ? '+' : ''}{score}</span>
                  </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Right player (P3) */}
        <div className="table-right">
          <Hand cards={revealedHands?.['3'] ?? []} faceDown={!revealedHands?.['3']} cardCount={state.hand_sizes[3]} position="right" label={names[3]} teamRole={teamOf(3)} isSolo={isSolo} />
        </div>

        {/* Bottom player (human, P0) */}
        <div className="table-bottom">
          <Hand
            cards={state.hand}
            legalPlays={isMyTurn && state.phase === 'trick_play' ? state.legal_plays : undefined}
            onCardClick={isMyTurn && state.phase === 'trick_play' ? onPlayCard : undefined}
            position="bottom"
            label={names[0]}
            teamRole={teamOf(0)}
            isSolo={isSolo}
          />
        </div>
      </div>

      {/* Turn indicator */}
      {state.phase === 'trick_play' && !showTrickAnimation && (
        <div className={`turn-indicator ${isMyTurn ? 'your-turn' : ''}`} data-testid="turn-indicator">
          {isMyTurn ? '🎯 Your turn — play a card' : `Waiting for ${names[state.current_player]}...`}
        </div>
      )}
    </div>
  );
}