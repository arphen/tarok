import React, { useState, useRef, useCallback } from 'react';
import type { CardData } from '../types/game';
import Card from './Card';
import { SUIT_SYMBOLS } from '../types/game';
import './CameraAgent.css';

// ---- Card data builders ----

const ALL_TAROKS: CardData[] = Array.from({ length: 22 }, (_, i) => ({
  card_type: 'tarok' as const,
  value: i + 1,
  suit: null,
  label: i + 1 === 22 ? 'Škis' : ['I','II','III','IV','V','VI','VII','VIII','IX','X','XI','XII','XIII','XIV','XV','XVI','XVII','XVIII','XIX','XX','XXI'][i],
  points: [1, 21, 22].includes(i + 1) ? 5 : 1,
}));

const SUITS: ('hearts' | 'diamonds' | 'clubs' | 'spades')[] = ['hearts', 'diamonds', 'clubs', 'spades'];
const SUIT_RANKS = [
  { value: 8, label: 'K', points: 5 },
  { value: 7, label: 'Q', points: 4 },
  { value: 6, label: 'C', points: 3 },
  { value: 5, label: 'J', points: 2 },
  { value: 4, label: '4', points: 1 },
  { value: 3, label: '3', points: 1 },
  { value: 2, label: '2', points: 1 },
  { value: 1, label: '1', points: 1 },
];

function makeSuitCard(suit: string, rank: typeof SUIT_RANKS[0]): CardData {
  const isRed = suit === 'hearts' || suit === 'diamonds';
  const pipLabels: Record<number, string> = isRed
    ? { 1: '1', 2: '2', 3: '3', 4: '4' }
    : { 1: '7', 2: '8', 3: '9', 4: '10' };
  const faceLabels: Record<number, string> = { 5: 'J', 6: 'C', 7: 'Q', 8: 'K' };
  const sym = SUIT_SYMBOLS[suit as keyof typeof SUIT_SYMBOLS] || suit;
  const lbl = rank.value <= 4 ? pipLabels[rank.value] : faceLabels[rank.value];
  return {
    card_type: 'suit',
    value: rank.value,
    suit: suit as CardData['suit'],
    label: `${lbl}${sym}`,
    points: rank.points,
  };
}

const CONTRACTS: { id: string; name: string; talon: number }[] = [
  { id: 'three', name: 'Three', talon: 3 },
  { id: 'two', name: 'Two', talon: 2 },
  { id: 'one', name: 'One', talon: 1 },
  { id: 'solo_three', name: 'Solo Three', talon: 3 },
  { id: 'solo_two', name: 'Solo Two', talon: 2 },
  { id: 'solo_one', name: 'Solo One', talon: 1 },
  { id: 'solo', name: 'Solo', talon: 0 },
];

const PLAYER_NAMES = ['You (P0)', 'Player 1', 'Player 2', 'Player 3'];

// ---- Types ----

type Step =
  | 'scan_hand'
  | 'bidding'
  | 'king_call'
  | 'talon'
  | 'trick_play'
  | 'finished';

interface BidEntry {
  player: number;
  contract: string | null;
}

interface TrickRecord {
  cards: { player: number; card: CardData }[];
  winner: number;
}

interface BidRecommendation {
  recommended: number | null;
  recommended_name: string;
  ranked_bids: { contract: number | null; name: string; probability: number }[];
  has_trained_model: boolean;
}

interface KingRecommendation {
  recommended: CardData | null;
  callable_kings: CardData[];
  has_trained_model: boolean;
}

interface PlayRecommendation {
  recommended: CardData;
  legal_plays: CardData[];
  ranked_plays: { card: CardData; probability: number }[];
  position_value: number | null;
  has_trained_model: boolean;
}

// ---- Reusable card picker ----

function CardPicker({
  selected,
  onToggle,
  disabled,
  maxCards,
}: {
  selected: CardData[];
  onToggle: (card: CardData) => void;
  disabled?: CardData[];
  maxCards?: number;
}) {
  const [tab, setTab] = useState<'taroks' | 'hearts' | 'diamonds' | 'clubs' | 'spades'>('taroks');

  const isSelected = (card: CardData) =>
    selected.some(c => c.card_type === card.card_type && c.value === card.value && c.suit === card.suit);
  const isDisabled = (card: CardData) =>
    disabled?.some(c => c.card_type === card.card_type && c.value === card.value && c.suit === card.suit) ?? false;

  const tabCards: Record<string, CardData[]> = {
    taroks: ALL_TAROKS,
    hearts: SUIT_RANKS.map(r => makeSuitCard('hearts', r)),
    diamonds: SUIT_RANKS.map(r => makeSuitCard('diamonds', r)),
    clubs: SUIT_RANKS.map(r => makeSuitCard('clubs', r)),
    spades: SUIT_RANKS.map(r => makeSuitCard('spades', r)),
  };

  const atMax = maxCards !== undefined && selected.length >= maxCards;

  return (
    <div className="cm-card-picker">
      <div className="ca-tabs">
        {(['taroks', 'hearts', 'diamonds', 'clubs', 'spades'] as const).map(t => (
          <button key={t} className={`ca-tab ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>
            {t === 'taroks' ? '🃏 Taroks' : `${SUIT_SYMBOLS[t]} ${t.charAt(0).toUpperCase() + t.slice(1)}`}
          </button>
        ))}
      </div>
      <div className="ca-card-grid">
        {tabCards[tab].map((card, i) => {
          const sel = isSelected(card);
          const dis = isDisabled(card) || (!sel && atMax);
          return (
            <div
              key={i}
              className={`ca-card-slot ${sel ? 'selected' : ''} ${dis ? 'disabled' : ''}`}
              onClick={() => !dis && onToggle(card)}
            >
              <Card card={card} />
              {sel && <span className="ca-badge hand-badge">✓</span>}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ---- Camera capture ----

function CameraCapture({ onCapture }: { onCapture: (imageData: string) => void }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setStreaming(true);
      }
    } catch {
      setError('Camera access denied or unavailable');
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (videoRef.current?.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach(t => t.stop());
      videoRef.current.srcObject = null;
      setStreaming(false);
    }
  }, []);

  const capture = useCallback(() => {
    if (!videoRef.current) return;
    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    canvas.getContext('2d')!.drawImage(videoRef.current, 0, 0);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
    onCapture(dataUrl);
    stopCamera();
  }, [onCapture, stopCamera]);

  return (
    <div className="cm-camera">
      {error && <div className="ca-error">{error}</div>}
      {!streaming ? (
        <button className="btn-secondary cm-camera-btn" onClick={startCamera}>
          📸 Open Camera
        </button>
      ) : (
        <div className="cm-camera-preview">
          <video ref={videoRef} autoPlay playsInline muted className="cm-video" />
          <div className="cm-camera-controls">
            <button className="btn-gold" onClick={capture}>📷 Capture</button>
            <button className="btn-secondary" onClick={stopCamera}>✕ Cancel</button>
          </div>
        </div>
      )}
      <p className="cm-camera-note">
        Card recognition coming soon — for now, use the manual picker below
      </p>
    </div>
  );
}

// ---- Main component ----

export default function CameraAgent({ onBack }: { onBack: () => void }) {
  const [step, setStep] = useState<Step>('scan_hand');
  const [hand, setHand] = useState<CardData[]>([]);
  const [bids, setBids] = useState<BidEntry[]>([]);
  const [contract, setContract] = useState<string | null>(null);
  const [declarer, setDeclarer] = useState<number>(0);
  const [calledKing, setCalledKing] = useState<CardData | null>(null);
  const [talonCards, setTalonCards] = useState<CardData[]>([]);
  const [position, setPosition] = useState(0);
  const [tricks, setTricks] = useState<TrickRecord[]>([]);
  const [currentTrick, setCurrentTrick] = useState<CardData[]>([]);
  const [playedCards, setPlayedCards] = useState<CardData[]>([]);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [bidRec, setBidRec] = useState<BidRecommendation | null>(null);
  const [kingRec, setKingRec] = useState<KingRecommendation | null>(null);
  const [playRec, setPlayRec] = useState<PlayRecommendation | null>(null);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);

  const [biddingPlayer, setBiddingPlayer] = useState(1);
  const [userBidDone, setUserBidDone] = useState(false);

  const cardEq = (a: CardData, b: CardData) =>
    a.card_type === b.card_type && a.value === b.value && a.suit === b.suit;

  const clearError = () => setError(null);

  const toggleInList = (card: CardData, list: CardData[], setList: React.Dispatch<React.SetStateAction<CardData[]>>) => {
    if (list.some(c => cardEq(c, card))) {
      setList(list.filter(c => !cardEq(c, card)));
    } else {
      setList([...list, card]);
    }
  };

  // ---- Step 1: Scan Hand ----

  const confirmHand = () => {
    if (hand.length === 0) {
      setError('Select at least one card for your hand');
      return;
    }
    clearError();
    setStep('bidding');
    fetchBidRecommendation([]);
  };

  // ---- Step 2: Bidding ----

  const fetchBidRecommendation = async (currentBids: BidEntry[]) => {
    setLoading(true);
    try {
      const resp = await fetch('/api/analyze-bid', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          hand: hand.map(c => ({ card_type: c.card_type, value: c.value, suit: c.suit })),
          bids: currentBids.map(b => ({ player: b.player, contract: b.contract })),
          dealer: 0,
        }),
      });
      if (!resp.ok) throw new Error(`Server error: ${resp.status}`);
      setBidRec(await resp.json());
    } catch (err: any) {
      setError(err.message || 'Failed to get bid recommendation');
    } finally {
      setLoading(false);
    }
  };

  const submitUserBid = (contractId: string | null) => {
    const newBids = [...bids, { player: 0, contract: contractId }];
    setBids(newBids);
    setUserBidDone(true);
    setBiddingPlayer(1);
  };

  const submitOpponentBid = (player: number, contractId: string | null) => {
    const newBids = [...bids, { player, contract: contractId }];
    setBids(newBids);
    setBiddingPlayer(prev => (prev % 3) + 1);

    const passes = newBids.filter(b => b.contract === null).length;
    const activeBidders = new Set(newBids.filter(b => b.contract !== null).map(b => b.player));

    if (passes === 4) {
      setContract('klop');
      setPosition(2);
      setStep('trick_play');
      fetchPlayRecommendation('klop', 2, [], 0, []);
      return;
    }

    const oneBidderLeft = activeBidders.size === 1 && newBids.length >= 4;
    const threePassedOneActive = passes >= 3 && activeBidders.size >= 1;

    if (oneBidderLeft || threePassedOneActive) {
      const bidsWithContract = newBids.filter(b => b.contract !== null);
      const contractStrength: Record<string, number> = {
        three: 1, two: 2, one: 3, solo_three: 4, solo_two: 5, solo_one: 6, solo: 7,
      };
      const winner = bidsWithContract.reduce((best, b) =>
        (contractStrength[b.contract!] || 0) > (contractStrength[best.contract!] || 0) ? b : best
      );
      setContract(winner.contract);
      setDeclarer(winner.player);
      const myPos = winner.player === 0 ? 0 : 2;
      setPosition(myPos);

      const isSolo = ['solo', 'solo_three', 'solo_two', 'solo_one'].includes(winner.contract!);
      if (isSolo) {
        const talonCount = CONTRACTS.find(c => c.id === winner.contract)?.talon ?? 0;
        if (talonCount > 0 && winner.player === 0) {
          setStep('talon');
        } else {
          setStep('trick_play');
          fetchPlayRecommendation(winner.contract!, myPos, [], 0, []);
        }
      } else if (winner.player === 0) {
        setStep('king_call');
        fetchKingRecommendation(winner.contract!);
      } else {
        setStep('trick_play');
        fetchPlayRecommendation(winner.contract!, myPos, [], 0, []);
      }
      return;
    }

    if (newBids.length % 4 === 0) {
      setUserBidDone(false);
      fetchBidRecommendation(newBids);
    }
  };

  // ---- Step 3: King Call ----

  const fetchKingRecommendation = async (contractId: string) => {
    setLoading(true);
    try {
      const resp = await fetch('/api/analyze-king', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          hand: hand.map(c => ({ card_type: c.card_type, value: c.value, suit: c.suit })),
          contract: contractId,
        }),
      });
      if (!resp.ok) throw new Error(`Server error: ${resp.status}`);
      setKingRec(await resp.json());
    } catch (err: any) {
      setError(err.message || 'Failed to get king recommendation');
    } finally {
      setLoading(false);
    }
  };

  const confirmKingCall = (king: CardData) => {
    setCalledKing(king);
    const talonCount = CONTRACTS.find(c => c.id === contract)?.talon ?? 0;
    if (talonCount > 0) {
      setStep('talon');
    } else {
      setStep('trick_play');
      fetchPlayRecommendation(contract!, position, [], 0, []);
    }
  };

  // ---- Step 4: Talon ----

  const confirmTalon = () => {
    setStep('trick_play');
    fetchPlayRecommendation(contract!, position, [], 0, []);
  };

  // ---- Step 5: Trick Play ----

  const fetchPlayRecommendation = async (
    ctr: string, pos: number, trick: CardData[], tricksPlayed: number, played: CardData[],
  ) => {
    setLoading(true);
    try {
      const resp = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          hand: hand.map(c => ({ card_type: c.card_type, value: c.value, suit: c.suit })),
          trick: trick.map(c => ({ card_type: c.card_type, value: c.value, suit: c.suit })),
          contract: ctr,
          position: pos,
          tricks_played: tricksPlayed,
          played_cards: played.map(c => ({ card_type: c.card_type, value: c.value, suit: c.suit })),
        }),
      });
      if (!resp.ok) throw new Error(`Server error: ${resp.status}`);
      setPlayRec(await resp.json());
    } catch (err: any) {
      setError(err.message || 'Failed to get play recommendation');
    } finally {
      setLoading(false);
    }
  };

  const addTrickCard = (card: CardData) => {
    if (currentTrick.length >= 3) return;
    setCurrentTrick(prev => [...prev, card]);
  };

  const removeTrickCard = (index: number) => {
    setCurrentTrick(prev => prev.filter((_, i) => i !== index));
  };

  const confirmMyPlay = (card: CardData) => {
    setHand(prev => prev.filter(c => !cardEq(c, card)));
    const allTrickCards = [...currentTrick, card];
    const newPlayed = [...playedCards, ...allTrickCards];
    setPlayedCards(newPlayed);
    const newTricks = [...tricks, { cards: allTrickCards.map((c, i) => ({ player: i, card: c })), winner: 0 }];
    setTricks(newTricks);
    setCurrentTrick([]);
    setPlayRec(null);

    if (newTricks.length >= 12) {
      setStep('finished');
    } else {
      fetchPlayRecommendation(contract || 'three', position, [], newTricks.length, newPlayed);
    }
  };

  const getRecommendation = () => {
    fetchPlayRecommendation(contract || 'three', position, currentTrick, tricks.length, playedCards);
  };

  // ---- Step progress bar ----

  const steps: { key: Step; label: string }[] = [
    { key: 'scan_hand', label: '1. Hand' },
    { key: 'bidding', label: '2. Bid' },
    { key: 'king_call', label: '3. King' },
    { key: 'talon', label: '4. Talon' },
    { key: 'trick_play', label: '5. Play' },
    { key: 'finished', label: 'Done' },
  ];
  const stepIndex = steps.findIndex(s => s.key === step);

  const resetGame = () => {
    setStep('scan_hand'); setHand([]); setBids([]); setContract(null);
    setDeclarer(0); setCalledKing(null); setTalonCards([]);
    setTricks([]); setCurrentTrick([]); setPlayedCards([]);
    setBidRec(null); setKingRec(null); setPlayRec(null);
    setCapturedImage(null); setUserBidDone(false); setError(null);
  };

  return (
    <div className="camera-agent">
      <div className="ca-header">
        <button className="btn-secondary btn-sm" onClick={onBack}>← Back</button>
        <h2>🎯 Game Assistant</h2>
        <p className="ca-subtitle">Step-by-step guidance for your real Tarok game</p>
      </div>

      {/* Progress */}
      <div className="cm-steps">
        {steps.map((s, i) => (
          <div key={s.key} className={`cm-step ${i === stepIndex ? 'active' : ''} ${i < stepIndex ? 'done' : ''}`}>
            <span className="cm-step-dot">{i < stepIndex ? '✓' : i + 1}</span>
            <span className="cm-step-label">{s.label}</span>
          </div>
        ))}
      </div>

      {error && (
        <div className="ca-error">
          {error} <button className="cm-dismiss" onClick={clearError}>✕</button>
        </div>
      )}

      {/* ===== STEP 1: Scan Hand ===== */}
      {step === 'scan_hand' && (
        <div className="cm-section">
          <div className="cm-instruction">
            <div className="cm-instruction-icon">📸</div>
            <div>
              <h3>Photograph your hand</h3>
              <p>Take a photo of your 12 cards, or select them manually below</p>
            </div>
          </div>

          <CameraCapture onCapture={(img) => setCapturedImage(img)} />

          {capturedImage && (
            <div className="cm-captured">
              <img src={capturedImage} alt="Captured hand" className="cm-captured-img" />
              <p className="cm-camera-note">Card recognition not yet available — please select cards manually</p>
            </div>
          )}

          <h4 className="cm-picker-label">Select your cards ({hand.length} selected)</h4>
          <CardPicker selected={hand} onToggle={(card) => toggleInList(card, hand, setHand)} />

          {hand.length > 0 && (
            <div className="cm-selected-summary">
              <h4>Your hand:</h4>
              <div className="ca-selected-cards">
                {hand.map((c, i) => <Card key={i} card={c} />)}
              </div>
            </div>
          )}

          <button className="btn-gold btn-large cm-confirm-btn" onClick={confirmHand} disabled={hand.length === 0}>
            Confirm Hand → Start Bidding
          </button>
        </div>
      )}

      {/* ===== STEP 2: Bidding ===== */}
      {step === 'bidding' && (
        <div className="cm-section">
          <div className="cm-instruction">
            <div className="cm-instruction-icon">🗣️</div>
            <div>
              <h3>Bidding Phase</h3>
              <p>The AI will recommend what to bid. Enter what each player bids.</p>
            </div>
          </div>

          {bidRec && !userBidDone && (
            <div className="cm-recommendation">
              <h4>{bidRec.has_trained_model ? '🎯 AI recommends you bid:' : '🎲 Random suggestion:'}</h4>
              <div className="cm-rec-value">{bidRec.recommended_name}</div>
              {bidRec.ranked_bids.length > 1 && (
                <div className="cm-rec-breakdown">
                  {bidRec.ranked_bids.map((b, i) => (
                    <div key={i} className="cm-rec-option">
                      <span>{b.name}</span>
                      <span className="cm-prob">{(b.probability * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {bids.length > 0 && (
            <div className="cm-bid-history">
              <h4>Bids so far:</h4>
              {bids.map((b, i) => (
                <div key={i} className="cm-bid-entry">
                  <span className="cm-bid-player">{PLAYER_NAMES[b.player]}</span>
                  <span className="cm-bid-value">
                    {b.contract ? CONTRACTS.find(c => c.id === b.contract)?.name || b.contract : 'Pass'}
                  </span>
                </div>
              ))}
            </div>
          )}

          {!userBidDone && (
            <div className="cm-bid-actions">
              <h4>Your bid:</h4>
              <div className="cm-btn-row">
                <button className="btn-secondary" onClick={() => submitUserBid(null)}>Pass</button>
                {CONTRACTS.map(c => (
                  <button key={c.id} className="btn-primary" onClick={() => submitUserBid(c.id)}>{c.name}</button>
                ))}
              </div>
            </div>
          )}

          {userBidDone && (
            <div className="cm-bid-actions">
              <h4>What did {PLAYER_NAMES[biddingPlayer]} bid?</h4>
              <div className="cm-btn-row">
                <button className="btn-secondary" onClick={() => submitOpponentBid(biddingPlayer, null)}>Pass</button>
                {CONTRACTS.map(c => (
                  <button key={c.id} className="btn-primary" onClick={() => submitOpponentBid(biddingPlayer, c.id)}>{c.name}</button>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ===== STEP 3: King Call ===== */}
      {step === 'king_call' && (
        <div className="cm-section">
          <div className="cm-instruction">
            <div className="cm-instruction-icon">👑</div>
            <div>
              <h3>Call a King</h3>
              <p>You won the bid! Choose which king to call — the holder becomes your secret partner.</p>
            </div>
          </div>

          {kingRec && (
            <div className="cm-recommendation">
              <h4>{kingRec.has_trained_model ? '🎯 AI recommends:' : '🎲 Suggestion:'}</h4>
              {kingRec.recommended && (
                <div className="cm-rec-card-row">
                  <Card card={kingRec.recommended} highlighted />
                  <span>Call the King of {kingRec.recommended.suit}</span>
                </div>
              )}
            </div>
          )}

          <div className="cm-king-options">
            <h4>Select which king to call:</h4>
            <div className="cm-btn-row">
              {(['hearts', 'diamonds', 'clubs', 'spades'] as const)
                .filter(s => !hand.some(c => c.suit === s && c.value === 8))
                .map(s => (
                  <button
                    key={s}
                    className={`btn-gold cm-king-btn ${kingRec?.recommended?.suit === s ? 'recommended' : ''}`}
                    onClick={() => confirmKingCall({
                      card_type: 'suit', value: 8, suit: s,
                      label: `K${SUIT_SYMBOLS[s]}`, points: 5,
                    })}
                  >
                    {SUIT_SYMBOLS[s]} King of {s}
                  </button>
                ))}
            </div>
          </div>
        </div>
      )}

      {/* ===== STEP 4: Talon ===== */}
      {step === 'talon' && (
        <div className="cm-section">
          <div className="cm-instruction">
            <div className="cm-instruction-icon">🎴</div>
            <div>
              <h3>Talon Exchange</h3>
              <p>Select the talon cards that were revealed, then adjust your hand.</p>
            </div>
          </div>

          <h4 className="cm-picker-label">Talon cards ({talonCards.length} selected)</h4>
          <CardPicker
            selected={talonCards}
            onToggle={(card) => toggleInList(card, talonCards, setTalonCards)}
            disabled={hand}
            maxCards={6}
          />

          <h4 className="cm-picker-label" style={{ marginTop: '1rem' }}>
            Update your hand (add picked, remove discarded):
          </h4>
          <CardPicker selected={hand} onToggle={(card) => toggleInList(card, hand, setHand)} />

          {hand.length > 0 && (
            <div className="cm-selected-summary">
              <h4>Your updated hand ({hand.length} cards):</h4>
              <div className="ca-selected-cards">
                {hand.map((c, i) => <Card key={i} card={c} />)}
              </div>
            </div>
          )}

          <button className="btn-gold btn-large cm-confirm-btn" onClick={confirmTalon}>
            Confirm → Start Playing
          </button>
        </div>
      )}

      {/* ===== STEP 5: Trick Play ===== */}
      {step === 'trick_play' && (
        <div className="cm-section">
          <div className="cm-instruction">
            <div className="cm-instruction-icon">🃏</div>
            <div>
              <h3>Trick {tricks.length + 1} of 12</h3>
              <p>
                {contract === 'klop'
                  ? 'Klop — avoid taking tricks!'
                  : `Contract: ${CONTRACTS.find(c => c.id === contract)?.name || contract}`}
                {calledKing && ` | Called: ${calledKing.label}`}
              </p>
            </div>
          </div>

          <div className="cm-hand-display">
            <h4>Your hand ({hand.length} cards):</h4>
            <div className="ca-selected-cards">
              {hand.map((c, i) => (
                <div key={i} className={`cm-hand-card ${playRec?.recommended?.label === c.label ? 'recommended' : ''}`}>
                  <Card card={c} highlighted={playRec?.recommended?.label === c.label} />
                </div>
              ))}
            </div>
          </div>

          <div className="cm-trick-input">
            <h4>Cards played before you ({currentTrick.length}):</h4>
            {currentTrick.length > 0 && (
              <div className="cm-trick-cards">
                {currentTrick.map((c, i) => (
                  <div key={i} className="cm-trick-card-entry" onClick={() => removeTrickCard(i)}>
                    <Card card={c} />
                    <span className="cm-remove-hint">✕</span>
                  </div>
                ))}
              </div>
            )}
            {currentTrick.length < 3 && (
              <>
                <p className="cm-hint">Select cards played by opponents:</p>
                <CardPicker
                  selected={currentTrick}
                  onToggle={addTrickCard}
                  disabled={[...hand, ...playedCards]}
                  maxCards={3}
                />
              </>
            )}
          </div>

          <button className="btn-gold btn-large cm-confirm-btn" onClick={getRecommendation} disabled={loading}>
            {loading ? 'Analyzing...' : '🤖 What should I play?'}
          </button>

          {playRec && (
            <div className="cm-recommendation cm-play-rec">
              <h4>{playRec.has_trained_model ? '🎯 AI recommends:' : '🎲 Suggestion:'}</h4>
              <div className="cm-rec-card-row">
                <div className="cm-rec-card-highlight">
                  <Card card={playRec.recommended} highlighted />
                </div>
                <span className="cm-rec-label">Play <strong>{playRec.recommended.label}</strong></span>
              </div>

              {playRec.position_value !== null && (
                <p className="cm-eval">Position eval: <strong>{playRec.position_value > 0 ? '+' : ''}{playRec.position_value}</strong></p>
              )}

              {playRec.ranked_plays.length > 1 && (
                <div className="cm-rec-breakdown">
                  <h5>All options:</h5>
                  {playRec.ranked_plays.map((rp, i) => (
                    <div key={i} className="cm-rec-option">
                      <span className="cm-rank">#{i + 1}</span>
                      <Card card={rp.card} highlighted={rp.card.label === playRec.recommended.label} />
                      <span className="cm-prob">{(rp.probability * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              )}

              <div className="cm-play-confirm">
                <h4>Confirm which card you played:</h4>
                <div className="cm-btn-row cm-play-cards">
                  {(playRec.legal_plays.length > 0 ? playRec.legal_plays : hand).map((c, i) => (
                    <div key={i} className="cm-play-option" onClick={() => confirmMyPlay(c)}>
                      <Card card={c} highlighted={c.label === playRec.recommended.label} />
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ===== FINISHED ===== */}
      {step === 'finished' && (
        <div className="cm-section cm-finished">
          <div className="cm-instruction">
            <div className="cm-instruction-icon">🏆</div>
            <div>
              <h3>Game Complete!</h3>
              <p>{tricks.length} tricks played.</p>
            </div>
          </div>
          <button className="btn-gold btn-large" onClick={onBack}>Back to Menu</button>
          <button className="btn-secondary btn-large" style={{ marginTop: '0.5rem' }} onClick={resetGame}>New Game</button>
        </div>
      )}
    </div>
  );
}
