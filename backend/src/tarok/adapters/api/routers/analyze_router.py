"""Camera-agent analysis endpoints — hand, bid, and king recommendation."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from tarok.adapters.players.neural_player import NeuralPlayer
from tarok.adapters.api.checkpoint_utils import resolve_checkpoint_or_default
from tarok.entities import Card, CardType, Suit, SuitRank, DECK, Contract, GameState, Phase, PlayerRole, Trick
from tarok.entities.game_types import suit_card

router = APIRouter(tags=["analyze"])


# ---- Models ----

class CardInput(BaseModel):
    card_type: str  # "tarok" or "suit"
    value: int
    suit: str | None = None


class AnalyzeRequest(BaseModel):
    """Cards the user holds + the current trick + game context."""
    hand: list[CardInput]
    trick: list[CardInput] = []  # cards already played in current trick
    contract: str = "three"  # contract name
    position: int = 0  # 0=declarer, 1=partner, 2/3=opponent
    tricks_played: int = 0
    played_cards: list[CardInput] = []  # all previously played cards


class AnalyzeBidRequest(BaseModel):
    """Hand cards + bidding context → AI bid recommendation."""
    hand: list[CardInput]
    bids: list[dict] = []  # [{player: int, contract: str|null}]
    dealer: int = 0


class AnalyzeKingRequest(BaseModel):
    hand: list[CardInput]
    contract: str = "three"


# ---- Helpers ----

def _parse_card(ci: CardInput) -> Card:
    ct = CardType(ci.card_type)
    s = Suit(ci.suit) if ci.suit else None
    return Card(ct, ci.value, s)


def _card_to_dict(card: Card) -> dict:
    return {
        "card_type": card.card_type.value,
        "value": card.value,
        "suit": card.suit.value if card.suit else None,
        "label": card.label,
        "points": card.points,
    }


# Map of known cards in the DECK for quick label-to-card lookup
_LABEL_TO_CARD: dict[str, Card] = {c.label: c for c in DECK}

CONTRACT_NAME_MAP = {
    "three": Contract.THREE,
    "two": Contract.TWO,
    "one": Contract.ONE,
    "solo_three": Contract.SOLO_THREE,
    "solo_two": Contract.SOLO_TWO,
    "solo_one": Contract.SOLO_ONE,
    "solo": Contract.SOLO,
}


# ---- Endpoints ----

@router.post("/api/analyze")
async def analyze_hand(req: AnalyzeRequest):
    """Given a hand of cards and game context, return the AI's recommended play.

    This endpoint lets users photograph a real Tarok hand, input the cards,
    and get the trained agent's recommendation for what to play.
    """
    # Parse cards
    hand = [_parse_card(c) for c in req.hand]
    trick_cards = [_parse_card(c) for c in req.trick]
    played = [_parse_card(c) for c in req.played_cards]

    # Build a synthetic game state for the agent
    state = GameState(phase=Phase.TRICK_PLAY)
    contract = CONTRACT_NAME_MAP.get(req.contract, Contract.THREE)
    state.contract = contract
    state.declarer = 0 if req.position == 0 else 1

    # Set player 0 as the user
    state.hands[0] = list(hand)
    state.current_player = 0
    state.roles = {
        0: PlayerRole.DECLARER if req.position == 0 else (
            PlayerRole.PARTNER if req.position == 1 else PlayerRole.OPPONENT
        ),
        1: PlayerRole.PARTNER if req.position != 1 else PlayerRole.DECLARER,
        2: PlayerRole.OPPONENT,
        3: PlayerRole.OPPONENT,
    }

    # Build current trick if cards have been played
    if trick_cards:
        state.current_trick = Trick(lead_player=(4 - len(trick_cards)) % 4)
        for i, card in enumerate(trick_cards):
            player_idx = (state.current_trick.lead_player + i) % 4
            state.current_trick.cards.append((player_idx, card))
    else:
        state.current_trick = Trick(lead_player=0)

    # Record previously played tricks
    state.tricks = []
    # Approximate: create empty trick records for played tricks count
    for i in range(req.tricks_played):
        t = Trick(lead_player=0)
        # Minimal trick data
        state.tricks.append(t)

    # Compute legal plays
    legal = state.legal_plays(0)

    # Load the trained agent and get its recommendation
    agent = NeuralPlayer(name="Advisor")
    agent.set_training(False)
    checkpoint_path = resolve_checkpoint_or_default(None)
    if checkpoint_path and checkpoint_path.exists():
        import torch
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        agent.network.load_state_dict(checkpoint["model_state_dict"])

    # Get the agent's card choice
    recommended = await agent.choose_card(state, 0)

    # Also rank all legal plays by the agent's policy
    from tarok_model.encoding import encode_state, encode_legal_mask, CARD_TO_IDX
    import torch

    state_tensor = torch.tensor(encode_state(state, 0), dtype=torch.float32).unsqueeze(0)
    legal_mask = torch.tensor(encode_legal_mask(legal), dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits, value = agent.network(state_tensor)

    # Mask illegal actions
    masked = logits.clone()
    masked[legal_mask == 0] = float('-inf')
    probs = torch.softmax(masked, dim=-1).squeeze(0)

    # Build ranked recommendations
    ranked = []
    for card in legal:
        idx = CARD_TO_IDX.get(card)
        if idx is not None:
            prob = probs[idx].item()
            ranked.append({
                "card": _card_to_dict(card),
                "probability": round(prob, 4),
            })

    ranked.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "recommended": _card_to_dict(recommended),
        "legal_plays": [_card_to_dict(c) for c in legal],
        "ranked_plays": ranked,
        "position_value": round(value.item(), 4) if value is not None else None,
        "has_trained_model": checkpoint_path.exists(),
    }


@router.post("/api/analyze-bid")
async def analyze_bid(req: AnalyzeBidRequest):
    """Given a hand and bidding history, return the AI's recommended bid."""
    from tarok_model.encoding import (
        encode_state, encode_bid_mask, BID_ACTIONS, BID_TO_IDX, DecisionType,
    )
    import torch

    hand = [_parse_card(c) for c in req.hand]

    # Build synthetic game state in BIDDING phase
    from tarok.entities import Bid
    state = GameState(phase=Phase.BIDDING)
    state.dealer = req.dealer
    state.hands[0] = list(hand)
    state.current_player = 0
    state.current_bidder = 0

    # Replay bid history
    for b in req.bids:
        c = CONTRACT_NAME_MAP.get(b.get("contract", ""), None) if b.get("contract") else None
        state.bids.append(Bid(player=b["player"], contract=c))

    # Legal bids for player 0
    legal = state.legal_bids(0)

    # Load agent
    agent = NeuralPlayer(name="BidAdvisor")
    agent.set_training(False)
    checkpoint_path = resolve_checkpoint_or_default(None)
    if checkpoint_path and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        agent.network.load_state_dict(checkpoint["model_state_dict"])

    # Get bid recommendation
    recommended = await agent.choose_bid(state, 0, legal)

    # Get ranked probabilities
    state_tensor = encode_state(state, 0, DecisionType.BID).unsqueeze(0)
    mask = encode_bid_mask(legal).unsqueeze(0)
    with torch.no_grad():
        logits, value = agent.network(state_tensor, DecisionType.BID)
    masked = logits.clone()
    masked[mask == 0] = float('-inf')
    probs = torch.softmax(masked, dim=-1).squeeze(0)

    ranked = []
    for bid_option in legal:
        if bid_option is None:
            idx = BID_TO_IDX.get(None, 0)
            label = "Pass"
        else:
            idx = BID_TO_IDX.get(bid_option, 0)
            label = bid_option.value if isinstance(bid_option.value, str) else bid_option.name.replace("_", " ").title()
        ranked.append({
            "contract": bid_option.value if bid_option else None,
            "name": label,
            "probability": round(probs[idx].item(), 4),
        })
    ranked.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "recommended": recommended.value if recommended else None,
        "recommended_name": (recommended.name.replace("_", " ").title() if recommended else "Pass"),
        "legal_bids": [{"value": b.value if b else None, "name": b.name.replace("_", " ").title() if b else "Pass"} for b in legal],
        "ranked_bids": ranked,
        "position_value": round(value.item(), 4) if value is not None else None,
        "has_trained_model": checkpoint_path.exists(),
    }


@router.post("/api/analyze-king")
async def analyze_king(req: AnalyzeKingRequest):
    """Given a hand and contract, recommend which king to call."""
    from tarok_model.encoding import (
        encode_state, encode_king_mask, KING_ACTIONS, SUIT_TO_IDX, DecisionType,
    )
    import torch

    hand = [_parse_card(c) for c in req.hand]
    contract = CONTRACT_NAME_MAP.get(req.contract, Contract.THREE)

    state = GameState(phase=Phase.KING_CALLING)
    state.hands[0] = list(hand)
    state.contract = contract
    state.declarer = 0
    state.current_player = 0
    state.roles = {0: PlayerRole.DECLARER, 1: PlayerRole.OPPONENT, 2: PlayerRole.OPPONENT, 3: PlayerRole.OPPONENT}

    # Find callable kings (kings NOT in hand)
    callable_kings = []
    for suit in [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]:
        king = suit_card(suit, SuitRank.KING)
        if king not in hand:
            callable_kings.append(king)

    if not callable_kings:
        return {"recommended": None, "callable_kings": [], "has_trained_model": False}

    agent = NeuralPlayer(name="KingAdvisor")
    agent.set_training(False)
    checkpoint_path = resolve_checkpoint_or_default(None)
    if checkpoint_path and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        agent.network.load_state_dict(checkpoint["model_state_dict"])

    recommended = await agent.choose_king(state, 0, callable_kings)

    return {
        "recommended": _card_to_dict(recommended),
        "callable_kings": [_card_to_dict(k) for k in callable_kings],
        "has_trained_model": checkpoint_path is not None and checkpoint_path.exists(),
    }
