"""Pydantic schemas for the API adapter."""

from __future__ import annotations

from pydantic import BaseModel


class CardSchema(BaseModel):
    card_type: str  # "tarok" or "suit"
    value: int
    suit: str | None = None
    label: str
    points: int


class TrickSchema(BaseModel):
    cards: list[tuple[int, CardSchema]]
    lead_player: int
    winner: int | None = None


class GameStateSchema(BaseModel):
    phase: str
    hand: list[CardSchema]
    hand_sizes: list[int]
    talon_groups: list[list[CardSchema]] | None = None
    bids: list[dict]
    contract: str | None = None
    declarer: int | None = None
    called_king: CardSchema | None = None
    partner_revealed: bool = False
    partner: int | None = None
    current_trick: list[tuple[int, CardSchema]]
    tricks_played: int = 0
    current_player: int = 0
    scores: dict[str, int] | None = None
    legal_plays: list[CardSchema] = []
    player_names: list[str] = []


class PlayCardRequest(BaseModel):
    card_type: str
    value: int
    suit: str | None = None


class BidRequest(BaseModel):
    contract: int | None  # Contract value or None for pass


class CallKingRequest(BaseModel):
    suit: str


class TalonChoiceRequest(BaseModel):
    group_index: int


class DiscardRequest(BaseModel):
    cards: list[PlayCardRequest]


class NewGameRequest(BaseModel):
    """Request to create a new human-vs-AI game with per-opponent model selection."""
    opponents: list[str] = ["latest", "latest", "latest"]  # 3 entries: filename, "latest", or "random"
    num_rounds: int = 1


class TrainingRequest(BaseModel):
    num_sessions: int = 100
    games_per_session: int = 100
    learning_rate: float = 3e-4
    hidden_size: int = 256
    resume: bool = False
    resume_from: str | None = None
    stockskis_ratio: float = 0.0
    stockskis_strength: float = 1.0
    use_rust_engine: bool = False
    warmup_games: int = 0
    batch_concurrency: int = 32


class LabTrainingRequest(BaseModel):
    """Request to start training via the training-lab (GPU lab) package."""
    num_sessions: int = 1000
    games_per_session: int = 20
    learning_rate: float = 3e-4
    hidden_size: int = 256
    resume_from: str | None = None
    concurrency: int = 128
    buffer_capacity: int = 50_000
    min_experiences: int = 5_000
    ppo_epochs: int = 6
    batch_size: int = 256
    explore_rate: float = 0.1
    checkpoint_interval: int = 50
    device: str = "auto"


class TrainingMetricsSchema(BaseModel):
    episode: int = 0
    total_episodes: int = 0
    session: int = 0
    avg_reward: float = 0.0
    avg_loss: float = 0.0
    win_rate: float = 0.0
    entropy: float = 0.0
    value_loss: float = 0.0
    policy_loss: float = 0.0
    games_per_second: float = 0.0
    bid_rate: float = 0.0
    klop_rate: float = 0.0
    solo_rate: float = 0.0
    reward_history: list[float] = []
    win_rate_history: list[float] = []
    loss_history: list[float] = []
