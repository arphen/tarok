"""FastAPI server — app setup, middleware, and router wiring only."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tarok.adapters.api.routers import (
    arena_router,
    checkpoint_router,
    game_router,
    spectate_router,
    tournament_router,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="Tarok API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(arena_router.router)
app.include_router(checkpoint_router.router)
app.include_router(game_router.router)
app.include_router(spectate_router.router)
app.include_router(tournament_router.router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}
