# 3-Player Tarok Support

This document captures how the 3-player Tarok variant is wired through the
stack. The 4-player variant remains the default; `variant="four_player"`
is implicit unless callers override it.

## Variants in code

| Layer | Identifier |
| --- | --- |
| HTTP API (`/api/game/new`, `/api/arena/start`, `/api/arena/duplicate/start`) | `variant: "four_player" \| "three_player"` |
| Frontend localStorage | `tarok.variant` (lobby) / `tarok.arena.variant` (arena) |
| Rust engine (`tarok_engine.RustGameState`, `te.run_self_play`) | `variant: u8` — `0` = four-player, `1` = three-player |
| Network checkpoints | `model_arch == "v3p"` ↔ three-player; anything else (`v4`, `v5`, …) ↔ four-player |

## Game-rule deltas (3-player)

| Concept | 4-player | 3-player |
| --- | --- | --- |
| Players at the table | 4 | 3 |
| Hand size | 12 | 16 |
| Tricks per hand | 12 | 16 |
| Talon | 6 | 6 |
| King-call phase | yes (declarer picks a partner via king) | **skipped** — declarer always plays solo |
| Biddable contracts | Klop, Three, Two, One, Solo Three/Two/One/Solo, Berač, Barvni Valat | Klop (implicit), Solo Three/Two/One, Berač, Barvni Valat |
| Partner score (non-Klop) | partner shares point-diff + bonuses | n/a |

The scoring rules themselves are unchanged: declarer still gets
`sign × (contract_base + point_diff) × kontra + bonuses`, and (for 4-player
2v2 contracts) the partner gets `sign × point_diff × kontra + bonuses`.
Valat replaces all scoring for everyone on the declaring team. See the
top-level `.github/copilot-instructions.md` for the locked-in scoring
contract.

## Auto-detection

Several entry points auto-detect variant from checkpoint metadata:

* **`/api/checkpoints`** annotates each entry with `model_arch` and
  `variant`. The frontend disables checkpoints whose `variant` does not
  match the user's current selection (with a tooltip explaining why).
* **`/api/arena/start`** scans every `rl` / `centaur` agent's checkpoint
  before launching the run. Mixing 3p and 4p checkpoints in one arena is
  a hard error.
* **`/api/arena/duplicate/start`** requires both `challenger` and
  `defender` to share a variant, and validates that the chosen pairing
  (`rotation_8game`, `rotation_4game`, `single_seat_2game`,
  `rotation_6game`) is compatible. `rotation_6game` is 3p-only;
  `rotation_8game`/`rotation_4game` are 4p-only.

## Bots in the 3-player arena

The Rust engine ships a 3p-aware heuristic bot named `bot_v3_3p`. The
arena's `agent_type_to_seat_label` mapping accepts `stockskis_v3_3p` /
`bot_v3_3p` / `v3_3p` and routes them to that seat. When the user
selects the 3-player radio in the Bot Arena UI, the agent dropdowns hide
4p-only options (`stockskis_*`, `centaur`) and surface
`stockskis_v3_3p` instead.

Neural-network seats (`rl`) work in both variants — the checkpoint's
`model_arch` is what determines which network is loaded. The
`export_checkpoint_to_torchscript` helper in `tarok.use_cases.arena`
wraps both 4p (`TarokNetV4`) and 3p (`TarokNet3`) checkpoints into the
same 5-tuple TorchScript module, so the Rust `NeuralNetPlayer` does not
need variant-specific code.

## Frontend surfaces

* **Play-vs-AI lobby** (`frontend/src/App.tsx`): radio above the
  opponent dropdowns toggles between 4-player (3 AI slots) and 3-player
  (2 AI slots). Choice persists in `localStorage` under
  `tarok.variant`.
* **Bot Arena** (`frontend/src/components/BotArena.tsx`): radio above
  the agent grid toggles between 4 and 3 seat cards. Choice persists
  under `tarok.arena.variant`. Disabled while a run is in progress.
* **GameBoard** (`frontend/src/components/GameBoard.tsx`): hides the
  right-seat hand and renders the trick counter as `n/16` instead of
  `n/12` whenever `state.player_names.length === 3`.

## Tests

* `backend/tests/test_arena_router_variant.py` — patches
  `tarok_engine.run_self_play` and verifies that
  `variant="three_player"` produces a 3-seat `seat_config` and forwards
  `variant=1` to the engine.
* `backend/tests/test_arena_use_case.py::test_agent_type_to_seat_label_maps_v3_3p`
  — covers the 3p bot label mapping.
* `backend/tests/test_duplicate_arena_router.py::test_duplicate_start_accepts_rotation_6game_pairing`
  — covers the new 3p pairing token.
* `frontend/e2e/three-player-ui.spec.ts` — Playwright smoke test for
  variant radio + dynamic seat counts.
