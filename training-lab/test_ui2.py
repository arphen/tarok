from training.adapters.presenter import TerminalPresenter, _format_time
from training.entities.training_config import TrainingConfig
cfg = TrainingConfig(iterations=100, games=10000, seats='nn,nn,nn,nn', ppo_epochs=6,
    batch_size=8192, lr=0.0001, gamma=0.995, gae_lambda=0.98, explore_rate=0.05,
    bench_seats='nn,bot_v1,bot_v3,bot_m6', bench_games=3000, benchmark_checkpoints=[0, 10])
p = TerminalPresenter()
p.on_iteration_start(2, 100, 60.0)

# Simulate what we want on_selfplay_start/done to look like
seats_str = "nn,checkpoints/Andreja_Starič/league_pool/iter_005.pt,bot_m6,nn"
explore_rate = 0.1

print(f"  ① Self-play   {10000:,} games  (ε={explore_rate})")
print(f"      seats: {seats_str}")
print("      stats: ", end="", flush=True)
print(f"{537644:,} exp  [{_format_time(51.0)}]")

print(f"  ② PPO update   6ep  batch=8192")
print("      stats: ", end="", flush=True)
print(f"loss=1.2345  (π=0.123 v=1.111)  [{_format_time(10.0)}]")

print(f"  ③ Benchmark    {3000:,} games  (greedy)")
print(f"      seats: nn,bot_v1,bot_v3,bot_m6")
print("      stats: ", end="", flush=True)
print(f"placement=2.532  [{_format_time(4.2)}]")

print("────────────────────────────────────────────────────────────────────────")
print(f"  placement  2.600 → 2.532  (▲ 0.068 better)    took {_format_time(65.2)}")

print()
# Simulate memory stats table
stats = {"rss_mb": 745, "footprint_mb": 3994, "py_heap_mb": 4, "py_heap_peak_mb": 852, "mps_alloc_mb": 15, "mps_driver_mb": 62}
deltas = {"rss_mb": -71, "footprint_mb": 102, "py_heap_mb": 0, "py_heap_peak_mb": 0, "mps_alloc_mb": 0, "mps_driver_mb": -51}

print("  mem │ rrs: 745(+0)   footprint: 3994(+102)  py_heap: 4        py_peak: 852")
print("      │ mps_a: 15      mps_d: 62")
