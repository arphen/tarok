from training.adapters.presenter import TerminalPresenter, _format_time
from training.entities.training_config import TrainingConfig
cfg = TrainingConfig(iterations=100, games=10000, seats='nn,nn,nn,nn', ppo_epochs=6,
    batch_size=8192, lr=0.0001, gamma=0.995, gae_lambda=0.98, explore_rate=0.05,
    bench_seats='nn,bot_v1,bot_v3,bot_m6', bench_games=3000, benchmark_checkpoints=[0, 10])
p = TerminalPresenter()
p.on_iteration_start(2, 100, 60.0)

seats_str = "nn,checkpoints/Andreja_Starič/league_pool/iter_005.pt,bot_m6,nn"

p.on_selfplay_start(cfg, seats_str)
p.on_selfplay_done(537644, 51.0)
p.on_ppo_start(cfg)
metrics = {"policy_loss": 0.123, "value_loss": 1.111, "entropy": 0.444, "total_loss": 1.2345}
p.on_ppo_done(metrics, 10.0)
p.on_benchmark_start(cfg)
p.on_benchmark_done(2.532, 4.2)
p.on_iteration_done(2.600, 2.532, 65.2)

stats = {"rss_mb": 745, "footprint_mb": 3994, "py_heap_mb": 4, "py_heap_peak_mb": 852, "mps_alloc_mb": 15, "mps_driver_mb": 62}
deltas = {"rss_mb": -71, "footprint_mb": 102, "py_heap_mb": 0, "py_heap_peak_mb": 0, "mps_alloc_mb": 0, "mps_driver_mb": -51}
p.on_memory_stats(2, stats, deltas)
