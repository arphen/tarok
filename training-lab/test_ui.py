from training.adapters.presentation import TerminalPresenter, _format_time
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
print(f"      Seats:  {seats_str}")
print(f"      Result: {537644:,} exp  [{_format_time(51.0)}]")

print(f"  ② PPO update   6ep  batch=8192")
print(f"      Result: loss=1.2345  (π=0.123 v=1.111)  [{_format_time(10.0)}]")

print()
