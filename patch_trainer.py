import re

with open("backend/src/tarok/adapters/ai/trainer.py", "r") as f:
    text = f.read()

# Change _save_checkpoint call after the loop
old_call = """        self._save_checkpoint(game_count, is_snapshot=False)
        return self.metrics"""

new_call = """        # Save final completed snapshot
        custom_name = f"tarok_agent_completed_S{num_sessions}_ep{game_count}.pt"
        self._save_checkpoint(game_count, is_snapshot=True, custom_name=custom_name)
        return self.metrics"""
text = text.replace(old_call, new_call)

# Update _save_checkpoint definition
old_def = """    def _save_checkpoint(self, episode: int, is_snapshot: bool = False) -> dict:
        data = {
            "episode": episode,
            "session": self.metrics.session,
            "model_state_dict": self.shared_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": self.metrics.to_dict(),
        }

        # Always save as 'latest'
        latest = self.save_dir / "tarok_agent_latest.pt"
        torch.save(data, latest)

        # Numbered snapshot
        path = self.save_dir / f"tarok_agent_ep{episode}.pt"
        torch.save(data, path)

        info = {"""

new_def = """    def _save_checkpoint(self, episode: int, is_snapshot: bool = False, custom_name: str | None = None) -> dict:
        data = {
            "episode": episode,
            "session": self.metrics.session,
            "model_state_dict": self.shared_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": self.metrics.to_dict(),
        }

        # Always save as 'latest'
        latest = self.save_dir / "tarok_agent_latest.pt"
        torch.save(data, latest)

        if not is_snapshot and not custom_name:
            return {}

        file_name = custom_name if custom_name else f"tarok_agent_ep{episode}.pt"
        path = self.save_dir / file_name
        torch.save(data, path)

        info = {"""
text = text.replace(old_def, new_def)

with open("backend/src/tarok/adapters/ai/trainer.py", "w") as f:
    f.write(text)
