with open("backend/src/tarok/adapters/ai/trainer.py", "r") as f:
    text = f.read()

old_call = """        # Save final completed snapshot
        custom_name = f"tarok_agent_completed_S{num_sessions}_ep{game_count}.pt"
        self._save_checkpoint(game_count, is_snapshot=True, custom_name=custom_name)
        return self.metrics"""

new_call = """        # Save final completed snapshot
        custom_name = f"tarok_agent_completed_S{num_sessions}_ep{game_count}.pt"
        snap_info = self._save_checkpoint(game_count, is_snapshot=True, custom_name=custom_name)
        if snap_info:
            self.metrics.snapshots.append(snap_info)
        return self.metrics"""

text = text.replace(old_call, new_call)

with open("backend/src/tarok/adapters/ai/trainer.py", "w") as f:
    f.write(text)
