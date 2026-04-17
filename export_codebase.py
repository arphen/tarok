import os
from pathlib import Path

OUTPUT_FILE = "gemini_research_export.md"

CONTEXT = """# Tarok AI Project Codebase

## Context Information for LLM
This is a codebase for a Tarok card game AI. The system comprises a Python backend featuring a Reinforcement Learning (PPO) agent that learns through self-play. It evaluates the agent's performance dynamically, splitting stats into Declarer vs. Defender win rates. The backend uses a single-core asynchronous Python event loop combining the CPU-heavy PPO simulation steps with a FastAPI web server (yielding control via `asyncio.sleep` to stream real-time metrics). 
The frontend is a React/Vite application that provides a real-time training dashboard with live charts to monitor the AI's learning progress.

## Research Questions
Please do a deep research pass over this codebase and answer the following questions:
1. **Rule correctness**: Are the Tarok game rules implemented correctly in this codebase (e.g., card ranking, bidding phases, legal move constraints, and trick scoring)?
2. **Modern academic improvements**: What are the most recent academic improvements for self-play learning in trick-taking or imperfect-information card games that could be relevant to our learning here?
3. **Applying recent methods**: How exactly can we improve the agent's learning performance, training stability, and policy convergence using these recent methods?
4. **State & Reward shaping**: How can we optimize our state representation (observations) and reward shaping to better capture Tarok's hidden information dynamics?
5. **Architectural improvements**: Are there any critical flaws or bottlenecks in how the PPO algorithm is currently integrated with the Tarok game loop, and how can we safely scale this architecture?

---
## Codebase
"""

def main():
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write(CONTEXT + "\n")
        
        # Collect relevant backend and frontend files
        workspace_root = Path("/Users/swozny/work/tarok")
        backend_files = list((workspace_root / "backend/src/tarok").rglob("*.py"))
        frontend_ts = list((workspace_root / "frontend/src").rglob("*.ts"))
        frontend_tsx = list((workspace_root / "frontend/src").rglob("*.tsx"))
        
        all_files = sorted(backend_files + frontend_ts + frontend_tsx)
        
        for filepath in all_files:
            filepath_str = str(filepath)
            # Skip common cached or compiled directories just in case
            if "node_modules" in filepath_str or "__pycache__" in filepath_str or ".venv" in filepath_str:
                continue
            try:
                content = filepath.read_text(encoding="utf-8")
                
                ext = filepath.suffix.lstrip('.')
                lang = "python" if ext == "py" else ("typescript" if ext == "ts" else "tsx")
                
                out.write(f"### File: `{filepath_str}`\n")
                out.write(f"```{lang}\n")
                out.write(content)
                out.write("\n```\n\n")
            except Exception as e:
                print(f"Skipping {filepath_str}: {e}")

    print(f"Successfully exported codebase to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
