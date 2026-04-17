import re

with open("frontend/src/components/TrainingDashboard.tsx", "r") as f:
    content = f.read()

# State
content = content.replace(
    "const [resume, setResume] = useState(false);",
    """const [resume, setResume] = useState(false);
  const [resumeFrom, setResumeFrom] = useState('tarok_agent_latest.pt');
  const [availableSnapshots, setAvailableSnapshots] = useState<{filename: string, session: number, episode: number}[]>([]);"""
)

# Poll logic
old_poll = """  const poll = useCallback(async () => {
    try {
      const [mRes, sRes] = await Promise.all([
        fetch(`${API}/api/training/metrics`),
        fetch(`${API}/api/training/status`),
      ]);
      const mData = await mRes.json();
      const sData = await sRes.json();
      setMetrics(mData);
      setIsTraining(sData.running);
    } catch { /* server not up */ }
  }, []);"""

new_poll = """  const poll = useCallback(async () => {
    try {
      const [mRes, sRes, cRes] = await Promise.all([
        fetch(`${API}/api/training/metrics`),
        fetch(`${API}/api/training/status`),
        fetch(`${API}/api/checkpoints`),
      ]);
      const mData = await mRes.json();
      const sData = await sRes.json();
      const cData = await cRes.json();
      setMetrics(mData);
      setIsTraining(sData.running);
      setAvailableSnapshots(cData.checkpoints || []);
    } catch { /* server not up */ }
  }, []);"""
content = content.replace(old_poll, new_poll)

# Start training
old_start = """  const startTraining = async () => {
    await fetch(`${API}/api/training/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ num_sessions: sessions, games_per_session: gamesPerSession, resume }),
    });
    setIsTraining(true);
  };"""

new_start = """  const startTraining = async () => {
    const isLatest = resumeFrom === '' || resumeFrom === 'tarok_agent_latest.pt';
    await fetch(`${API}/api/training/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        num_sessions: sessions, 
        games_per_session: gamesPerSession, 
        resume,
        resume_from: resume && !isLatest ? resumeFrom : undefined
      }),
    });
    setIsTraining(true);
  };"""
content = content.replace(old_start, new_start)

# Checkbox
old_check = """        <label className="td-check">
          <input type="checkbox" checked={resume} onChange={e => setResume(e.target.checked)} disabled={isTraining} />
          <span>Resume from checkpoint</span>
        </label>"""

new_check = """        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <label className="td-check" style={{ margin: 0 }}>
            <input type="checkbox" checked={resume} onChange={e => setResume(e.target.checked)} disabled={isTraining} />
            <span>Resume</span>
          </label>
          {resume && (
            <select 
              value={resumeFrom} 
              onChange={e => setResumeFrom(e.target.value)} 
              disabled={isTraining}
              style={{
                background: '#232529',
                color: '#fff',
                border: '1px solid #444',
                padding: '4px 8px',
                borderRadius: '4px',
                fontSize: '12px'
              }}
            >
              <option value="tarok_agent_latest.pt">Latest (tarok_agent_latest.pt)</option>
              {availableSnapshots.map(s => (
                <option key={s.filename} value={s.filename}>{s.filename} (Session {s.session}, Ep {s.episode})</option>
              ))}
            </select>
          )}
        </div>"""
content = content.replace(old_check, new_check)

with open("frontend/src/components/TrainingDashboard.tsx", "w") as f:
    f.write(content)
