#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║          Face Recognition Dashboard — Flask Server           ║
╚══════════════════════════════════════════════════════════════╝

INSTALL:
  pip install flask

USAGE:
  1. Start the main app:   python face_emotion_cv.py
  2. Start this dashboard: python dashboard.py
  3. Open browser:         http://localhost:5000
"""

from flask import Flask, render_template_string, jsonify, send_from_directory
from pathlib import Path
import json, os

app = Flask(__name__)

LIVE_DATA_FILE    = Path("live_data.json")
ATTENDANCE_FILE   = Path("attendance_log.csv")
SCREENSHOTS_DIR   = Path("screenshots")
UNKNOWN_FACES_DIR = Path("unknown_faces")

# ── HTML Template ─────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FaceCV Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:      #0a0c0f;
    --surface: #111418;
    --border:  #1e2530;
    --accent:  #00e87a;
    --warn:    #ff4444;
    --blue:    #3d9eff;
    --yellow:  #ffd166;
    --text:    #e2e8f0;
    --muted:   #4a5568;
    --font-mono: 'IBM Plex Mono', monospace;
    --font-sans: 'IBM Plex Sans', sans-serif;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-sans);
    min-height: 100vh;
    padding: 24px;
  }

  /* Subtle grid background */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(var(--border) 1px, transparent 1px),
      linear-gradient(90deg, var(--border) 1px, transparent 1px);
    background-size: 40px 40px;
    opacity: 0.3;
    pointer-events: none;
    z-index: 0;
  }

  .wrap { position: relative; z-index: 1; max-width: 1400px; margin: 0 auto; }

  /* Header */
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 32px;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--border);
  }

  .logo {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .logo-icon {
    width: 40px; height: 40px;
    background: var(--accent);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
  }

  h1 {
    font-family: var(--font-mono);
    font-size: 1.3rem;
    font-weight: 600;
    letter-spacing: -0.02em;
  }

  h1 span { color: var(--accent); }

  .status-pill {
    display: flex; align-items: center; gap: 8px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 100px;
    padding: 6px 16px;
    font-family: var(--font-mono);
    font-size: 0.75rem;
  }

  .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--muted);
    animation: pulse 2s infinite;
  }
  .dot.live { background: var(--accent); }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }

  /* Stats row */
  .stats {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
  }

  .stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
  }

  .stat-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent);
    opacity: 0.5;
  }

  .stat-card.warn::after { background: var(--warn); }
  .stat-card.blue::after { background: var(--blue); }
  .stat-card.yellow::after { background: var(--yellow); }

  .stat-label {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 10px;
  }

  .stat-value {
    font-family: var(--font-mono);
    font-size: 2.2rem;
    font-weight: 600;
    line-height: 1;
  }

  .stat-value.green  { color: var(--accent); }
  .stat-value.red    { color: var(--warn); }
  .stat-value.blue   { color: var(--blue); }
  .stat-value.yellow { color: var(--yellow); }

  /* Main grid */
  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-bottom: 24px;
  }

  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
  }

  .panel-title {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .panel-title::before {
    content: '';
    display: block;
    width: 3px; height: 12px;
    background: var(--accent);
    border-radius: 2px;
  }

  /* Emotion bars */
  .emotion-list { display: flex; flex-direction: column; gap: 10px; }

  .emotion-row {
    display: grid;
    grid-template-columns: 80px 1fr 60px;
    align-items: center;
    gap: 12px;
  }

  .emotion-name {
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--text);
    text-transform: capitalize;
  }

  .bar-track {
    height: 6px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
  }

  .bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.5s ease;
  }

  .bar-pct {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--muted);
    text-align: right;
  }

  /* Activity log */
  .log-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
    max-height: 280px;
    overflow-y: auto;
  }

  .log-list::-webkit-scrollbar { width: 4px; }
  .log-list::-webkit-scrollbar-track { background: transparent; }
  .log-list::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  .log-row {
    display: grid;
    grid-template-columns: 70px 1fr 80px 60px;
    gap: 10px;
    align-items: center;
    padding: 8px 12px;
    background: var(--bg);
    border-radius: 6px;
    border: 1px solid var(--border);
    font-family: var(--font-mono);
    font-size: 0.72rem;
    animation: fadeIn 0.3s ease;
  }

  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(-4px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .log-time { color: var(--muted); }
  .log-name { color: var(--accent); font-weight: 600; }
  .log-emotion { color: var(--text); text-transform: capitalize; }
  .log-age { color: var(--muted); text-align: right; }

  /* Full-width panel */
  .full { grid-column: 1 / -1; }

  /* Attendance table */
  .table-wrap { overflow-x: auto; }

  table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--font-mono);
    font-size: 0.75rem;
  }

  th {
    text-align: left;
    padding: 8px 16px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.65rem;
    border-bottom: 1px solid var(--border);
  }

  td {
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    color: var(--text);
  }

  tr:last-child td { border-bottom: none; }
  tr:hover td { background: rgba(255,255,255,0.02); }

  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.65rem;
    text-transform: capitalize;
  }

  .badge.happy    { background: #00e87a22; color: #00e87a; }
  .badge.sad      { background: #ff444422; color: #ff8888; }
  .badge.angry    { background: #ff222222; color: #ff6666; }
  .badge.neutral  { background: #ffffff11; color: #aaaaaa; }
  .badge.surprise { background: #3d9eff22; color: #3d9eff; }
  .badge.fear     { background: #a855f722; color: #c084fc; }
  .badge.disgust  { background: #22c55e22; color: #4ade80; }

  /* Footer */
  footer {
    text-align: center;
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--muted);
    margin-top: 24px;
    padding-top: 16px;
    border-top: 1px solid var(--border);
  }

  .empty {
    text-align: center;
    color: var(--muted);
    font-family: var(--font-mono);
    font-size: 0.8rem;
    padding: 32px;
  }

  @media (max-width: 900px) {
    .stats { grid-template-columns: repeat(2, 1fr); }
    .grid  { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<div class="wrap">

  <header>
    <div class="logo">
      <div class="logo-icon">👁</div>
      <div>
        <h1>Face<span>CV</span> Dashboard</h1>
        <div style="font-size:0.72rem; color:var(--muted); margin-top:2px;" id="session-time">Loading…</div>
      </div>
    </div>
    <div class="status-pill">
      <div class="dot" id="status-dot"></div>
      <span id="status-text">Connecting…</span>
    </div>
  </header>

  <!-- Stats -->
  <div class="stats">
    <div class="stat-card">
      <div class="stat-label">Active Faces</div>
      <div class="stat-value green" id="active-faces">—</div>
    </div>
    <div class="stat-card blue">
      <div class="stat-label">FPS</div>
      <div class="stat-value blue" id="fps">—</div>
    </div>
    <div class="stat-card warn">
      <div class="stat-label">Unknown Alerts</div>
      <div class="stat-value red" id="unknown-count">—</div>
    </div>
    <div class="stat-card yellow">
      <div class="stat-label">Last Updated</div>
      <div class="stat-value yellow" style="font-size:1rem; padding-top:8px;" id="last-updated">—</div>
    </div>
  </div>

  <!-- Charts + Log -->
  <div class="grid">

    <div class="panel">
      <div class="panel-title">Emotion Distribution</div>
      <div class="emotion-list" id="emotion-bars">
        <div class="empty">Waiting for data…</div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-title">Recognition Log</div>
      <div class="log-list" id="log-list">
        <div class="empty">No recognitions yet…</div>
      </div>
    </div>

    <div class="panel full">
      <div class="panel-title">Attendance Log</div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Name</th><th>Date</th><th>Time</th>
              <th>Emotion</th><th>Age</th><th>Gender</th>
            </tr>
          </thead>
          <tbody id="attendance-body">
            <tr><td colspan="6" class="empty">No attendance records yet…</td></tr>
          </tbody>
        </table>
      </div>
    </div>

  </div>

  <footer>FaceCV Dashboard — Auto-refreshes every 2 seconds — Open-source Computer Vision Project</footer>
</div>

<script>
const EMOTION_COLORS = {
  happy:'#00e87a', sad:'#ff8888', angry:'#ff6666',
  fear:'#c084fc', surprise:'#3d9eff', disgust:'#4ade80', neutral:'#aaaaaa'
};

async function refresh() {
  try {
    const res  = await fetch('/api/live');
    const data = await res.json();

    // Status
    document.getElementById('status-dot').classList.add('live');
    document.getElementById('status-text').textContent = 'Live';

    // Session
    const st = new Date(data.session_start);
    document.getElementById('session-time').textContent =
      'Session started ' + st.toLocaleTimeString();

    // Stats
    document.getElementById('active-faces').textContent  = data.active_faces;
    document.getElementById('fps').textContent           = data.fps;
    document.getElementById('unknown-count').textContent = data.unknown_count;
    const lu = new Date(data.last_updated);
    document.getElementById('last-updated').textContent  = lu.toLocaleTimeString();

    // Emotion bars
    const counts = data.emotion_counts;
    const total  = Object.values(counts).reduce((a,b) => a+b, 0) || 1;
    const sorted = Object.entries(counts).sort((a,b) => b[1]-a[1]);
    const barsEl = document.getElementById('emotion-bars');
    if (total > 0) {
      barsEl.innerHTML = sorted.map(([emo, count]) => {
        const pct   = ((count/total)*100).toFixed(1);
        const color = EMOTION_COLORS[emo] || '#aaa';
        return `
          <div class="emotion-row">
            <div class="emotion-name">${emo}</div>
            <div class="bar-track">
              <div class="bar-fill" style="width:${pct}%; background:${color}"></div>
            </div>
            <div class="bar-pct">${pct}%</div>
          </div>`;
      }).join('');
    }

    // Recognition log
    const logEl  = document.getElementById('log-list');
    const logData = [...data.recognized_log].reverse().slice(0, 20);
    if (logData.length > 0) {
      logEl.innerHTML = logData.map(e => `
        <div class="log-row">
          <div class="log-time">${e.time}</div>
          <div class="log-name">${e.name}</div>
          <div class="log-emotion">${e.emotion}</div>
          <div class="log-age">${e.age} / ${e.gender ? e.gender[0] : '?'}</div>
        </div>`).join('');
    }

  } catch(e) {
    document.getElementById('status-dot').classList.remove('live');
    document.getElementById('status-text').textContent = 'Disconnected';
  }

  // Attendance
  try {
    const res2 = await fetch('/api/attendance');
    const rows = await res2.json();
    const tbody = document.getElementById('attendance-body');
    if (rows.length > 0) {
      tbody.innerHTML = rows.reverse().map(r => `
        <tr>
          <td style="color:var(--accent); font-weight:600">${r.Name || '—'}</td>
          <td>${r.Date || '—'}</td>
          <td>${r.Time || '—'}</td>
          <td><span class="badge ${r.Emotion}">${r.Emotion || '—'}</span></td>
          <td>${r.Age || '—'}</td>
          <td>${r.Gender || '—'}</td>
        </tr>`).join('');
    }
  } catch(e) {}
}

refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>
"""

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/api/live")
def api_live():
    if LIVE_DATA_FILE.exists():
        with open(LIVE_DATA_FILE) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "No data yet — is face_emotion_cv.py running?"})

@app.route("/api/attendance")
def api_attendance():
    if not ATTENDANCE_FILE.exists():
        return jsonify([])
    import csv as csv_mod
    rows = []
    with open(ATTENDANCE_FILE) as f:
        reader = csv_mod.DictReader(f)
        rows = list(reader)
    return jsonify(rows)

@app.route("/screenshots/<path:filename>")
def screenshots(filename):
    return send_from_directory(str(SCREENSHOTS_DIR), filename)

@app.route("/unknown/<path:filename>")
def unknown_faces(filename):
    return send_from_directory(str(UNKNOWN_FACES_DIR), filename)

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║              FaceCV Dashboard — Flask Server                 ║
╚══════════════════════════════════════════════════════════════╝

  1. Make sure face_emotion_cv.py is running in another Terminal
  2. Open your browser at: http://localhost:5000

""")
    app.run(debug=False, port=5000)
