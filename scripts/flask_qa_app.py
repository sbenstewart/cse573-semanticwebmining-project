#!/usr/bin/env python3
"""
TrendScout AI 2.0 — Flask Interface
=====================================
A polished, full-featured Flask web app for the KG-RAG Q&A system.

Run:
    python tredscout_flask_app.py

Endpoints:
    GET  /              → chat UI
    POST /api/ask       → JSON API  {question, mode: "A"|"B"|"both"}
    GET  /api/health    → service health check
    GET  /api/graph/stats → KG statistics

Env vars (or .env):
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
"""

from __future__ import annotations

import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime

# Resolve project root regardless of whether the file lives in the project
# root itself or inside a scripts/ subdirectory.
_here = Path(__file__).resolve().parent
ROOT = next(
    (p for p in [_here, _here.parent] if (p / "src").is_dir()),
    _here,
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, render_template_string, request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Backend initialisation
# ─────────────────────────────────────────────────────────────────────────────
_systems: dict = {}
_init_error: str | None = None

def _init_backends():
    global _systems, _init_error
    try:
        from src.kg.neo4j_client import Neo4jClient
        client = Neo4jClient()
        client.connect()

        from src.rag.text_to_cypher import TextToCypherQA
        from src.rag.graph_rag import GraphRAG

        _systems = {
            "A": TextToCypherQA(neo4j_client=client),
            "B": GraphRAG(neo4j_client=client),
        }
        logger.info("Both QA systems initialised successfully.")
    except Exception as e:
        _init_error = str(e)
        logger.error("Backend init failed: %s", e)

_init_backends()


def _serialize_answer(ans) -> dict:
    return {
        "approach": ans.approach,
        "text": ans.text,
        "error": ans.error,
        "latency_ms": round(ans.latency_ms, 1),
        "cited_doc_ids": ans.cited_doc_ids,
        "trace": ans.trace,
    }


# ─────────────────────────────────────────────────────────────────────────────
# API routes
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    if _init_error:
        return jsonify({"status": "degraded", "error": _init_error}), 503
    return jsonify({
        "status": "ok",
        "systems": list(_systems.keys()),
        "ts": datetime.utcnow().isoformat(),
    }), 200


@app.post("/api/ask")
def ask():
    if _init_error:
        return jsonify({"error": f"Backend unavailable: {_init_error}"}), 503

    payload  = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    mode     = (payload.get("mode") or "both").strip().upper()

    if not question:
        return jsonify({"error": "Missing required field: question"}), 400
    if mode not in {"A", "B", "BOTH"}:
        return jsonify({"error": "mode must be one of: A, B, both"}), 400

    answers = {}
    if mode in {"A", "BOTH"} and "A" in _systems:
        answers["A"] = _serialize_answer(_systems["A"].answer(question))
    if mode in {"B", "BOTH"} and "B" in _systems:
        answers["B"] = _serialize_answer(_systems["B"].answer(question))

    return jsonify({"question": question, "answers": answers, "ts": datetime.utcnow().isoformat()}), 200


@app.get("/api/graph/stats")
def graph_stats():
    """Return basic KG counts for the dashboard stat panel."""
    if _init_error or not _systems:
        return jsonify({"error": "Backend unavailable"}), 503
    try:
        # Pull the Neo4j client from one of the systems
        sys_a = _systems.get("A")
        client = getattr(sys_a, "neo4j_client", None) or getattr(sys_a, "_client", None)
        if client is None:
            return jsonify({"error": "Cannot access Neo4j client"}), 500

        counts_query = """
        CALL apoc.meta.stats() YIELD labels
        RETURN labels
        """
        # fallback to simple queries if apoc not available
        stats = {}
        for label in ["Startup", "Investor", "FundingRound", "Product", "Technology", "Document"]:
            try:
                res = client.query(f"MATCH (n:{label}) RETURN count(n) AS c")
                stats[label] = res[0]["c"] if res else 0
            except Exception:
                stats[label] = "?"
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# HTML template — dark terminal + neon accent aesthetic
# ─────────────────────────────────────────────────────────────────────────────
HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>TrendScout AI 2.0</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet"/>
  <style>
    :root {
      --bg:      #0d1117;
      --bg2:     #161b22;
      --bg3:     #21262d;
      --border:  #30363d;
      --accent:  #00e5a0;
      --acb:     #58a6ff;
      --danger:  #ff7b72;
      --warn:    #f0883e;
      --text:    #e6edf3;
      --muted:   #8b949e;
      --mono:    'Space Mono', monospace;
      --sans:    'DM Sans', sans-serif;
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    html, body { height: 100%; background: var(--bg); color: var(--text); font-family: var(--sans); font-size: 15px; line-height: 1.6; }

    /* ── Layout ─────────────────────────────────── */
    .shell { display: flex; height: 100vh; overflow: hidden; }
    .sidebar { width: 280px; min-width: 280px; background: var(--bg2); border-right: 1px solid var(--border); display: flex; flex-direction: column; padding: 20px; overflow-y: auto; }
    .main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
    .topbar { padding: 16px 28px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 12px; }
    .chat-area { flex: 1; overflow-y: auto; padding: 24px 28px; }
    .input-bar { padding: 16px 28px; border-top: 1px solid var(--border); background: var(--bg); }

    /* ── Sidebar ────────────────────────────────── */
    .logo { font-family: var(--mono); font-size: 1.15rem; font-weight: 700; color: var(--accent); margin-bottom: 2px; }
    .logo-sub { font-family: var(--mono); font-size: 0.65rem; color: var(--muted); margin-bottom: 24px; }
    .sidebar-section { font-family: var(--mono); font-size: 0.65rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); margin: 20px 0 8px; }
    .radio-group { display: flex; flex-direction: column; gap: 6px; }
    .radio-opt { display: flex; align-items: center; gap: 8px; padding: 8px 10px; border-radius: 6px; cursor: pointer; font-size: 0.85rem; border: 1px solid transparent; transition: all 0.15s; }
    .radio-opt:hover { background: var(--bg3); border-color: var(--border); }
    .radio-opt.active { background: var(--bg3); border-color: var(--accent); color: var(--accent); }
    .radio-opt input { display: none; }
    .dot { width: 8px; height: 8px; border-radius: 50%; border: 2px solid currentColor; flex-shrink: 0; }
    .radio-opt.active .dot { background: var(--accent); border-color: var(--accent); }

    .stat-row { display: flex; gap: 8px; margin-bottom: 10px; }
    .stat-box { flex: 1; background: var(--bg3); border: 1px solid var(--border); border-radius: 8px; padding: 10px; text-align: center; }
    .stat-val { font-family: var(--mono); font-size: 1.1rem; font-weight: 700; color: var(--accent); }
    .stat-lbl { font-size: 0.68rem; color: var(--muted); margin-top: 2px; }

    .status-item { display: flex; align-items: center; gap: 6px; font-size: 0.8rem; margin-bottom: 6px; }
    .dot-ok   { width: 7px; height: 7px; border-radius: 50%; background: var(--accent); flex-shrink: 0; }
    .dot-fail { width: 7px; height: 7px; border-radius: 50%; background: var(--danger); flex-shrink: 0; }

    .sq-list { display: flex; flex-direction: column; gap: 6px; }
    .sq-btn { background: var(--bg3); border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; font-size: 0.8rem; cursor: pointer; color: var(--text); text-align: left; transition: border-color 0.15s, background 0.15s; font-family: var(--sans); }
    .sq-btn:hover { border-color: var(--accent); background: var(--bg2); }

    /* ── Topbar ─────────────────────────────────── */
    .topbar-title { font-family: var(--mono); font-size: 1.5rem; font-weight: 700; color: var(--accent); }
    .topbar-sub { font-size: 0.75rem; color: var(--muted); }
    .live-badge { margin-left: auto; font-family: var(--mono); font-size: 0.6rem; padding: 3px 10px; border: 1px solid var(--accent); border-radius: 20px; color: var(--accent); animation: pulse 2s infinite; }
    @keyframes pulse { 0%,100% { opacity:1 } 50% { opacity:0.5 } }

    /* ── KG stat chips ──────────────────────────── */
    .kg-chips { display: flex; gap: 10px; flex-wrap: wrap; padding: 12px 0; }
    .kg-chip { background: var(--bg3); border: 1px solid var(--border); border-radius: 20px; padding: 3px 12px; font-family: var(--mono); font-size: 0.7rem; color: var(--muted); }
    .kg-chip span { color: var(--accent); }

    /* ── Chat ───────────────────────────────────── */
    .empty-state { text-align: center; padding: 80px 20px; color: var(--muted); }
    .empty-icon { font-size: 2.5rem; margin-bottom: 12px; opacity: 0.5; }
    .empty-title { font-family: var(--mono); font-size: 0.85rem; letter-spacing: 0.06em; margin-bottom: 8px; }
    .empty-body { font-size: 0.82rem; line-height: 1.7; }

    .turn { margin-bottom: 24px; }
    .user-msg { display: flex; justify-content: flex-end; margin-bottom: 14px; }
    .user-bubble { background: var(--bg3); border: 1px solid var(--border); border-radius: 12px 12px 4px 12px; padding: 12px 16px; max-width: 75%; font-size: 0.9rem; }
    .bubble-meta { font-family: var(--mono); font-size: 0.65rem; color: var(--muted); margin-bottom: 4px; text-align: right; }

    .answers { display: grid; gap: 14px; }
    .answers.dual { grid-template-columns: 1fr 1fr; }

    .ans-card { background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; padding: 18px; transition: border-color 0.2s; overflow: hidden; }
    .ans-card:hover { border-color: var(--border); }
    .ans-card.approach-a { border-left: 3px solid var(--accent); }
    .ans-card.approach-b { border-left: 3px solid var(--acb); }
    .ans-card.error-card  { border-left: 3px solid var(--danger); }

    .ans-title { font-family: var(--mono); font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 10px; display: flex; align-items: center; gap: 6px; }
    .ans-title.a { color: var(--accent); }
    .ans-title.b { color: var(--acb); }
    .ans-title.err { color: var(--danger); }

    .ans-body { font-size: 0.88rem; line-height: 1.7; color: var(--text); margin-bottom: 12px; white-space: pre-wrap; word-break: break-word; }

    .ans-meta { display: flex; gap: 12px; flex-wrap: wrap; font-family: var(--mono); font-size: 0.65rem; color: var(--muted); padding-top: 10px; border-top: 1px solid var(--border); }

    .expander { margin-top: 10px; }
    .exp-toggle { background: none; border: 1px solid var(--border); color: var(--muted); font-family: var(--mono); font-size: 0.68rem; padding: 4px 10px; border-radius: 6px; cursor: pointer; transition: border-color 0.15s; }
    .exp-toggle:hover { border-color: var(--accent); color: var(--accent); }
    .exp-body { display: none; margin-top: 8px; }
    .exp-body.open { display: block; }

    .cypher-block { font-family: var(--mono); font-size: 0.75rem; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 12px; color: var(--accent); overflow-x: auto; white-space: pre; line-height: 1.5; }
    .doc-item { display: flex; align-items: center; gap: 8px; background: var(--bg3); border: 1px solid var(--border); border-radius: 6px; padding: 6px 12px; margin: 3px 0; font-size: 0.8rem; text-decoration: none; color: var(--text); transition: border-color 0.15s, background 0.15s; width: 100%; }
    .doc-item[href]:not([href="#"]):hover { border-color: var(--acb); color: var(--acb); background: rgba(88,166,255,0.06); cursor: pointer; }
    .doc-item[href="#"] { cursor: default; opacity: 0.7; }
    .doc-score-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--acb); flex-shrink: 0; opacity: 0.6; }
    .doc-title { flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .doc-pub { font-family: var(--mono); font-size: 0.65rem; color: var(--muted); flex-shrink: 0; }
    .doc-ext { font-family: var(--mono); font-size: 0.65rem; color: var(--acb); flex-shrink: 0; opacity: 0; transition: opacity 0.15s; }
    .doc-item:hover .doc-ext { opacity: 1; }

    /* ── Input bar ──────────────────────────────── */
    .input-wrap { display: flex; gap: 10px; align-items: flex-end; }
    .input-wrap textarea { flex: 1; background: var(--bg2); border: 1px solid var(--border); border-radius: 10px; color: var(--text); font-family: var(--sans); font-size: 0.9rem; padding: 12px 14px; resize: none; min-height: 46px; max-height: 140px; outline: none; transition: border-color 0.15s; }
    .input-wrap textarea:focus { border-color: var(--accent); }
    .input-wrap textarea::placeholder { color: var(--muted); }
    .send-btn { background: var(--accent); border: none; border-radius: 8px; color: var(--bg); font-family: var(--mono); font-size: 0.8rem; font-weight: 700; padding: 12px 18px; cursor: pointer; white-space: nowrap; transition: opacity 0.15s; }
    .send-btn:hover { opacity: 0.85; }
    .send-btn:disabled { opacity: 0.4; cursor: not-allowed; }

    .input-hint { font-family: var(--mono); font-size: 0.62rem; color: var(--muted); margin-top: 6px; }

    /* ── Spinner ────────────────────────────────── */
    .spinner-wrap { display: flex; align-items: center; gap: 10px; padding: 14px 0; color: var(--muted); font-family: var(--mono); font-size: 0.75rem; }
    .spinner { width: 16px; height: 16px; border: 2px solid var(--border); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.8s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* ── Clear button ───────────────────────────── */
    .clear-btn { background: none; border: 1px solid var(--border); color: var(--muted); font-family: var(--mono); font-size: 0.68rem; padding: 5px 12px; border-radius: 6px; cursor: pointer; transition: border-color 0.15s; width: 100%; margin-top: 6px; }
    .clear-btn:hover { border-color: var(--danger); color: var(--danger); }

    /* ── Scrollbar ──────────────────────────────── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--bg3); border-radius: 4px; }

    @media (max-width: 780px) {
      .sidebar { display: none; }
      .answers.dual { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
<div class="shell">

  <!-- ── Sidebar ─────────────────────────────────────── -->
  <aside class="sidebar">
    <div class="logo">🔭 TrendScout AI</div>
    <div class="logo-sub">v2.0 · Knowledge Graph Q&A</div>

    <div class="sidebar-section">Approach</div>
    <div class="radio-group" id="modeGroup">
      <label class="radio-opt active" data-mode="both">
        <input type="radio" name="mode" value="both" checked/>
        <span class="dot"></span> Both side-by-side
      </label>
      <label class="radio-opt" data-mode="A">
        <input type="radio" name="mode" value="A"/>
        <span class="dot"></span> Approach A — Text-to-Cypher
      </label>
      <label class="radio-opt" data-mode="B">
        <input type="radio" name="mode" value="B"/>
        <span class="dot"></span> Approach B — Graph RAG
      </label>
    </div>

    <div class="sidebar-section">Session Stats</div>
    <div class="stat-row">
      <div class="stat-box">
        <div class="stat-val" id="statQ">0</div>
        <div class="stat-lbl">Queries</div>
      </div>
      <div class="stat-box">
        <div class="stat-val" id="statL">—</div>
        <div class="stat-lbl">Avg ms</div>
      </div>
    </div>

    <div class="sidebar-section">System Status</div>
    <div id="statusPanel">
      <div class="status-item"><span class="dot-ok"></span> Checking…</div>
    </div>

    <div class="sidebar-section">Knowledge Graph</div>
    <div id="kgStats" style="font-size:0.78rem; color:var(--muted);">Loading…</div>

    <div class="sidebar-section">Suggested Queries</div>
    <div class="sq-list" id="sqList"></div>

    <div style="margin-top: auto; padding-top: 20px;">
      <button class="clear-btn" onclick="clearChat()">🗑 Clear chat history</button>
    </div>
  </aside>

  <!-- ── Main ────────────────────────────────────────── -->
  <div class="main">
    <header class="topbar">
      <div>
        <div class="topbar-title">TrendScout AI 2.0</div>
        <div class="topbar-sub">Knowledge Graph-Augmented Market Intelligence</div>
      </div>
      <div class="live-badge">LIVE</div>
    </header>

    <div class="chat-area" id="chatArea">
      <div id="emptyState" class="empty-state">
        <div class="empty-icon">🔭</div>
        <div class="empty-title">AWAITING QUERY</div>
        <div class="empty-body">Ask anything about AI startups, funding rounds,<br>investors, products, or technologies.</div>
      </div>
    </div>

    <div class="input-bar">
      <div class="input-wrap">
        <textarea id="questionInput" rows="1" placeholder="Ask about AI startups, funding, investors, or technologies…"></textarea>
        <button class="send-btn" id="sendBtn" onclick="submitQuery()">Ask ⏎</button>
      </div>
      <div class="input-hint">⌘↵ or click Ask to submit · Approach selected on left</div>
    </div>
  </div>
</div>

<script>
// ── State ───────────────────────────────────────────────────────────────
let totalQ = 0, totalLat = 0;

const SUGGESTED = [
  "Who invested in Replit?",
  "What funding rounds does Anthropic have?",
  "Which technologies appear most?",
  "Which startups raised Seed rounds?",
  "What products did Anthropic announce?",
  "Which investors back LLM startups?",
];

// ── Init ─────────────────────────────────────────────────────────────────
(async function init() {
  // Suggested queries
  const sqList = document.getElementById('sqList');
  SUGGESTED.forEach(q => {
    const btn = document.createElement('button');
    btn.className = 'sq-btn';
    btn.textContent = q;
    btn.onclick = () => { setInput(q); submitQuery(); };
    sqList.appendChild(btn);
  });

  // Mode toggles
  document.querySelectorAll('#modeGroup .radio-opt').forEach(el => {
    el.addEventListener('click', () => {
      document.querySelectorAll('#modeGroup .radio-opt').forEach(e => e.classList.remove('active'));
      el.classList.add('active');
    });
  });

  // Auto-resize textarea
  const ta = document.getElementById('questionInput');
  ta.addEventListener('input', () => {
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 140) + 'px';
  });
  ta.addEventListener('keydown', e => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault();
      submitQuery();
    }
  });

  // Health + KG stats
  await checkHealth();
  await loadKGStats();
})();

function setInput(text) {
  const ta = document.getElementById('questionInput');
  ta.value = text;
  ta.style.height = 'auto';
  ta.style.height = Math.min(ta.scrollHeight, 140) + 'px';
}

// ── Health check ─────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const r = await fetch('/api/health');
    const data = await r.json();
    const panel = document.getElementById('statusPanel');
    if (data.status === 'ok') {
      panel.innerHTML = `
        <div class="status-item"><span class="dot-ok"></span> Neo4j connected</div>
        <div class="status-item"><span class="dot-ok"></span> Approach A (Text-to-Cypher)</div>
        <div class="status-item"><span class="dot-ok"></span> Approach B (Graph RAG)</div>`;
    } else {
      panel.innerHTML = `<div class="status-item"><span class="dot-fail"></span> ${data.error || 'Backend unavailable'}</div>`;
    }
  } catch (e) {
    document.getElementById('statusPanel').innerHTML =
      '<div class="status-item"><span class="dot-fail"></span> Cannot reach server</div>';
  }
}

// ── KG stats ─────────────────────────────────────────────────────────────
async function loadKGStats() {
  const el = document.getElementById('kgStats');
  try {
    const r = await fetch('/api/graph/stats');
    const data = await r.json();
    if (data.error) { el.textContent = 'Stats unavailable'; return; }
    const chips = Object.entries(data)
      .map(([k, v]) => `<span class="kg-chip"><span>${v}</span> ${k}</span>`)
      .join('');
    el.innerHTML = `<div class="kg-chips" style="padding:0;">${chips}</div>`;
  } catch { el.textContent = 'Stats unavailable'; }
}

// ── Submit ────────────────────────────────────────────────────────────────
async function submitQuery() {
  const ta = document.getElementById('questionInput');
  const question = ta.value.trim();
  if (!question) return;

  const mode = document.querySelector('#modeGroup .radio-opt.active')?.dataset.mode || 'both';

  // Clear empty state
  document.getElementById('emptyState')?.remove();

  const chatArea = document.getElementById('chatArea');
  const sendBtn  = document.getElementById('sendBtn');
  sendBtn.disabled = true;

  // User bubble
  const ts = new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'});
  const turn = document.createElement('div');
  turn.className = 'turn';
  turn.innerHTML = `
    <div class="user-msg">
      <div class="user-bubble">
        <div class="bubble-meta">YOU · ${ts}</div>
        ${escHtml(question)}
      </div>
    </div>
    <div class="answers${mode === 'both' ? ' dual' : ''}" id="answerGrid_${Date.now()}">
      <div class="spinner-wrap"><div class="spinner"></div>Running ${mode === 'both' ? 'both approaches' : 'Approach ' + mode}…</div>
    </div>`;
  chatArea.appendChild(turn);
  chatArea.scrollTop = chatArea.scrollHeight;
  ta.value = '';
  ta.style.height = 'auto';

  const grid = turn.querySelector('.answers');

  try {
    const res = await fetch('/api/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question, mode}),
    });
    const data = await res.json();

    grid.innerHTML = '';

    if (data.error) {
      grid.innerHTML = `<div class="ans-card error-card">
        <div class="ans-title err">✗ Error</div>
        <div class="ans-body" style="color:var(--danger);">${escHtml(data.error)}</div>
      </div>`;
    } else {
      let sumLat = 0;
      Object.entries(data.answers).forEach(([k, ans]) => {
        sumLat += ans.latency_ms || 0;
        grid.appendChild(buildCard(k, ans));
      });
      totalQ++;
      totalLat += sumLat / Object.keys(data.answers).length;
      document.getElementById('statQ').textContent = totalQ;
      document.getElementById('statL').textContent = Math.round(totalLat / totalQ);
    }
  } catch (e) {
    grid.innerHTML = `<div class="ans-card error-card">
      <div class="ans-title err">✗ Network Error</div>
      <div class="ans-body">${escHtml(e.message)}</div>
    </div>`;
  }

  chatArea.scrollTop = chatArea.scrollHeight;
  sendBtn.disabled = false;
  ta.focus();
}

// ── Card builder ──────────────────────────────────────────────────────────
function buildCard(key, ans) {
  const card = document.createElement('div');
  const cls = ans.error ? 'error-card' : `approach-${key.toLowerCase()}`;
  const titleCls = ans.error ? 'err' : key.toLowerCase();
  const icon = key === 'A' ? '◈' : '◉';
  const label = key === 'A' ? 'Approach A — Text-to-Cypher' : 'Approach B — Graph RAG';

  let body = '';
  if (ans.error) {
    body = `<div class="ans-body" style="color:var(--danger);">⚠ ${escHtml(ans.error)}</div>`;
  } else {
    body = `<div class="ans-body">${escHtml(ans.text || 'No answer returned.')}</div>`;
    body += `<div class="ans-meta">
      <span>⏱ ${Math.round(ans.latency_ms)} ms</span>
      <span>📎 ${(ans.cited_doc_ids || []).length} source(s)</span>
      <span>🔀 ${ans.approach || key}</span>
    </div>`;

    // Cypher expander (A)
    if (key === 'A') {
      const attempts = (ans.trace || {}).attempts || [];
      const cypher = attempts.length ? (attempts[attempts.length-1].cypher || '') : '';
      if (cypher) {
        body += `<div class="expander">
          <button class="exp-toggle" onclick="toggleExp(this)">⟨/⟩ Generated Cypher</button>
          <div class="exp-body">
            <div class="cypher-block">${escHtml(cypher)}</div>
          </div>
        </div>`;
      }
    }

    // Docs expander (B)
    if (key === 'B') {
      const docs = (ans.trace || {}).retrieved_docs || [];
      if (docs.length) {
        const items = docs.map(d => {
          const score = (parseFloat(d.score) || 0).toFixed(3);
          const title = escHtml(d.title || 'Untitled');
          const pub   = d.publisher ? escHtml(d.publisher) : '';
          const url   = d.url || '#';
          const target = url !== '#' ? ' target="_blank" rel="noopener noreferrer"' : '';
          const hasLink = url !== '#';
          return `<a class="doc-item" href="${escHtml(url)}"${target} title="Similarity: ${score}${pub ? ' · ' + d.publisher : ''}">
            <span class="doc-score-dot"></span>
            <span class="doc-title">${title}</span>
            ${pub ? `<span class="doc-pub">${pub}</span>` : ''}
            ${hasLink ? `<span class="doc-ext">↗</span>` : ''}
          </a>`;
        }).join('');
        body += `<div class="expander">
          <button class="exp-toggle" onclick="toggleExp(this)">📄 Retrieved documents (${docs.length})</button>
          <div class="exp-body" style="margin-top:8px; display:none; flex-direction:column; gap:4px;">${items}</div>
        </div>`;
      }
    }
  }

  card.className = `ans-card ${cls}`;
  card.innerHTML = `<div class="ans-title ${titleCls}">${icon} ${label}</div>${body}`;
  return card;
}

// ── Helpers ───────────────────────────────────────────────────────────────
function toggleExp(btn) {
  const body = btn.nextElementSibling;
  const isOpen = body.style.display === 'flex' || body.classList.contains('open');
  if (isOpen) {
    body.style.display = 'none';
    body.classList.remove('open');
  } else {
    body.style.display = body.style.flexDirection ? 'flex' : 'block';
    body.classList.add('open');
  }
}

function clearChat() {
  const ca = document.getElementById('chatArea');
  ca.innerHTML = `<div id="emptyState" class="empty-state">
    <div class="empty-icon">🔭</div>
    <div class="empty-title">AWAITING QUERY</div>
    <div class="empty-body">Ask anything about AI startups, funding rounds,<br>investors, products, or technologies.</div>
  </div>`;
  totalQ = 0; totalLat = 0;
  document.getElementById('statQ').textContent = '0';
  document.getElementById('statL').textContent = '—';
}

function escHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/\n/g, '<br>');
}
</script>
</body>
</html>
"""


@app.get("/")
def index():
    return render_template_string(HTML)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8502, debug=True)