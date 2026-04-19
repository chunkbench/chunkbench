import React, { useState, useEffect } from 'react'

const API = '/api'

const STRATEGIES = ['S1', 'S2', 'S3', 'S4', 'B0']
const STRATEGY_NAMES = {
  S1: 'Fixed-size', S2: 'Recursive', S3: 'Semantic',
  S4: 'Proposition', B0: 'No-RAG Baseline'
}
const STRATEGY_COLORS = {
  S1: '#3b82f6', S2: '#8b5cf6', S3: '#10b981',
  S4: '#f59e0b', B0: '#6b7280'
}

const FROZEN_PARAMS = {
  k: 10, context_mode: 'fixed-budget', context_budget_tokens: 1800,
  context_budget_chars: 6000, temperature: 0.0, max_tokens: 512,
  dedup: true, dedup_threshold: 0.95,
}

function App() {
  const [mode, setMode] = useState('phase2')
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [indexStats, setIndexStats] = useState(null)
  const [error, setError] = useState(null)

  // Demo mode controls
  const [k, setK] = useState(10)
  const [temperature, setTemperature] = useState(0.0)
  const [maxTokens, setMaxTokens] = useState(512)
  const [dedup, setDedup] = useState(true)
  const [dedupThreshold, setDedupThreshold] = useState(0.95)
  const [contextMode, setContextMode] = useState('fixed-budget')
  const [contextBudgetTokens, setContextBudgetTokens] = useState(1800)
  const [contextBudgetChars, setContextBudgetChars] = useState(6000)
  const [selectedStrategies, setSelectedStrategies] = useState(STRATEGIES)

  useEffect(() => {
    const m = mode === 'demo' ? 'phase2' : mode
    fetch(`${API}/index-stats?mode=${m}`).then(r => r.json()).then(setIndexStats).catch(() => {})
  }, [mode])

  const handleQuery = async () => {
    if (!question.trim()) return
    setLoading(true)
    setError(null)
    try {
      let body
      if (mode === 'phase1' || mode === 'phase2') {
        body = { question: question.trim(), mode, ...FROZEN_PARAMS, strategies: null }
      } else {
        body = {
          question: question.trim(),
          mode: 'demo',
          k,
          context_mode: contextMode,
          context_budget_tokens: contextBudgetTokens,
          context_budget_chars: contextBudgetChars,
          temperature,
          max_tokens: maxTokens,
          dedup,
          dedup_threshold: dedup ? dedupThreshold : null,
          strategies: selectedStrategies,
        }
      }
      const res = await fetch(`${API}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      setResults(await res.json())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const toggleStrategy = (sid) => {
    setSelectedStrategies(prev =>
      prev.includes(sid) ? prev.filter(s => s !== sid) : [...prev, sid]
    )
  }

  const modeLabel = mode === 'phase1' ? 'Phase 1' : mode === 'phase2' ? 'Phase 2' : 'Demo'

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <h1 style={styles.title}>HAE-RAG Chunking Benchmark</h1>
        <p style={styles.subtitle}>
          Comparing chunking strategies for medical RAG in Hereditary Angioedema
        </p>
        <div style={styles.modeToggle}>
          {[
            { id: 'phase1', label: 'Phase 1' },
            { id: 'phase2', label: 'Phase 2' },
            { id: 'demo',   label: 'Demo' },
          ].map(({ id, label }) => (
            <button key={id} onClick={() => { setMode(id); setResults(null) }}
              style={{ ...styles.modeBtn, ...(mode === id ? styles.modeBtnActive : {}) }}>
              {label}
            </button>
          ))}
        </div>
      </header>

      {(mode === 'phase1' || mode === 'phase2') && (
        <div style={styles.frozenPanel}>
          <h3 style={styles.frozenTitle}>
            {mode === 'phase1'
              ? 'Phase 1 — Starting-point Benchmark (Frozen)'
              : 'Phase 2 — Best-config Benchmark (Frozen)'}
          </h3>
          <div style={styles.frozenGrid}>
            <span>Candidate depth: <strong>k=10</strong></span>
            <span>Context: <strong>1800 tokens (fixed-budget)</strong></span>
            <span>Packing: <strong>rank-order, whole-chunk</strong></span>
            <span>Dedup: <strong>cosine ≥ 0.95</strong></span>
            <span>Embedding: <strong>BAAI/bge-m3</strong></span>
            <span>LLM: <strong>DeepSeek-V3, T=0.0</strong></span>
          </div>
          <div style={styles.strategyParamGrid}>
            {mode === 'phase1' && <>
              <div style={styles.stratParamCard}><strong>S1</strong> Fixed-size · size=512, overlap=50 (10%) · 17,886 chunks</div>
              <div style={styles.stratParamCard}><strong>S2</strong> Recursive · size=512, overlap=50 (10%) · 18,543 chunks</div>
              <div style={styles.stratParamCard}><strong>S3</strong> Semantic · percentile/95 · 3,920 chunks</div>
              <div style={styles.stratParamCard}><strong>S4</strong> Proposition · LLM decomp, default · 85,537 chunks</div>
              <div style={styles.stratParamCard}><strong>B0</strong> No-RAG baseline · no retrieval</div>
            </>}
            {mode === 'phase2' && <>
              <div style={styles.stratParamCard}><strong>S1</strong> Fixed-size · size=512, overlap=102 (20%) · 20,141 chunks</div>
              <div style={styles.stratParamCard}><strong>S2</strong> Recursive · size=1024, overlap=102 (10%) · 9,638 chunks</div>
              <div style={styles.stratParamCard}><strong>S3</strong> Semantic · threshold=85, max=2000 · 11,305 chunks</div>
              <div style={styles.stratParamCard}><strong>S4</strong> Proposition · LLM decomp, no tunable params · 85,537 chunks</div>
              <div style={styles.stratParamCard}><strong>B0</strong> No-RAG baseline · no retrieval</div>
            </>}
          </div>
        </div>
      )}

      {mode === 'demo' && (
        <div style={styles.paramPanel}>
          <h3 style={styles.paramTitle}>Query Parameters</h3>
          <div style={styles.paramGrid}>
            <label>
              Retrieval k: <strong>{k}</strong>
              <input type="range" min="1" max="20" value={k}
                onChange={e => setK(Number(e.target.value))} style={styles.slider} />
            </label>
            <label>
              Temperature: <strong>{temperature.toFixed(1)}</strong>
              <input type="range" min="0" max="10" value={temperature * 10}
                onChange={e => setTemperature(Number(e.target.value) / 10)} style={styles.slider} />
            </label>
            <label>
              Max output tokens: <strong>{maxTokens}</strong>
              <input type="range" min="256" max="1024" step="64" value={maxTokens}
                onChange={e => setMaxTokens(Number(e.target.value))} style={styles.slider} />
            </label>
            <div>
              <span style={{ fontSize: 14 }}>Context mode: </span>
              <button onClick={() => setContextMode('fixed-k')}
                style={{ ...styles.ctxBtn, ...(contextMode === 'fixed-k' ? styles.ctxBtnActive : {}) }}>
                fixed-k
              </button>
              <button onClick={() => setContextMode('fixed-budget')}
                style={{ ...styles.ctxBtn, ...(contextMode === 'fixed-budget' ? styles.ctxBtnActive : {}) }}>
                fixed-budget
              </button>
            </div>
            {contextMode === 'fixed-budget' && (
              <label>
                Token budget: <strong>{contextBudgetTokens}</strong>
                <input type="range" min="500" max="4000" step="100" value={contextBudgetTokens}
                  onChange={e => setContextBudgetTokens(Number(e.target.value))} style={styles.slider} />
              </label>
            )}
            {contextMode === 'fixed-k' && (
              <label>
                Char limit: <strong>{contextBudgetChars}</strong>
                <input type="range" min="2000" max="20000" step="500" value={contextBudgetChars}
                  onChange={e => setContextBudgetChars(Number(e.target.value))} style={styles.slider} />
              </label>
            )}
            <label style={styles.checkLabel}>
              <input type="checkbox" checked={dedup} onChange={e => setDedup(e.target.checked)} />
              Deduplication
            </label>
            {dedup && (
              <label>
                Dedup threshold: <strong>{dedupThreshold.toFixed(2)}</strong>
                <input type="range" min="80" max="99" value={dedupThreshold * 100}
                  onChange={e => setDedupThreshold(Number(e.target.value) / 100)} style={styles.slider} />
              </label>
            )}
          </div>
          <div style={styles.strategyCheckboxes}>
            {STRATEGIES.map(sid => (
              <label key={sid} style={styles.checkLabel}>
                <input type="checkbox" checked={selectedStrategies.includes(sid)}
                  onChange={() => toggleStrategy(sid)} />
                <span style={{ color: STRATEGY_COLORS[sid], fontWeight: 'bold' }}>{sid}</span>
                {' '}{STRATEGY_NAMES[sid]}
              </label>
            ))}
          </div>
        </div>
      )}

      <div style={styles.querySection}>
        <textarea value={question} onChange={e => setQuestion(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleQuery() } }}
          placeholder="Ask a question about HAE..."
          style={styles.textarea} rows={3} />
        <button onClick={handleQuery} disabled={loading || !question.trim()} style={styles.queryBtn}>
          {loading ? 'Querying...' : 'Query All Strategies'}
        </button>
        {(mode === 'phase1' || mode === 'phase2') && (
          <span style={styles.frozenBadge}>{modeLabel}: k=10, budget=1800tok, dedup=0.95</span>
        )}
      </div>

      {error && <div style={styles.error}>{error}</div>}

      {results && (
        <div style={styles.resultsSection}>
          <h2 style={styles.resultsTitle}>Results</h2>
          <div style={styles.strategyGrid}>
            {Object.entries(results.results).map(([sid, r]) => (
              <div key={sid} style={{ ...styles.strategyCard, borderTopColor: STRATEGY_COLORS[sid] }}>
                <div style={styles.cardHeader}>
                  <span style={{ ...styles.strategyBadge, backgroundColor: STRATEGY_COLORS[sid] }}>{sid}</span>
                  <span style={styles.strategyName}>{r.strategy_name}</span>
                  <span style={styles.latency}>{r.latency_s}s</span>
                </div>
                <div style={styles.answer}>{r.answer}</div>
                {r.retrieved_chunks && r.retrieved_chunks.length > 0 && (
                  <details style={styles.chunksDetails}>
                    <summary style={styles.chunksSummary}>
                      {r.retrieved_chunks.length} chunks | ~{r.context_tokens_est} tokens
                    </summary>
                    <div style={styles.chunksList}>
                      {r.retrieved_chunks.map((c, i) => (
                        <div key={i} style={styles.chunk}>
                          <div style={styles.chunkMeta}>
                            Rank {c.rank} | {c.doc_id} | dist: {c.distance?.toFixed(4)}
                          </div>
                          <div style={styles.chunkText}>{c.text.substring(0, 300)}...</div>
                        </div>
                      ))}
                    </div>
                  </details>
                )}
                {r.error && <div style={styles.cardError}>{r.error}</div>}
              </div>
            ))}
          </div>
        </div>
      )}

      {indexStats && Object.keys(indexStats).length > 0 && (
        <div style={styles.statsPanel}>
          <h3 style={styles.statsTitle}>Index Statistics</h3>
          <div style={styles.statsGrid}>
            {Object.entries(indexStats).map(([sid, s]) => (
              <div key={sid} style={styles.statCard}>
                <strong style={{ color: STRATEGY_COLORS[sid] }}>{sid}: {s.name}</strong>
                <br />{s.total_documents} docs | {s.total_chunks} chunks
              </div>
            ))}
          </div>
        </div>
      )}

      <footer style={styles.footer}>
        HAE-RAG Chunking Benchmark v4.0
      </footer>
    </div>
  )
}

const styles = {
  container: { maxWidth: 1200, margin: '0 auto', padding: '20px', fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', color: '#1a1a2e', backgroundColor: '#f8f9fa', minHeight: '100vh' },
  header: { textAlign: 'center', marginBottom: 24, padding: '24px 0', borderBottom: '2px solid #e2e8f0' },
  title: { fontSize: 28, fontWeight: 700, margin: 0, color: '#1a1a2e' },
  subtitle: { fontSize: 14, color: '#64748b', marginTop: 8 },
  modeToggle: { display: 'flex', justifyContent: 'center', gap: 8, marginTop: 16 },
  modeBtn: { padding: '8px 20px', border: '1px solid #cbd5e1', borderRadius: 6, cursor: 'pointer', background: 'white', fontSize: 14 },
  modeBtnActive: { background: '#1a1a2e', color: 'white', borderColor: '#1a1a2e' },
  frozenPanel: { background: '#f1f5f9', border: '1px solid #e2e8f0', borderRadius: 8, padding: 16, marginBottom: 16 },
  frozenTitle: { margin: '0 0 8px', fontSize: 14, color: '#475569' },
  frozenGrid: { display: 'flex', flexWrap: 'wrap', gap: '8px 24px', fontSize: 13, color: '#64748b', marginBottom: 10 },
  strategyParamGrid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 6, marginTop: 4 },
  stratParamCard: { background: 'white', border: '1px solid #e2e8f0', borderRadius: 5, padding: '5px 10px', fontSize: 12, color: '#475569' },
  paramPanel: { background: 'white', border: '1px solid #e2e8f0', borderRadius: 8, padding: 16, marginBottom: 16 },
  paramTitle: { margin: '0 0 12px', fontSize: 14, color: '#475569' },
  paramGrid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12 },
  slider: { width: '100%', marginTop: 4 },
  checkLabel: { display: 'flex', alignItems: 'center', gap: 6, fontSize: 14, cursor: 'pointer' },
  strategyCheckboxes: { display: 'flex', flexWrap: 'wrap', gap: 16, marginTop: 12, paddingTop: 12, borderTop: '1px solid #e2e8f0' },
  ctxBtn: { padding: '4px 12px', border: '1px solid #cbd5e1', borderRadius: 4, cursor: 'pointer', background: 'white', fontSize: 13, marginLeft: 4 },
  ctxBtnActive: { background: '#1a1a2e', color: 'white', borderColor: '#1a1a2e' },
  querySection: { marginBottom: 24 },
  textarea: { width: '100%', padding: 12, fontSize: 15, border: '2px solid #e2e8f0', borderRadius: 8, resize: 'vertical', fontFamily: 'inherit', boxSizing: 'border-box' },
  queryBtn: { marginTop: 8, padding: '10px 24px', fontSize: 15, fontWeight: 600, background: '#1a1a2e', color: 'white', border: 'none', borderRadius: 8, cursor: 'pointer' },
  frozenBadge: { marginLeft: 12, fontSize: 12, color: '#64748b', background: '#f1f5f9', padding: '4px 10px', borderRadius: 4 },
  error: { background: '#fef2f2', color: '#dc2626', padding: 12, borderRadius: 8, marginBottom: 16 },
  resultsSection: { marginTop: 16 },
  resultsTitle: { fontSize: 20, marginBottom: 16 },
  strategyGrid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: 16 },
  strategyCard: { background: 'white', borderRadius: 8, padding: 16, border: '1px solid #e2e8f0', borderTop: '3px solid #ccc' },
  cardHeader: { display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 },
  strategyBadge: { color: 'white', padding: '2px 8px', borderRadius: 4, fontSize: 12, fontWeight: 700 },
  strategyName: { fontSize: 14, fontWeight: 600 },
  latency: { marginLeft: 'auto', fontSize: 12, color: '#94a3b8' },
  answer: { fontSize: 14, lineHeight: 1.6, color: '#334155', padding: '8px 0' },
  chunksDetails: { marginTop: 8, borderTop: '1px solid #f1f5f9', paddingTop: 8 },
  chunksSummary: { fontSize: 12, color: '#64748b', cursor: 'pointer' },
  chunksList: { marginTop: 8 },
  chunk: { background: '#f8fafc', borderRadius: 4, padding: 8, marginBottom: 6, fontSize: 12 },
  chunkMeta: { color: '#94a3b8', marginBottom: 4, fontSize: 11 },
  chunkText: { color: '#475569', lineHeight: 1.4 },
  cardError: { color: '#dc2626', fontSize: 12, marginTop: 8 },
  statsPanel: { background: '#f1f5f9', borderRadius: 8, padding: 16, marginTop: 24 },
  statsTitle: { margin: '0 0 12px', fontSize: 14 },
  statsGrid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12 },
  statCard: { background: 'white', padding: 12, borderRadius: 6, fontSize: 13 },
  footer: { textAlign: 'center', fontSize: 12, color: '#94a3b8', marginTop: 40, paddingTop: 16, borderTop: '1px solid #e2e8f0' },
}

export default App
