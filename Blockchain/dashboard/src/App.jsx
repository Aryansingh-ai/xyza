import { useEffect, useRef, useState } from "react";
import {
  ShieldCheck, ShieldAlert, RefreshCw, XCircle, Link2, Cpu, Hash, Radar,
  AlertTriangle, ExternalLink, Eye, EyeOff, Key, Sparkles, CheckCircle, Bot
} from "lucide-react";
import "./App.css";

const API_URL = (import.meta.env.VITE_API_URL || "http://localhost:3000").replace(/\/+$/, "");

function useCountUp(target, ms = 900) {
  const [val, setVal] = useState(target);
  const from = useRef(target);
  useEffect(() => {
    const start = performance.now();
    const a = from.current;
    let raf;
    const tick = (t) => {
      const p = Math.min((t - start) / ms, 1);
      setVal(Math.round(a + (target - a) * (1 - Math.pow(1 - p, 3))));
      if (p < 1) raf = requestAnimationFrame(tick);
      else from.current = target;
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, ms]);
  return val;
}

/* ---------- helpers (real data only) ---------- */
const recNo = (r, i) => r.recordNumber ?? r.id ?? i + 1;
const shortHash = (r, i) => {
  const h = r.proofHash || r.originalProofHash || r.hash || r.currentHash || r.sha256 || r.proof || "";
  return h ? `${String(h).slice(0, 6)}…${String(h).slice(-4)}` : `0x${Number(recNo(r, i)).toString(16).padStart(4, "0")}`;
};
const clip = (h) => (h ? `${String(h).slice(0, 10)}…${String(h).slice(-6)}` : null);
const money = (v) => (v == null || v === "" ? "—" : String(v).startsWith("₹") ? v : `₹${v}`);
const when = (r) => {
  const t = r.record?.timestamp ?? r.timestamp;
  const d = t ? new Date(t) : null;
  return d && !isNaN(d) ? d.toLocaleString([], { dateStyle: "short", timeStyle: "short" }) : null;
};
const details = (d) => (!d ? null : Array.isArray(d) ? d.join(" · ") : typeof d === "object" ? JSON.stringify(d) : String(d));

/* ---------- decorative: gopuram silhouette ---------- */
function Gopuram() {
  return (
    <svg className="gopuram" viewBox="0 0 300 380" aria-hidden="true">
      <defs>
        <radialGradient id="halo" cx="50%" cy="38%" r="55%">
          <stop offset="0" stopColor="#f3b24a" stopOpacity="0.55" />
          <stop offset="1" stopColor="#f3b24a" stopOpacity="0" />
        </radialGradient>
      </defs>
      <circle cx="150" cy="150" r="150" fill="url(#halo)" className="halo" />
      <g className="body">
        <rect x="30" y="338" width="240" height="30" />
        <rect x="50" y="308" width="200" height="30" />
        {[0, 1, 2, 3, 4, 5, 6].map((i) => (
          <g key={i}>
            <rect x={70 + i * 11} y={282 - i * 38} width={160 - i * 22} height="38" rx="2" />
            <rect x={70 + i * 11} y={282 - i * 38} width={160 - i * 22} height="3" className="gold" />
            <rect x="143" y={292 - i * 38} width="14" height="20" rx="7" className="door" />
          </g>
        ))}
        <path d="M150 -6 L158 22 H142 Z" className="gold" />
        <rect x="140" y="22" width="20" height="10" className="gold" />
      </g>
    </svg>
  );
}

/* ---------- gauge + stats ---------- */
function Insights({ total, verified, tampered, pct }) {
  const [armed, setArmed] = useState(false);
  useEffect(() => { const t = setTimeout(() => setArmed(true), 80); return () => clearTimeout(t); }, []);
  const nTotal = useCountUp(armed ? total : 0, 1100);
  const nOk = useCountUp(armed ? verified : 0, 1100);
  const nBad = useCountUp(armed ? tampered : 0, 1100);
  const shown = useCountUp(armed ? Math.round(pct * 10) : 0, 1300) / 10;
  const bad = tampered > 0;
  const r = 86, c = 2 * Math.PI * r;
  return (
    <section className={`panel ${bad ? "is-bad" : "is-ok"}`} aria-label="Integrity">
      <div className="gauge-box">
        <svg className="gauge" viewBox="0 0 240 240" role="img" aria-label={`${pct.toFixed(1)}% intact`}>
          <defs>
            <linearGradient id="gOk" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stopColor="#e08a1e" /><stop offset="1" stopColor="#1f7a5a" /></linearGradient>
            <linearGradient id="gBad" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stopColor="#e08a1e" /><stop offset="1" stopColor="#b3261e" /></linearGradient>
          </defs>
          <circle cx="120" cy="120" r="108" className="g-ticks" />
          <circle cx="120" cy="120" r="70" className="g-inner" />
          <circle cx="120" cy="120" r={r} className="g-track" />
          <g transform="rotate(-90 120 120)">
            <circle cx="120" cy="120" r={r} className="g-fill ok" strokeDasharray={c} strokeDashoffset={c - (c * shown) / 100} />
            <circle cx="120" cy="120" r={r} className="g-fill badarc" strokeDasharray={c} strokeDashoffset={c - (c * shown) / 100} />
          </g>
          <text x="120" y="124" className="g-num">{shown.toFixed(1)}</text>
          <text x="120" y="148" className="g-sub">% intact</text>
        </svg>
      </div>
      <div className="numbers">
        <div className="stat"><strong>{nTotal.toLocaleString()}</strong><span>records registered</span></div>
        <div className="stat g"><strong>{nOk.toLocaleString()}</strong><span>match their proof</span></div>
        <div className={`stat r ${bad ? "dominant" : ""}`}><strong>{nBad.toLocaleString()}</strong><span>failed verification</span></div>
      </div>
    </section>
  );
}

/* ---------- tampered record comparison ---------- */
function TamperCard({ r, i }) {
  const note = details(r.tamperDetails);
  return (
    <article className="tcard">
      <header>
        <b><AlertTriangle size={16} /> Record #{recNo(r, i)}</b>
        <span className="mismatch">Blockchain proof mismatch</span>
      </header>
      <div className="cmp">
        <div className="side">
          <small>Original</small>
          <strong>{money(r.originalDenomination)}</strong>
          <code>{r.originalBin ?? "—"}</code>
        </div>
        <div className="neq" aria-label="does not equal"><i /><b>≠</b><i /></div>
        <div className="side now">
          <small>Current</small>
          <strong>{money(r.currentDenomination)}</strong>
          <code>{r.currentBin ?? "—"}</code>
        </div>
      </div>
      {(r.originalProofHash || r.proofHash) && (
        <div className="hashes">
          <div><small>Original proof</small><code>{clip(r.originalProofHash) ?? "—"}</code></div>
          <span className="hbreak">→</span>
          <div className="now"><small>Current calculated proof</small><code>{clip(r.proofHash) ?? "—"}</code></div>
        </div>
      )}
      {note && <p className="tnote">{note}</p>}
      {r.explorerUrl && (
        <a className="tx" href={r.explorerUrl} target="_blank" rel="noopener noreferrer">
          View blockchain transaction <ExternalLink size={14} />
        </a>
      )}
    </article>
  );
}

/* ---------- envio proof analytics & explorer ---------- */
function ProofAnalytics() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const load = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/proof-analytics`);
      if (!res.ok) throw new Error("Unavailable");
      const json = await res.json();
      if (!json.success) throw new Error(json.error);
      setData(json);
      setError(null);
    } catch (err) {
      setError("On-chain analytics temporarily unavailable.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  if (loading && !data) return null;
  if (error) return <div className="panel muted" style={{ marginTop: '2rem' }}><AlertTriangle size={14} /> {error}</div>;

  return (
    <section className="analytics panel" style={{ marginTop: '2rem', padding: '2rem' }}>
      <div className="sec-head row">
        <div>
          <h2>On-Chain Analytics</h2>
          <span className="muted">Global HyperIndex Statistics</span>
        </div>
        {data.latestProcessedBlock && (
          <span className="pill" style={{ fontSize: '0.8rem', background: 'var(--bg-sub)' }}>
            <span style={{ color: 'var(--ok)' }}>●</span> Synced to block {data.latestProcessedBlock}
          </span>
        )}
      </div>
      <div className="numbers" style={{ justifyContent: 'flex-start', gap: '3rem', marginTop: '1.5rem', flexWrap: 'wrap' }}>
        <div className="stat">
          <strong>{data.total}{data.isPartial ? '+' : ''}</strong>
          <span>Total Indexed Proofs</span>
        </div>
        {Object.entries(data.byIssuer || {}).map(([issuer, count]) => (
          <div className="stat" key={issuer}>
            <strong>{count}{data.isPartial ? '+' : ''}</strong>
            <span>Proofs by Issuer ({clip(issuer)})</span>
          </div>
        ))}
      </div>
      {data.isPartial && (
        <div className="muted" style={{ marginTop: '1.5rem', fontSize: '0.85rem' }}>
          * Displaying aggregated results for the most recent 10,000 proofs.
        </div>
      )}
    </section>
  );
}

function ProofVerifier({ selectedHash }) {
  const [hash, setHash] = useState(selectedHash || "");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [rpcError, setRpcError] = useState(null);

  const verify = async (targetHash) => {
    const queryHash = (targetHash || hash).trim();
    if (!queryHash) return;
    setLoading(true);
    setRpcError(null);
    setResult(null);

    try {
      const res = await fetch(`${API_URL}/verify-proof/${encodeURIComponent(queryHash)}`);
      if (!res.ok) {
        throw new Error(`API returned HTTP ${res.status}`);
      }
      const data = await res.json();
      if (!data.success) {
        throw new Error(data.error || "Verification failed");
      }
      setResult(data);
    } catch (err) {
      console.error("Verification error:", err);
      setRpcError(err.message || "Failed to reach RPC / API");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (selectedHash) {
      setHash(selectedHash);
      verify(selectedHash);
    }
  }, [selectedHash]);

  const handleSample = () => {
    const sample = "0x1b79ba86c84e91772811ac5677b2c80897b16c7a580c85d9514a22b4cbbd50d8";
    setHash(sample);
    verify(sample);
  };

  return (
    <section className="verifier panel" style={{ marginTop: '2.5rem', padding: '2rem' }}>
      <div className="sec-head row">
        <div>
          <h2>Public Proof Verifier</h2>
          <span className="muted">Direct Monad Testnet On-Chain Verification (Read-Only)</span>
        </div>
        <span className="pill" style={{ fontSize: '0.8rem', background: 'var(--parchment)' }}>
          <ShieldCheck size={14} color="var(--emerald)" /> V1 Contract: 0xBCD9…d7e6
        </span>
      </div>

      <form onSubmit={(e) => { e.preventDefault(); verify(); }} style={{ display: 'flex', gap: '0.75rem', marginTop: '1.25rem', flexWrap: 'wrap' }}>
        <input
          type="text"
          className="v-input"
          value={hash}
          onChange={(e) => setHash(e.target.value)}
          placeholder="Enter 0x-prefixed 32-byte proof hash..."
          style={{
            flex: '1',
            minWidth: '280px',
            padding: '10px 14px',
            borderRadius: '6px',
            border: '1px solid var(--line)',
            background: '#fffdf6',
            color: 'var(--ink)',
            fontFamily: '"JetBrains Mono", monospace',
            fontSize: '13px'
          }}
        />
        <button type="submit" className="btn" disabled={loading} style={{ margin: 0 }}>
          {loading ? <RefreshCw size={14} className="refreshed" /> : <ShieldCheck size={16} />}
          {loading ? "Verifying..." : "Verify Proof"}
        </button>
        <button type="button" onClick={handleSample} className="btn" style={{ margin: 0, background: 'var(--parchment)', color: 'var(--maroon)', border: '1px solid var(--line)' }}>
          Sample Proof
        </button>
      </form>

      {rpcError && (
        <div className="verdict bad" style={{ marginTop: '1.5rem', width: '100%' }}>
          <span className="v-icon"><AlertTriangle size={24} /></span>
          <div>
            <strong>RPC / BACKEND UNREACHABLE</strong>
            <span>{rpcError} (Note: Network failure is distinct from an unverified proof).</span>
          </div>
        </div>
      )}

      {result && result.exists && (
        <div className="verdict ok" style={{ marginTop: '1.5rem', width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <span className="v-icon"><ShieldCheck size={24} /></span>
            <div>
              <strong style={{ color: 'var(--emerald)' }}>VERIFIED ON MONAD TESTNET</strong>
              <span>Authoritative contract storage confirmed proof existence.</span>
            </div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', width: '100%', background: '#fffdf6', padding: '1rem', borderRadius: '6px', border: '1px solid var(--line)' }}>
            <div>
              <small className="muted" style={{ display: 'block', fontSize: '11px', textTransform: 'uppercase' }}>Source ID</small>
              <strong>{result.sourceId || "—"}</strong>
            </div>
            <div>
              <small className="muted" style={{ display: 'block', fontSize: '11px', textTransform: 'uppercase' }}>Registered By</small>
              <code style={{ fontSize: '12px' }}>{clip(result.registeredBy) || "—"}</code>
            </div>
            <div>
              <small className="muted" style={{ display: 'block', fontSize: '11px', textTransform: 'uppercase' }}>Registration Time</small>
              <span style={{ fontSize: '12px' }}>{result.timestamp ? new Date(Number(result.timestamp) * 1000).toLocaleString() : "—"}</span>
            </div>
            {result.transactionHash && (
              <div>
                <small className="muted" style={{ display: 'block', fontSize: '11px', textTransform: 'uppercase' }}>Tx Hash (Envio)</small>
                <a href={result.explorerUrl || `https://testnet.monadexplorer.com/tx/${result.transactionHash}`} target="_blank" rel="noopener noreferrer" style={{ color: 'var(--maroon)', display: 'inline-flex', alignItems: 'center', gap: '4px', fontSize: '12px', fontWeight: 600 }}>
                  {clip(result.transactionHash)} <ExternalLink size={12} />
                </a>
              </div>
            )}
          </div>
        </div>
      )}

      {result && !result.exists && (
        <div className="verdict bad" style={{ marginTop: '1.5rem', width: '100%' }}>
          <span className="v-icon"><AlertTriangle size={24} /></span>
          <div>
            <strong>NOT REGISTERED ON MONAD</strong>
            <span>This proof hash was not found in DonationProvenanceRegistry storage. It has not been registered or is pending confirmation.</span>
          </div>
        </div>
      )}
    </section>
  );
}

function ProofExplorer({ onVerifyProof }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const load = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/proof-history?limit=10`);
      if (!res.ok) throw new Error("Unavailable");
      const json = await res.json();
      if (!json.success) throw new Error(json.error);
      setData(json.records || []);
      setError(null);
    } catch {
      setError("On-chain history temporarily unavailable; indexer synchronization may be interrupted.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  if (loading && !data) return <div className="muted" style={{ padding: '2rem' }}>Loading Explorer...</div>;

  if (error) {
    return (
      <section className="explorer" style={{ marginTop: '3rem', padding: '2rem', border: '1px solid var(--border)', borderRadius: '8px' }}>
        <div className="sec-head row">
          <div>
            <h2>On-Chain Proof Explorer</h2>
            <span className="muted">Live Envio HyperIndex Feed</span>
          </div>
          <button className="btn" onClick={load}><RefreshCw size={12} /> Retry</button>
        </div>
        <div className="verdict bad" style={{ marginTop: '1rem', padding: '1rem' }}>
          <span className="v-icon"><AlertTriangle size={20} /></span>
          <div>
            <strong>INDEXER OFFLINE</strong>
            <span>{error}</span>
          </div>
        </div>
      </section>
    );
  }

  return (
    <section className="explorer" style={{ marginTop: '3rem', paddingTop: '2rem' }}>
      <div className="sec-head row">
        <div>
          <h2>On-Chain Proof Explorer</h2>
          <span className="muted">Live Envio HyperIndex Feed</span>
        </div>
        <button className="btn" onClick={load}><RefreshCw size={12} /> Refresh Feed</button>
      </div>

      {data.length === 0 ? (
        <p className="muted">No proofs indexed yet.</p>
      ) : (
        <div className="tgrid" style={{ marginTop: '1rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          {data.map((r) => (
            <article key={r.proofHash} className="tcard" style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', alignItems: 'center', padding: '1rem 1.5rem' }}>
              <div>
                <small>Proof Hash</small>
                <code style={{ display: 'block' }}>{clip(r.proofHash)}</code>
              </div>
              <div>
                <small>Source ID</small>
                <strong>{r.sourceId}</strong>
              </div>
              <div>
                <small>Block</small>
                <strong>#{r.blockNumber}</strong>
                <div className="muted" style={{ fontSize: '0.75rem' }}>{new Date(Number(r.blockTimestamp) * 1000).toLocaleString([], { dateStyle: 'short', timeStyle: 'short' })}</div>
              </div>
              <div style={{ textAlign: 'right', display: 'flex', gap: '8px', justifyContent: 'flex-end', alignItems: 'center' }}>
                <button
                  type="button"
                  className="btn"
                  onClick={() => onVerifyProof && onVerifyProof(r.proofHash)}
                  style={{ margin: 0, padding: '6px 12px', fontSize: '11px', background: 'var(--parchment)', color: 'var(--maroon)', border: '1px solid var(--line)' }}
                >
                  <ShieldCheck size={12} /> Verify
                </button>
                <a className="tx" href={r.explorerUrl} target="_blank" rel="noopener noreferrer" style={{ marginTop: 0, padding: '6px 12px' }}>
                  Tx <ExternalLink size={12} />
                </a>
              </div>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}

/* ---------- Qwen AI Investigation Agent ---------- */
function QwenAgent() {
  const [apiKey, setApiKey] = useState("");
  const [showKey, setShowKey] = useState(false);
  const [targetId, setTargetId] = useState("");
  const [query, setQuery] = useState("");

  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState(null);

  const [running, setRunning] = useState(false);
  const [runResult, setRunResult] = useState(null);
  const [runError, setRunError] = useState(null);

  const handleTestKey = async () => {
    const key = apiKey.trim();
    if (!key) return;
    setTesting(true);
    setTestResult(null);
    setRunError(null);

    try {
      const res = await fetch(`${API_URL}/api/investigate/test-key`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ apiKey: key })
      });
      const data = await res.json();
      if (!res.ok || !data.success) {
        setTestResult({
          success: false,
          error: data.error || `HTTP ${res.status}: Key validation failed`
        });
      } else {
        setTestResult({
          success: true,
          message: data.message || "Qwen API key is valid and connected.",
          details: data.details
        });
      }
    } catch (err) {
      setTestResult({
        success: false,
        error: `Network failure connecting to API: ${err.message}`
      });
    } finally {
      setTesting(false);
    }
  };

  const handleRunAgent = async (e) => {
    if (e && e.preventDefault) e.preventDefault();
    const key = apiKey.trim();
    if (!key || running) return;

    setRunning(true);
    setRunError(null);
    setRunResult(null);
    setTestResult(null);

    try {
      const payload = {
        apiKey: key,
        query: query.trim() || undefined,
        targetId: targetId.trim() || undefined,
        maxTokens: 700
      };

      const res = await fetch(`${API_URL}/api/investigate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const data = await res.json();

      if (!res.ok || !data.success) {
        let errorMsg = data.error || `Investigation failed (HTTP ${res.status})`;
        if (res.status === 401 || data.code === "AUTH_ERROR" || data.code === "INVALID_KEY") {
          errorMsg = "Authentication failed: Invalid Qwen / OpenRouter API key.";
        } else if (res.status === 402 || data.code === "PAYMENT_REQUIRED" || data.code === "QUOTA_EXCEEDED") {
          errorMsg = "Quota / Credits exceeded: Your OpenRouter account requires credits.";
        } else if (res.status === 429 || data.code === "RATE_LIMIT") {
          errorMsg = "Rate limit reached: Too many requests on this API key.";
        } else if (res.status === 504 || data.code === "TIMEOUT") {
          errorMsg = "Investigation timeout: OpenRouter took longer than deadline.";
        }
        setRunError({ message: errorMsg, code: data.code, diagnostics: data.diagnostics });
      } else {
        setRunResult(data);
      }
    } catch (err) {
      setRunError({
        message: `Network failure: Unable to connect to backend server. ${err.message}`,
        code: "NETWORK_ERROR"
      });
    } finally {
      setRunning(false);
    }
  };

  const hasKey = apiKey.trim().length > 0;
  const report = runResult?.report;

  return (
    <section className="qwen-section panel" style={{ marginTop: "3rem", padding: "2.2rem" }}>
      <div className="sec-head row">
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <span className="qwen-avatar"><Bot size={22} color="var(--gold-light)" /></span>
            <h2 style={{ margin: 0 }}>Qwen AI Investigation Agent</h2>
          </div>
          <span className="muted" style={{ marginTop: "6px" }}>
            Autonomous Forensic Integrity Auditor · Powered by Alibaba Qwen via OpenRouter
          </span>
        </div>
        <span className="pill" style={{ background: "var(--parchment)" }}>
          <Sparkles size={14} color="var(--saffron)" /> Read-Only Agent
        </span>
      </div>

      <p className="qwen-desc" style={{ margin: "1rem 0 1.5rem", fontSize: "13.5px", lineHeight: "1.7", color: "#5a4e45" }}>
        The Qwen agent operates in read-only mode to independently cross-reference the local SQLite donation ledger,
        SHA-256 fingerprints, Monad smart contract state, and Envio HyperIndex proofs.
        <strong> Your API key is held exclusively in browser memory and sent per-request over HTTPS. It is never stored or logged.</strong>
      </p>

      {/* API Key Input and Action Bar */}
      <div className="qwen-key-box" style={{ background: "#fffdf6", padding: "1.5rem", borderRadius: "8px", border: "1px solid var(--line)" }}>
        <label style={{ display: "block", fontSize: "12px", fontWeight: "700", textTransform: "uppercase", letterSpacing: "0.1em", color: "var(--maroon)", marginBottom: "8px" }}>
          <Key size={14} style={{ verticalAlign: "middle", marginRight: "6px" }} />
          Qwen / OpenRouter API Key <span style={{ color: "var(--red)" }}>*</span>
        </label>

        <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap", alignItems: "center" }}>
          <div style={{ position: "relative", flex: "1", minWidth: "260px" }}>
            <input
              type={showKey ? "text" : "password"}
              value={apiKey}
              onChange={(e) => {
                setApiKey(e.target.value);
                setTestResult(null);
                setRunError(null);
              }}
              placeholder="sk-or-v1-..."
              autoComplete="off"
              spellCheck="false"
              className="v-input qwen-input"
              style={{
                width: "100%",
                padding: "11px 44px 11px 14px",
                borderRadius: "6px",
                border: "1px solid var(--line)",
                background: "#ffffff",
                color: "var(--ink)",
                fontFamily: '"JetBrains Mono", monospace',
                fontSize: "13px"
              }}
            />
            <button
              type="button"
              onClick={() => setShowKey(!showKey)}
              aria-label={showKey ? "Hide API key" : "Show API key"}
              title={showKey ? "Hide API key" : "Show API key"}
              style={{
                position: "absolute",
                right: "8px",
                top: "50%",
                transform: "translateY(-50%)",
                background: "transparent",
                border: "none",
                cursor: "pointer",
                padding: "6px",
                color: "var(--muted)",
                display: "flex",
                alignItems: "center"
              }}
            >
              {showKey ? <EyeOff size={16} /> : <Eye size={16} />}
            </button>
          </div>

          <button
            type="button"
            onClick={handleTestKey}
            disabled={!hasKey || testing || running}
            className="btn qwen-btn-secondary"
            style={{
              margin: 0,
              background: "var(--parchment)",
              color: "var(--maroon)",
              border: "1px solid var(--line)",
              opacity: !hasKey || testing || running ? 0.6 : 1,
              cursor: !hasKey || testing || running ? "not-allowed" : "pointer"
            }}
          >
            {testing ? <RefreshCw size={14} className="refreshed" /> : <Sparkles size={14} />}
            {testing ? "Testing Key..." : "Test Key"}
          </button>

          <button
            type="button"
            onClick={handleRunAgent}
            disabled={!hasKey || running || testing}
            className="btn qwen-btn-primary"
            style={{
              margin: 0,
              background: hasKey ? "var(--maroon)" : "#888",
              color: "#fff",
              opacity: !hasKey || running || testing ? 0.6 : 1,
              cursor: !hasKey || running || testing ? "not-allowed" : "pointer"
            }}
          >
            {running ? <RefreshCw size={14} className="refreshed" /> : <Radar size={15} />}
            {running ? "Auditing Ledger..." : "Run Agent"}
          </button>
        </div>

        {/* Required Key Warning Prompt */}
        {!hasKey && (
          <div className="qwen-key-missing" style={{ marginTop: "1rem", display: "flex", alignItems: "center", gap: "8px", padding: "10px 14px", borderRadius: "6px", background: "#fff6e6", border: "1px solid #ffd480", color: "#8a5700", fontSize: "12.5px", fontWeight: "600" }}>
            <AlertTriangle size={16} color="#d9822b" />
            <span>Enter your Qwen API key to activate the agent</span>
          </div>
        )}

        {/* Optional Target / Filter Fields */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "1rem", marginTop: "1rem", paddingTop: "1rem", borderTop: "1px dashed var(--line)" }}>
          <div>
            <label style={{ display: "block", fontSize: "11px", fontWeight: "600", textTransform: "uppercase", color: "var(--muted)", marginBottom: "4px" }}>
              Target Record ID (Optional)
            </label>
            <input
              type="text"
              value={targetId}
              onChange={(e) => setTargetId(e.target.value)}
              placeholder="e.g. 1 or leave empty for all records"
              className="v-input"
              style={{
                width: "100%",
                padding: "8px 12px",
                borderRadius: "6px",
                border: "1px solid var(--line)",
                background: "#ffffff",
                fontSize: "12px"
              }}
            />
          </div>
          <div>
            <label style={{ display: "block", fontSize: "11px", fontWeight: "600", textTransform: "uppercase", color: "var(--muted)", marginBottom: "4px" }}>
              Inquiry / Focus Query (Optional)
            </label>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g. Audit denomination tampering"
              className="v-input"
              style={{
                width: "100%",
                padding: "8px 12px",
                borderRadius: "6px",
                border: "1px solid var(--line)",
                background: "#ffffff",
                fontSize: "12px"
              }}
            />
          </div>
        </div>
      </div>

      {/* Test Key Feedback State */}
      {testResult && (
        <div
          className={`verdict ${testResult.success ? "ok" : "bad"}`}
          style={{ marginTop: "1.25rem", width: "100%", animation: "none" }}
        >
          <span className="v-icon">
            {testResult.success ? <CheckCircle size={22} /> : <AlertTriangle size={22} />}
          </span>
          <div>
            <strong>{testResult.success ? "KEY VALIDATION SUCCESSFUL" : "KEY VALIDATION FAILED"}</strong>
            <span>{testResult.message || testResult.error}</span>
            {testResult.details && (
              <div style={{ marginTop: "6px", fontSize: "11.5px", color: "var(--muted)", display: "flex", gap: "14px", flexWrap: "wrap" }}>
                {testResult.details.label && <span>Key: <b>{testResult.details.label}</b></span>}
                {testResult.details.usage && <span>Usage: <b>{testResult.details.usage}</b></span>}
                {testResult.details.limit && <span>Limit: <b>{testResult.details.limit}</b></span>}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Loading State */}
      {running && (
        <div className="verdict" style={{ marginTop: "1.5rem", width: "100%", borderColor: "var(--saffron)" }}>
          <span className="v-icon" style={{ borderColor: "var(--saffron)", color: "var(--saffron)" }}>
            <RefreshCw size={22} className="refreshed" />
          </span>
          <div>
            <strong style={{ color: "var(--saffron)" }}>QWEN INVESTIGATION AGENT EXECUTING</strong>
            <span>Running multi-turn forensic tools against SQLite, SHA-256 hashes, and Monad RPC...</span>
          </div>
        </div>
      )}

      {/* Error State */}
      {runError && (
        <div className="verdict bad" style={{ marginTop: "1.5rem", width: "100%" }}>
          <span className="v-icon"><XCircle size={22} /></span>
          <div>
            <strong>INVESTIGATION ERROR {runError.code ? `(${runError.code})` : ""}</strong>
            <span>{runError.message}</span>
            {runError.diagnostics && (
              <div style={{ marginTop: "6px", fontSize: "11px", color: "var(--muted)" }}>
                Status: {runError.diagnostics.status || "N/A"} · Content-Type: {runError.diagnostics.contentType || "N/A"}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Success / Report Display */}
      {report && (
        <div className="qwen-report" style={{ marginTop: "2rem", background: "#fffdf6", borderRadius: "8px", border: "1px solid var(--line)", padding: "1.8rem" }}>
          {/* Header Row: Status, Confidence, Model Metadata */}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "1rem", marginBottom: "1.5rem", paddingBottom: "1.2rem", borderBottom: "1px solid var(--line)" }}>
            <div>
              <span style={{ fontSize: "11px", textTransform: "uppercase", letterSpacing: "0.12em", color: "var(--muted)", display: "block" }}>
                Agent Finding Verdict
              </span>
              <div style={{ display: "flex", alignItems: "center", gap: "10px", marginTop: "4px" }}>
                <span
                  className="pill"
                  style={{
                    background:
                      report.status === "VERIFIED" ? "rgba(31, 122, 90, 0.12)" :
                        report.status === "TAMPERED" ? "rgba(179, 38, 30, 0.12)" :
                          report.status === "HASH_MISMATCH" ? "rgba(224, 138, 30, 0.12)" : "rgba(124, 111, 99, 0.12)",
                    color:
                      report.status === "VERIFIED" ? "var(--emerald)" :
                        report.status === "TAMPERED" ? "var(--red)" :
                          report.status === "HASH_MISMATCH" ? "var(--saffron)" : "var(--muted)",
                    borderColor: "currentColor",
                    fontWeight: "800",
                    fontSize: "13px"
                  }}
                >
                  {report.status}
                </span>
                <span className="pill" style={{ fontSize: "12px" }}>
                  Confidence: <b>{report.confidence || "MEDIUM"}</b>
                </span>
              </div>
            </div>

            <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
              {runResult.model && (
                <span className="pill" style={{ fontSize: "11px", fontFamily: '"JetBrains Mono", monospace' }}>
                  {runResult.model}
                </span>
              )}
              <span className="pill" style={{ fontSize: "11px" }}>
                {runResult.iterations || 1} Turn{(runResult.iterations || 1) === 1 ? "" : "s"} · {runResult.toolsInvoked || 0} Tools
              </span>
              {runResult.usage?.totalTokens && (
                <span className="pill" style={{ fontSize: "11px" }}>
                  {runResult.usage.totalTokens} Tokens
                </span>
              )}
            </div>
          </div>

          {/* Executive Summary */}
          <div style={{ marginBottom: "1.5rem" }}>
            <h3 style={{ margin: "0 0 8px", fontSize: "16px", color: "var(--maroon)", fontFamily: "var(--serif)", fontWeight: 700 }}>
              Executive Summary
            </h3>
            <p style={{ margin: 0, fontSize: "14px", lineHeight: "1.7", color: "var(--ink)" }}>
              {report.summary}
            </p>
          </div>

          {/* Key Findings */}
          {Array.isArray(report.keyFindings) && report.keyFindings.length > 0 && (
            <div style={{ marginBottom: "1.5rem" }}>
              <h3 style={{ margin: "0 0 10px", fontSize: "15px", color: "var(--maroon)", fontFamily: "var(--serif)", fontWeight: 700 }}>
                Forensic Findings ({report.keyFindings.length})
              </h3>
              <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
                {report.keyFindings.map((kf, idx) => (
                  <div
                    key={idx}
                    style={{
                      padding: "12px 16px",
                      borderRadius: "6px",
                      background: "#ffffff",
                      border: "1px solid var(--line)",
                      borderLeft: `4px solid ${kf.status === "VERIFIED" ? "var(--emerald)" :
                          kf.status === "TAMPERED" ? "var(--red)" :
                            kf.status === "HASH_MISMATCH" ? "var(--saffron)" : "var(--muted)"
                        }`
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" }}>
                      <span style={{ fontSize: "11px", fontWeight: "700", textTransform: "uppercase", color: "var(--muted)" }}>
                        Finding #{idx + 1}
                      </span>
                      <span style={{ fontSize: "11px", fontWeight: "700", color: kf.status === "TAMPERED" ? "var(--red)" : "var(--emerald)" }}>
                        {kf.status} ({kf.confidence || "MEDIUM"} CONFIDENCE)
                      </span>
                    </div>
                    <p style={{ margin: 0, fontSize: "13.5px", lineHeight: "1.6", color: "var(--ink)" }}>
                      {kf.finding}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Risks / Anomaly Vectors */}
          {Array.isArray(report.risks) && report.risks.length > 0 && (
            <div style={{ marginBottom: "1.2rem", padding: "12px 16px", borderRadius: "6px", background: "#fff5f1", border: "1px solid rgba(179, 38, 30, 0.25)" }}>
              <strong style={{ display: "block", fontSize: "12px", textTransform: "uppercase", color: "var(--red)", marginBottom: "4px" }}>
                <AlertTriangle size={13} style={{ verticalAlign: "middle", marginRight: "4px" }} />
                Identified Risks & Anomalies
              </strong>
              <ul style={{ margin: 0, paddingLeft: "1.2rem", fontSize: "13px", color: "#6e2920" }}>
                {report.risks.map((risk, i) => (
                  <li key={i}>{risk}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Recommended Actions */}
          {Array.isArray(report.recommendedActions) && report.recommendedActions.length > 0 && (
            <div style={{ padding: "12px 16px", borderRadius: "6px", background: "#f2f9f5", border: "1px solid rgba(31, 122, 90, 0.25)" }}>
              <strong style={{ display: "block", fontSize: "12px", textTransform: "uppercase", color: "var(--emerald)", marginBottom: "4px" }}>
                <ShieldCheck size={13} style={{ verticalAlign: "middle", marginRight: "4px" }} />
                Recommended Actions
              </strong>
              <ul style={{ margin: 0, paddingLeft: "1.2rem", fontSize: "13px", color: "#164d39" }}>
                {report.recommendedActions.map((act, i) => (
                  <li key={i}>{act}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Audit trail / Tools invoked */}
          {Array.isArray(runResult.audit?.toolsUsed) && runResult.audit.toolsUsed.length > 0 && (
            <div style={{ marginTop: "1.2rem", paddingTop: "1rem", borderTop: "1px dashed var(--line)", display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap", fontSize: "11px", color: "var(--muted)" }}>
              <span>Tools invoked:</span>
              {runResult.audit.toolsUsed.map((tool, idx) => (
                <code key={idx} style={{ padding: "2px 6px", background: "#f0ebe1", borderRadius: "4px", fontSize: "11px" }}>
                  {tool}
                </code>
              ))}
            </div>
          )}
        </div>
      )}
    </section>
  );
}

const STEPS = [
  [Cpu, "Model scans", "ML screens each record for anomalies"],
  [Hash, "SHA-256", "The record becomes a fixed fingerprint"],
  [Link2, "Monad", "The fingerprint is anchored on-chain"],
  [Radar, "Re-check", "We recompute it every 3 seconds"],
];

export default function App() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [updated, setUpdated] = useState(null);
  const [fresh, setFresh] = useState(() => new Set());
  const [selectedProof, setSelectedProof] = useState("");
  const seen = useRef(null);

  const load = async () => {
    try {
      setError("");
      const res = await fetch(`${API_URL}/all-records`);
      if (!res.ok) throw new Error("Failed to load records");
      const json = await res.json();
      if (!json.success) throw new Error(json.error || "API error");
      const keys = (json.records || []).map((r, i) => String(recNo(r, i)));
      if (seen.current) {
        const added = keys.filter((k) => !seen.current.has(k));
        if (added.length) {
          setFresh(new Set(added));
          setTimeout(() => setFresh(new Set()), 2200);
        }
      }
      seen.current = new Set(keys);
      setData(json);
      setUpdated(new Date());
    } catch (e) {
      console.error(e);
      setError("Can't reach the Smart Donation System API. Start the backend on port 3000, then retry.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    const id = setInterval(load, 3000);
    return () => clearInterval(id);
  }, []);

  const records = data?.records || [];
  const total = Number(data?.totalRecords || 0);
  const tampered = records.filter((r) => r.status === "TAMPERED").length;
  const verified = Math.max(total - tampered, 0);
  const pct = total > 0 ? (verified / total) * 100 : 100;

  if (loading || error)
    return (
      <div className="app center">
        <div className={`seal ${error ? "bad" : "spin"}`}>
          {error ? <XCircle size={40} /> : <ShieldCheck size={40} />}
        </div>
        <h2>{error ? "Connection lost" : "Checking the ledger"}</h2>
        <p className="muted">{error || "Comparing every record with its on-chain proof…"}</p>
        {error && <button className="btn" onClick={load}><RefreshCw size={15} /> Retry</button>}
      </div>
    );

  const chain = records.slice(-12);
  const offset = records.length - chain.length;
  const firstBroken = chain.findIndex((r) => r.status === "TAMPERED");
  const badRecords = records.map((r, i) => [r, i]).filter(([r]) => r.status === "TAMPERED").reverse();

  return (
    <div className="app">
      <div className="torana" aria-hidden="true" />

      <header className="nav">
        <div className="brand">
          <img
            src="/images/daandrishti-logo.png"
            alt="Smart Donation System"
            className="brand-logo"
            width="44"
            height="44"
          />
          <div>
            <b>Smart Donation System</b>
            <span lang="hi">दानदृष्टि</span>
          </div>
          <em className="muted">donation provenance</em>
        </div>
        <div className="pill"><i className="dot" /> Monad testnet · live</div>
      </header>

      <main>
        <section className="hero">
          <div className="hero-copy">
            <p className="label"><i /> SMART DONATION SYSTEM / DONATION PROVENANCE</p>
            <h1>
              Every rupee given<br />
              leaves a <em>proof</em><br />
              that can't be edited.
            </h1>
            <p className="lede">
              Each donation record is fingerprinted with SHA-256 and anchored on Monad.
              If a single digit changes later, the fingerprint no longer matches and we flag it here.
            </p>

            <div className={`verdict ${tampered ? "bad" : "ok"}`} role="status" aria-live="polite">
              <span className="v-icon">
                {tampered ? <ShieldAlert size={24} /> : <ShieldCheck size={24} />}
                <i className="v-ping" />
              </span>
              <div>
                <strong>{tampered ? "TAMPERING DETECTED" : "SYSTEM SECURE"}</strong>
                <span>
                  {tampered
                    ? `${tampered} record${tampered === 1 ? "" : "s"} no longer match their proof. Review them now.`
                    : "All records match their on-chain proof."}
                </span>
              </div>
            </div>
          </div>
          <div className="hero-art">
            <Gopuram />
            {Array.from({ length: 9 }).map((_, i) => (
              <i key={i} className="mote" style={{ left: `${10 + i * 10}%`, animationDelay: `${i * -2.1}s`, animationDuration: `${11 + (i % 4) * 3}s` }} />
            ))}
          </div>
        </section>

        <section className="ledger" aria-label="Latest records on the chain">
          <div className="sec-head row">
            <div>
              <h2>Latest blocks</h2>
              <span className="muted">Live provenance chain</span>
            </div>
            <span className="muted refreshed">
              <RefreshCw size={12} /> {updated && `Refreshed ${updated.toLocaleTimeString()} · every 3s`}
            </span>
          </div>
          <div className="chain">
            {chain.length === 0 && <p className="muted">No records yet. New donations appear here as they are registered.</p>}
            {chain.map((r, i) => {
              const bad = r.status === "TAMPERED";
              const idx = offset + i;
              const key = String(recNo(r, idx));
              const ts = when(r);
              return (
                <div className="link-group" key={key}>
                  {i > 0 && (
                    <span className={`wire ${i >= firstBroken && firstBroken > -1 && (bad || chain[i - 1].status === "TAMPERED") ? "cut" : ""}`} />
                  )}
                  <div className={`slip ${bad ? "bad" : "ok"} ${fresh.has(key) ? "fresh" : ""}`} style={{ "--d": `${i * 60}ms` }}>
                    <article className="block">
                      <small>Block</small>
                      <b>#{recNo(r, idx)}</b>
                      <code>{shortHash(r, idx)}</code>
                      {ts && <time>{ts}</time>}
                      <span className="stamp">
                        {bad ? <AlertTriangle size={12} /> : <ShieldCheck size={12} />}
                        {bad ? "Tampered" : "Verified"}
                      </span>
                    </article>
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        {badRecords.length > 0 && (
          <section className="tamper" aria-label="Tampered records">
            <div className="sec-head">
              <h2>Original record ≠ current record</h2>
              <span className="muted">Each proof was anchored on Monad before the change</span>
            </div>
            <div className="tgrid">
              {badRecords.slice(0, 4).map(([r, i]) => <TamperCard key={recNo(r, i)} r={r} i={i} />)}
            </div>
            {badRecords.length > 4 && <p className="muted more">+ {badRecords.length - 4} more tampered records</p>}
          </section>
        )}

        <Insights total={total} verified={verified} tampered={tampered} pct={pct} />

        <ProofAnalytics />
        <ProofVerifier selectedHash={selectedProof} />
        <ProofExplorer onVerifyProof={(h) => setSelectedProof(h)} />
        <QwenAgent />

        <section className="flow" aria-label="How verification works">
          <div className="track"><i className="particle" /></div>
          {STEPS.map(([Icon, t, d], i) => (
            <div className="step" key={t} style={{ "--i": i }}>
              <span className="s-icon" aria-hidden="true"><Icon size={22} /></span>
              <div><b>{t}</b><span className="s-desc">{d}</span></div>
            </div>
          ))}
        </section>
      </main>

      <footer className="foot muted">Smart Donation System · Records are checked against Monad testnet proofs</footer>
    </div>
  );
}
