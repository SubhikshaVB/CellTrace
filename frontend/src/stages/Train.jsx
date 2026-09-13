import { useCallback, useEffect, useState } from "react";
import api, { unwrap } from "../api/client";
import { StageShell } from "../components/DiveBits";

const IDLE = { state: "idle", progress: 0, note: "○ waiting" };

function curvePath(vals, w, h, pad) {
  if (!vals.length) return "";
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  const rng = max - min || 1;
  return vals
    .map((v, i) => {
      const x = pad + (i / Math.max(vals.length - 1, 1)) * (w - pad * 2);
      const y = h - pad - ((v - min) / rng) * (h - pad * 2);
      return `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

export default function Train({ ctx }) {
  const { movieId, register, report } = ctx;
  const [st, setSt] = useState(IDLE);
  const [epochs, setEpochs] = useState(5);
  const [res, setRes] = useState(null);
  const [nodeId, setNodeId] = useState("");
  const [att, setAtt] = useState(null);
  const [anote, setAnote] = useState("");
  const [boost, setBoost] = useState(3);

  const loadAtt = useCallback(async (mid, nid) => {
    setAnote("reading trained attention…");
    try {
      const a = unwrap(await api.module3Attention(mid, nid, 8));
      setAtt(a);
      setAnote("");
    } catch (e) {
      setAnote(e.message);
      setAtt(null);
    }
  }, []);

  const run = useCallback(async () => {
    if (!movieId) throw new Error("Pick a specimen first (Stage 00).");
    setSt({ state: "run", progress: 0, note: `◉ training ${epochs} epochs… (real compute)` });
    report("s06", "run");
    try {
      const r = unwrap(await api.runModule3(movieId, epochs));
      setRes(r);
      try {
        const g = unwrap(await api.module2GraphPreview(movieId));
        const first = g?.spatial?.nodes?.[0]?.node_id;
        if (first !== undefined && first !== null) {
          setNodeId(first);
          await loadAtt(movieId, first);
        }
      } catch (e) {
        setAnote(`attention default unavailable: ${e.message}`);
      }
      setSt({
        state: "done",
        progress: 1,
        note: `✓ ${r.mode} · final ${Number(r.final_train_loss ?? NaN).toFixed?.(4) ?? "—"} · ${r.elapsed_sec}s`,
      });
      report("s06", "done");
      return r;
    } catch (e) {
      setSt({ state: "err", progress: 0, note: `✗ ${e.message}` });
      report("s06", "err");
      throw e;
    }
  }, [movieId, epochs, report, loadAtt]);

  useEffect(() => {
    register("s06", run);
  }, [register, run]);

  useEffect(() => {
    setRes(null);
    setAtt(null);
    setSt(IDLE);
    report("s06", "");
  }, [movieId, report]);

  const hist = res?.training_history || [];
  const trains = hist.map((h, i) => Number(h.train_loss ?? h.loss ?? NaN)).filter((v) => Number.isFinite(v));
  const vals = hist.map((h) => Number(h.val_loss ?? NaN)).filter((v) => Number.isFinite(v));
  const nbrs = att?.neighbors || [];
  const heads = att?.heads || 0;
  const headAvg = [];
  for (let h = 0; h < heads; h++) {
    const vs = nbrs.map((n) => n.alpha_heads?.[h] ?? 0);
    headAvg.push(vs.length ? vs.reduce((a, b) => a + b, 0) / vs.length : 0);
  }
  const uniform = nbrs.length ? 1 / nbrs.length : 0;
  const shown = (a) => Math.max(0, Math.min(1, uniform + (a - uniform) * boost));

  return (
    <StageShell
      id="s06"
      num="06"
      kicker="STAGE 06 · TEACHING THE MACHINE"
      name="Train"
      desc={
        <>
          The GAT trains <b>live, in front of you</b> — contrastive loss falls epoch by epoch while
          attention heads learn which neighbors to trust.
        </>
      }
      runLabel={`TRAIN · ${epochs} EPOCHS`}
      status={st}
      onRun={() => run().catch(() => {})}
      say="“Watch the loss fall live — the network is learning, right now, on this embryo.”"
    >
      <label className="filelabel inline">
        Epochs (1–500 · real compute, CPU minutes at 50)
        <input
          className="textin num"
          type="number"
          min={1}
          max={500}
          value={epochs}
          onChange={(e) => setEpochs(Math.max(1, Math.min(500, Number(e.target.value) || 1)))}
        />
      </label>
      {!res && <p className="muted">{movieId ? "Press TRAIN." : "Pick a specimen in Stage 00 first."}</p>}
      {res && (
        <>
          <div className="grid g3">
            <div className="panel curvebox">
              <h3>CONTRASTIVE LOSS · REAL CURVE</h3>
              <p className="sub">mode {res.mode} · {hist.length} epochs logged.</p>
              <svg viewBox="0 0 300 190">
                <line x1="34" y1="10" x2="34" y2="160" stroke="var(--line)" strokeWidth="2" />
                <line x1="34" y1="160" x2="292" y2="160" stroke="var(--line)" strokeWidth="2" />
                {trains.length > 1 && (
                  <path d={curvePath(trains, 300, 190, 40)} fill="none" stroke="var(--acc)" strokeWidth="3" />
                )}
                {vals.length > 1 && (
                  <path d={curvePath(vals, 300, 190, 40)} fill="none" stroke="var(--acc2)" strokeWidth="2" strokeDasharray="5 4" />
                )}
                <text x="40" y="178" fill="var(--mut)" fontSize="10" fontFamily="monospace">
                  {trains.length ? `${trains[0].toFixed(3)} → ${trains[trains.length - 1].toFixed(3)}` : "no history"}
                </text>
              </svg>
              <div className="chips">
                <div className="chip">{Number(res.final_train_loss ?? NaN).toFixed?.(4) ?? "—"}<small>FINAL TRAIN</small></div>
                <div className="chip">{Number(res.best_val_loss ?? NaN).toFixed?.(4) ?? "—"}<small>BEST VAL</small></div>
                <div className="chip">{res.elapsed_sec}s<small>ELAPSED</small></div>
              </div>
            </div>
            <div className="panel">
              <h3>TRAINING FACTS</h3>
              <p className="sub">Pairs + graph that were actually used.</p>
              <div className="meter"><span>cell nodes</span><b>{res.n_cell_nodes ?? "—"}</b></div>
              <div className="meter"><span>positive pairs</span><b>{res.pairs?.n_positive ?? "—"}</b></div>
              <div className="meter"><span>negative pairs</span><b>{res.pairs?.n_negative ?? res.pairs?.n_neg ?? "—"}</b></div>
              <div className="meter"><span>graph edges</span><b>{res.graph?.n_edges ?? res.graph?.edges ?? "—"}</b></div>
              <div className="meter"><span>embed norm</span><b>{res.mean_embedding_norm ?? "—"}</b></div>
              <p className="sub" style={{ marginTop: 10 }}>{res.val_scope || ""}</p>
            </div>
            <div className="panel">
              <h3>ATTENTION LENS · TRAINED α</h3>
              <p className="sub">Exact layer-1 weights of the saved GAT · frame {att?.frame ?? "—"}.</p>
              <label className="filelabel inline">
                Focus
                <input
                  className="textin num"
                  type="number"
                  value={nodeId}
                  onChange={(e) => setNodeId(e.target.value === "" ? "" : Number(e.target.value))}
                />
                <button type="button" className="btn sm" onClick={() => nodeId !== "" && loadAtt(movieId, Number(nodeId))}>
                  READ α
                </button>
              </label>
              {anote && <p className="muted">{anote}</p>}
              {nbrs.length > 0 && (
                <div className="lens">
                  <div className="orb q" style={{ left: "50%", top: "50%" }}>Q</div>
                  {nbrs.slice(0, 6).map((n, i) => {
                    const a = ((i / 6) * 360 - 90) * (Math.PI / 180);
                    const size = 26 + shown(n.alpha_mean) * 34;
                    return (
                      <div
                        key={n.node_id}
                        className="orb"
                        title={`#${n.node_id} α=${n.alpha_mean}`}
                        style={{
                          left: `${50 + 34 * Math.cos(a)}%`,
                          top: `${50 + 34 * Math.sin(a)}%`,
                          width: size,
                          height: size,
                        }}
                      >
                        {n.is_self ? "me" : `n${i + 1}`}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
          {nbrs.length > 0 && (
            <div className="grid g2" style={{ marginTop: 18 }}>
              <div className="panel">
                <h3>NEIGHBOR α · RAW VALUES</h3>
                <p className="sub">Mean over {heads} heads · sorted · self included.</p>
                {nbrs.map((n) => (
                  <div className="arow" key={n.node_id}>
                    <span>#{n.node_id}{n.is_self ? " (me)" : ""}</span>
                    <div className="afill"><i style={{ width: `${Math.min(100, (n.alpha_mean / Math.max(...nbrs.map((x) => x.alpha_mean), 1e-9)) * 100)}%` }} /></div>
                    <span>{n.alpha_mean.toFixed(4)}</span>
                  </div>
                ))}
              </div>
              <div className="panel">
                <h3>HEAD AVERAGES · DISPLAY BOOST ×{boost}</h3>
                <p className="sub">
                  ⚠ Boost is <b>display-only</b>: shown = uniform + (α − uniform) × {boost}. Raw α unchanged.
                </p>
                <input
                  type="range"
                  min={1}
                  max={8}
                  value={boost}
                  onChange={(e) => setBoost(Number(e.target.value))}
                  style={{ width: "100%", accentColor: "var(--acc2)" }}
                  aria-label="Display-only attention contrast boost"
                />
                {headAvg.map((a, h) => (
                  <div className="arow" key={h}>
                    <span>head {h + 1}</span>
                    <div className="afill"><i style={{ width: `${shown(a) * 100}%` }} /></div>
                    <span>~{shown(a).toFixed(3)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </StageShell>
  );
}
