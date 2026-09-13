import { useCallback, useEffect, useRef, useState } from "react";
import api, { unwrap } from "../api/client";
import { StageShell } from "../components/DiveBits";

const IDLE = { state: "idle", progress: 0, note: "○ waiting" };

function confColor(c) {
  if (c >= 0.66) return "#22c55e";
  if (c >= 0.33) return "#f59e0b";
  return "#ef4444";
}

function TrackCanvas({ bundle, frame, tilt3d }) {
  const ref = useRef(null);
  const live = useRef({ bundle, frame, tilt3d });
  live.current = { bundle, frame, tilt3d };

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas.getContext("2d");
    let raf = 0;
    const render = () => {
      const { bundle: b, frame: f, tilt3d: t3 } = live.current;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      ctx.clearRect(0, 0, w, h);
      if (!b) {
        raf = requestAnimationFrame(render);
        return;
      }
      const bx = b.bounds?.x_max || 256;
      const by = b.bounds?.y_max || 256;
      const bz = b.bounds?.z_max || 64;
      const px = (x) => 20 + (x / bx) * (w - 40);
      const py = (y, z) => {
        const t = t3 ? 0.5 : 0;
        return 20 + ((y - (z / bz) * by * 0.35 * t) / by) * (h - 40);
      };
      const at = new Map();
      for (const c of b.cells || []) {
        if (c.t === f || c.t === f + 1) at.set(c.id, c);
      }
      for (const m of b.matches || []) {
        if (m.t_src !== f) continue;
        const s = at.get(m.src);
        const d = at.get(m.dst);
        if (!s || !d) continue;
        ctx.strokeStyle = confColor(m.confidence);
        ctx.globalAlpha = 0.75;
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.moveTo(px(s.x), py(s.y, s.z));
        ctx.lineTo(px(d.x), py(d.y, d.z));
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
      for (const c of at.values()) {
        const cur = c.t === f;
        ctx.fillStyle = cur ? "#22c55e" : "rgba(148,163,184,0.6)";
        ctx.beginPath();
        ctx.arc(px(c.x), py(c.y, c.z), cur ? 4 : 2.6, 0, 6.2832);
        ctx.fill();
      }
      raf = requestAnimationFrame(render);
    };
    raf = requestAnimationFrame(render);
    return () => cancelAnimationFrame(raf);
  }, []);

  return <canvas ref={ref} className="tracklive" role="img" aria-label="Tracking matches" />;
}

export default function Track({ ctx }) {
  const { movieId, register, report } = ctx;
  const [st, setSt] = useState(IDLE);
  const [fit, setFit] = useState(null);
  const [run, setRun] = useState(null);
  const [bundle, setBundle] = useState(null);
  const [frame, setFrame] = useState(0);
  const [tilt3d, setTilt3d] = useState(false);

  const runAll = useCallback(async () => {
    if (!movieId) throw new Error("Pick a specimen first (Stage 00).");
    setSt({ state: "run", progress: 0, note: "◉ fitting group weights…" });
    report("s07", "run");
    try {
      const fw = unwrap(await api.fitTrackingWeights([movieId]));
      setFit(fw);
      setSt({ state: "run", progress: 0.35, note: "◉ Hungarian matching…" });
      const tr = unwrap(await api.runTracking(movieId));
      setRun(tr);
      setSt({ state: "run", progress: 0.7, note: "◉ building theatre…" });
      const th = unwrap(await api.theatre(movieId, "fitted"));
      setBundle(th);
      if (th.frames?.length) setFrame(th.frames[0]);
      setSt({
        state: "done",
        progress: 1,
        note: `✓ ${tr.n_associations} links · conf ${tr.mean_confidence}`,
      });
      report("s07", "done");
      return tr;
    } catch (e) {
      setSt({ state: "err", progress: 0, note: `✗ ${e.message}` });
      report("s07", "err");
      throw e;
    }
  }, [movieId, report]);

  useEffect(() => {
    register("s07", runAll);
  }, [register, runAll]);

  useEffect(() => {
    setFit(null);
    setRun(null);
    setBundle(null);
    setSt(IDLE);
    report("s07", "");
  }, [movieId, report]);

  const frames = bundle?.frames || [];
  const frameMatches = (bundle?.matches || []).filter((m) => m.t_src === frame);
  const confs = (bundle?.matches || []).map((m) => m.confidence);
  const bins = new Array(10).fill(0);
  confs.forEach((c) => {
    bins[Math.min(9, Math.floor(c * 10))]++;
  });
  const binMax = Math.max(...bins, 1);
  const low5 = [...(bundle?.matches || [])].sort((a, b) => a.confidence - b.confidence).slice(0, 5);
  const wEntries = fit ? Object.entries(fit.weights || {}) : [];
  const wMax = Math.max(...wEntries.map(([, v]) => Math.abs(v)), 1e-9);
  const mc = run?.mean_confidence ?? 0;
  const ang = Math.PI * (1 - mc);
  const nx = 100 + 62 * Math.cos(ang);
  const ny = 105 - 62 * Math.sin(ang);
  const preview = run?.preview || [];

  return (
    <StageShell
      id="s07"
      num="07"
      kicker="STAGE 07 · THE VERDICT"
      name="Track"
      desc={
        <>
          Embeddings + graph + trained weights → <b>frame-to-frame matches</b>, each with an honest
          confidence. Did it actually work? Count the links.
        </>
      }
      runLabel="MATCH FRAMES"
      status={st}
      onRun={() => runAll().catch(() => {})}
      say="“Hundreds of links, each scored — and we show the uncertain ones too. That's honesty.”"
    >
      {!run && <p className="muted">{movieId ? "Press MATCH FRAMES." : "Pick a specimen in Stage 00 first."}</p>}
      {run && (
        <>
          <div className="grid" style={{ gridTemplateColumns: "2fr 1fr" }}>
            <div className="panel">
              <div className="seg">
                <button type="button" className={`btn sm${!tilt3d ? " on" : ""}`} onClick={() => setTilt3d(false)}>2D</button>
                <button type="button" className={`btn sm${tilt3d ? " on" : ""}`} onClick={() => setTilt3d(true)}>3D</button>
                <span style={{ flex: 1 }} />
                <span className="status">T = <b>{frame}</b> · {frameMatches.length} links → T{frame + 1}</span>
              </div>
              <TrackCanvas bundle={bundle} frame={frame} tilt3d={tilt3d} />
              {frames.length > 1 && (
                <input
                  type="range"
                  min={frames[0]}
                  max={frames[frames.length - 1]}
                  value={frame}
                  onChange={(e) => setFrame(Number(e.target.value))}
                  style={{ width: "100%", accentColor: "var(--acc)" }}
                  aria-label="Frame scrubber"
                />
              )}
              <div className="hist">
                {bins.map((v, i) => (
                  <i key={i} title={`${(i / 10).toFixed(1)}–${((i + 1) / 10).toFixed(1)}: ${v}`} style={{ height: `${Math.max(3, (v / binMax) * 100)}%` }} />
                ))}
              </div>
              <p className="sub">Confidence histogram · {confs.length} links · green ≥0.66 · amber ≥0.33 · red below.</p>
            </div>
            <div className="panel gauge">
              <h3>MEAN CONFIDENCE</h3>
              <svg viewBox="0 0 200 120" width="180">
                <path d="M20,105 A80,80 0 0 1 180,105" fill="none" stroke="var(--chip)" strokeWidth="14" strokeLinecap="round" />
                <line x1="100" y1="105" x2={nx} y2={ny} stroke="var(--acc)" strokeWidth="4" strokeLinecap="round" />
                <circle cx="100" cy="105" r="7" fill="var(--acc)" />
              </svg>
              <div className="bignum">{mc.toFixed(3)}</div>
              <div className="meter"><span>matched links</span><b>{run.n_associations}</b></div>
              <div className="meter"><span>frame pairs</span><b>{run.n_frame_pairs}</b></div>
              <div className="meter"><span>gated matches</span><b>{run.n_gated_matches}</b></div>
              <div className="meter"><span>weight source</span><b className="tiny">{run.group_weight_source}</b></div>
            </div>
          </div>
          <div className="grid g2" style={{ marginTop: 18 }}>
            <div className="panel">
              <h3>GROUP WEIGHTS · FITTED</h3>
              <p className="sub">loss {fit?.final_loss} · {fit?.n_positives}/{fit?.n_candidates} positives · {fit?.elapsed_sec}s.</p>
              {wEntries.map(([g, w]) => (
                <div className="arow" key={g}>
                  <span>{g}</span>
                  <div className="afill"><i style={{ width: `${(Math.abs(w) / wMax) * 100}%` }} /></div>
                  <span>{w.toFixed(4)}</span>
                </div>
              ))}
            </div>
            <div className="panel">
              <h3>LOWEST-CONFIDENCE LINKS · HONEST FLAGS</h3>
              <p className="sub">The 5 links the model trusts least — review advised.</p>
              {low5.map((m, i) => (
                <div className="flag" key={i}>
                  <span className="fdot" style={{ background: confColor(m.confidence) }} />
                  #{m.src}→#{m.dst} · T{m.t_src}→T{m.t_dst} · conf {m.confidence.toFixed(3)} · cost {m.cost.toFixed(3)}
                </div>
              ))}
              <h3 style={{ marginTop: 12 }}>RUN PREVIEW · FIRST 8</h3>
              {preview.slice(0, 4).map((m, i) => (
                <div className="flag" key={`p${i}`}>
                  <span className="fdot" style={{ background: "var(--acc)" }} />
                  #{m.src_node_id ?? m.src}→#{m.dst_node_id ?? m.dst} · conf {(m.confidence ?? 0).toFixed(3)}
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </StageShell>
  );
}
