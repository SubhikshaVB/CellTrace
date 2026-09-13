import { useEffect, useMemo, useRef, useState } from "react";
import api, { unwrap } from "../api/client";
import { Say } from "./DiveBits";

function Observatory({ cloud }) {
  const canvasRef = useRef(null);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [frame, setFrame] = useState(cloud.t_min);
  const [trail, setTrail] = useState(6);
  const [follow, setFollow] = useState(null);
  const live = useRef({ frame: cloud.t_min, trail: 6, follow: null, ang: 0.7 });
  live.current.frame = frame;
  live.current.trail = trail;
  live.current.follow = follow;

  const model = useMemo(() => {
    const pts = cloud.points;
    const byId = new Map(pts.map((p) => [p.id, p]));
    const parents = new Map();
    (cloud.links || []).forEach(([s, d]) => {
      if (!parents.has(d)) parents.set(d, s);
    });
    const rootOf = (id) => {
      let c = id;
      let g = 0;
      while (parents.has(c) && g++ < 400) c = parents.get(c);
      return c;
    };
    const hue = new Map();
    pts.forEach((p) => {
      const r = rootOf(p.id);
      if (!hue.has(r)) hue.set(r, (Math.abs(r) * 47 + 13) % 360);
    });
    const b = { minX: 1e9, maxX: -1e9, minY: 1e9, maxY: -1e9, minZ: 1e9, maxZ: -1e9 };
    pts.forEach((p) => {
      if (p.x < b.minX) b.minX = p.x;
      if (p.x > b.maxX) b.maxX = p.x;
      if (p.y < b.minY) b.minY = p.y;
      if (p.y > b.maxY) b.maxY = p.y;
      if (p.z < b.minZ) b.minZ = p.z;
      if (p.z > b.maxZ) b.maxZ = p.z;
    });
    const byFrame = new Map();
    pts.forEach((p) => {
      if (!byFrame.has(p.t)) byFrame.set(p.t, []);
      byFrame.get(p.t).push(p);
    });
    return { byId, parents, hue, byFrame, b };
  }, [cloud]);
  const modelRef = useRef(model);
  modelRef.current = model;
  const projRef = useRef([]);

  useEffect(() => {
    setFrame(cloud.t_min);
    setFollow(null);
  }, [cloud]);

  useEffect(() => {
    if (!playing) return undefined;
    const h = setInterval(() => {
      setFrame((f) => (f >= cloud.t_max ? cloud.t_min : f + 1));
    }, Math.max(60, 420 / speed));
    return () => clearInterval(h);
  }, [playing, speed, cloud]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    let raf = 0;
    const drag = { on: false, x: 0, off: 0 };
    const project = (p, W, H, m, a, cx, cy, s) => {
      const dx = p.x - (m.b.minX + m.b.maxX) / 2;
      const dz = (p.z - (m.b.minZ + m.b.maxZ) / 2) * 3;
      const rx = dx * Math.cos(a) - dz * Math.sin(a);
      const depth = dx * Math.sin(a) + dz * Math.cos(a);
      return {
        x: cx + rx * s,
        y: cy + (p.y - (m.b.minY + m.b.maxY) / 2) * s,
        near: Math.max(0, Math.min(1, depth / 300 + 0.5)),
      };
    };
    const render = () => {
      const m = modelRef.current;
      const L = live.current;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      if (!drag.on) L.ang += 0.0022;
      const a = L.ang + drag.off;
      const span = Math.max(m.b.maxX - m.b.minX, m.b.maxY - m.b.minY, 1);
      const s = Math.min(w, h) / (1.6 * span);
      ctx.clearRect(0, 0, w, h);
      let cx = w / 2;
      let cy = h / 2;
      const fp = L.follow !== null ? m.byId.get(L.follow) : null;
      if (fp) {
        const pr = project(fp, w, h, m, a, w / 2, h / 2, s);
        cx += w / 2 - pr.x;
        cy += h / 2 - pr.y;
      }
      const seg = (p1, p2, hue, alpha, width) => {
        const q1 = project(p1, w, h, m, a, cx, cy, s);
        const q2 = project(p2, w, h, m, a, cx, cy, s);
        ctx.strokeStyle = `hsla(${hue}, 90%, 60%, ${alpha})`;
        ctx.lineWidth = width;
        ctx.beginPath();
        ctx.moveTo(q1.x, q1.y);
        ctx.lineTo(q2.x, q2.y);
        ctx.stroke();
      };
      m.parents.forEach((srcId, dstId) => {
        const pd = m.byId.get(dstId);
        const ps = m.byId.get(srcId);
        if (!pd || !ps || pd.t > L.frame) return;
        let c = dstId;
        let g = 0;
        while (m.parents.has(c) && g++ < 400) c = m.parents.get(c);
        seg(ps, pd, m.hue.get(c) ?? 200, 0.14, 1);
      });
      const here = m.byFrame.get(L.frame) || [];
      projRef.current = [];
      for (const p of here) {
        let c = p.id;
        let g = 0;
        while (m.parents.has(c) && g++ < 400) c = m.parents.get(c);
        const hue = m.hue.get(c) ?? 200;
        const chain = [p];
        let cur = p;
        for (let k = 0; k < L.trail; k++) {
          const par = m.parents.get(cur.id);
          const pp = par !== undefined ? m.byId.get(par) : null;
          if (!pp) break;
          chain.unshift(pp);
          cur = pp;
        }
        for (let k = 0; k < chain.length - 1; k++) {
          seg(chain[k], chain[k + 1], hue, 0.25 + (0.65 * k) / Math.max(chain.length - 1, 1), 2);
        }
        const q = project(p, w, h, m, a, cx, cy, s);
        projRef.current.push({ id: p.id, x: q.x, y: q.y });
        ctx.fillStyle = `hsla(${hue}, 95%, 65%, 0.28)`;
        ctx.beginPath();
        ctx.arc(q.x, q.y, 9 + q.near * 4, 0, 6.2832);
        ctx.fill();
        ctx.fillStyle = `hsla(${hue}, 95%, 72%, 0.95)`;
        ctx.beginPath();
        ctx.arc(q.x, q.y, 3.4 + q.near * 1.6, 0, 6.2832);
        ctx.fill();
        ctx.fillStyle = "rgba(255,255,255,0.9)";
        ctx.beginPath();
        ctx.arc(q.x, q.y, 1.3, 0, 6.2832);
        ctx.fill();
      }
      if (fp) {
        const q = project(fp, w, h, m, a, cx, cy, s);
        ctx.strokeStyle = "#facc15";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(q.x, q.y, 18, 0, 6.2832);
        ctx.stroke();
      }
      raf = requestAnimationFrame(render);
    };
    const onDown = (e) => {
      drag.on = true;
      drag.x = e.clientX;
    };
    const onMove = (e) => {
      if (!drag.on) return;
      drag.off += (e.clientX - drag.x) * 0.008;
      drag.x = e.clientX;
    };
    const onUp = () => {
      drag.on = false;
    };
    canvas.addEventListener("pointerdown", onDown);
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    raf = requestAnimationFrame(render);
    return () => {
      cancelAnimationFrame(raf);
      canvas.removeEventListener("pointerdown", onDown);
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, []);

  const pick = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    let best = null;
    let bestD = 20;
    for (const q of projRef.current) {
      const d = Math.hypot(q.x - mx, q.y - my);
      if (d < bestD) {
        bestD = d;
        best = q.id;
      }
    }
    setFollow(best);
  };

  const here = model.byFrame.get(frame) || [];
  const fp = follow !== null ? model.byId.get(follow) : null;

  return (
    <div className="obs">
      <div className="obs-stage">
        <canvas ref={canvasRef} onClick={pick} className="obs-canvas" role="img" aria-label="Embryo time-lapse" />
        <div className="obs-hud">
          <div className="obs-t">
            <small>FRAME</small>
            <b>T = {frame}</b>
          </div>
          <div className="obs-counts">
            <span><b>{here.length}</b> live cells</span>
            <span><b>{model.hue.size}</b> lineages</span>
            <span><b>{model.parents.size}</b> truth links</span>
          </div>
        </div>
        {fp && (
          <div className="obs-dossier">
            <span className="eyebrow">CELL DOSSIER</span>
            <b>Cell {fp.id} · T{fp.t}</b>
            <span>x {fp.x} · y {fp.y} · z {fp.z}</span>
            <button type="button" className="btn" onClick={() => setFollow(null)}>Release</button>
          </div>
        )}
      </div>
      <div className="obs-controls">
        <button type="button" className="btn hot" onClick={() => setPlaying((p) => !p)}>
          {playing ? "❚❚ Pause" : "▶ Play"}
        </button>
        <input type="range" min={cloud.t_min} max={cloud.t_max} value={frame}
          onChange={(e) => setFrame(Number(e.target.value))} aria-label="Frame scrubber" />
        <span className="obs-speed">
          {[0.5, 1, 2, 4].map((v) => (
            <button key={v} type="button" className={speed === v ? "active" : ""} onClick={() => setSpeed(v)}>
              {v}×
            </button>
          ))}
        </span>
        <label>
          Trails {trail}
          <input type="range" min={0} max={12} value={trail} onChange={(e) => setTrail(Number(e.target.value))} />
        </label>
      </div>
    </div>
  );
}

export default function Hero({ movieId, diving, onDive }) {
  const [cloud, setCloud] = useState(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    if (!movieId) {
      setCloud(null);
      return;
    }
    let cancelled = false;
    setErr("");
    api
      .geffCloud(movieId, 3000)
      .then((p) => {
        if (!cancelled) setCloud(unwrap(p));
      })
      .catch((e) => {
        if (!cancelled) setErr(e.message);
      });
    return () => {
      cancelled = true;
    };
  }, [movieId]);

  const goSpecimen = () => document.getElementById("s00")?.scrollIntoView({ behavior: "smooth" });

  return (
    <section className="hero" id="surface">
      <div className="hero-inner">
        <div className="kicker">CELLTRACE · GUIDED JOURNEY</div>
        <h1>
          Tracking the origin of life
          <span className="thin">a dive through a living zebrafish embryo</span>
        </h1>
        <p className="lead">
          Every cell is a suspect. Every frame is a snapshot. Below, a <b>living embryo</b> plays
          its first 100 frames — pick a specimen and descend through the full pipeline, from raw
          pixels to the final verdict.
        </p>
        {cloud && (
          <div className="ribbon">
            <div className="stat"><b>{cloud.n_total_nodes}</b><small>ANNOTATED CELLS</small></div>
            <div className="stat"><b>{cloud.n_total_edges}</b><small>TRUTH LINKS</small></div>
            <div className="stat"><b>T{cloud.t_min}–T{cloud.t_max}</b><small>FRAMES</small></div>
            <div className="stat">
              <b>{cloud.spacing_um?.z_um}·{cloud.spacing_um?.y_um}·{cloud.spacing_um?.x_um}</b>
              <small>VOXEL µM</small>
            </div>
          </div>
        )}
        <div className="hero-cta">
          <button type="button" className="btn hot" onClick={onDive}>
            {diving ? "■ ABORT THE DIVE" : "▼ BEGIN THE DIVE"}
          </button>
          <button type="button" className="btn" onClick={goSpecimen}>◎ CHOOSE SPECIMEN</button>
        </div>
        {err && <p className="err-text">{err}</p>}
        {!movieId && <p className="muted">↑ Pick a specimen in Stage 00 to summon the embryo.</p>}
        {movieId && !cloud && !err && <p className="muted">Summoning the embryo…</p>}
        {cloud && <Observatory cloud={cloud} />}
        <Say>
          <b>Say:</b> “This is a real zebrafish embryo — watch 100 frames of life. Every color is
          one cell family. Our job: teach AI to track them automatically.”
        </Say>
      </div>
      <div className="cue">▼ SCROLL TO DESCEND ▼</div>
    </section>
  );
}
