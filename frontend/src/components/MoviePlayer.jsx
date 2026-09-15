import { useEffect, useRef, useState } from "react";
import api, { unwrap } from "../api/client";

// Specimen Cinema: the .zarr film reel as a playable movie, with optional
// .geff annotation orbs pinned on the film. All frames are real slices.
export default function MoviePlayer({ movieId }) {
  const [bounds, setBounds] = useState(null);
  const [dims, setDims] = useState([256, 256]);
  const [t, setT] = useState(0);
  const [z, setZ] = useState(32);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [annotated, setAnnotated] = useState(true);
  const [img, setImg] = useState("");
  const [nodes, setNodes] = useState([]);
  const [note, setNote] = useState("");
  const canvasRef = useRef(null);
  const seq = useRef(0);

  useEffect(() => {
    setT(0);
    setImg("");
    setNodes([]);
    setNote("");
    setPlaying(false);
    setBounds(null);
    if (!movieId) return;
    let cancelled = false;
    api
      .slice(movieId, 0, 0)
      .then(unwrap)
      .then((d) => {
        if (cancelled) return;
        if (d.bounds) {
          setBounds(d.bounds);
          setZ(Math.floor(((d.bounds.z_max || 1) - 1) / 2));
        }
        if (d.shape) setDims(d.shape);
      })
      .catch((e) => {
        if (!cancelled) setNote(e.message);
      });
    return () => {
      cancelled = true;
    };
  }, [movieId]);

  useEffect(() => {
    if (!playing || !bounds) return undefined;
    const tMax = bounds.t_max - 1;
    const h = setInterval(() => {
      setT((v) => (v >= tMax ? 0 : v + 1));
    }, Math.max(80, 500 / speed));
    return () => clearInterval(h);
  }, [playing, speed, bounds]);

  useEffect(() => {
    if (!movieId) return undefined;
    const my = ++seq.current;
    let cancelled = false;
    (annotated ? api.geffOverlay(movieId, t, z) : api.slice(movieId, t, z))
      .then(unwrap)
      .then((d) => {
        if (cancelled || my !== seq.current) return;
        setImg(`data:${d.mime || "image/jpeg"};base64,${d.image_base64}`);
        setNodes(d.visible_nodes || []);
      })
      .catch((e) => {
        if (cancelled || my !== seq.current) return;
        if (annotated) {
          setNote(`Annotations need the .geff folder (${e.message}) — showing raw film.`);
          setAnnotated(false);
        } else {
          setNote(e.message);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [movieId, t, z, annotated]);

  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const Y = dims[0] || 256;
    const X = dims[1] || 256;
    cv.width = X;
    cv.height = Y;
    const c = cv.getContext("2d");
    c.clearRect(0, 0, X, Y);
    if (!annotated) return;
    const r = Math.max(6, X / 42);
    for (const n of nodes) {
      const g = c.createRadialGradient(n.x, n.y, 1, n.x, n.y, r);
      g.addColorStop(0, "rgba(255,255,255,0.95)");
      g.addColorStop(0.35, "rgba(61,255,158,0.9)");
      g.addColorStop(1, "rgba(61,255,158,0)");
      c.fillStyle = g;
      c.beginPath();
      c.arc(n.x, n.y, r, 0, 6.2832);
      c.fill();
      c.strokeStyle = "rgba(61,255,158,0.8)";
      c.lineWidth = 1.5;
      c.beginPath();
      c.arc(n.x, n.y, r * 0.55, 0, 6.2832);
      c.stroke();
    }
    if (nodes.length > 0 && nodes.length <= 40) {
      c.font = `${Math.max(9, X / 60)}px monospace`;
      c.fillStyle = "#e8fcee";
      nodes.forEach((n) => {
        c.fillText(String(n.node_id).slice(-4), n.x + r * 0.6, n.y - r * 0.6);
      });
    }
  }, [img, nodes, dims, annotated]);

  if (!movieId) {
    return (
      <div className="panel">
        <h3>SPECIMEN CINEMA · THE FILM REEL</h3>
        <p className="sub">Select a specimen above to play its movie.</p>
      </div>
    );
  }

  const tMax = bounds ? bounds.t_max - 1 : 0;
  const zMax = bounds ? bounds.z_max - 1 : 0;

  return (
    <div className="panel">
      <h3>SPECIMEN CINEMA · {movieId}</h3>
      <p className="sub">
        The .zarr film reel, frame by frame.{" "}
        {annotated
          ? `Each glowing orb = one annotated cell at this depth (${nodes.length} visible).`
          : "Raw fluorescence — toggle ANNOTATED to pin the .geff notes on the film."}
      </p>
      <div className="cinema-stage">
        {img ? (
          <img src={img} alt={`Frame T=${t} Z=${z}`} className="cinema-img" draggable={false} />
        ) : (
          <div className="image-placeholder">Loading film…</div>
        )}
        <canvas ref={canvasRef} className="cinema-orbs" />
        <div className="cinema-badge">T = {t} · Z = {z}</div>
      </div>
      <div className="cinema-controls">
        <button type="button" className="btn sm" onClick={() => setT(0)}>|◀</button>
        <button type="button" className="btn sm" onClick={() => setT((v) => Math.max(0, v - 1))}>◀</button>
        <button type="button" className="btn sm hot" onClick={() => setPlaying((p) => !p)}>
          {playing ? "❚❚" : "▶"}
        </button>
        <button type="button" className="btn sm" onClick={() => setT((v) => Math.min(tMax, v + 1))}>▶</button>
        <button type="button" className="btn sm" onClick={() => setT(tMax)}>▶|</button>
        <input
          type="range"
          min={0}
          max={Math.max(tMax, 0)}
          value={Math.min(t, Math.max(tMax, 0))}
          onChange={(e) => setT(Number(e.target.value))}
          aria-label="Time scrubber"
        />
        <span className="obs-speed">
          {[0.5, 1, 2, 4].map((v) => (
            <button key={v} type="button" className={speed === v ? "active" : ""} onClick={() => setSpeed(v)}>
              {v}×
            </button>
          ))}
        </span>
      </div>
      <div className="sliderrow" style={{ marginTop: 10, marginBottom: 0 }}>
        <label>
          Depth Z = {z} (cells drift in and out as you slice)
          <input
            type="range"
            min={0}
            max={Math.max(zMax, 0)}
            value={Math.min(z, Math.max(zMax, 0))}
            onChange={(e) => setZ(Number(e.target.value))}
          />
        </label>
        <label className="checkline">
          <input type="checkbox" checked={annotated} onChange={(e) => setAnnotated(e.target.checked)} />
          ANNOTATED — pin .geff orbs on the film
        </label>
      </div>
      {note && <p className="muted">{note}</p>}
    </div>
  );
}
