import { useEffect, useMemo, useState } from "react";
import api, { unwrap } from "../api/client";

// WHAT ARE THESE TWO FOLDERS? Plain-language explainer + real GEFF life
// stories, so panel members from any domain get the essence in one glance.
export default function DataStory({ movieId }) {
  const [rep, setRep] = useState(null);
  const [cloud, setCloud] = useState(null);
  const [note, setNote] = useState("");

  useEffect(() => {
    if (!movieId) {
      setRep(null);
      setCloud(null);
      return;
    }
    let cancelled = false;
    setNote("reading the two folders…");
    Promise.all([api.validate(movieId).then(unwrap), api.geffCloud(movieId, 3000).then(unwrap)])
      .then(([v, c]) => {
        if (cancelled) return;
        setRep(v);
        setCloud(c);
        setNote("");
      })
      .catch((e) => {
        if (cancelled) return;
        setNote(e.message);
      });
    return () => {
      cancelled = true;
    };
  }, [movieId]);

  const story = useMemo(() => {
    if (!cloud) return null;
    const byId = new Map(cloud.points.map((p) => [p.id, p]));
    const child = new Map();
    const hasParent = new Set();
    (cloud.links || []).forEach(([s, d]) => {
      if (!child.has(s)) child.set(s, d);
      hasParent.add(d);
    });
    const chains = [];
    for (const p of cloud.points) {
      if (hasParent.has(p.id)) continue;
      const chain = [p];
      let cur = p.id;
      let g = 0;
      while (child.has(cur) && g++ < 300) {
        const nx = byId.get(child.get(cur));
        if (!nx) break;
        chain.push(nx);
        cur = nx.id;
      }
      if (chain.length >= 2) chains.push(chain);
    }
    chains.sort((a, b) => b.length - a.length);
    return { chains: chains.slice(0, 5), nStories: chains.length };
  }, [cloud]);

  if (!movieId) {
    return (
      <div className="panel">
        <h3>WHAT ARE THESE TWO FOLDERS?</h3>
        <p className="sub">Select a specimen above — the story of its data appears here.</p>
      </div>
    );
  }

  const c = rep?.checks || {};
  const geff = c.geff || {};
  const t0 = cloud?.t_min ?? 0;
  const t1 = cloud?.t_max ?? 0;
  const span = Math.max(t1 - t0, 1);
  const W = 620;
  const H = 40 + (story?.chains.length || 0) * 56;
  const X = (t) => 60 + ((t - t0) / span) * (W - 100);

  return (
    <>
      <div className="grid g2">
        <div className="panel">
          <h3>🎞 .ZARR = THE FILM REEL</h3>
          <p className="sub">
            Pixels. A 4-D movie: time × depth × height × width of microscope light. This is what
            plays in the cinema above.
          </p>
          <div className="chips">
            <div className="chip">{(c.shape || []).join("×") || "—"}<small>FRAMES × DEPTH × H × W</small></div>
            <div className="chip">{c.dtype || "—"}<small>PIXEL TYPE</small></div>
            <div className="chip">{(c.n_voxels ?? 0).toLocaleString()}<small>VOXELS</small></div>
          </div>
        </div>
        <div className="panel">
          <h3>🕸 .GEFF = THE DETECTIVE'S NOTES</h3>
          <p className="sub">
            A graph, not pixels. Each dot = one cell sighting (time + 3-D position). Each link =
            "same cell, next frame" (ground truth).
          </p>
          <div className="chips">
            <div className="chip">{(geff.n_nodes ?? 0).toLocaleString()}<small>SIGHTINGS</small></div>
            <div className="chip">{(geff.n_edges ?? 0).toLocaleString()}<small>TRUTH LINKS</small></div>
            <div className="chip">T{t0}–T{t1}<small>TIME SPAN</small></div>
          </div>
        </div>
      </div>
      <div className="panel" style={{ marginTop: 18 }}>
        <h3>🔗 HOW THEY LINK · REAL LIFE STORIES</h3>
        <p className="sub">
          Every note points <b>at</b> the film: same (x, y, z, t). Below, {story?.nStories ?? 0} real
          cell life stories traced through the notes — each row is one cell, frame by frame. Toggle
          ANNOTATED in the cinema to see these dots pinned on the moving film.
        </p>
        {note && <p className="muted">{note}</p>}
        {story && story.chains.length > 0 && (
          <svg viewBox={`0 0 ${W} ${H}`} className="stories">
            <line x1="60" y1="14" x2={W - 40} y2="14" stroke="var(--line)" strokeWidth="2" />
            <text x="60" y="10" fill="var(--mut)" fontSize="9" fontFamily="monospace">T{t0}</text>
            <text x={W - 40} y="10" fill="var(--mut)" fontSize="9" fontFamily="monospace" textAnchor="end">T{t1}</text>
            {story.chains.map((chain, i) => {
              const hue = (i * 72 + 150) % 360;
              const y = 44 + i * 56;
              return (
                <g key={i}>
                  <text x="8" y={y + 4} fill="var(--mut)" fontSize="10" fontFamily="monospace">
                    cell {i + 1} · {chain.length}f
                  </text>
                  <polyline
                    points={chain.map((p) => `${X(p.t)},${y}`).join(" ")}
                    fill="none"
                    stroke={`hsl(${hue},80%,55%)`}
                    strokeWidth="2"
                    opacity="0.7"
                  />
                  {chain.map((p) => (
                    <circle key={p.id} cx={X(p.t)} cy={y} r="6" fill={`hsl(${hue},85%,60%)`} opacity="0.9">
                      <title>cell sighting #{p.id} · T{p.t} · x{p.x} y{p.y} z{p.z}</title>
                    </circle>
                  ))}
                </g>
              );
            })}
          </svg>
        )}
        {story && story.chains.length === 0 && (
          <p className="muted">No linked chains found in this sample — the notes may be unlinked sightings.</p>
        )}
      </div>
    </>
  );
}
