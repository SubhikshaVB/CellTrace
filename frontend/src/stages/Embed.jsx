import { useCallback, useEffect, useState } from "react";
import api, { unwrap } from "../api/client";
import { StageShell, loadSnap, saveSnap } from "../components/DiveBits";

const IDLE = { state: "idle", progress: 0, note: "○ waiting" };

export default function Embed({ ctx }) {
  const { movieId, register, report } = ctx;
  const [st, setSt] = useState(IDLE);
  const [maxCells, setMaxCells] = useState(64);
  const [meta, setMeta] = useState(null);
  const [tens, setTens] = useState(null);
  const [nodeId, setNodeId] = useState(null);
  const [vec, setVec] = useState(null);
  const [nbrs, setNbrs] = useState(null);
  const [selNote, setSelNote] = useState("");

  const loadSel = useCallback(
    async (mid, nid) => {
      setSelNote("loading fingerprint…");
      try {
        const [v, n] = await Promise.all([
          api.embeddingVector(mid, nid).then(unwrap),
          api.embeddingNeighbors(mid, nid, 5).then(unwrap),
        ]);
        setVec(v);
        setNbrs(n);
        setSelNote("");
      } catch (e) {
        setSelNote(e.message);
      }
    },
    []
  );

  const run = useCallback(async () => {
    if (!movieId) throw new Error("Pick a specimen first (Stage 00).");
    setSt({ state: "run", progress: 0, note: "◉ tensorizing cells…" });
    report("s04", "run");
    try {
      const tz = unwrap(await api.tensorize(movieId, maxCells));
      setTens(tz);
      setSt({ state: "run", progress: 0.4, note: `◉ encoding ${tz.n_tensors} cells…` });
      const r = unwrap(await api.runCellDino(movieId, maxCells));
      setMeta(r);
      const first = r.preview?.[0]?.node_id;
      if (first !== undefined && first !== null) {
        setNodeId(first);
        await loadSel(movieId, first);
      }
      setSt({
        state: "done",
        progress: 1,
        note: `✓ ${r.n_embeddings} fingerprints · dim ${r.embed_dim} · ${r.elapsed_sec.toFixed(1)}s`,
      });
      report("s04", "done");
      return r;
    } catch (e) {
      setSt({ state: "err", progress: 0, note: `✗ ${e.message}` });
      report("s04", "err");
      throw e;
    }
  }, [movieId, maxCells, report, loadSel]);

  useEffect(() => {
    register("s04", run);
  }, [register, run]);

  if (meta && meta.movie_id === movieId) saveSnap("s04", movieId, { meta, tens, vec, nbrs, nodeId, st });
  useEffect(() => {
    const snap = loadSnap("s04", movieId);
    if (snap && snap.meta && snap.meta.movie_id === movieId && snap.st.state === "done") {
      setMeta(snap.meta);
      setTens(snap.tens);
      setVec(snap.vec);
      setNbrs(snap.nbrs);
      setNodeId(snap.nodeId);
      setSt(snap.st);
      report("s04", "done");
    } else {
      setMeta(null);
      setTens(null);
      setVec(null);
      setNbrs(null);
      setNodeId(null);
      setSt(IDLE);
      report("s04", "");
    }
  }, [movieId, report]);

  const pick = (nid) => {
    setNodeId(nid);
    loadSel(movieId, nid);
  };

  const v = vec?.vector || [];
  const vmax = Math.max(...v.map(Math.abs), 1e-9);

  return (
    <StageShell
      id="s04"
      num="04"
      kicker="STAGE 04 · EVERY CELL GETS A FACE"
      name="Embed"
      desc={
        <>
          The prototype encoder reads each patch and writes a <b>fingerprint vector</b> — cells that
          look alike land close together. Pick a cell, meet its lookalikes.
        </>
      }
      runLabel={`ENCODER · ${maxCells} CELLS`}
      status={st}
      onRun={() => run().catch(() => {})}
      say="“The AI compresses each cell into a fingerprint — and fingerprints of the same cell agree.”"
    >
      <span className="tag-proto">PROTOTYPE ENCODER · research prototype — not official CellDINO weights</span>
      <label className="filelabel inline">
        Cells to encode
        <input
          className="textin num"
          type="number"
          min={1}
          max={1000}
          value={maxCells}
          onChange={(e) => setMaxCells(Math.max(1, Number(e.target.value) || 1))}
        />
      </label>
      {!meta && <p className="muted">{movieId ? "Press ENCODER." : "Pick a specimen in Stage 00 first."}</p>}
      {meta && (
        <div className="grid g2">
          <div className="panel">
            <h3>CELL #{nodeId} · EMBEDDING FINGERPRINT</h3>
            <p className="sub">Full {vec?.dim || meta.embed_dim}-dim vector · relative scale · real values.</p>
            <label className="filelabel">
              Fingerprint of
              <select className="textin" value={nodeId ?? ""} onChange={(e) => pick(Number(e.target.value))}>
                {(meta.preview || []).map((p) => (
                  <option key={p.node_id} value={p.node_id}>
                    cell #{p.node_id} · T{p.t}
                  </option>
                ))}
              </select>
            </label>
            {selNote && <p className="muted">{selNote}</p>}
            <div className="bars">
              {v.map((x, i) => (
                <i
                  key={i}
                  title={`dim ${i}: ${x.toFixed(4)}`}
                  style={{ height: `${Math.max(4, (Math.abs(x) / vmax) * 100)}%`, opacity: 0.35 + 0.65 * (Math.abs(x) / vmax) }}
                />
              ))}
            </div>
            <div className="chips">
              <div className="chip">{vec?.dim ?? "—"}<small>DIMENSIONS</small></div>
              <div className="chip">{vec ? vec.norm.toFixed(3) : "—"}<small>L2 NORM</small></div>
              <div className="chip">{meta.backend}<small>BACKEND</small></div>
            </div>
          </div>
          <div className="panel">
            <h3>LOOKALIKES · COSINE SIMILARITY</h3>
            <p className="sub">Top-5 neighbors in embedding space · NaN {meta.nan_count} · Inf {meta.inf_count}.</p>
            {(nbrs?.neighbors || []).map((n) => (
              <div className="lrow" key={n.node_id}>
                <span>#{n.node_id} T{n.t}</span>
                <div className="lbar"><i style={{ width: `${Math.max(0, Math.min(100, n.similarity * 100))}%` }} /></div>
                <span>{n.similarity.toFixed(4)}</span>
              </div>
            ))}
            <div className="chips">
              <div className="chip">{tens?.n_tensors ?? "—"}<small>TENSORS</small></div>
              <div className="chip">{meta.n_embeddings}<small>FINGERPRINTS</small></div>
              <div className="chip">{meta.elapsed_sec.toFixed(1)}s<small>ENCODE TIME</small></div>
            </div>
          </div>
        </div>
      )}
    </StageShell>
  );
}
