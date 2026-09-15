import { useCallback, useEffect, useRef, useState } from "react";
import api, { unwrap } from "../api/client";
import { StageShell, loadSnap, saveSnap } from "../components/DiveBits";

const IDLE = { state: "idle", progress: 0, note: "○ waiting" };

function GraphLive({ graph }) {
  const ref = useRef(null);
  const gref = useRef(graph);
  gref.current = graph;

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas.getContext("2d");
    let raf = 0;
    let t = 0;
    const render = () => {
      t += 0.03;
      const g = gref.current?.spatial;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      ctx.clearRect(0, 0, w, h);
      if (!g || !g.nodes?.length) {
        raf = requestAnimationFrame(render);
        return;
      }
      const xs = g.nodes.map((n) => n.x);
      const ys = g.nodes.map((n) => n.y);
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);
      const px = (x) => 30 + ((x - minX) / Math.max(maxX - minX, 1e-6)) * (w - 60);
      const py = (y) => 30 + ((y - minY) / Math.max(maxY - minY, 1e-6)) * (h - 60);
      const pos = new Map(g.nodes.map((n) => [n.node_id, [px(n.x), py(n.y)]]));
      for (const e of g.edges || []) {
        const a = pos.get(e.source_node_id);
        const b = pos.get(e.target_node_id);
        if (!a || !b) continue;
        ctx.strokeStyle = "rgba(34,197,94,0.45)";
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
        ctx.stroke();
        const f = (t * 0.4 + (e.source_node_id % 10) / 10) % 1;
        ctx.fillStyle = "#f59e0b";
        ctx.beginPath();
        ctx.arc(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, 3, 0, 6.2832);
        ctx.fill();
      }
      for (const n of g.nodes) {
        const [x, y] = pos.get(n.node_id);
        if (n.is_focus) {
          ctx.strokeStyle = "#f59e0b";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(x, y, 11 + Math.sin(t * 2) * 1.5, 0, 6.2832);
          ctx.stroke();
          ctx.fillStyle = "#f59e0b";
        } else {
          ctx.fillStyle = "#22c55e";
        }
        ctx.beginPath();
        ctx.arc(x, y, n.is_focus ? 7 : 5, 0, 6.2832);
        ctx.fill();
      }
      raf = requestAnimationFrame(render);
    };
    raf = requestAnimationFrame(render);
    return () => cancelAnimationFrame(raf);
  }, []);

  return <canvas ref={ref} className="graphlive" role="img" aria-label="Neighbor graph" />;
}

export default function Connect({ ctx }) {
  const { movieId, register, report } = ctx;
  const [st, setSt] = useState(IDLE);
  const [res, setRes] = useState(null);
  const [graph, setGraph] = useState(null);
  const [focus, setFocus] = useState("");
  const [gnote, setGnote] = useState("");

  const loadGraph = useCallback(
    async (mid, nid) => {
      setGnote("loading neighborhood…");
      try {
        const g = unwrap(await api.module2GraphPreview(mid, nid));
        setGraph(g);
        setGnote("");
        return g;
      } catch (e) {
        setGnote(e.message);
        return null;
      }
    },
    []
  );

  const run = useCallback(async () => {
    if (!movieId) throw new Error("Pick a specimen first (Stage 00).");
    setSt({ state: "run", progress: 0, note: "◉ fusing appearance + context…" });
    report("s05", "run");
    try {
      const r = unwrap(await api.runModule2(movieId));
      setRes(r);
      const first = r.preview?.[0]?.node_id;
      if (first !== undefined && first !== null) {
        setFocus(first);
        await loadGraph(movieId, first);
      }
      setSt({
        state: "done",
        progress: 1,
        note: `✓ ${r.n_cell_observations} fused · dim ${r.fused_feature_dimension}`,
      });
      report("s05", "done");
      return r;
    } catch (e) {
      setSt({ state: "err", progress: 0, note: `✗ ${e.message}` });
      report("s05", "err");
      throw e;
    }
  }, [movieId, report, loadGraph]);

  useEffect(() => {
    register("s05", run);
  }, [register, run]);

  if (res && res.movie_id === movieId) saveSnap("s05", movieId, { res, graph, focus, st });
  useEffect(() => {
    const snap = loadSnap("s05", movieId);
    if (snap && snap.res && snap.res.movie_id === movieId && snap.st.state === "done") {
      setRes(snap.res);
      setGraph(snap.graph);
      setFocus(snap.focus);
      setSt(snap.st);
      report("s05", "done");
    } else {
      setRes(null);
      setGraph(null);
      setSt(IDLE);
      report("s05", "");
    }
  }, [movieId, report]);

  return (
    <StageShell
      id="s05"
      num="05"
      kicker="STAGE 05 · NO CELL IS AN ISLAND"
      name="Connect"
      desc={
        <>
          Each cell is wired to its <b>K nearest neighbors</b> — the social graph the GAT reasons
          over. Appearance fuses with neighborhood context.
        </>
      }
      runLabel="FUSE GRAPH"
      status={st}
      onRun={() => run().catch(() => {})}
      say="“A cell is known by the company it keeps — the graph fuses each cell with its neighborhood.”"
    >
      {!res && <p className="muted">{movieId ? "Press FUSE GRAPH." : "Pick a specimen in Stage 00 first."}</p>}
      {res && (
        <div className="grid" style={{ gridTemplateColumns: "2fr 1fr" }}>
          <div className="panel">
            <h3>QUERY CELL + NEIGHBORS · FRAME {graph?.spatial?.frame ?? "—"}</h3>
            <p className="sub">Real positions · real edges · pulse motion decorative.</p>
            <label className="filelabel inline">
              Focus node
              <input
                className="textin num"
                type="number"
                value={focus}
                onChange={(e) => setFocus(e.target.value === "" ? "" : Number(e.target.value))}
              />
              <button
                type="button"
                className="btn sm"
                onClick={() => focus !== "" && loadGraph(movieId, Number(focus))}
              >
                RE-CENTER
              </button>
            </label>
            {gnote && <p className="muted">{gnote}</p>}
            {graph && <GraphLive graph={graph} />}
            <div className="legend">
              <span><b>◉ amber</b> focus cell</span>
              <span><b>● green</b> neighbors</span>
              <span>{graph?.spatial?.edges?.length ?? 0} edges · {graph?.spatial?.nodes?.length ?? 0} nodes</span>
            </div>
          </div>
          <div className="panel">
            <h3>FUSION REPORT</h3>
            <p className="sub">Adaptive appearance-context fusion.</p>
            <div className="meter"><span>observations</span><b>{res.n_cell_observations}</b></div>
            <div className="meter"><span>temporal links</span><b>{res.n_geff_temporal_links}</b></div>
            <div className="meter"><span>spatial links</span><b>{res.n_spatial_neighbour_links}</b></div>
            <div className="meter"><span>appearance coverage</span><b>{res.appearance_coverage_percent}%</b></div>
            <div className="meter"><span>fused dim</span><b>{res.fused_feature_dimension}</b></div>
            <div className="meter"><span>mean app weight</span><b>{res.mean_appearance_weight}</b></div>
            <h3 style={{ marginTop: 14 }}>SAMPLE CELLS</h3>
            {(res.preview || []).slice(0, 4).map((p) => (
              <div className="flag" key={p.node_id}>
                <span className="fdot" style={{ background: "var(--acc)" }} />#{p.node_id} T{p.frame} · {p.speed_um_per_frame} µm/f · {p.neighbours} nbrs
              </div>
            ))}
          </div>
        </div>
      )}
    </StageShell>
  );
}
