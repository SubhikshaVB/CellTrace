import { useCallback, useEffect, useState } from "react";
import api, { unwrap } from "../api/client";
import { StageShell, loadSnap, saveSnap } from "../components/DiveBits";

const IDLE = { state: "idle", progress: 0, note: "○ waiting" };

export default function Extract({ ctx }) {
  const { movieId, register, report } = ctx;
  const [st, setSt] = useState(IDLE);
  const [maxCells, setMaxCells] = useState(24);
  const [res, setRes] = useState(null);

  const run = useCallback(async () => {
    if (!movieId) throw new Error("Pick a specimen first (Stage 00).");
    setSt({ state: "run", progress: 0, note: `◉ cutting ${maxCells} patches…` });
    report("s02", "run");
    try {
      const r = unwrap(await api.extract(movieId, maxCells));
      setRes(r);
      setSt({
        state: "done",
        progress: 1,
        note: `✓ ${r.n_extracted} patches · ${r.n_failed} failed`,
      });
      report("s02", "done");
      return r;
    } catch (e) {
      setSt({ state: "err", progress: 0, note: `✗ ${e.message}` });
      report("s02", "err");
      throw e;
    }
  }, [movieId, maxCells, report]);

  useEffect(() => {
    register("s02", run);
  }, [register, run]);

  if (res && res.movie_id === movieId) saveSnap("s02", movieId, { res, st });
  useEffect(() => {
    const snap = loadSnap("s02", movieId);
    if (snap && snap.res && snap.res.movie_id === movieId && snap.st.state === "done") {
      setRes(snap.res);
      setSt(snap.st);
      report("s02", "done");
    } else {
      setRes(null);
      setSt(IDLE);
      report("s02", "");
    }
  }, [movieId, report]);

  const previews = res?.patches_preview || [];

  return (
    <StageShell
      id="s02"
      num="02"
      kicker="STAGE 02 · CUT OUT EVERY SUSPECT"
      name="Extract"
      desc={
        <>
          A 3-D box is cropped around <b>every annotated cell</b> — the raw material for everything
          the networks will ever see.
        </>
      }
      runLabel={`EXTRACT ${maxCells} PATCHES`}
      status={st}
      onRun={() => run().catch(() => {})}
      say="“Real cells, cut out of a billion-voxel volume — this is exactly what our networks see.”"
    >
      <label className="filelabel inline">
        Cells to extract
        <input
          className="textin num"
          type="number"
          min={1}
          max={1000}
          value={maxCells}
          onChange={(e) => setMaxCells(Math.max(1, Number(e.target.value) || 1))}
        />
      </label>
      {!res && <p className="muted">{movieId ? "Press EXTRACT." : "Pick a specimen in Stage 00 first."}</p>}
      {res && (
        <>
          <div className="grid" style={{ gridTemplateColumns: "2fr 1fr" }}>
            <div className="panel">
              <h3>PATCH WALL · 3-D CROPS</h3>
              <p className="sub">Max-intensity projections of the real extracted patches.</p>
              <div className="tiles">
                {previews.map((p) => (
                  <figure className="tile" key={`${p.node_id}-${p.t}`}>
                    <img
                      src={`data:${p.mime || "image/png"};base64,${p.patch_image_base64}`}
                      alt={`Cell ${p.node_id}`}
                    />
                    <figcaption>#{p.node_id} · T{p.t}</figcaption>
                  </figure>
                ))}
              </div>
            </div>
            <div className="panel">
              <h3>EXTRACTION TALLY</h3>
              <p className="sub">Counts from the real run.</p>
              <div className="meter"><span>requested</span><b>{res.n_requested}</b></div>
              <div className="meter"><span>extracted</span><b>{res.n_extracted}</b></div>
              <div className="meter"><span>failed (bounds)</span><b>{res.n_failed}</b></div>
              <div className="meter"><span>manifest</span><b className="tiny">{String(res.manifest).split(/[/\\]/).slice(-2).join("/")}</b></div>
            </div>
          </div>
          {res.failures?.length > 0 && (
            <div className="panel" style={{ marginTop: 18, maxWidth: 1100 }}>
              <h3>HONEST FAILURES · SAMPLE</h3>
              <p className="sub">Boundary crops that could not be cut — shown, not hidden.</p>
              {res.failures.slice(0, 4).map((f, i) => (
                <div className="flag" key={i}>
                  <span className="fdot" style={{ background: "var(--acc2)" }} />
                  node {f.node_id} · {f.error}
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </StageShell>
  );
}
