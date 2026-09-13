import { useCallback, useEffect, useRef, useState } from "react";
import api, { unwrap } from "../api/client";
import { StageShell } from "../components/DiveBits";

const IDLE = { state: "idle", progress: 0, note: "○ waiting" };

function Wipe({ raw, dev }) {
  const ref = useRef(null);
  const [pos, setPos] = useState(50);
  const drag = useRef(false);

  const update = (clientX) => {
    const r = ref.current.getBoundingClientRect();
    setPos(Math.max(4, Math.min(96, ((clientX - r.left) / r.width) * 100)));
  };

  return (
    <div
      className="wipe"
      ref={ref}
      onPointerDown={(e) => {
        drag.current = true;
        e.target.setPointerCapture(e.pointerId);
        update(e.clientX);
      }}
      onPointerMove={(e) => drag.current && update(e.clientX)}
      onPointerUp={() => {
        drag.current = false;
      }}
    >
      <img className="wipe-img" src={raw} alt="raw slice" draggable={false} />
      <div className="wipe-dev" style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}>
        <img src={dev} alt="standardized slice" draggable={false} />
      </div>
      <div className="knob" style={{ left: `${pos}%` }} />
      <span className="cap" style={{ left: 10 }}>RAW</span>
      <span className="cap" style={{ right: 10 }}>DEVELOPED</span>
    </div>
  );
}

export default function Standardize({ ctx }) {
  const { movieId, register, report } = ctx;
  const [st, setSt] = useState(IDLE);
  const [t, setT] = useState(0);
  const [z, setZ] = useState(32);
  const [res, setRes] = useState(null);

  const run = useCallback(async () => {
    if (!movieId) throw new Error("Pick a specimen first (Stage 00).");
    setSt({ state: "run", progress: 0, note: `◉ developing T=${t} Z=${z}…` });
    report("s03", "run");
    try {
      const r = unwrap(await api.standardize(movieId, t, z));
      setRes(r);
      setSt({ state: "done", progress: 1, note: `✓ T=${r.t} Z=${r.z} · μ=${r.normalized.mean.toFixed(3)}` });
      report("s03", "done");
      return r;
    } catch (e) {
      setSt({ state: "err", progress: 0, note: `✗ ${e.message}` });
      report("s03", "err");
      throw e;
    }
  }, [movieId, t, z, report]);

  useEffect(() => {
    register("s03", run);
  }, [register, run]);

  useEffect(() => {
    setRes(null);
    setSt(IDLE);
    report("s03", "");
  }, [movieId, report]);

  const hist = res?.histogram || [];
  const histMax = Math.max(...hist, 1);
  const np = res?.nearest_cell_patch;

  return (
    <StageShell
      id="s03"
      num="03"
      kicker="STAGE 03 · DEVELOP THE EVIDENCE"
      name="Standardize"
      desc={
        <>
          Microscope brightness drifts. <b>Percentile clipping + rescaling</b> develops every slice
          like photographic film — comparable intensity, honest contrast.
        </>
      }
      runLabel={`DEVELOP T=${t} Z=${z}`}
      status={st}
      onRun={() => run().catch(() => {})}
      say="“Same cells, same slice — now comparable. Networks can't learn from drifting brightness.”"
    >
      <div className="sliderrow">
        <label>
          Time T = {t}
          <input type="range" min={0} max={99} value={t} onChange={(e) => setT(Number(e.target.value))} />
        </label>
        <label>
          Depth Z = {z}
          <input type="range" min={0} max={63} value={z} onChange={(e) => setZ(Number(e.target.value))} />
        </label>
      </div>
      {!res && <p className="muted">{movieId ? "Press DEVELOP." : "Pick a specimen in Stage 00 first."}</p>}
      {res && (
        <div className="grid g2">
          <div className="panel">
            <h3>RAW → DEVELOPED · DRAG THE DIVIDER</h3>
            <p className="sub">Server-clipped to T={res.t} Z={res.z} · shape {(res.shape || []).join("×")}.</p>
            <Wipe
              raw={`data:image/png;base64,${res.raw_image_base64}`}
              dev={`data:image/png;base64,${res.standardized_image_base64}`}
            />
            <div className="hist">
              {hist.map((v, i) => (
                <i key={i} style={{ height: `${Math.max(3, (v / histMax) * 100)}%` }} title={`bin ${i}: ${v}`} />
              ))}
            </div>
            <p className="sub">Normalized-intensity histogram · 32 bins · real counts.</p>
          </div>
          <div className="panel">
            <h3>NORMALIZATION LOG</h3>
            <p className="sub">Parameters computed from real pixels.</p>
            <div className="meter"><span>p-low → p-high</span><b>{res.standardization.p_low.toFixed(1)} → {res.standardization.p_high.toFixed(1)}</b></div>
            <div className="meter"><span>raw μ ± σ</span><b>{res.raw.mean.toFixed(1)} ± {res.raw.std.toFixed(1)}</b></div>
            <div className="meter"><span>output μ ± σ</span><b>{res.normalized.mean.toFixed(3)} ± {res.normalized.std.toFixed(3)}</b></div>
            <div className="meter"><span>output range</span><b>{res.normalized.min.toFixed(3)} – {res.normalized.max.toFixed(3)}</b></div>
            {np && (
              <>
                <h3 style={{ marginTop: 16 }}>NEAREST CELL · NODE {np.node_id}</h3>
                <p className="sub">Closest real patch to this slice · T{np.t ?? res.t}.</p>
                <div className="patchpair">
                  <figure>
                    <img src={`data:${np.mime || "image/png"};base64,${np.patch_image_base64}`} alt="raw patch" />
                    <figcaption>raw MIP</figcaption>
                  </figure>
                  <figure>
                    <img src={`data:${np.mime || "image/png"};base64,${np.normalized_patch_image_base64}`} alt="normalized patch" />
                    <figcaption>developed MIP</figcaption>
                  </figure>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </StageShell>
  );
}
