import { useCallback, useEffect, useState } from "react";
import api, { unwrap } from "../api/client";
import { StageShell, loadSnap, saveSnap } from "../components/DiveBits";

const IDLE = { state: "idle", progress: 0, note: "○ waiting" };

export default function Validate({ ctx }) {
  const { movieId, register, report } = ctx;
  const [st, setSt] = useState(IDLE);
  const [rep, setRep] = useState(null);

  const run = useCallback(async () => {
    if (!movieId) throw new Error("Pick a specimen first (Stage 00).");
    setSt({ state: "run", progress: 0, note: "◉ running checks…" });
    report("s01", "run");
    try {
      const res = unwrap(await api.validate(movieId));
      setRep(res);
      const ok = res.ok;
      setSt({
        state: "done",
        progress: 1,
        note: ok ? `✓ valid · ${res.warnings?.length || 0} warnings` : `✓ done · ${res.issues?.length || 0} issues`,
      });
      report("s01", "done");
      return res;
    } catch (e) {
      setSt({ state: "err", progress: 0, note: `✗ ${e.message}` });
      report("s01", "err");
      throw e;
    }
  }, [movieId, report]);

  useEffect(() => {
    register("s01", run);
  }, [register, run]);

  if (rep && rep.movie_id === movieId) saveSnap("s01", movieId, { res: rep, st });
  useEffect(() => {
    const snap = loadSnap("s01", movieId);
    if (snap && snap.res && snap.res.movie_id === movieId && snap.st.state === "done") {
      setRep(snap.res);
      setSt(snap.st);
      report("s01", "done");
    } else {
      setRep(null);
      setSt(IDLE);
      report("s01", "");
    }
  }, [movieId, report]);

  const c = rep?.checks || {};
  const geff = c.geff || {};
  const avgPerFrame =
    geff.n_nodes && geff.t_max !== null && geff.t_min !== null && geff.t_max > geff.t_min
      ? (geff.n_nodes / (geff.t_max - geff.t_min + 1)).toFixed(1)
      : null;

  return (
    <StageShell
      id="s01"
      num="01"
      kicker="STAGE 01 · TRUST, THEN COMPUTE"
      name="Validate"
      desc={
        <>
          Before a single number is computed: is the volume intact, are the annotations aligned, do
          the voxel scales match the microscope? <b>No green lights, no dive.</b>
        </>
      }
      runLabel="RUN CHECKS"
      status={st}
      onRun={() => run().catch(() => {})}
      say="“Science starts with distrust — we verify every byte before believing a single result.”"
    >
      {!rep && <p className="muted">{movieId ? "Press RUN CHECKS." : "Pick a specimen in Stage 00 first."}</p>}
      {rep && (
        <div className="grid g2">
          <div className="panel">
            <h3>INTEGRITY CHECKLIST · REAL BYTES</h3>
            <p className="sub">{rep.movie_id} · {rep.ok ? "VALID" : "HAS ISSUES"}</p>
            <div className="check"><span className="ck">✓</span> Volume shape {(c.shape || []).join(" × ") || "—"}</div>
            <div className="check"><span className="ck">✓</span> dtype {c.dtype || "—"} · {(c.n_voxels ?? 0).toLocaleString()} voxels</div>
            <div className="check">
              <span className={c.sample_finite ? "ck" : "ck warn"}>{c.sample_finite ? "✓" : "!"}</span>
              Sample slice finite · range {c.sample_min?.toFixed?.(1)} – {c.sample_max?.toFixed?.(1)}
            </div>
            <div className="check">
              <span className={c.spacing?.ok ? "ck" : "ck warn"}>{c.spacing?.ok ? "✓" : "!"}</span>
              Voxel spacing matches microscope log
            </div>
            <div className="check">
              <span className={geff.n_nodes ? "ck" : "ck warn"}>{geff.n_nodes ? "✓" : "!"}</span>
              GEFF: {(geff.n_nodes ?? 0).toLocaleString()} nodes · {(geff.n_edges ?? 0).toLocaleString()} edges
            </div>
            {(rep.issues?.length > 0 || rep.warnings?.length > 0) && (
              <div className="flags">
                {(rep.issues || []).map((m, i) => (
                  <div className="flag" key={`i${i}`}><span className="fdot" style={{ background: "var(--alert)" }} />{m}</div>
                ))}
                {(rep.warnings || []).map((m, i) => (
                  <div className="flag" key={`w${i}`}><span className="fdot" style={{ background: "var(--acc2)" }} />{m}</div>
                ))}
              </div>
            )}
          </div>
          <div className="panel">
            <h3>SPECIMEN AT A GLANCE</h3>
            <p className="sub">Shape, scale and annotation density.</p>
            <div className="ringwrap">
              <svg width="120" height="120" viewBox="0 0 120 120">
                <circle cx="60" cy="60" r="52" fill="none" stroke="var(--chip)" strokeWidth="12" />
                <circle cx="60" cy="60" r="52" fill="none" stroke="var(--acc)" strokeWidth="12"
                  strokeDasharray="327" strokeDashoffset={rep.ok ? 40 : 160} strokeLinecap="round"
                  transform="rotate(-90 60 60)" />
              </svg>
              <div>
                <div className="bignum">{avgPerFrame ?? "—"}</div>
                <div className="muted mono">CELLS / FRAME (AVG)</div>
              </div>
            </div>
            <div className="chips">
              <div className="chip">{(c.shape || []).join("×") || "—"}<small>T × Z × Y × X</small></div>
              <div className="chip">
                {rep.spacing_um ? `${rep.spacing_um.z_um}·${rep.spacing_um.y_um}·${rep.spacing_um.x_um}` : "—"}
                <small>VOXEL µM</small>
              </div>
              <div className="chip">{(geff.n_nodes ?? "—").toLocaleString?.() ?? geff.n_nodes ?? "—"}<small>GEFF NODES</small></div>
            </div>
          </div>
        </div>
      )}
    </StageShell>
  );
}
