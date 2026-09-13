import { useCallback, useEffect, useState } from "react";
import api, { unwrap } from "../api/client";
import { StageShell } from "../components/DiveBits";

const IDLE = { state: "idle", progress: 0, note: "○ waiting" };

const ARTIFACTS = [
  { name: "embedding_meta", label: "Encoder fingerprints", stage: "Stage 04" },
  { name: "module2_summary", label: "Graph fusion", stage: "Stage 05" },
  { name: "module3_summary", label: "GAT training", stage: "Stage 06" },
  { name: "tracking_summary", label: "Tracking verdict", stage: "Stage 07" },
];

export default function Export({ ctx }) {
  const { movieId, register, report } = ctx;
  const [st, setSt] = useState(IDLE);
  const [rows, setRows] = useState([]);
  const [manifest, setManifest] = useState(null);

  const run = useCallback(async () => {
    if (!movieId) throw new Error("Pick a specimen first (Stage 00).");
    setSt({ state: "run", progress: 0, note: "◉ sealing manifest…" });
    report("s08", "run");
    try {
      const settled = await Promise.all(
        ARTIFACTS.map(async (a) => {
          try {
            const d = unwrap(await api.journeyArtifact(a.name, movieId));
            return { ...a, ok: true, bytes: JSON.stringify(d).length, data: d.data ?? d };
          } catch (e) {
            return { ...a, ok: false, error: e.message };
          }
        })
      );
      setRows(settled);
      const doc = {
        app: "CellTrace Dive",
        movie_id: movieId,
        exported_at: new Date().toISOString(),
        artifacts: Object.fromEntries(settled.map((r) => [r.name, r.ok ? r.data : { missing: r.error }])),
      };
      setManifest(doc);
      const got = settled.filter((r) => r.ok).length;
      setSt({ state: "done", progress: 1, note: `✓ sealed · ${got}/${settled.length} artifacts` });
      report("s08", "done");
      return doc;
    } catch (e) {
      setSt({ state: "err", progress: 0, note: `✗ ${e.message}` });
      report("s08", "err");
      throw e;
    }
  }, [movieId, report]);

  useEffect(() => {
    register("s08", run);
  }, [register, run]);

  useEffect(() => {
    setRows([]);
    setManifest(null);
    setSt(IDLE);
    report("s08", "");
  }, [movieId, report]);

  const download = () => {
    if (!manifest) return;
    const text = JSON.stringify(manifest, null, 2);
    const blob = new Blob([text], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `celltrace_manifest_${movieId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const pretty = manifest ? JSON.stringify(manifest, null, 2) : "";
  const shown = pretty.split("\n").slice(0, 18).join("\n");

  return (
    <StageShell
      id="s08"
      num="08"
      kicker="STAGE 08 · BAG THE EVIDENCE"
      name="Export"
      desc={
        <>
          Every run seals a <b>manifest</b> — data, parameters, metrics, weights — so any result can
          be reproduced. Then: the road ahead.
        </>
      }
      runLabel="SEAL MANIFEST"
      status={st}
      onRun={() => run().catch(() => {})}
      say="“Everything reproducible from one file — and here's exactly what comes next.”"
    >
      {!manifest && <p className="muted">{movieId ? "Press SEAL MANIFEST." : "Pick a specimen in Stage 00 first."}</p>}
      {manifest && (
        <div className="grid g2">
          <div className="panel">
            <h3>RUN MANIFEST · JSON · REAL BYTES</h3>
            <p className="sub">{pretty.length.toLocaleString()} bytes · assembled from this run's artifacts.</p>
            <div className="manifest">{shown}{pretty.split("\n").length > 18 ? "\n  …" : ""}</div>
            <div style={{ height: 12 }} />
            <button type="button" className="btn hot" onClick={download}>
              ⬇ DOWNLOAD {(pretty.length / 1024).toFixed(1)} KB
            </button>
            <div style={{ height: 12 }} />
            {rows.map((r) => (
              <div className="flag" key={r.name}>
                <span
                  className="fdot"
                  style={{ background: r.ok ? "var(--acc)" : "var(--alert)" }}
                />
                {r.label} · {r.ok ? `${r.bytes.toLocaleString()} B` : `missing — run ${r.stage}`}
              </div>
            ))}
          </div>
          <div className="panel">
            <h3>THE ROAD AHEAD</h3>
            <p className="sub">Marked future work — never claimed done.</p>
            <div className="timeline">
              <div className="tl"><b>Division &amp; lineage submodule</b><span>mitosis events → true family trees</span></div>
              <div className="tl"><b>Pretrained encoder integration</b><span>swap prototype stem for validated weights</span></div>
              <div className="tl"><b>Multi-embryo atlas</b><span>cross-specimen track statistics</span></div>
              <div className="tl"><b>Real-time tracking</b><span>streaming inference as frames land</span></div>
              <div className="tl"><b>Results database</b><span>SQLite run-store for every dive</span></div>
            </div>
          </div>
        </div>
      )}
    </StageShell>
  );
}
