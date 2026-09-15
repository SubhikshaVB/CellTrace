import { useCallback, useEffect, useRef, useState } from "react";
import api, { unwrap } from "../api/client";
import { StageShell, fmtBytes } from "../components/DiveBits";
import MoviePlayer from "../components/MoviePlayer";
import DataStory from "../components/DataStory";

const IDLE = { state: "idle", progress: 0, note: "○ waiting" };

export default function Specimen({ ctx }) {
  const { movieId, selectMovie, register, report } = ctx;
  const [st, setSt] = useState(IDLE);
  const [datasets, setDatasets] = useState([]);
  const [registry, setRegistry] = useState({});
  const [movieName, setMovieName] = useState("");
  const [busy, setBusy] = useState(null);
  const [prog, setProg] = useState(0);
  const [note, setNote] = useState("");
  const [lastUp, setLastUp] = useState(null);
  const [serverFiles, setServerFiles] = useState(null);
  const zarrInput = useRef(null);
  const geffInput = useRef(null);

  useEffect(() => {
    zarrInput.current?.setAttribute("webkitdirectory", "");
    geffInput.current?.setAttribute("webkitdirectory", "");
  }, []);

  const refresh = useCallback(async () => {
    const [ds, reg] = await Promise.all([
      api.listDatasets().then(unwrap),
      api.uploadRegistry().then(unwrap),
    ]);
    setDatasets(Array.isArray(ds) ? ds : []);
    setRegistry(reg || {});
  }, []);

  useEffect(() => {
    refresh().catch((e) => setNote(e.message));
  }, [refresh]);

  useEffect(() => {
    if (movieId) {
      setSt({ state: "done", progress: 1, note: `✓ specimen: ${movieId}` });
      report("s00", "done");
    } else {
      setSt(IDLE);
      report("s00", "");
    }
  }, [movieId, report]);

  const run = useCallback(async () => {
    if (!movieId) throw new Error("Pick a specimen first (Stage 00).");
    return true;
  }, [movieId]);

  useEffect(() => {
    register("s00", run);
  }, [register, run]);

  const pickFolder = async (kind, fileList) => {
    const files = Array.from(fileList || []);
    if (files.length === 0) return;
    const top = (files[0].webkitRelativePath || files[0].name).split("/")[0];
    if (!top.endsWith(`.${kind}`)) {
      setNote(`That folder is not a .${kind} folder (got “${top}”).`);
      return;
    }
    const name = movieName.trim() || top.replace(/\.(zarr|geff)$/, "");
    setBusy(kind);
    setProg(0);
    setNote(`Uploading ${files.length} files…`);
    try {
      const fn = kind === "zarr" ? api.uploadZarr : api.uploadGeff;
      const res = await fn(name, files, setProg);
      setLastUp(res);
      setNote(
        res.pair.ready
          ? `✓ ${res.movie_id} complete — zarr + geff paired.`
          : `✓ ${res.movie_id} ${kind} stored — now upload the matching .${kind === "zarr" ? "geff" : "zarr"}.`
      );
      await refresh();
      selectMovie(res.movie_id);
    } catch (e) {
      setNote(`Upload failed: ${e.message}`);
    } finally {
      setBusy(null);
    }
  };

  const removeMovie = async (id) => {
    if (!window.confirm(`Delete uploaded movie “${id}”?`)) return;
    try {
      await api.deleteMovie(id);
      if (movieId === id) selectMovie("");
      await refresh();
      setNote(`Deleted ${id}.`);
    } catch (e) {
      setNote(e.message);
    }
  };

  const loadServerFiles = async () => {
    try {
      setServerFiles(unwrap(await api.trainContents()));
    } catch (e) {
      setNote(e.message);
    }
  };

  const sel = datasets.find((d) => d.movie_id === movieId);
  const incomplete = sel && (!sel.has_zarr || !sel.has_geff);

  return (
    <StageShell
      id="s00"
      num="00"
      kicker="STAGE 00 · THE DIVE BEGINS"
      name="Specimen"
      desc={
        <>
          Bring your own embryo: drop a <b>.zarr folder</b> and its <b>.geff folder</b> — folders,
          not files — or pick a movie from the lab vault. Everything downstream runs on <b>this</b> specimen.
        </>
      }
      runLabel="CONFIRM SPECIMEN"
      status={st}
      onRun={() => run().catch((e) => setNote(e.message))}
      say="“Any embryo works — drop the two folders and the whole pipeline runs on your data.”"
    >
      {incomplete && (
        <div className="panel warn" style={{ maxWidth: 1100, marginBottom: 18 }}>
          <h3>⚠ SPECIMEN INCOMPLETE · {movieId}</h3>
          <p className="sub">
            {!sel.has_zarr && "Missing the .zarr film reel. "}
            {!sel.has_geff && "Missing the .geff annotation notes — the time-lapse, patches, graph and tracking all need it. "}
            Upload the missing folder above (same movie name) to complete the pair.
          </p>
        </div>
      )}
      <div className="grid g2">
        <div className="panel">
          <h3>YOUR DATA · FOLDERS ONLY</h3>
          <p className="sub">Directory upload — structure is validated before anything runs.</p>
          <label className="filelabel">
            Movie name (optional — defaults to folder name)
            <input
              className="textin"
              value={movieName}
              onChange={(e) => setMovieName(e.target.value)}
              placeholder="e.g. my_embryo"
            />
          </label>
          <div
            className={`dz${busy === "zarr" ? " busy" : ""}`}
            onClick={() => !busy && zarrInput.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => e.key === "Enter" && zarrInput.current?.click()}
          >
            <div className="big">📁</div>
            <b>{busy === "zarr" ? `Uploading… ${Math.round(prog * 100)}%` : "Drop .zarr folder"}</b>
            <span>expects array meta · chunk grid · ~GBs</span>
            {busy === "zarr" && (
              <div className="pbar"><i style={{ width: `${Math.round(prog * 100)}%` }} /></div>
            )}
          </div>
          <div style={{ height: 12 }} />
          <div
            className={`dz${busy === "geff" ? " busy" : ""}`}
            onClick={() => !busy && geffInput.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => e.key === "Enter" && geffInput.current?.click()}
          >
            <div className="big">📁</div>
            <b>{busy === "geff" ? `Uploading… ${Math.round(prog * 100)}%` : "Drop .geff folder"}</b>
            <span>expects nodes/ids · props · edges</span>
            {busy === "geff" && (
              <div className="pbar"><i style={{ width: `${Math.round(prog * 100)}%` }} /></div>
            )}
          </div>
          <input
            ref={zarrInput}
            type="file"
            multiple
            hidden
            onChange={(e) => {
              pickFolder("zarr", e.target.files);
              e.target.value = "";
            }}
          />
          <input
            ref={geffInput}
            type="file"
            multiple
            hidden
            onChange={(e) => {
              pickFolder("geff", e.target.files);
              e.target.value = "";
            }}
          />
          {note && <p className="muted">{note}</p>}
          {lastUp && (
            <div className="chips">
              <div className="chip">{lastUp.n_files}<small>FILES</small></div>
              <div className="chip">{fmtBytes(lastUp.bytes)}<small>BYTES</small></div>
              <div className="chip">{lastUp.pair.ready ? "READY" : "HALF"}<small>PAIR</small></div>
            </div>
          )}
        </div>
        <div className="panel">
          <h3>SPECIMEN VAULT · CLICK TO SWITCH MOVIES</h3>
          <p className="sub">Lab movies + your uploads. Each movie keeps its own results — hop back and forth freely.</p>
          {datasets.length === 0 && <p className="muted">No movies found — upload folders or add vault data.</p>}
          {datasets.map((d) => {
            const inc = !d.has_zarr || !d.has_geff;
            return (
              <div key={d.movie_id} className={`vrow${movieId === d.movie_id ? " sel" : ""}`}>
                {registry[d.movie_id] && <span className="pill">UPLOADED</span>}
                {inc && <span className="pill warn">INCOMPLETE</span>}
                <span className="nm" onClick={() => selectMovie(d.movie_id)} role="button" tabIndex={0}
                  onKeyDown={(e) => e.key === "Enter" && selectMovie(d.movie_id)}>
                  {d.movie_id}
                </span>
                <span className="sz">
                  {d.has_zarr ? ".zarr ✓" : ".zarr ✗"} · {d.has_geff ? ".geff ✓" : ".geff ✗"}
                </span>
                {registry[d.movie_id] && (
                  <button type="button" className="btn sm" onClick={() => removeMovie(d.movie_id)}>
                    ✕
                  </button>
                )}
              </div>
            );
          })}
          <div className="chips">
            <div className="chip">{datasets.length}<small>MOVIES</small></div>
            <div className="chip">{Object.keys(registry).length}<small>UPLOADED</small></div>
          </div>
        </div>
      </div>

      <div style={{ maxWidth: 1100, marginTop: 18 }}>
        <MoviePlayer movieId={movieId} />
      </div>
      <div style={{ maxWidth: 1100, marginTop: 18 }}>
        <DataStory movieId={movieId} />
      </div>

      <div className="panel" style={{ maxWidth: 1100, marginTop: 18 }}>
        <h3>SERVER FILES · WHAT'S REALLY ON DISK</h3>
        <p className="sub">Diagnostics: the exact folders the backend sees{serverFiles ? ` at ${serverFiles.root}` : ""}.</p>
        {!serverFiles && (
          <button type="button" className="btn sm" onClick={loadServerFiles}>INSPECT SERVER FOLDER</button>
        )}
        {serverFiles && (
          <>
            {(serverFiles.entries || []).map((e) => (
              <div className="flag" key={e.name}>
                <span className="fdot" style={{ background: e.has_zarr_json ? "var(--acc)" : "var(--acc2)" }} />
                {e.name} · {e.is_dir ? "folder" : "file"} · {e.n_children} items · {e.has_zarr_json ? "zarr.json ✓" : "no zarr.json"}
              </div>
            ))}
            {(serverFiles.entries || []).length === 0 && <p className="muted">Train folder is empty.</p>}
          </>
        )}
      </div>
    </StageShell>
  );
}
