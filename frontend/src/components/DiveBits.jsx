export function fmtBytes(n) {
  if (n === null || n === undefined) return "···";
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

export function Say({ children }) {
  return <p className="say">🎤 {children}</p>;
}

export function DepthMeter({ meters }) {
  return (
    <div className="depth">
      DEPTH <b>{meters}</b> m · T-SPAN <b>100</b>
    </div>
  );
}

export function StageShell({
  id,
  num,
  kicker,
  name,
  desc,
  runLabel,
  status,
  onRun,
  say,
  children,
}) {
  const running = status.state === "run";
  return (
    <section className="stage" id={id}>
      <div className="ghost">{num}</div>
      <div className="stage-head">
        <div className="num">{kicker}</div>
        <h2>{name}</h2>
        <p>{desc}</p>
        <div className="headrow">
          <button type="button" className="btn hot" onClick={onRun} disabled={running}>
            {running ? "◉ RUNNING…" : runLabel}
          </button>
          <div className={`pbar${running ? " indet" : ""}`}>
            <i style={{ width: `${Math.round((status.progress || 0) * 100)}%` }} />
          </div>
          <span className={`status${status.state === "done" ? " done" : ""}`}>
            {status.note || "○ waiting"}
          </span>
        </div>
      </div>
      {children}
      {say && <Say>{say}</Say>}
    </section>
  );
}

// Per-movie result memory: switching specimens restores each movie's last
// results instead of wiping them. Keyed snapshots, no React state inside.
const MOVIE_CACHE = new Map();
export function saveSnap(stage, mid, snap) {
  if (!mid || !snap) return;
  if (!MOVIE_CACHE.has(stage)) MOVIE_CACHE.set(stage, new Map());
  MOVIE_CACHE.get(stage).set(mid, snap);
}
export function loadSnap(stage, mid) {
  return (MOVIE_CACHE.get(stage)?.get(mid) || null);
}
