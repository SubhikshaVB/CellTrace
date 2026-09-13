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
