export default function TopBar({
  theme,
  onTheme,
  diving,
  onDive,
  auto,
  onAuto,
  talk,
  onTalk,
}) {
  return (
    <div className="topbar">
      <div className="brand">◉ CELLTRACE</div>
      <span className="livebadge">● LIVE DATA · every number computed</span>
      <span className="sp" />
      <button type="button" className="btn" onClick={onTalk} title="Toggle presenter talk-tracks">
        {talk ? "🎤 ON" : "🎤 OFF"}
      </button>
      <button type="button" className="btn" onClick={onAuto} title="Toggle auto-advance scrolling">
        {auto ? "▼ AUTO" : "■ MANUAL"}
      </button>
      <button type="button" className="btn" onClick={onTheme}>
        {theme === "dark" ? "☀ LIGHT MODE" : "🌙 DARK MODE"}
      </button>
      <button type="button" className="btn hot" onClick={onDive}>
        {diving ? "■ ABORT DIVE" : "▼ BEGIN DIVE"}
      </button>
    </div>
  );
}
