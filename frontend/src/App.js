import React from "react";
import "./App.css";

// Placeholder shell — the Dive (stages 00–08) lands here next.
export default function App() {
  return (
    <div className="ct-shell">
      <header className="ct-top">
        <span className="ct-brand">◉ CELLTRACE</span>
        <span className="ct-note">Dive build in progress</span>
      </header>
      <main className="ct-main">
        <h1>CellTrace</h1>
        <p>
          The new Dive experience is being built here — stages 00–08 land next,
          all wired to the real pipeline backend.
        </p>
        <p>
          Backend API docs: <code>http://127.0.0.1:8000/docs</code>
        </p>
      </main>
    </div>
  );
}
