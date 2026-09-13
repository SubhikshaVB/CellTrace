import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import "./dive/stages.css";
import { STAGES } from "./dive/stages";
import TopBar from "./components/TopBar";
import ProcessRail from "./components/ProcessRail";
import Hero from "./components/Hero";
import Specimen from "./stages/Specimen";
import Validate from "./stages/Validate";
import Extract from "./stages/Extract";
import Standardize from "./stages/Standardize";
import Embed from "./stages/Embed";
import Connect from "./stages/Connect";
import Train from "./stages/Train";
import Track from "./stages/Track";
import Export from "./stages/Export";
import { DepthMeter, Say } from "./components/DiveBits";

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

export default function App() {
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem("ct-theme") || "dark";
    } catch (e) {
      return "dark";
    }
  });
  const [movieId, setMovieId] = useState("");
  const [status, setStatus] = useState({});
  const [active, setActive] = useState("");
  const [meters, setMeters] = useState(0);
  const [diving, setDiving] = useState(false);
  const [auto, setAuto] = useState(true);
  const [talk, setTalk] = useState(true);
  const runners = useRef({});
  const abortRef = useRef(false);
  const autoRef = useRef(true);
  autoRef.current = auto;

  useEffect(() => {
    document.documentElement.dataset.theme = theme === "dark" ? "abyss" : "surface";
    try {
      localStorage.setItem("ct-theme", theme);
    } catch (e) {}
  }, [theme]);

  const register = useCallback((id, fn) => {
    runners.current[id] = fn;
  }, []);
  const report = useCallback((id, st) => {
    setStatus((m) => ({ ...m, [id]: st }));
  }, []);
  const selectMovie = useCallback((id) => setMovieId(id), []);

  const ctx = useMemo(
    () => ({ movieId, selectMovie, register, report }),
    [movieId, selectMovie, register, report]
  );

  useEffect(() => {
    const obs = new IntersectionObserver(
      (es) => {
        es.forEach((e) => {
          if (e.isIntersecting) setActive(e.target.id);
        });
      },
      { rootMargin: "-40% 0px -40% 0px" }
    );
    STAGES.forEach((s) => {
      const el = document.getElementById(s.id);
      if (el) obs.observe(el);
    });
    return () => obs.disconnect();
  }, []);

  useEffect(() => {
    const onScroll = () => {
      const h = document.documentElement;
      const f = h.scrollTop / Math.max(h.scrollHeight - h.clientHeight, 1);
      setMeters(Math.round(f * 100));
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const dive = useCallback(async () => {
    if (diving) {
      abortRef.current = true;
      return;
    }
    setDiving(true);
    abortRef.current = false;
    for (const s of STAGES) {
      if (abortRef.current) break;
      if (autoRef.current) {
        document.getElementById(s.id)?.scrollIntoView({ behavior: "smooth", block: "start" });
        await wait(1400);
        if (abortRef.current) break;
      }
      try {
        const fn = runners.current[s.id];
        if (fn) await fn();
      } catch (e) {
        break;
      }
    }
    setDiving(false);
  }, [diving]);

  return (
    <div className={talk ? "" : "talk-off"}>
      <TopBar
        theme={theme}
        onTheme={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
        diving={diving}
        onDive={dive}
        auto={auto}
        onAuto={() => setAuto((a) => !a)}
        talk={talk}
        onTalk={() => setTalk((t) => !t)}
      />
      <ProcessRail status={status} active={active} />
      <DepthMeter meters={meters} />
      <main>
        <Hero movieId={movieId} diving={diving} onDive={dive} />
        <Specimen ctx={ctx} />
        <Validate ctx={ctx} />
        <Extract ctx={ctx} />
        <Standardize ctx={ctx} />
        <Embed ctx={ctx} />
        <Connect ctx={ctx} />
        <Train ctx={ctx} />
        <Track ctx={ctx} />
        <Export ctx={ctx} />
        <footer className="exit">
          <div>
            <div className="kicker">DIVE COMPLETE · 100 M</div>
            <h2>You surfaced with the verdict.</h2>
            <p className="muted">Every number above was computed live from your specimen.</p>
            <Say>
              <b>Say:</b> “From raw pixels to matched tracks — the whole pipeline, on real data, in
              one dive.”
            </Say>
          </div>
        </footer>
      </main>
    </div>
  );
}
