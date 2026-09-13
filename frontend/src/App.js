import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import { STAGES } from "./dive/stages";
import TopBar from "./components/TopBar";
import ProcessRail from "./components/ProcessRail";
import Hero from "./components/Hero";
import Specimen from "./stages/Specimen";
import Validate from "./stages/Validate";
import { DepthMeter, Say } from "./components/DiveBits";

const COMING = {
  s02: "A 3-D box is cropped around every annotated cell — the raw material for everything the networks will ever see.",
  s03: "Percentile clipping + rescaling develops every patch like photographic film.",
  s04: "The prototype encoder writes a fingerprint vector per cell — lookalikes land close together.",
  s05: "Each cell is wired to its K nearest neighbors — the social graph the GAT reasons over.",
  s06: "The GAT trains live, in front of you — loss falls while attention heads learn whom to trust.",
  s07: "Embeddings + graph + trained weights → frame-to-frame matches, each with an honest confidence.",
  s08: "Every run seals a manifest — data, parameters, metrics, weights. Then: the road ahead.",
};

function StageComing({ id, num, kicker, name }) {
  return (
    <section className="stage coming" id={id}>
      <div className="ghost">{num}</div>
      <div className="stage-head">
        <div className="num">{kicker}</div>
        <h2>{name}</h2>
        <p>{COMING[id]}</p>
        <div className="headrow">
          <span className="status">○ landing next build</span>
        </div>
      </div>
    </section>
  );
}

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
    report("dive", "");
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
  }, [diving, report]);

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
        {STAGES.slice(2).map((s) => (
          <StageComing key={s.id} id={s.id} num={s.num} kicker={s.kicker} name={s.name} />
        ))}
        <footer className="exit">
          <div>
            <div className="kicker">CELLTRACE · DIVE BUILD 1</div>
            <h2>Shell + Specimen + Validate are live.</h2>
            <p className="muted">Stages 02–08 land in the next builds — same shell, same rail.</p>
            <Say>
              <b>Say:</b> “Two stages down, the pipeline already runs on your data — and the dive
              keeps going.”
            </Say>
          </div>
        </footer>
      </main>
    </div>
  );
}
