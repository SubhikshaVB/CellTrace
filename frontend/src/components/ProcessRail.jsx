import { STAGES } from "../dive/stages";

export default function ProcessRail({ status, active }) {
  return (
    <aside className="rail" aria-label="Process rail">
      {STAGES.map((s) => (
        <a
          key={s.id}
          href={`#${s.id}`}
          className={`${active === s.id ? "on " : ""}${status[s.id] || ""}`}
        >
          <span className="d" />
          {s.num} {s.name}
        </a>
      ))}
    </aside>
  );
}
