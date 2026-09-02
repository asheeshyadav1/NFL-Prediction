"use client";

import { useEffect, useRef } from "react";

type Play = { d: string; num: string; title: string; body: string };

const PLAYS: Play[] = [
  {
    d: "M40 100 L40 62 L40 24",
    num: "First",
    title: "Run the number",
    body:
      "A model that has watched nine seasons of box scores projects both players, before it is shown a single headline.",
  },
  {
    d: "M40 100 L40 62 L120 62 L196 30",
    num: "Then",
    title: "Read the report",
    body:
      "It pulls the official injury report: who practiced, who is questionable, who is a game-time call. It cannot see the projection while it looks.",
  },
  {
    d: "M40 100 L40 70 L96 44 L150 70 L206 26",
    num: "Only then",
    title: "Say it in English",
    body:
      "The write-up is checked against the projection before you ever see it. Quote a number nobody gave it and the whole thing gets flagged.",
  },
];

export function Plays() {
  const ref = useRef<HTMLDivElement>(null);

  // Routes draw themselves when they scroll into view. The dash length is the
  // path's own measured length, so it has to be read from the live DOM.
  useEffect(() => {
    const root = ref.current;
    if (!root) return;
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((en) => {
          if (!en.isIntersecting) return;
          en.target.classList.add("in");
          en.target.querySelectorAll<SVGPathElement>("[data-route]").forEach((path) => {
            path.style.setProperty("--len", String(path.getTotalLength()));
            path.classList.add("drawn");
          });
          io.unobserve(en.target);
        });
      },
      { threshold: 0.35 },
    );
    root.querySelectorAll(".fadein").forEach((el) => io.observe(el));
    return () => io.disconnect();
  }, []);

  return (
    <div className="plays" ref={ref}>
      {PLAYS.map((p) => (
        <div className="play fadein" key={p.num}>
          <svg viewBox="0 0 240 118" aria-hidden="true">
            <line className="hash" x1="0" y1="96" x2="240" y2="96" />
            <line className="hash" x1="0" y1="62" x2="240" y2="62" />
            <line className="hash" x1="0" y1="28" x2="240" y2="28" />
            <path className="route" data-route d={p.d} />
            <circle className="dot" cx="40" cy="100" r="5" />
          </svg>
          <div className="num">{p.num}</div>
          <h3>{p.title}</h3>
          <p>{p.body}</p>
        </div>
      ))}
    </div>
  );
}
