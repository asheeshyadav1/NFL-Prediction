"use client";

import { useEffect, useRef } from "react";
import { useReducedMotion } from "./useReducedMotion";

type Stat = {
  target: number;
  suffix: string;
  decimals: number;
  title: string;
  detail: string;
};

const STATS: Stat[] = [
  {
    target: 78.1,
    suffix: "%",
    decimals: 1,
    title: "Start/sit calls it gets right",
    detail: "Graded on the 2025 season, which it never saw during training.",
  },
  {
    target: 8.6,
    suffix: "%",
    decimals: 1,
    title: 'Closer than "he’s been hot lately"',
    detail: "Beats the last-three-games average everyone actually uses.",
  },
  {
    target: 51,
    suffix: "k",
    decimals: 0,
    title: "Player-weeks on the tape",
    detail: "Ten seasons of box scores, 2016 through 2025.",
  },
];

function Counter({ stat }: { stat: Stat }) {
  const ref = useRef<HTMLDivElement>(null);
  const reduce = useReducedMotion();

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const { target, suffix, decimals } = stat;

    if (reduce) {
      el.innerHTML = `${target.toFixed(decimals)}<small>${suffix}</small>`;
      return;
    }

    let raf = 0;
    let t0: number | null = null;
    const dur = 1400;
    const step = (ts: number) => {
      if (t0 === null) t0 = ts;
      const k = Math.min(1, (ts - t0) / dur);
      const e = 1 - Math.pow(1 - k, 3);
      el.innerHTML = `${(target * e).toFixed(decimals)}<small>${suffix}</small>`;
      if (k < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [stat, reduce]);

  return (
    <div className="cell">
      <div className="n" ref={ref}>
        0<small>{stat.suffix}</small>
      </div>
      <div className="t">{stat.title}</div>
      <div className="d">{stat.detail}</div>
    </div>
  );
}

export function Scoreboard() {
  return (
    <div className="board">
      {STATS.map((s) => (
        <Counter key={s.title} stat={s} />
      ))}
    </div>
  );
}
