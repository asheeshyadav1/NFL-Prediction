"use client";

import { useEffect, useState } from "react";

type Target = "home" | "app" | "how";

const TARGETS: Record<Target, string> = {
  home: "view-home",
  app: "view-app",
  how: "how",
};

function show(name: Target) {
  const el = document.getElementById(TARGETS[name]);
  if (!el) return;
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  el.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
  history.replaceState(null, "", `#${name}`);
}

export function Go({
  to,
  className = "btn",
  children,
}: {
  to: Target;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <button className={className} onClick={() => show(to)}>
      {children}
    </button>
  );
}

export function TopBar() {
  // The underline follows whichever half of the page you are looking at.
  const [atApp, setAtApp] = useState(false);

  useEffect(() => {
    const el = document.getElementById("view-app");
    if (!el) return;
    const mark = () =>
      setAtApp(el.getBoundingClientRect().top <= window.innerHeight * 0.45);
    mark();
    window.addEventListener("scroll", mark, { passive: true });
    return () => window.removeEventListener("scroll", mark);
  }, []);

  // Deep links land on the section they name.
  useEffect(() => {
    const hash = window.location.hash.slice(1);
    if (hash === "app" || hash === "how") show(hash);
  }, []);

  return (
    <div className="top">
      <div className="wrap">
        <div className="logo">
          <span className="pill" />
          Fourth &amp; Goal
        </div>
        <button className="tab" aria-current={!atApp} onClick={() => show("home")}>
          The Pitch
        </button>
        <button className="tab" aria-current={atApp} onClick={() => show("app")}>
          Set Your Lineup
        </button>
      </div>
    </div>
  );
}
