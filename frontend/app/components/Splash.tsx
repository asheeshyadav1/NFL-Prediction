"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useReducedMotion } from "./useReducedMotion";

// Once per tab session: a refresh inside the session goes straight to the app,
// a fresh tab plays it again. Kept out of localStorage on purpose so the thing
// is demoable without clearing site data.
const SEEN_KEY = "fg-splash-seen";
const RUN_MS = 2600;
const REDUCED_MS = 900;

function alreadySeen(): boolean {
  try {
    return sessionStorage.getItem(SEEN_KEY) === "1";
  } catch {
    // Private windows and blocked site data throw here. Showing the splash is
    // the harmless outcome, so treat a failed read as "not seen".
    return false;
  }
}

function markSeen(): void {
  try {
    sessionStorage.setItem(SEEN_KEY, "1");
  } catch {
    /* nothing to do -- it just plays again next time */
  }
}

export function Splash() {
  // null = undecided. Rendering nothing until the effect runs keeps the server
  // markup and the first client paint identical, and stops a returning visitor
  // seeing a frame of splash before it is dismissed.
  const [show, setShow] = useState<boolean | null>(null);
  const [leaving, setLeaving] = useState(false);
  const reduce = useReducedMotion();
  const timers = useRef<number[]>([]);

  const dismiss = useCallback(() => {
    setLeaving(true);
    markSeen();
    // Let the wipe finish before the node goes away.
    timers.current.push(window.setTimeout(() => setShow(false), 420));
  }, []);

  useEffect(() => {
    if (alreadySeen()) {
      setShow(false);
      return;
    }
    setShow(true);
  }, []);

  useEffect(() => {
    if (show !== true) return;
    const t = timers.current;
    t.push(window.setTimeout(dismiss, reduce ? REDUCED_MS : RUN_MS));

    const onKey = () => dismiss();
    window.addEventListener("keydown", onKey);
    // The overlay covers the page; keep the app behind it from scrolling.
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    return () => {
      t.forEach(clearTimeout);
      t.length = 0;
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [show, reduce, dismiss]);

  if (show !== true) return null;

  return (
    <div
      className={`splash${leaving ? " out" : ""}${reduce ? " still" : ""}`}
      onClick={dismiss}
      role="button"
      tabIndex={0}
      aria-label="Skip intro"
    >
      <div className="splash-turf" aria-hidden="true" />

      <div className="splash-stage" aria-hidden="true">
        <svg className="splash-posts" viewBox="0 0 220 130">
          <path className="post-base" d="M110 130 L110 74" />
          <path className="post-bar" d="M52 74 L168 74" />
          <path className="post-arm" d="M52 74 L52 12" />
          <path className="post-arm" d="M168 74 L168 12" />
        </svg>
        <div className="splash-ball" />
      </div>

      <div className="splash-copy">
        <div className="splash-mark">
          <span>Fourth</span>
          <span className="amp">&amp;</span>
          <span>Goal</span>
        </div>
        <p className="splash-tag">Start / sit, settled</p>
      </div>

      <div className="splash-clock" aria-hidden="true">
        <i />
      </div>

      <span className="splash-skip">Click anywhere to skip</span>
    </div>
  );
}
