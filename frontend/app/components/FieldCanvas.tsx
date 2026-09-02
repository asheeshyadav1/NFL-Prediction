"use client";

import { useEffect, useRef } from "react";
import { useReducedMotion } from "./useReducedMotion";

// The hero backdrop: yard lines, and a ball that keeps getting thrown downfield.
export function FieldCanvas() {
  const ref = useRef<HTMLCanvasElement>(null);
  const reduce = useReducedMotion();

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;

    let w = 0;
    let h = 0;
    const dpr = Math.min(2, window.devicePixelRatio || 1);

    function size() {
      const r = cv!.getBoundingClientRect();
      w = r.width;
      h = r.height;
      cv!.width = w * dpr;
      cv!.height = h * dpr;
      ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    size();
    window.addEventListener("resize", size);

    const css = getComputedStyle(document.documentElement);
    const tone = (name: string) => css.getPropertyValue(name).trim();

    function drawField() {
      ctx!.clearRect(0, 0, w, h);
      ctx!.strokeStyle = tone("--grid");
      ctx!.lineWidth = 1;
      const step = Math.max(56, w / 14);
      for (let x = step; x < w; x += step) {
        ctx!.beginPath();
        ctx!.moveTo(x, 0);
        ctx!.lineTo(x, h);
        ctx!.stroke();
      }
      // hash marks along two rows, the way a broadcast frame reads
      ctx!.beginPath();
      [h * 0.38, h * 0.72].forEach((y) => {
        for (let x = step / 2; x < w; x += step / 5) {
          ctx!.moveTo(x, y - 4);
          ctx!.lineTo(x, y + 4);
        }
      });
      ctx!.stroke();
    }

    function drawBall(x: number, y: number, spin: number, alpha: number) {
      ctx!.save();
      ctx!.translate(x, y);
      ctx!.rotate(spin);
      ctx!.globalAlpha = alpha;
      ctx!.fillStyle = tone("--accent-fill");
      ctx!.beginPath();
      ctx!.ellipse(0, 0, 15, 9.5, 0, 0, Math.PI * 2);
      ctx!.fill();
      ctx!.strokeStyle = tone("--accent-ink");
      ctx!.lineWidth = 1.6;
      ctx!.beginPath();
      ctx!.moveTo(-6, 0);
      ctx!.lineTo(6, 0);
      ctx!.stroke();
      ctx!.beginPath();
      for (let i = -4; i <= 4; i += 2.6) {
        ctx!.moveTo(i, -2.6);
        ctx!.lineTo(i, 2.6);
      }
      ctx!.stroke();
      ctx!.restore();
    }

    if (reduce) {
      drawField();
      return () => window.removeEventListener("resize", size);
    }

    let raf = 0;
    let start: number | null = null;
    const PERIOD = 5200;

    function frame(ts: number) {
      if (start === null) start = ts;
      const k = ((ts - start) % PERIOD) / PERIOD;
      drawField();

      // a throw: flat in x, parabolic in y, released just past the left hash
      const ease = Math.min(1, k / 0.72);
      const x = -40 + (w + 90) * ease;
      const y = h * 0.86 - Math.sin(Math.PI * ease) * h * 0.52;
      const alpha = ease >= 1 ? Math.max(0, 1 - (k - 0.72) / 0.14) : 1;

      if (alpha > 0) {
        ctx!.save();
        ctx!.setLineDash([2, 9]);
        ctx!.strokeStyle = tone("--grid");
        ctx!.lineWidth = 2;
        ctx!.beginPath();
        for (let s = 0; s <= ease; s += 0.02) {
          const tx = -40 + (w + 90) * s;
          const ty = h * 0.86 - Math.sin(Math.PI * s) * h * 0.52;
          if (s === 0) ctx!.moveTo(tx, ty);
          else ctx!.lineTo(tx, ty);
        }
        ctx!.stroke();
        ctx!.restore();
        drawBall(x, y, ease * Math.PI * 7, alpha);
      }
      raf = requestAnimationFrame(frame);
    }
    raf = requestAnimationFrame(frame);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", size);
    };
  }, [reduce]);

  return <canvas id="field" ref={ref} aria-hidden="true" />;
}
