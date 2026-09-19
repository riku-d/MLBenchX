import { useEffect, useState } from "react";

const PHASES = [
  "Reading your file",
  "Checking data quality",
  "Preparing columns",
  "Splitting the data",
];

/**
 * The working state.
 *
 * The server usually finishes in well under a second. These labels are the real
 * phases, stepped on a timer so the screen reads as progress rather than a
 * flash — App holds the state for a short floor before showing results.
 */
export function Working() {
  const [phase, setPhase] = useState(0);

  useEffect(() => {
    const timer = window.setInterval(
      () => setPhase((current) => Math.min(current + 1, PHASES.length - 1)),
      300,
    );
    return () => window.clearInterval(timer);
  }, []);

  return (
    <div className="mx-auto max-w-[640px] px-5 pt-28 pb-24">
      <ol className="space-y-3.5" aria-live="polite" aria-label="Progress">
        {PHASES.map((label, index) => {
          const done = index < phase;
          const active = index === phase;
          return (
            <li key={label} className="flex items-center gap-3">
              <span
                aria-hidden
                className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-full border text-[11px] ${
                  done
                    ? "border-xray bg-xray text-paper"
                    : active
                      ? "border-xray text-xray"
                      : "border-slate/30 text-transparent"
                }`}
              >
                {done ? "✓" : "•"}
              </span>
              <span
                className={`font-display text-[19px] ${
                  done || active ? "text-ink" : "text-slate/55"
                }`}
              >
                {label}
              </span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
