import type { Step } from "../api";
import { SplitBar, TransformStrip } from "./TransformStrip";

/**
 * One step in the sequence.
 *
 * The number is on a rail down the left edge with a rule connecting it to the
 * next card, because the order genuinely matters — these things happened one
 * after another, and the layout should say so.
 */
export function StepCard({ step, isLast, index }: { step: Step; isLast: boolean; index: number }) {
  return (
    <li
      className="rise relative grid grid-cols-[2.25rem_1fr] gap-x-4 pb-7 sm:grid-cols-[2.75rem_1fr] sm:gap-x-5"
      style={{ animationDelay: `${index * 60}ms` }}
    >
      {/* The rail: number, then a rule running down to the next step. */}
      <div className="relative flex flex-col items-center">
        <span className="z-10 flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-slate/25 bg-paper font-mono text-[14px] text-ink">
          {step.n}
        </span>
        {!isLast && <span aria-hidden className="mt-1 w-px flex-1 bg-slate/25" />}
      </div>

      <div className="min-w-0 rounded-xl border border-slate/15 bg-paper px-5 py-5 sm:px-6">
        <div className="flex flex-wrap items-start justify-between gap-x-4 gap-y-2">
          <h3 className="font-display text-[19px] leading-snug font-medium text-ink">
            {step.title}
          </h3>
          {step.fitted && (
            <span className="mt-0.5 shrink-0 rounded-full border border-slate/25 px-2.5 py-0.5 text-[12px] whitespace-nowrap text-slate">
              learned from training rows only
            </span>
          )}
        </div>

        {step.columns.length > 0 && (
          <p className="mt-1.5 flex flex-wrap gap-x-2 gap-y-1">
            {step.columns.map((column) => (
              <span key={column} className="font-mono text-[13px] text-slate">
                {column}
              </span>
            ))}
          </p>
        )}

        <p className="mt-3 max-w-[62ch] text-[17px] leading-[1.65] text-ink">{step.explanation}</p>

        {step.strip && <TransformStrip strip={step.strip} />}
        {step.split && <SplitBar {...step.split} />}
      </div>
    </li>
  );
}
