import { useState } from "react";
import type { ColumnProfile, Profile } from "../api";

/**
 * The per-column profile table.
 *
 * One row per column of the user's file, showing what we measured and what it means
 * for that column's fate. The ordering decision worth stating: rows are *not* sorted
 * by severity. They keep the file's own column order, because someone reading this is
 * looking for a column they already have in mind, and hunting for it in a table that
 * has been rearranged on their behalf is worse than seeing the worst one second.
 *
 * Colour follows the app's rule. `xray` is reserved for a changed value and the active
 * control, so it appears nowhere here; `ochre` carries every warning, and everything
 * else is `slate`. A flag chip is not decoration — a column with no chips has nothing
 * wrong with it, and the eye should be able to skip it.
 */

function percent(rate: number): string {
  if (rate === 0) return "0";
  if (rate < 0.001) return "<0.1%";
  return `${(rate * 100).toFixed(rate < 0.1 ? 1 : 0)}%`;
}

/** Warnings get the warm tone; statements of fact stay quiet. */
const QUIET_FLAGS = new Set(["free text", "date/time"]);

function Flag({ label }: { label: string }) {
  const quiet = QUIET_FLAGS.has(label);
  return (
    <span
      className={`rounded px-1.5 py-0.5 text-[12px] whitespace-nowrap ${
        quiet ? "bg-slate/10 text-slate" : "bg-ochre/10 text-ochre"
      }`}
    >
      {label}
    </span>
  );
}

function Row({ record }: { record: ColumnProfile }) {
  const { column, kind_label, n_missing, missing_rate, n_unique, outliers } = record;

  return (
    <tr className="border-t border-slate/12 align-top">
      <td className="py-2.5 pr-3">
        <span className="font-mono text-[14px] break-all text-ink">{column}</span>
        {record.is_target && (
          <span className="mt-1 block text-[12px] text-slate">being predicted</span>
        )}
        {!record.used && !record.is_target && (
          <span className="mt-1 block text-[12px] text-slate">set aside</span>
        )}
      </td>

      <td className="py-2.5 pr-3 text-[14px] whitespace-nowrap text-slate">{kind_label}</td>

      <td className="py-2.5 pr-3 text-right font-mono text-[14px] tabular-nums whitespace-nowrap">
        {n_missing === 0 ? (
          <span className="text-slate/50">—</span>
        ) : (
          <span className={missing_rate > 0.5 ? "text-ochre" : "text-ink"}>
            {percent(missing_rate)}
          </span>
        )}
      </td>

      <td className="py-2.5 pr-3 text-right font-mono text-[14px] text-ink tabular-nums whitespace-nowrap">
        {n_unique.toLocaleString()}
      </td>

      <td className="py-2.5 pr-3 text-right font-mono text-[14px] tabular-nums whitespace-nowrap">
        {outliers === null ? (
          <span className="text-slate/50">—</span>
        ) : outliers.count === 0 ? (
          <span className="text-slate/50">0</span>
        ) : (
          <span className="text-ink" title={`outside ${outliers.lower} to ${outliers.upper}`}>
            {outliers.count.toLocaleString()}
          </span>
        )}
      </td>

      <td className="py-2.5">
        <div className="flex flex-wrap gap-1">
          {record.flags.map((flag) => (
            <Flag key={flag} label={flag} />
          ))}
        </div>
      </td>
    </tr>
  );
}

export function ColumnTable({ profile }: { profile: Profile }) {
  const [open, setOpen] = useState(false);
  const flagged = profile.columns.filter((c) => c.flags.length > 0).length;

  return (
    <div className="rounded-xl border border-slate/15 bg-paper">
      <button
        onClick={() => setOpen((was) => !was)}
        aria-expanded={open}
        className="flex w-full items-center justify-between px-5 py-4 text-left sm:px-6"
      >
        <span className="text-[16px] text-ink">
          {profile.columns.length} column{profile.columns.length === 1 ? "" : "s"}
          {flagged > 0 && (
            <span className="text-slate">
              {" "}
              · {flagged} with something worth a look
            </span>
          )}
        </span>
        <span aria-hidden className="font-mono text-[13px] text-slate">
          {open ? "hide" : "show"}
        </span>
      </button>

      {open && (
        <div className="overflow-x-auto border-t border-slate/15 px-5 pb-4 sm:px-6">
          <table className="w-full min-w-[520px] border-collapse">
            <thead>
              <tr className="text-[12px] tracking-[0.08em] text-slate uppercase">
                <th className="py-2.5 pr-3 text-left font-medium">Column</th>
                <th className="py-2.5 pr-3 text-left font-medium">Holds</th>
                <th className="py-2.5 pr-3 text-right font-medium">Blank</th>
                <th className="py-2.5 pr-3 text-right font-medium">Distinct</th>
                <th className="py-2.5 pr-3 text-right font-medium" title="Values far outside the usual range, by the interquartile rule">
                  Outliers
                </th>
                <th className="py-2.5 text-left font-medium">Notes</th>
              </tr>
            </thead>
            <tbody>
              {profile.columns.map((record) => (
                <Row key={record.column} record={record} />
              ))}
            </tbody>
          </table>

          <p className="mt-3 max-w-[70ch] text-[13px] text-slate">
            Outliers are values more than 1.5 interquartile ranges outside the middle
            half of the column. They are counted, never removed — an implausible value
            and an unusually large real one look identical from here.
          </p>
        </div>
      )}
    </div>
  );
}

/**
 * Dataset-wide warnings: a column that looks like the answer, a lopsided target, two
 * columns saying the same thing. Each is a finding the user has to act on themselves,
 * so each says what we saw and leaves the decision with them.
 */
export function ProfileWarnings({ profile }: { profile: Profile }) {
  const { leakage_flags, class_balance, correlated_pairs } = profile;
  const pairs = correlated_pairs.slice(0, 3);

  if (leakage_flags.length === 0 && !class_balance && pairs.length === 0) return null;

  return (
    <div className="mt-4 space-y-3">
      {leakage_flags.map((flag) => (
        <div
          key={flag.column}
          className="rounded-xl border border-ochre/35 bg-ochre/8 px-5 py-4 sm:px-6"
        >
          <p className="font-display text-[15px] text-ochre">
            <span className="font-mono">{flag.column}</span> may already contain the answer
          </p>
          <p className="mt-1.5 max-w-[68ch] text-[15px] text-ink">{flag.reason}</p>
        </div>
      ))}

      {class_balance && (
        <div className="rounded-xl border border-ochre/35 bg-ochre/8 px-5 py-4 sm:px-6">
          <p className="font-display text-[15px] text-ochre">
            One answer is {class_balance.ratio}× more common than the other
          </p>
          <p className="mt-1.5 max-w-[68ch] text-[15px] text-ink">{class_balance.reason}</p>
        </div>
      )}

      {pairs.length > 0 && (
        <div className="rounded-xl border border-slate/15 bg-paper px-5 py-4 sm:px-6">
          <p className="font-display text-[15px] text-ink">
            {pairs.length === 1 ? "Two columns tell" : "Some columns tell"} you nearly the
            same thing
          </p>
          <ul className="mt-2 space-y-1.5">
            {pairs.map((pair) => (
              <li key={pair.columns.join("~")} className="text-[15px] text-slate">
                <span className="font-mono text-[14px] text-ink">
                  {pair.columns[0]} · {pair.columns[1]}
                </span>{" "}
                — correlation {pair.correlation.toFixed(2)}
              </li>
            ))}
          </ul>
          <p className="mt-2 max-w-[68ch] text-[14px] text-slate">
            Both were kept. Which one to drop depends on what they mean, which is
            yours to decide.
          </p>
        </div>
      )}
    </div>
  );
}

/** The health score's arithmetic, itemised — shown on demand, not by default. */
export function HealthBreakdown({ profile }: { profile: Profile }) {
  const [open, setOpen] = useState(false);
  const deductions = profile.quality.breakdown;
  if (deductions.length === 0) return null;

  return (
    <div className="mt-4 border-t border-slate/15 pt-3">
      <button
        onClick={() => setOpen((was) => !was)}
        aria-expanded={open}
        className="text-[14px] text-slate underline-offset-4 hover:underline"
      >
        {open ? "Hide the arithmetic" : "How was this worked out?"}
      </button>

      {open && (
        <table className="mt-3 w-full border-collapse">
          <tbody>
            <tr className="text-[14px]">
              <td className="py-1 text-slate">Starting score</td>
              <td className="py-1 text-right font-mono text-ink tabular-nums">100</td>
            </tr>
            {deductions.map((item) => (
              <tr key={item.factor} className="text-[14px]">
                <td className="py-1 text-slate">{item.label}</td>
                <td className="py-1 text-right font-mono text-ink tabular-nums">
                  −{item.points_lost.toFixed(1)}
                </td>
              </tr>
            ))}
            <tr className="border-t border-slate/15 text-[14px]">
              <td className="py-1.5 text-ink">Health score</td>
              <td className="py-1.5 text-right font-mono text-ink tabular-nums">
                {profile.quality.score}
              </td>
            </tr>
          </tbody>
        </table>
      )}
    </div>
  );
}
