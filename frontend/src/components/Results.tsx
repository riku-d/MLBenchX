import { useState } from "react";
import type { Analysis } from "../api";
import { downloadUrl } from "../api";
import { StepCard } from "./StepCard";

function Section({
  title,
  children,
  aside,
}: {
  title: string;
  children: React.ReactNode;
  aside?: React.ReactNode;
}) {
  return (
    <section className="mt-14">
      <div className="flex items-baseline justify-between gap-4">
        <h2 className="font-display text-[13px] tracking-[0.14em] text-slate uppercase">{title}</h2>
        {aside}
      </div>
      <div className="mt-4">{children}</div>
    </section>
  );
}

function Fact({
  label,
  value,
  title,
  wide,
}: {
  label: string;
  value: React.ReactNode;
  title?: string;
  wide?: boolean;
}) {
  return (
    <div className={`min-w-0 ${wide ? "col-span-2" : ""}`}>
      <dt className="text-[13px] text-slate">{label}</dt>
      <dd className="mt-0.5 truncate font-mono text-[15px] text-ink" title={title}>
        {value}
      </dd>
    </div>
  );
}

function scoreReading(score: number): string {
  if (score >= 90) return "Very good — nothing here needs your attention";
  if (score >= 75) return "Good, with a couple of things worth checking";
  if (score >= 55) return "Usable, but there are gaps you should know about";
  return "Rough — the results below will only be as good as this data";
}

export function Results({
  data,
  onRetarget,
  onReset,
  busy,
}: {
  data: Analysis;
  onRetarget: (column: string) => void;
  onReset: () => void;
  busy: boolean;
}) {
  const [showDropped, setShowDropped] = useState(false);
  const [changing, setChanging] = useState(false);

  return (
    <div className="mx-auto max-w-[800px] px-5 pt-12 pb-28">
      <div className="flex items-baseline justify-between gap-4">
        <h1 className="font-display text-[26px] font-semibold tracking-tight text-ink">
          ML-BenchX
        </h1>
        <button
          onClick={onReset}
          className="text-[15px] text-xray underline-offset-4 hover:underline"
        >
          Start over
        </button>
      </div>

      {/* 1 — What we found */}
      <Section title="What we found">
        <div className="rounded-xl border border-slate/15 bg-paper px-5 py-5 sm:px-6">
          <dl className="grid grid-cols-2 gap-x-6 gap-y-5 sm:grid-cols-5">
            <Fact label="File" value={data.file.name} title={data.file.name} wide />
            <Fact label="Rows" value={data.file.rows.toLocaleString()} />
            <Fact label="Columns" value={data.file.columns} />
            <Fact
              label="Split"
              value={`${data.result.train_rows.toLocaleString()} / ${data.result.test_rows.toLocaleString()}`}
            />
          </dl>

          <div className="mt-5 border-t border-slate/15 pt-5">
            <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
              <span className="text-[15px] text-slate">Predicting</span>
              {changing ? (
                <select
                  autoFocus
                  defaultValue={data.target.column}
                  onChange={(e) => {
                    setChanging(false);
                    if (e.target.value !== data.target.column) onRetarget(e.target.value);
                  }}
                  onBlur={() => setChanging(false)}
                  className="rounded-md border border-xray bg-paper px-2 py-1 font-mono text-[15px] text-ink focus:outline-none"
                >
                  {data.columns.map((column) => (
                    <option key={column} value={column}>
                      {column}
                    </option>
                  ))}
                </select>
              ) : (
                <>
                  <span className="rounded-md bg-bone px-2 py-1 font-mono text-[15px] text-ink">
                    {data.target.column}
                  </span>
                  <button
                    onClick={() => setChanging(true)}
                    disabled={busy}
                    className="text-[14px] text-xray underline-offset-4 hover:underline disabled:opacity-40"
                  >
                    change
                  </button>
                </>
              )}
              <span className="text-[15px] text-slate">· {data.target.task}</span>
            </div>

            <p className="mt-2 text-[14px] text-slate">
              {data.target.inferred ? data.target.note : "You chose this column."}{" "}
              {Object.entries(data.target.values).length > 0 && (
                <>
                  Values:{" "}
                  {Object.entries(data.target.values)
                    .map(([key, count]) => `${key} (${count.toLocaleString()})`)
                    .join(", ")}
                  .
                </>
              )}
            </p>
          </div>

          {data.cleaning.length > 0 && (
            <ul className="mt-5 space-y-1.5 border-t border-slate/15 pt-5">
              {data.cleaning.map((line) => (
                <li key={line} className="text-[15px] text-slate">
                  {line}
                </li>
              ))}
            </ul>
          )}
        </div>
      </Section>

      {/* 2 — Data quality */}
      <Section title="Data quality">
        <div className="rounded-xl border border-slate/15 bg-paper px-5 py-5 sm:px-6">
          <div className="flex items-baseline gap-3">
            <span className="font-mono text-[38px] leading-none text-ink tabular-nums">
              {data.quality.score}
            </span>
            <span className="font-mono text-[15px] text-slate">/ 100</span>
          </div>
          <p className="mt-2 text-[16px] text-ink">{scoreReading(data.quality.score)}</p>

          {data.quality.issues.length > 0 && (
            <ul className="mt-4 space-y-2 border-t border-slate/15 pt-4">
              {data.quality.issues.map((issue) => (
                <li key={issue.text} className="flex gap-2.5 text-[15px]">
                  <span
                    aria-hidden
                    className={`mt-[7px] h-1.5 w-1.5 shrink-0 rounded-full ${
                      issue.severity === "warning" ? "bg-ochre" : "bg-slate/45"
                    }`}
                  />
                  <span className={issue.severity === "warning" ? "text-ochre" : "text-slate"}>
                    {issue.text}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </Section>

      {/* 3 — Columns we set aside */}
      {data.dropped.length > 0 && (
        <Section title="Columns we set aside">
          <div className="rounded-xl border border-slate/15 bg-paper">
            <button
              onClick={() => setShowDropped((open) => !open)}
              aria-expanded={showDropped}
              className="flex w-full items-center justify-between px-5 py-4 text-left sm:px-6"
            >
              <span className="text-[16px] text-ink">
                {data.dropped.length} column{data.dropped.length === 1 ? "" : "s"} set aside
              </span>
              <span aria-hidden className="font-mono text-[13px] text-slate">
                {showDropped ? "hide" : "show"}
              </span>
            </button>
            {showDropped && (
              <ul className="space-y-3.5 border-t border-slate/15 px-5 py-4 sm:px-6">
                {data.dropped.map((item) => (
                  <li key={item.column}>
                    <span className="font-mono text-[14px] text-ink">{item.column}</span>
                    <p className="mt-0.5 max-w-[62ch] text-[15px] text-slate">{item.reason}</p>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </Section>
      )}

      {/* 4 — The centrepiece */}
      <Section
        title="What we did to your data"
        aside={
          <span className="font-mono text-[13px] text-slate">
            {data.steps.length} steps · {data.result.features} columns out
          </span>
        }
      >
        <ol className="mt-1">
          {data.steps.map((step, index) => (
            <StepCard
              key={step.n}
              step={step}
              index={index}
              isLast={index === data.steps.length - 1}
            />
          ))}
        </ol>
      </Section>

      {/* 5 — Take it with you */}
      <Section title="Take it with you">
        <div className="flex flex-wrap gap-3">
          <a
            href={downloadUrl(data.run_id, "data.csv")}
            className="rounded-lg border border-slate/25 bg-paper px-4 py-3 text-[16px] text-ink hover:border-slate/50"
          >
            Download the cleaned data
            <span className="mt-0.5 block font-mono text-[13px] text-slate">
              {data.result.train_rows.toLocaleString()} rows · {data.result.features} columns
            </span>
          </a>
          <a
            href={downloadUrl(data.run_id, "script.py")}
            className="rounded-lg border border-slate/25 bg-paper px-4 py-3 text-[16px] text-ink hover:border-slate/50"
          >
            Download the Python
            <span className="mt-0.5 block font-mono text-[13px] text-slate">
              runs on its own · reproduces all of the above
            </span>
          </a>
        </div>
      </Section>
    </div>
  );
}
