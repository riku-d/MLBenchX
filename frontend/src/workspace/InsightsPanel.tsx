/**
 * The right panel: health, warnings, and what the last compile changed.
 *
 * The health score is the number the whole workspace is oriented around, so it is never
 * shown alone. Under it sit the six dimensions that produced it, and under those the
 * arithmetic — because a score a user cannot take apart is a score they have to take on
 * faith, and this product's entire argument is against that.
 *
 * When a compile has run, every dimension shows its before and after. That comparison
 * is the honest answer to "did this help?", including when the answer is no.
 */

import { useState } from "react";
import {
  Activity,
  AlertTriangle,
  ChevronDown,
  Info,
  Scale,
  ShieldCheck,
  ShieldAlert,
} from "lucide-react";
import type { Health, HealthDimension, Workspace } from "../api";
import { Badge, Delta, Meter, ScoreRing, Tabs } from "../components/ui";

function toneFor(score: number) {
  if (score >= 90) return "good" as const;
  if (score >= 75) return "accent" as const;
  if (score >= 55) return "warn" as const;
  return "danger" as const;
}

/** The evidence behind one dimension, rendered from whatever facts it reported. */
function DimensionFacts({ dimension }: { dimension: HealthDimension }) {
  const facts = dimension.facts as Record<string, unknown>;
  const rows: [string, string][] = [];

  const push = (label: string, value: unknown) => {
    if (value === undefined || value === null) return;
    if (Array.isArray(value)) {
      if (value.length === 0) return;
      rows.push([label, value.slice(0, 4).map(String).join(", ")]);
      return;
    }
    if (typeof value === "object") return;
    rows.push([label, typeof value === "number" ? value.toLocaleString() : String(value)]);
  };

  push("Missing cells", facts.missing_cells);
  push("Columns with gaps", facts.columns_with_gaps);
  push("Outliers", facts.outliers);
  push("Infinite values", facts.infinite);
  push("Duplicate rows", facts.duplicate_rows);
  push("Constant columns", facts.constant_columns);
  push("Identifier-like", facts.identifier_columns);
  push("Numeric columns", facts.numeric_columns);
  push("Scale disparity", facts.scale_disparity);
  push("Features", facts.feature_count);

  const redundant = facts.redundant_pairs as { columns: string[]; correlation: number }[] | undefined;
  const leaks = facts.leakage_suspects as { column: string; correlation: number }[] | undefined;

  return (
    <div className="mt-2 space-y-1">
      {rows.map(([label, value]) => (
        <div key={label} className="flex justify-between gap-3 text-[11.5px]">
          <span className="text-faint">{label}</span>
          <span className="tnum truncate font-mono text-muted">{value}</span>
        </div>
      ))}
      {leaks?.map((leak) => (
        <p key={leak.column} className="text-[11.5px] text-danger">
          {leak.column} correlates {leak.correlation} with the target
        </p>
      ))}
      {redundant?.slice(0, 3).map((pair) => (
        <p key={pair.columns.join()} className="font-mono text-[11.5px] text-faint">
          {pair.columns[0]} ≈ {pair.columns[1]} ({pair.correlation})
        </p>
      ))}
    </div>
  );
}

function Dimensions({
  health,
  comparison,
}: {
  health: Health;
  comparison?: { key: string; before: number; after: number; delta: number }[];
}) {
  const [expanded, setExpanded] = useState<string | null>(null);
  const deltas = new Map((comparison ?? []).map((d) => [d.key, d]));

  return (
    <ul className="space-y-1">
      {health.dimensions.map((dimension) => {
        const open = expanded === dimension.key;
        const change = deltas.get(dimension.key);
        return (
          <li key={dimension.key} className="rounded-lg px-2 py-1.5 transition-colors hover:bg-raised">
            <button
              onClick={() => setExpanded(open ? null : dimension.key)}
              aria-expanded={open}
              className="w-full text-left"
            >
              <div className="mb-1.5 flex items-baseline gap-2">
                <span className="flex-1 truncate text-[12.5px] text-text">{dimension.label}</span>
                {change && change.delta !== 0 ? (
                  <Delta value={Math.round(change.delta)} />
                ) : null}
                <span className="tnum font-mono text-[11.5px] text-muted">
                  {dimension.score.toFixed(0)}
                </span>
                <ChevronDown
                  size={11}
                  className={`shrink-0 text-faint transition-transform ${open ? "rotate-180" : ""}`}
                />
              </div>
              <Meter
                value={dimension.score}
                label={dimension.label}
                showValue={false}
                tone={toneFor(dimension.score)}
              />
            </button>

            {open && (
              <div className="fade mt-2.5 border-t border-line pt-2.5">
                <p className="text-[12px] leading-relaxed text-muted">{dimension.summary}</p>
                <p className="mt-1.5 text-[11.5px] text-faint">{dimension.description}</p>
                <DimensionFacts dimension={dimension} />
                <p className="mt-2 font-mono text-[11px] text-faint">
                  {dimension.score.toFixed(1)} × {dimension.weight.toFixed(2)} ={" "}
                  {dimension.contribution.toFixed(2)} points
                </p>
              </div>
            )}
          </li>
        );
      })}
    </ul>
  );
}

export function InsightsPanel({ workspace }: { workspace: Workspace }) {
  const [tab, setTab] = useState<"health" | "issues" | "changes">("health");

  const compilation = workspace.latest_compilation;
  const succeeded = compilation?.status === "success";
  const after = succeeded ? (compilation!.health_after as Health) : null;
  const shown = after ?? workspace.baseline_health;
  const comparison = compilation?.result?.comparison;

  const issues = shown.issues ?? [];
  const balance = shown.class_balance;

  return (
    <div className="flex h-full flex-col">
      <div className="flex h-11 shrink-0 items-center justify-between gap-2 border-b border-line px-3.5">
        <h2 className="font-display text-[11px] tracking-[0.13em] text-faint uppercase">
          Insights
        </h2>
        <Tabs
          label="Insights view"
          value={tab}
          onChange={setTab}
          options={[
            { value: "health", label: "Health" },
            { value: "issues", label: "Issues", count: issues.length },
            { value: "changes", label: "Changes" },
          ]}
        />
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        {/* --------------------------------------------------------- health */}
        {tab === "health" && (
          <div className="p-3.5">
            <div className="flex items-center gap-4">
              <ScoreRing score={shown.score} size={92} />
              <div className="min-w-0">
                <p className="font-display text-[15px] font-medium text-text">{shown.rating}</p>
                <p className="mt-0.5 text-[12px] text-muted">
                  {after ? "after preprocessing" : "as uploaded"}
                </p>
                {comparison && (
                  <div className="mt-2">
                    <Delta value={comparison.score.delta} />
                    <p className="tnum mt-0.5 font-mono text-[11px] text-faint">
                      {comparison.score.before} → {comparison.score.after}
                    </p>
                  </div>
                )}
              </div>
            </div>

            <div className="mt-4 border-t border-line pt-3">
              <Dimensions health={shown} comparison={comparison?.dimensions} />
            </div>

            <details className="mt-3 border-t border-line pt-3">
              <summary className="cursor-pointer list-none text-[12px] text-muted transition-colors hover:text-text">
                How this number was worked out ›
              </summary>
              <div className="fade mt-2.5">
                <p className="mb-2 text-[11.5px] leading-relaxed text-faint">
                  {shown.formula.description}
                </p>
                <table className="w-full">
                  <tbody>
                    {shown.formula.terms.map((term) => (
                      <tr key={term.label} className="text-[11.5px]">
                        <td className="py-0.5 text-muted">{term.label}</td>
                        <td className="tnum py-0.5 text-right font-mono text-faint">
                          {term.score.toFixed(1)} × {term.weight.toFixed(2)}
                        </td>
                        <td className="tnum py-0.5 pl-2 text-right font-mono text-text">
                          {term.contribution.toFixed(2)}
                        </td>
                      </tr>
                    ))}
                    <tr className="border-t border-line text-[11.5px]">
                      <td className="py-1 text-text">Total</td>
                      <td />
                      <td className="tnum py-1 pl-2 text-right font-mono text-text">
                        {shown.formula.total.toFixed(2)}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </details>
          </div>
        )}

        {/* --------------------------------------------------------- issues */}
        {tab === "issues" && (
          <div className="space-y-2 p-3.5">
            {issues.length === 0 && !balance?.imbalanced ? (
              <div className="flex flex-col items-center py-10 text-center">
                <ShieldCheck size={20} className="text-good" />
                <p className="mt-2.5 text-[13.5px] text-text">Nothing needs your attention</p>
                <p className="mt-1 max-w-[30ch] text-[12.5px] text-muted">
                  Every dimension scored above the concern threshold.
                </p>
              </div>
            ) : (
              <>
                {issues.map((issue) => (
                  <div key={issue.dimension} className="card p-3">
                    <div className="flex items-center gap-2">
                      {issue.severity === "warning" ? (
                        <AlertTriangle size={12} className="shrink-0 text-warn" />
                      ) : (
                        <Info size={12} className="shrink-0 text-info" />
                      )}
                      <span className="flex-1 truncate text-[13px] font-medium text-text">
                        {issue.label}
                      </span>
                      <span className="tnum font-mono text-[11.5px] text-muted">
                        {issue.score.toFixed(0)}
                      </span>
                    </div>
                    <p className="mt-1.5 text-[12.5px] leading-relaxed text-muted">{issue.text}</p>
                  </div>
                ))}

                {balance?.imbalanced && (
                  <div className="card p-3">
                    <div className="flex items-center gap-2">
                      <Scale size={12} className="shrink-0 text-warn" />
                      <span className="flex-1 text-[13px] font-medium text-text">
                        Class imbalance
                      </span>
                      <Badge tone="warn">{balance.ratio}:1</Badge>
                    </div>
                    <p className="mt-1.5 text-[12.5px] leading-relaxed text-muted">
                      {balance.summary}
                    </p>
                    <div className="mt-2.5 space-y-1">
                      {balance.classes.slice(0, 5).map((klass) => (
                        <div key={klass.value} className="flex items-center gap-2">
                          <span className="w-16 truncate font-mono text-[11px] text-muted">
                            {klass.value}
                          </span>
                          <Meter
                            value={klass.share * 100}
                            label={`${klass.value} share`}
                            showValue={false}
                            tone="info"
                          />
                          <span className="tnum w-9 text-right font-mono text-[11px] text-faint">
                            {(klass.share * 100).toFixed(0)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {/* -------------------------------------------------------- changes */}
        {tab === "changes" && (
          <div className="p-3.5">
            {!compilation ? (
              <div className="flex flex-col items-center py-10 text-center">
                <Activity size={20} className="text-faint" />
                <p className="mt-2.5 text-[13.5px] text-text">Nothing compiled yet</p>
                <p className="mt-1 max-w-[32ch] text-[12.5px] text-muted">
                  Run the pipeline to see what it changed and what it cost.
                </p>
              </div>
            ) : compilation.status === "error" ? (
              <div className="card border-[color-mix(in_oklab,var(--color-danger)_35%,transparent)] p-3">
                <div className="flex items-center gap-2">
                  <ShieldAlert size={13} className="text-danger" />
                  <span className="text-[13px] font-medium text-text">
                    The last run failed
                  </span>
                </div>
                <p className="mt-1.5 font-mono text-[12px] text-danger">
                  {compilation.error?.type}: {compilation.error?.message}
                </p>
              </div>
            ) : (
              <>
                <div className="grid grid-cols-2 gap-2">
                  {[
                    ["Rows", compilation.result.rows_before, compilation.result.rows_after],
                    ["Columns", compilation.result.columns_before, compilation.result.columns_after],
                  ].map(([label, before, after]) => (
                    <div key={String(label)} className="card p-2.5">
                      <p className="text-[11px] text-faint">{label as string}</p>
                      <p className="tnum mt-1 font-mono text-[15px] text-text">
                        {(before as number)?.toLocaleString()}
                        <span className="mx-1 text-faint">→</span>
                        {(after as number)?.toLocaleString()}
                      </p>
                    </div>
                  ))}
                </div>

                <p className="mt-4 mb-2 font-display text-[11px] tracking-[0.13em] text-faint uppercase">
                  What each step did
                </p>
                <ol className="space-y-1">
                  {(compilation.result.steps ?? []).map((step, index) => (
                    <li key={`${step.step}-${index}`} className="card px-2.5 py-2">
                      <div className="flex items-baseline gap-2">
                        <span className="tnum w-3.5 shrink-0 font-mono text-[10px] text-faint">
                          {index + 1}
                        </span>
                        <span className="flex-1 font-mono text-[11.5px] text-accent">
                          {step.step}
                        </span>
                        <span className="tnum shrink-0 font-mono text-[10.5px] text-faint">
                          {step.rows.toLocaleString()}×{step.columns}
                        </span>
                      </div>
                      <p className="mt-0.5 pl-5 text-[12px] leading-snug text-muted">
                        {step.action}
                      </p>
                    </li>
                  ))}
                </ol>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
