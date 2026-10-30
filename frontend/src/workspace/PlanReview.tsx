/**
 * The plan review screen: see every decision, disagree with any of it, then run.
 *
 * This is the screen the product's claim rests on. "Every automated recommendation is
 * visible, explained, optional and reversible" is a statement about what happens here:
 * each column shows what will be done to it, the numbers that drove that, and a dropdown
 * offering exactly the alternatives that make sense for its kind.
 *
 * Three decisions worth stating, because the obvious alternatives are worse.
 *
 * **Every column is listed, including the ones nothing happens to.** A table of only the
 * interesting columns leaves a reader unable to tell "nothing needed doing" from "we
 * forgot about it".
 *
 * **Edits are collected, then sent once.** Each save writes a new immutable plan version
 * on the server, so sending one request per dropdown would turn five changes into five
 * versions and bury the one the user actually meant. The pending edits are held here and
 * applied together, which is also what makes "discard" a local operation.
 *
 * **The dropdown offers what the server sent.** `actions_for_kind` travels with the plan
 * rather than being duplicated in this file; a copy here would be the one that goes
 * stale, and the failure would be an action the backend rejects after the user chose it.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  api,
  ApiError,
  type ColumnPlan,
  type ColumnProfile,
  type Histogram,
  type Plan,
  type PlanAction,
  type PlanWarning,
  type Run,
} from "../api";
import {
  Badge,
  Button,
  Dialog,
  EmptyState,
  Fact,
  PanelHeading,
  Select,
  Spinner,
} from "../components/ui";

/** Plain-English labels. The user never needs to read an action's internal name. */
const ACTION_LABELS: Record<PlanAction, string> = {
  keep: "Keep as it is",
  drop: "Remove the column",
  impute_median: "Fill gaps with the median",
  impute_mean: "Fill gaps with the mean",
  impute_mode: "Fill gaps with the most common value",
  impute_constant: "Fill gaps with a fixed value",
  encode_binary: "Turn into a single 0/1 column",
  encode_onehot: "One column per value",
  encode_ordinal: "One number per value",
  encode_target: "Average outcome per value",
  encode_frequency: "How often each value occurs",
  datetime_expand: "Split the date into parts",
  passthrough: "Leave untouched",
};

const KIND_LABELS: Record<string, string> = {
  numeric: "number",
  categorical: "category",
  boolean: "true/false",
  datetime: "date/time",
  text: "free text",
  constant: "single value",
};

const SEVERITY_TONE: Record<string, "neutral" | "warn" | "danger"> = {
  info: "neutral",
  warn: "warn",
  critical: "danger",
};

function pct(value: number): string {
  return `${(value * 100).toFixed(value < 0.01 && value > 0 ? 1 : 0)}%`;
}

/**
 * One warning, rendered by severity.
 *
 * The suggestion is shown, never hidden behind a disclosure: a warning that names a
 * problem and stops leaves the reader to work out what to do, which is the kind of
 * warning people learn to click past.
 */
function WarningCard({ warning }: { warning: PlanWarning }) {
  const tone = SEVERITY_TONE[warning.severity] ?? "neutral";
  const border =
    tone === "danger"
      ? "border-danger/40 bg-danger/5"
      : tone === "warn"
        ? "border-warn/40 bg-warn/5"
        : "border-line bg-raised";

  return (
    <div className={`rounded-md border px-3 py-2 ${border}`}>
      <div className="flex items-start gap-2">
        <Badge tone={tone}>{warning.severity}</Badge>
        <div className="min-w-0 flex-1">
          <p className="text-sm text-text">{warning.message}</p>
          {warning.suggestion ? (
            <p className="mt-1 text-xs text-muted">{warning.suggestion}</p>
          ) : null}
        </div>
      </div>
    </div>
  );
}

/**
 * The distribution of one column, drawn from the bins the backend computed.
 *
 * Server-side on purpose: the browser holds one page of rows and would draw a different
 * distribution on every page. A histogram of 25 rows is not a histogram of the column.
 */
function ColumnHistogram({ histogram }: { histogram: Histogram | null | undefined }) {
  const bars = useMemo(() => {
    if (!histogram?.counts?.length) return [];
    if (histogram.kind === "categorical") {
      return histogram.labels.map((label, index) => ({
        label,
        count: histogram.counts[index] ?? 0,
      }));
    }
    return histogram.counts.map((count, index) => ({
      label: histogram.edges[index]?.toPrecision(3) ?? "",
      count,
    }));
  }, [histogram]);

  // An empty or absent distribution is a real state, not a failure: a column that is
  // entirely blank, or one the profile could not summarise, has nothing to draw.
  if (!bars.length) {
    return (
      <p className="text-xs text-faint">
        No distribution to draw for this column.
      </p>
    );
  }

  return (
    <div className="h-32 w-full">
      <ResponsiveContainer width="100%" height="100%">
        {/* Colours are passed as props, not classes: recharts renders SVG and sets
            `fill` on the element directly, so a utility class is overwritten. The
            values come from the same design tokens the rest of the app uses. */}
        <BarChart data={bars} margin={{ top: 4, right: 4, bottom: 0, left: -24 }}>
          <CartesianGrid
            strokeDasharray="2 4"
            stroke="var(--color-line)"
            vertical={false}
          />
          <XAxis
            dataKey="label"
            tick={{ fontSize: 9, fill: "var(--color-muted)" }}
            stroke="var(--color-line-strong)"
            interval="preserveStartEnd"
          />
          <YAxis
            tick={{ fontSize: 9, fill: "var(--color-muted)" }}
            stroke="var(--color-line-strong)"
            width={36}
          />
          <Tooltip
            contentStyle={{
              fontSize: 11,
              background: "var(--color-raised)",
              border: "1px solid var(--color-line-strong)",
              borderRadius: 4,
            }}
            cursor={{ fill: "var(--color-line)" }}
            labelFormatter={(label) => `from ${label}`}
          />
          <Bar dataKey="count" radius={[2, 2, 0, 0]} fill="var(--color-accent)" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/** One row of the plan table. */
function ColumnRow({
  column,
  pending,
  actions,
  onChange,
  expanded,
  onToggle,
  profile,
}: {
  column: ColumnPlan;
  pending: PlanAction | undefined;
  actions: PlanAction[];
  onChange: (action: PlanAction) => void;
  expanded: boolean;
  onToggle: () => void;
  profile: ColumnProfile | undefined;
}) {
  const action = pending ?? column.action;
  const changed = pending !== undefined && pending !== column.action;
  const isTarget = column.alternatives.length === 0 && action === "passthrough";

  const options = useMemo(
    () =>
      // The current action is always offered even when it is not in the list for this
      // kind, so a plan can always be left exactly as it is.
      Array.from(new Set<PlanAction>([action, ...actions])).map((value) => ({
        value,
        label: ACTION_LABELS[value] ?? value,
      })),
    [action, actions],
  );

  return (
    <>
      <tr
        className={`border-t border-line align-top ${changed ? "bg-accent/5" : ""}`}
        data-testid={`column-${column.column}`}
      >
        <td className="py-2 pr-3">
          <button
            type="button"
            onClick={onToggle}
            className="text-left font-mono text-sm text-text hover:text-accent"
            aria-expanded={expanded}
          >
            {column.column}
          </button>
          {column.original_name && column.original_name !== column.column ? (
            <p className="font-mono text-[10px] text-faint">
              was “{column.original_name}”
            </p>
          ) : null}
        </td>

        <td className="py-2 pr-3 text-xs text-muted">
          {KIND_LABELS[column.kind] ?? column.kind}
        </td>

        <td className="py-2 pr-3 font-mono text-xs text-muted">
          {column.stats.n_missing ? `${pct(Number(column.stats.missing_rate))} blank` : "—"}
        </td>

        <td className="py-2 pr-3 font-mono text-xs text-muted">
          {column.stats.n_unique ?? "—"}
        </td>

        <td className="py-2 pr-3">
          {isTarget ? (
            <Badge tone="accent">the answer</Badge>
          ) : (
            <Select
              value={action}
              options={options}
              onChange={(next) => onChange(next)}
              label={`What happens to ${column.column}`}
            />
          )}
          <div className="mt-1 flex flex-wrap gap-1">
            {column.overridden && !changed ? <Badge tone="accent">your choice</Badge> : null}
            {changed ? <Badge tone="accent">unsaved</Badge> : null}
            {column.add_indicator ? <Badge>+ was-missing flag</Badge> : null}
            {column.rare_categories.length ? (
              <Badge>{column.rare_categories.length} rare → Other</Badge>
            ) : null}
          </div>
        </td>

        <td className="py-2 text-xs text-muted">
          {column.reason}
          {column.warning && column.warning.message.trim() !== column.reason.trim() ? (
            <span className="mt-1 block">
              <Badge tone={SEVERITY_TONE[column.warning.severity] ?? "neutral"}>
                {column.warning.message}
              </Badge>
            </span>
          ) : null}
          {column.warning?.suggestion &&
          column.warning.message.trim() === column.reason.trim() ? (
            <span className="mt-1 block text-[11px] text-faint">
              {column.warning.suggestion}
            </span>
          ) : null}
        </td>
      </tr>

      {expanded ? (
        <tr className="border-t border-line/50 bg-sunken">
          <td colSpan={6} className="px-3 py-3">
            <div className="grid gap-4 md:grid-cols-2">
              <div>
                <p className="mb-2 text-xs font-medium text-muted">
                  What this was measured from
                </p>
                <div className="grid grid-cols-2 gap-2">
                  {Object.entries(column.stats).map(([key, value]) => (
                    <Fact
                      key={key}
                      label={key.replace(/_/g, " ")}
                      value={String(value)}
                    />
                  ))}
                </div>
                <p className="mt-2 text-[11px] text-faint">
                  Measured on the training rows only — the held-out rows take no part in
                  deciding what happens to this column.
                </p>
              </div>
              <div>
                <p className="mb-2 text-xs font-medium text-muted">Distribution</p>
                <ColumnHistogram histogram={profile?.histogram} />
              </div>
            </div>
          </td>
        </tr>
      ) : null}
    </>
  );
}

/** Everything the run measured about the split itself. */
function RunReport({ run }: { run: Run }) {
  if (run.status === "failed") {
    return (
      <div className="rounded-md border border-danger/40 bg-danger/5 p-4">
        <PanelHeading>The run did not finish</PanelHeading>
        <p className="mt-2 text-sm text-text">{run.error?.message}</p>
        <p className="mt-1 text-xs text-muted">
          Nothing was produced, so there is nothing to download. The plan is unchanged —
          adjust a column and run it again.
        </p>
      </div>
    );
  }

  const { drift, balance, baseline } = run.checks;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Fact label="training rows" value={run.split.train_rows.toLocaleString()} />
        <Fact label="held out" value={run.split.test_rows.toLocaleString()} />
        <Fact label="features" value={String(run.n_features)} />
        <Fact
          label="stratified"
          value={run.split.stratified ? "yes" : "no"}
        />
      </div>

      {run.warnings.length ? (
        <div className="space-y-2">
          <PanelHeading>What the run found</PanelHeading>
          {run.warnings.map((warning, index) => (
            <WarningCard key={`${warning.kind}-${index}`} warning={warning} />
          ))}
        </div>
      ) : (
        <p className="text-sm text-muted">
          The run raised nothing worth flagging: the folds resemble each other and the
          target is learnable from the prepared columns.
        </p>
      )}

      {baseline ? (
        <div className="rounded-md border border-line bg-raised p-3">
          <PanelHeading>Sanity baseline</PanelHeading>
          <p className="mt-1 text-sm text-text">{baseline.verdict}</p>
          <div className="mt-2 grid grid-cols-3 gap-2">
            <Fact label="guessing" value={baseline.baseline.toFixed(3)} />
            <Fact label={baseline.model} value={baseline.model_score.toFixed(3)} />
            <Fact label={`lift (${baseline.metric})`} value={baseline.lift.toFixed(3)} />
          </div>
          <p className="mt-2 text-xs text-faint">{baseline.note}</p>
        </div>
      ) : null}

      {drift ? (
        <div className="rounded-md border border-line bg-raised p-3">
          <PanelHeading>Train/test drift</PanelHeading>
          <p className="mt-1 text-sm text-text">
            {drift.flagged} of {drift.tested} features are distributed differently across
            the two folds.
          </p>
          <p className="mt-1 text-xs text-faint">{drift.note}</p>
          {drift.features.length ? (
            <ul className="mt-2 space-y-1">
              {drift.features.slice(0, 5).map((feature) => (
                <li key={feature.column} className="font-mono text-xs text-muted">
                  {feature.column} — p={feature.p_value}
                </li>
              ))}
            </ul>
          ) : null}
        </div>
      ) : null}

      {balance?.classes?.length ? (
        <div className="rounded-md border border-line bg-raised p-3">
          <PanelHeading>Class balance in both folds</PanelHeading>
          <table className="mt-2 w-full text-xs">
            <thead className="text-faint">
              <tr>
                <th className="text-left font-normal">class</th>
                <th className="text-right font-normal">train</th>
                <th className="text-right font-normal">test</th>
                <th className="text-right font-normal">difference</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              {balance.classes.map((row) => (
                <tr key={row.value}>
                  <td className="py-0.5 text-text">{row.value}</td>
                  <td className="py-0.5 text-right text-muted">{pct(row.train_share)}</td>
                  <td className="py-0.5 text-right text-muted">{pct(row.test_share)}</td>
                  <td
                    className={`py-0.5 text-right ${
                      Math.abs(row.delta) > 0.05 ? "text-warn" : "text-faint"
                    }`}
                  >
                    {row.delta > 0 ? "+" : ""}
                    {pct(row.delta)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}

      <div>
        <PanelHeading>What this run produced</PanelHeading>
        <div className="mt-2 flex flex-wrap gap-2">
          {run.artifacts.map((name) => (
            <a
              key={name}
              href={api.artifactUrl(run.id, name)}
              className="rounded border border-line bg-raised px-2 py-1 font-mono text-xs text-muted hover:border-accent hover:text-accent"
            >
              {name}
            </a>
          ))}
        </div>
        {run.output_hash ? (
          <p className="mt-2 font-mono text-[10px] text-faint">
            output hash {run.output_hash.slice(0, 32)}… — the same plan on the same data
            produces this digest every time.
          </p>
        ) : null}
      </div>
    </div>
  );
}

export function PlanReview({
  datasetId,
  onExit,
}: {
  datasetId: string;
  onExit: () => void;
}) {
  const [plan, setPlan] = useState<Plan | null>(null);
  const [profiles, setProfiles] = useState<Record<string, ColumnProfile>>({});
  const [run, setRun] = useState<Run | null>(null);
  const [edits, setEdits] = useState<Record<string, PlanAction>>({});
  const [expanded, setExpanded] = useState<string | null>(null);
  const [busy, setBusy] = useState<"loading" | "saving" | "running" | null>("loading");
  const [error, setError] = useState<string | null>(null);
  const [confirmRun, setConfirmRun] = useState(false);

  // Only one load may be in flight. Without this, React's development double-invoke of
  // effects fires two loads at once; both find no plan, both propose one, and the user
  // arrives on version 2 of a plan they have never edited. A flag set inside the
  // effect would not help — the guard has to survive the re-invocation, which a ref
  // does and state does not.
  const loading = useRef(false);

  const load = useCallback(async () => {
    if (loading.current) return;
    loading.current = true;
    setBusy("loading");
    setError(null);
    try {
      let current: Plan;
      try {
        current = await api.readPlan(datasetId);
      } catch (exc) {
        // No plan has been proposed yet, which is the normal first visit.
        if (exc instanceof ApiError && exc.status === 404) {
          current = await api.proposePlan(datasetId, "");
        } else {
          throw exc;
        }
      }
      setPlan(current);
      setEdits({});

      // The distributions are the profile's, not the plan's. They are fetched
      // separately and joined by column name, so a profile that fails to load costs the
      // charts and nothing else — the plan is still reviewable without them.
      try {
        const profile = await api.readProfile(datasetId);
        setProfiles(
          Object.fromEntries(profile.columns.map((column) => [column.column, column])),
        );
      } catch {
        setProfiles({});
      }
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "The plan could not be loaded.");
    } finally {
      loading.current = false;
      setBusy(null);
    }
  }, [datasetId]);

  useEffect(() => {
    void load();
  }, [load]);

  const pendingCount = useMemo(
    () =>
      plan
        ? Object.entries(edits).filter(
            ([name, action]) => plan.columns.find((c) => c.column === name)?.action !== action,
          ).length
        : 0,
    [edits, plan],
  );

  const save = useCallback(async () => {
    if (!plan || !pendingCount) return;
    setBusy("saving");
    setError(null);
    try {
      const columns = Object.fromEntries(
        Object.entries(edits).map(([name, action]) => [name, { action }]),
      );
      setPlan(await api.editPlan(plan.id, columns));
      setEdits({});
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Those changes were refused.");
    } finally {
      setBusy(null);
    }
  }, [edits, pendingCount, plan]);

  const execute = useCallback(async () => {
    if (!plan) return;
    setConfirmRun(false);
    setBusy("running");
    setError(null);
    try {
      setRun(await api.runPlan(plan.id));
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "The run could not be started.");
    } finally {
      setBusy(null);
    }
  }, [plan]);

  if (busy === "loading") {
    return (
      <div className="flex h-screen items-center justify-center gap-2 text-muted">
        <Spinner /> Working out what to do with every column…
      </div>
    );
  }

  if (error && !plan) {
    return (
      <div className="flex h-screen items-center justify-center">
        <EmptyState
          title="The plan could not be loaded"
          description={error}
          action={<Button onClick={() => void load()}>Try again</Button>}
        />
      </div>
    );
  }

  if (!plan) return null;

  const kept = plan.columns.filter(
    (c) => c.action !== "drop" && c.action !== "passthrough",
  ).length;
  const dropped = plan.columns.filter((c) => c.action === "drop").length;

  return (
    <div className="min-h-screen bg-void text-text">
      <header className="sticky top-0 z-10 border-b border-line bg-surface/95 px-6 py-3 backdropliter">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-sm font-medium">
              Review the plan{" "}
              <span className="font-mono text-xs text-faint">v{plan.version}</span>
            </h1>
            <p className="text-xs text-muted">
              Predicting <span className="font-mono text-text">{plan.target}</span> ·{" "}
              {plan.task} · {kept} column{kept === 1 ? "" : "s"} kept, {dropped} removed
              {plan.n_overridden ? ` · ${plan.n_overridden} your choice` : ""}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="ghost" onClick={onExit}>
              Back
            </Button>
            <Button
              variant="ghost"
              onClick={() => setEdits({})}
              disabled={!pendingCount || busy !== null}
            >
              Discard changes
            </Button>
            <Button onClick={() => void save()} disabled={!pendingCount || busy !== null}>
              {busy === "saving" ? <Spinner /> : null}
              Save as v{plan.version + 1}
              {pendingCount ? ` (${pendingCount})` : ""}
            </Button>
            <Button
              variant="primary"
              onClick={() => setConfirmRun(true)}
              disabled={busy !== null}
            >
              {busy === "running" ? <Spinner /> : null}
              Run this plan
            </Button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1400px] space-y-6 px-6 py-6">
        {error ? (
          <WarningCard warning={{ severity: "critical", message: error }} />
        ) : null}

        {pendingCount ? (
          <WarningCard
            warning={{
              severity: "info",
              message: `${pendingCount} unsaved change${pendingCount === 1 ? "" : "s"}.`,
              suggestion:
                "Saving writes a new version of the plan. The version you are looking " +
                "at stays exactly as it is, so any run that used it still means what it said.",
            }}
          />
        ) : null}

        {plan.warnings.map((warning, index) => (
          <WarningCard key={`plan-${index}`} warning={warning} />
        ))}

        <section>
          <PanelHeading aside={
            <span className="text-[11px] normal-case tracking-normal text-faint">
              Including the ones nothing happens to
            </span>
          }>
            Every column, and what happens to it
          </PanelHeading>
          <div className="mt-3 overflow-x-auto rounded-md border border-line">
            <table className="w-full min-w-[900px] text-left">
              <thead className="bg-raised text-[11px] uppercase tracking-wide text-faint">
                <tr>
                  <th className="px-3 py-2 font-normal">column</th>
                  <th className="px-3 py-2 font-normal">holds</th>
                  <th className="px-3 py-2 font-normal">blank</th>
                  <th className="px-3 py-2 font-normal">distinct</th>
                  <th className="px-3 py-2 font-normal">what happens</th>
                  <th className="px-3 py-2 font-normal">why</th>
                </tr>
              </thead>
              <tbody>
                {plan.columns.map((column) => (
                  <ColumnRow
                    key={column.column}
                    column={column}
                    pending={edits[column.column]}
                    actions={plan.actions_for_kind[column.kind] ?? []}
                    onChange={(action) =>
                      setEdits((previous) => ({ ...previous, [column.column]: action }))
                    }
                    profile={profiles[column.column]}
                    expanded={expanded === column.column}
                    onToggle={() =>
                      setExpanded((previous) =>
                        previous === column.column ? null : column.column,
                      )
                    }
                  />
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {run ? (
          <section>
            <PanelHeading aside={
              <span className="text-[11px] normal-case tracking-normal text-faint">
                {run.duration_ms} ms · seed {run.split.seed} · {pct(run.split.test_size)} held out
              </span>
            }>
              Run of plan v{run.plan_version}
            </PanelHeading>
            <div className="mt-3">
              <RunReport run={run} />
            </div>
          </section>
        ) : null}
      </main>

      <Dialog
        open={confirmRun}
        title="Run this plan?"
        onClose={() => setConfirmRun(false)}
        footer={
          <>
            <Button variant="ghost" onClick={() => setConfirmRun(false)}>
              Cancel
            </Button>
            <Button variant="primary" onClick={() => void execute()}>
              Run it
            </Button>
          </>
        }
      >
        <p className="text-sm text-text">
          Version {plan.version} will be executed exactly as it stands: {kept} column
          {kept === 1 ? "" : "s"} prepared, {dropped} removed.
        </p>
        {pendingCount ? (
          <p className="mt-2 text-sm text-warn">
            You have {pendingCount} unsaved change{pendingCount === 1 ? "" : "s"}. They
            will <strong>not</strong> be included — save them first if you want them to
            run.
          </p>
        ) : null}
        <p className="mt-2 text-xs text-muted">
          Nothing is re-decided while it runs. The held-out rows are set aside before
          anything is measured or fitted.
        </p>
      </Dialog>
    </div>
  );
}
