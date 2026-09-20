/**
 * The landing page.
 *
 * It has one job: make it obvious, before anyone signs up to anything, that this tool
 * shows its working. So every section demonstrates rather than asserts — the pipeline
 * diagram is the real step list, the health panel shows the real six dimensions, the
 * code sample is what the generator actually emits. A page of claims about
 * transparency would be the wrong advertisement for a product about transparency.
 */

import { useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  Check,
  Code2,
  Download,
  FileSpreadsheet,
  GitCompare,
  Play,
  Shield,
  SlidersHorizontal,
  Sparkles,
  Upload,
} from "lucide-react";
import { Badge, Button, Meter } from "../components/ui";

const STEPS = [
  { id: "types", label: "Data Types", note: "13 columns typed" },
  { id: "missing", label: "Missing Values", note: "885 filled" },
  { id: "duplicates", label: "Duplicates", note: "none found" },
  { id: "outliers", label: "Outliers", note: "170 capped" },
  { id: "encoding", label: "Encoding", note: "5 columns" },
  { id: "scaling", label: "Scaling", note: "standard" },
  { id: "selection", label: "Feature Selection", note: "off" },
  { id: "imbalance", label: "Class Imbalance", note: "1.4 : 1" },
];

const DIMENSIONS = [
  { label: "Completeness", before: 92, after: 100 },
  { label: "Validity", before: 97, after: 99 },
  { label: "Consistency", before: 100, after: 100 },
  { label: "Uniqueness", before: 100, after: 100 },
  { label: "Feature Quality", before: 70, after: 70 },
  { label: "Distribution", before: 72, after: 67 },
];

const DETECTED = [
  "Missing values", "Duplicate rows", "Outliers", "Incorrect data types",
  "Constant columns", "High-cardinality columns", "Inconsistent formats",
  "Mixed-type columns", "Potential target leakage", "Highly correlated features",
  "Class imbalance", "Skewed distributions",
];

/** Reveals children once they scroll into view. Runs once; no exit animation. */
function Reveal({ children, delay = 0 }: { children: React.ReactNode; delay?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const [shown, setShown] = useState(false);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setShown(true);
          observer.disconnect();
        }
      },
      { rootMargin: "-60px" },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  return (
    <div
      ref={ref}
      style={{ transitionDelay: `${delay}ms` }}
      className={`transition-all duration-700 ease-out ${
        shown ? "translate-y-0 opacity-100" : "translate-y-4 opacity-0"
      }`}
    >
      {children}
    </div>
  );
}

function Section({
  eyebrow,
  title,
  lead,
  children,
  id,
}: {
  eyebrow: string;
  title: string;
  lead?: string;
  children: React.ReactNode;
  id?: string;
}) {
  return (
    <section id={id} className="border-t border-line px-6 py-20 sm:py-24">
      <div className="mx-auto max-w-[1100px]">
        <Reveal>
          <p className="font-display text-[11px] tracking-[0.16em] text-accent uppercase">
            {eyebrow}
          </p>
          <h2 className="mt-3 max-w-[20ch] font-display text-[30px] leading-[1.12] font-semibold tracking-tight text-text sm:text-[40px]">
            {title}
          </h2>
          {lead && (
            <p className="mt-4 max-w-[62ch] text-[16px] leading-relaxed text-muted sm:text-[17px]">
              {lead}
            </p>
          )}
        </Reveal>
        <div className="mt-10">{children}</div>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------------
// The product preview in the hero
// ---------------------------------------------------------------------------------

/**
 * A miniature of the real workspace.
 *
 * It cycles through the pipeline so the page is never static, but the numbers are the
 * ones the app actually produces on the bundled example. Inventing more impressive
 * figures for a screenshot would undercut the entire pitch.
 */
function HeroPreview() {
  const [active, setActive] = useState(0);
  const [score, setScore] = useState(90);

  useEffect(() => {
    const timer = setInterval(() => {
      setActive((current) => {
        const next = (current + 1) % (STEPS.length + 1);
        setScore(next === 0 ? 90 : Math.round(90 + (92 - 90) * (next / STEPS.length)));
        return next;
      });
    }, 1100);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="panel overflow-hidden shadow-2xl">
      {/* window chrome */}
      <div className="flex items-center gap-2 border-b border-line bg-raised/60 px-3.5 py-2.5">
        <div className="flex gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-line-strong" />
          <span className="h-2.5 w-2.5 rounded-full bg-line-strong" />
          <span className="h-2.5 w-2.5 rounded-full bg-line-strong" />
        </div>
        <span className="ml-2 font-mono text-[11px] text-faint">titanic.csv</span>
        <span className="ml-auto flex items-center gap-1.5 font-mono text-[11px] text-accent">
          <span className="h-1.5 w-1.5 rounded-full bg-accent" />
          {score} / 100
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-[150px_1fr] lg:grid-cols-[168px_1fr_150px]">
        {/* pipeline rail */}
        <div className="border-b border-line p-2.5 sm:border-r sm:border-b-0">
          <p className="px-1.5 pb-2 font-display text-[9px] tracking-[0.14em] text-faint uppercase">
            Pipeline
          </p>
          <ul className="space-y-0.5">
            {STEPS.map((step, index) => {
              const done = index < active;
              const running = index === active;
              return (
                <li
                  key={step.id}
                  className={`flex items-center gap-2 rounded-md px-1.5 py-1.5 transition-colors ${
                    running ? "bg-[color-mix(in_oklab,var(--color-accent)_11%,transparent)]" : ""
                  }`}
                >
                  <span
                    className={`flex h-3.5 w-3.5 shrink-0 items-center justify-center rounded-full border text-[8px] ${
                      done
                        ? "border-good bg-good/20 text-good"
                        : running
                          ? "border-accent text-accent"
                          : "border-line-strong text-faint"
                    }`}
                  >
                    {done ? <Check size={8} strokeWidth={3} /> : running ? "●" : ""}
                  </span>
                  <span
                    className={`truncate text-[11px] ${
                      done || running ? "text-text" : "text-faint"
                    }`}
                  >
                    {step.label}
                  </span>
                </li>
              );
            })}
          </ul>
        </div>

        {/* code */}
        <div className="min-w-0 border-b border-line bg-sunken p-3.5 lg:border-r lg:border-b-0">
          <div className="mb-2 flex items-center gap-2">
            <span className="font-mono text-[10px] text-muted">pipeline.py</span>
            <span className="font-mono text-[10px] text-faint">· generated</span>
          </div>
          <pre className="overflow-hidden font-mono text-[10.5px] leading-[1.75] text-muted">
            <span className="text-faint">{"# Split BEFORE any value is learned."}</span>
            {"\n"}
            <span className="text-info">train_index</span>
            {", "}
            <span className="text-info">test_index</span>
            {" = train_test_split(\n"}
            {"    df.index, test_size="}
            <span className="text-accent">0.2</span>
            {", random_state="}
            <span className="text-accent">42</span>
            {"\n)\n\n"}
            <span className="text-faint">{"# Fill values come from training rows only."}</span>
            {"\n"}
            {"imputer = SimpleImputer(strategy="}
            <span className="text-good">{'"median"'}</span>
            {")\n"}
            {"imputer.fit(df.loc["}
            <span className="text-info">train_index</span>
            {", NUMERIC])\n"}
            {"df[NUMERIC] = imputer.transform(df[NUMERIC])"}
          </pre>
        </div>

        {/* insights */}
        <div className="hidden p-3 lg:block">
          <p className="pb-2 font-display text-[9px] tracking-[0.14em] text-faint uppercase">
            Health
          </p>
          <div className="space-y-2">
            {DIMENSIONS.slice(0, 5).map((dimension, index) => (
              <div key={dimension.label}>
                <p className="mb-1 text-[10px] text-muted">{dimension.label}</p>
                <Meter
                  value={active > index ? dimension.after : dimension.before}
                  label={dimension.label}
                  tone={
                    (active > index ? dimension.after : dimension.before) >= 90 ? "good" : "warn"
                  }
                  showValue={false}
                />
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* footer strip */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-line bg-raised/40 px-3.5 py-2 font-mono text-[10.5px] text-faint">
        <span>891 rows</span>
        <span>→ 16 columns</span>
        <span className="text-good">no leakage detected</span>
        <span className="ml-auto text-accent">
          {active === STEPS.length ? "compiled in 0.99s" : "running…"}
        </span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------------

export function Landing({ onStart }: { onStart: () => void }) {
  return (
    <div className="min-h-screen bg-void">
      {/* ------------------------------------------------------------------ nav */}
      <header className="sticky top-0 z-30 border-b border-line bg-void/85 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-[1100px] items-center gap-6 px-6">
          <span className="font-display text-[15px] font-semibold tracking-tight text-text">
            ML<span className="text-accent">·</span>BenchX
          </span>
          <nav className="hidden gap-5 text-[13px] text-muted sm:flex">
            <a href="#how" className="transition-colors hover:text-text">How it works</a>
            <a href="#health" className="transition-colors hover:text-text">Health</a>
            <a href="#control" className="transition-colors hover:text-text">Control</a>
            <a href="#code" className="transition-colors hover:text-text">Your code</a>
          </nav>
          <Button variant="primary" size="sm" className="ml-auto" onClick={onStart}>
            Start preprocessing
          </Button>
        </div>
      </header>

      {/* ----------------------------------------------------------------- hero */}
      <section className="relative overflow-hidden px-6 pt-16 pb-20 sm:pt-24">
        <div className="aurora" aria-hidden />
        <div className="grid-veil" aria-hidden />
        <div className="relative mx-auto max-w-[1100px]">
          <Reveal>
            <Badge tone="accent" className="mb-5">
              <Sparkles size={11} /> Every transformation, visible and reversible
            </Badge>
            <h1 className="max-w-[16ch] font-display text-[42px] leading-[1.05] font-semibold tracking-tight text-text sm:text-[64px]">
              Transform your data.
              <br />
              <span className="text-muted">Understand every step.</span>
            </h1>
            <p className="mt-6 max-w-[64ch] text-[17px] leading-relaxed text-muted sm:text-[19px]">
              Upload a dataset, see exactly what is wrong with it, switch any preprocessing
              step on or off with the consequences spelled out first, edit the generated
              Python by hand, and compile a production-ready dataset — in one workspace.
            </p>
            <div className="mt-8 flex flex-wrap items-center gap-3">
              <Button variant="primary" size="lg" onClick={onStart}>
                Start preprocessing <ArrowRight size={16} />
              </Button>
              <Button variant="secondary" size="lg" onClick={onStart}>
                <Play size={14} /> Explore the demo
              </Button>
            </div>
            <p className="mt-4 flex items-center gap-1.5 text-[13px] text-faint">
              <Shield size={12} /> Runs locally. Your data never leaves your machine.
            </p>
          </Reveal>

          <Reveal delay={140}>
            <div className="mt-14">
              <HeroPreview />
            </div>
          </Reveal>
        </div>
      </section>

      {/* ------------------------------------------------------------ how it works */}
      <Section
        id="how"
        eyebrow="How it works"
        title="Five steps, and you can stop at any of them."
        lead="Nothing happens that you did not ask for, and nothing is hidden once it has."
      >
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
          {[
            { icon: Upload, title: "Upload", body: "CSV, Excel, JSON or Parquet. Read locally, never sent anywhere." },
            { icon: Sparkles, title: "Analyse", body: "Twelve classes of problem detected and scored across six dimensions." },
            { icon: SlidersHorizontal, title: "Customise", body: "Toggle any step. See what turning it off would cost before you do." },
            { icon: Play, title: "Compile", body: "Run the pipeline in a sandbox. Watch the health score move." },
            { icon: Download, title: "Download", body: "The data, the code, the config, and a report that explains itself." },
          ].map((item, index) => (
            <Reveal key={item.title} delay={index * 70}>
              <div className="card relative h-full p-4">
                <span className="tnum absolute top-3 right-3 font-mono text-[11px] text-faint">
                  {String(index + 1).padStart(2, "0")}
                </span>
                <item.icon size={17} className="text-accent" />
                <p className="mt-3 font-display text-[15px] font-medium text-text">{item.title}</p>
                <p className="mt-1.5 text-[13.5px] leading-relaxed text-muted">{item.body}</p>
              </div>
            </Reveal>
          ))}
        </div>
      </Section>

      {/* ---------------------------------------------------------------- health */}
      <Section
        id="health"
        eyebrow="Dataset health"
        title="A score you can take apart."
        lead="One number out of a hundred tells you nothing about what to do next. Six measured dimensions, each with the evidence behind it, tell you where to look."
      >
        <div className="grid gap-6 lg:grid-cols-[1fr_340px]">
          <Reveal>
            <div className="panel p-5">
              <div className="space-y-3.5">
                {DIMENSIONS.map((dimension) => (
                  <div key={dimension.label}>
                    <div className="mb-1.5 flex items-baseline justify-between">
                      <span className="text-[13.5px] text-text">{dimension.label}</span>
                      <span className="tnum font-mono text-[12px] text-muted">
                        {dimension.before} / 100
                      </span>
                    </div>
                    <Meter
                      value={dimension.before}
                      label={dimension.label}
                      showValue={false}
                      tone={dimension.before >= 90 ? "good" : dimension.before >= 75 ? "accent" : "warn"}
                    />
                  </div>
                ))}
              </div>
              <p className="mt-5 border-t border-line pt-4 text-[13px] leading-relaxed text-muted">
                Each dimension is measured independently, multiplied by its weight, and
                summed. The workspace shows you that arithmetic line by line — no part of
                the score is a constant hidden in a function.
              </p>
            </div>
          </Reveal>

          <Reveal delay={120}>
            <div className="panel h-full p-5">
              <p className="font-display text-[13px] font-medium text-text">
                Detected automatically
              </p>
              <ul className="mt-3 space-y-1.5">
                {DETECTED.map((item) => (
                  <li key={item} className="flex items-start gap-2 text-[13px] text-muted">
                    <Check size={12} className="mt-1 shrink-0 text-good" />
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          </Reveal>
        </div>
      </Section>

      {/* --------------------------------------------------------------- control */}
      <Section
        id="control"
        eyebrow="Full pipeline control"
        title="Every step is a switch, and every switch explains itself."
        lead="Turn one off and the app tells you what it was doing for this dataset, in this many rows and columns, before anything changes."
      >
        <div className="grid gap-6 lg:grid-cols-[300px_1fr]">
          <Reveal>
            <div className="panel p-4">
              <ul className="space-y-1">
                {STEPS.map((step, index) => (
                  <li
                    key={step.id}
                    className="flex items-center gap-2.5 rounded-lg px-2 py-2 transition-colors hover:bg-raised"
                  >
                    <span className="tnum w-4 shrink-0 font-mono text-[10px] text-faint">
                      {index + 1}
                    </span>
                    <span className="min-w-0 flex-1 truncate text-[13.5px] text-text">
                      {step.label}
                    </span>
                    <span className="font-mono text-[11px] text-faint">{step.note}</span>
                    <span
                      className={`h-3.5 w-6 shrink-0 rounded-full border ${
                        step.id === "selection"
                          ? "border-line-strong bg-sunken"
                          : "border-accent bg-accent/25"
                      }`}
                    />
                  </li>
                ))}
              </ul>
            </div>
          </Reveal>

          <Reveal delay={120}>
            <div className="panel overflow-hidden">
              <div className="flex items-center gap-2 border-b border-line px-4 py-3">
                <span className="h-1.5 w-1.5 rounded-full bg-warn" />
                <span className="font-display text-[13.5px] font-medium text-text">
                  Disable Missing Value Treatment?
                </span>
              </div>
              <div className="space-y-2.5 px-4 py-4">
                {[
                  "885 missing value(s) will remain, across 3 column(s).",
                  "Most scikit-learn estimators raise on the first blank they meet, so training will fail rather than degrade.",
                  "Any statistic computed downstream will silently skip those rows and be computed on a different subset per column.",
                ].map((line) => (
                  <p key={line} className="flex gap-2.5 text-[13.5px] leading-relaxed text-muted">
                    <span className="mt-[7px] h-1 w-1 shrink-0 rounded-full bg-warn" />
                    {line}
                  </p>
                ))}
                <div className="mt-4 flex items-center gap-4 rounded-lg border border-line bg-sunken px-3.5 py-3">
                  <div>
                    <p className="text-[11px] text-faint">Dataset health</p>
                    <p className="tnum mt-0.5 font-mono text-[18px] text-text">
                      87 <span className="text-faint">→</span>{" "}
                      <span className="text-warn">85</span>
                    </p>
                  </div>
                  <p className="border-l border-line pl-4 text-[12px] leading-relaxed text-faint">
                    Measured by running the pipeline both ways — not estimated.
                  </p>
                </div>
              </div>
              <div className="flex justify-end gap-2 border-t border-line px-4 py-3">
                <Button size="sm" variant="ghost">Keep enabled</Button>
                <Button size="sm" variant="danger">Disable anyway</Button>
              </div>
            </div>
          </Reveal>
        </div>
      </Section>

      {/* ------------------------------------------------------------------ code */}
      <Section
        id="code"
        eyebrow="Code ownership"
        title="The system generates the code. You own the code."
        lead="Open it in the built-in editor, change anything, and compile. From that moment the app runs what you wrote — and says so, rather than showing toggles that have stopped describing reality."
      >
        <div className="grid gap-6 lg:grid-cols-[1fr_290px]">
          <Reveal>
            <div className="panel overflow-hidden">
              <div className="flex items-center gap-2 border-b border-line bg-raised/50 px-3.5 py-2.5">
                <Code2 size={13} className="text-muted" />
                <span className="font-mono text-[12px] text-text">pipeline.py</span>
                <Badge tone="warn" className="ml-1">edited</Badge>
                <span className="ml-auto font-mono text-[11px] text-faint">Python</span>
              </div>
              <pre className="overflow-x-auto bg-sunken p-4 font-mono text-[12px] leading-[1.8] text-muted">
{`  38  `}<span className="text-faint">{`# Outlier Treatment`}</span>{`
  39  `}<span className="text-faint">{`# Bounds come from the training rows. An outlier is not`}</span>{`
  40  `}<span className="text-faint">{`# automatically an error, which is why this caps rather`}</span>{`
  41  `}<span className="text-faint">{`# than deletes.`}</span>{`
  42  `}<span className="text-info">for</span>{` _column `}<span className="text-info">in</span>{` OUTLIER_COLUMNS:
  43      _train = df.loc[train_index, _column]
  44      _q1, _q3 = _train.quantile(`}<span className="text-accent">0.25</span>{`), _train.quantile(`}<span className="text-accent">0.75</span>{`)
  45      _iqr = _q3 - _q1
  46      _low, _high = _q1 - `}<span className="text-accent">1.5</span>{` * _iqr, _q3 + `}<span className="text-accent">1.5</span>{` * _iqr
  47      df[_column] = df[_column].clip(lower=_low, upper=_high)`}
              </pre>
            </div>
          </Reveal>

          <Reveal delay={120}>
            <div className="panel h-full p-5">
              <p className="font-display text-[13px] font-medium text-text">In the editor</p>
              <ul className="mt-3 space-y-1.5">
                {[
                  "Syntax highlighting", "Line numbers", "Auto indentation",
                  "Find and replace", "Code folding", "Error markers on the failing line",
                  "Format", "Reset to generated", "Copy", "Download .py",
                ].map((item) => (
                  <li key={item} className="flex items-start gap-2 text-[13px] text-muted">
                    <Check size={12} className="mt-1 shrink-0 text-accent" />
                    {item}
                  </li>
                ))}
              </ul>
              <p className="mt-4 border-t border-line pt-3.5 text-[12.5px] leading-relaxed text-faint">
                Your code runs in a kernel-enforced sandbox with no network access and no
                ability to write outside its own scratch directory.
              </p>
            </div>
          </Reveal>
        </div>
      </Section>

      {/* ---------------------------------------------------------- before / after */}
      <Section
        eyebrow="Before and after"
        title="The difference, measured both ways."
        lead="The same six dimensions, computed on the data you uploaded and on the data that came out. Not a claim — a comparison."
      >
        <Reveal>
          <div className="panel overflow-hidden">
            <div className="grid grid-cols-2 divide-x divide-line">
              {(["before", "after"] as const).map((side) => (
                <div key={side} className="p-5">
                  <p className="font-display text-[11px] tracking-[0.13em] text-faint uppercase">
                    {side}
                  </p>
                  <p className="tnum mt-3 font-mono text-[34px] leading-none text-text">
                    {side === "before" ? 90 : 92}
                    <span className="ml-1.5 text-[15px] text-faint">/ 100</span>
                  </p>
                  <dl className="mt-5 space-y-1.5">
                    {[
                      ["Rows", side === "before" ? "891" : "891"],
                      ["Columns", side === "before" ? "13" : "16"],
                      ["Missing", side === "before" ? "885" : "0"],
                      ["Duplicates", "0"],
                      ["Outliers", side === "before" ? "170" : "130"],
                    ].map(([label, value]) => (
                      <div key={label} className="flex justify-between text-[13px]">
                        <dt className="text-muted">{label}</dt>
                        <dd className="tnum font-mono text-text">{value}</dd>
                      </div>
                    ))}
                  </dl>
                </div>
              ))}
            </div>
            <div className="space-y-2.5 border-t border-line bg-raised/40 p-5">
              {DIMENSIONS.map((dimension) => {
                const delta = dimension.after - dimension.before;
                return (
                  <div key={dimension.label} className="flex items-center gap-3">
                    <span className="w-32 shrink-0 text-[12.5px] text-muted">
                      {dimension.label}
                    </span>
                    <Meter
                      value={dimension.after}
                      label={dimension.label}
                      showValue={false}
                      tone={dimension.after >= 90 ? "good" : dimension.after >= 75 ? "accent" : "warn"}
                    />
                    <span
                      className={`tnum w-12 shrink-0 text-right font-mono text-[11.5px] ${
                        delta > 0 ? "text-good" : delta < 0 ? "text-danger" : "text-faint"
                      }`}
                    >
                      {delta > 0 ? "+" : ""}
                      {delta || "—"}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </Reveal>
      </Section>

      {/* -------------------------------------------------------------- downloads */}
      <Section
        eyebrow="Take it with you"
        title="Nothing here is locked in."
        lead="Everything the workspace produced, in formats that outlive it."
      >
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {[
            { icon: FileSpreadsheet, title: "Processed dataset", body: "The result, as CSV." },
            { icon: FileSpreadsheet, title: "Original dataset", body: "Exactly the bytes you uploaded." },
            { icon: Code2, title: "Python pipeline", body: "Standalone. Runs anywhere pandas does." },
            { icon: SlidersHorizontal, title: "Pipeline configuration", body: "JSON — reload it, diff it, commit it." },
            { icon: GitCompare, title: "Health report", body: "What was wrong, what was done, what was left." },
            { icon: Download, title: "Complete project", body: "All of the above, zipped." },
          ].map((item, index) => (
            <Reveal key={item.title} delay={index * 60}>
              <div className="card h-full p-4">
                <item.icon size={16} className="text-accent" />
                <p className="mt-3 font-display text-[14px] font-medium text-text">{item.title}</p>
                <p className="mt-1 text-[13px] text-muted">{item.body}</p>
              </div>
            </Reveal>
          ))}
        </div>
      </Section>

      {/* -------------------------------------------------------------- final CTA */}
      <section className="relative overflow-hidden border-t border-line px-6 py-24">
        <div className="aurora" aria-hidden />
        <div className="relative mx-auto max-w-[1100px] text-center">
          <Reveal>
            <h2 className="font-display text-[34px] leading-[1.1] font-semibold tracking-tight text-text sm:text-[46px]">
              Your dataset. Your pipeline.
              <br />
              <span className="text-muted">Your control.</span>
            </h2>
            <p className="mx-auto mt-5 max-w-[54ch] text-[16px] leading-relaxed text-muted">
              Every automated recommendation in this tool is visible, explained, optional
              and reversible. You are not handing your data to something that decides for
              you.
            </p>
            <Button variant="primary" size="lg" className="mt-8" onClick={onStart}>
              Start building your pipeline <ArrowRight size={16} />
            </Button>
          </Reveal>
        </div>
      </section>

      <footer className="border-t border-line px-6 py-8">
        <div className="mx-auto flex max-w-[1100px] flex-wrap items-center gap-x-4 gap-y-2">
          <span className="font-display text-[13px] font-medium text-muted">
            ML<span className="text-accent">·</span>BenchX
          </span>
          <span className="text-[12.5px] text-faint">
            Glass-box preprocessing for tabular data.
          </span>
          <span className="ml-auto font-mono text-[11.5px] text-faint">
            Runs locally · No account · No telemetry
          </span>
        </div>
      </footer>
    </div>
  );
}
