/**
 * The workspace shell: three panels, a dock, and the state that binds them.
 *
 * All workspace state lives here, and there is exactly one copy of it. Every mutating
 * call returns the whole workspace and we replace it wholesale, so the toggles, the
 * editor and the health panel are always rendering the same object. They cannot drift
 * apart, because there is nothing for them to drift between.
 *
 * Layout adapts rather than collapsing: three columns on a desktop, two with the
 * insights folded into a drawer on a tablet, and a single column with a tab bar on a
 * phone. The bottom dock is resizable by dragging its edge, since how much of it you
 * want depends entirely on whether you are reading code or reading rows.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  Check,
  ChevronDown,
  Database,
  Download,
  FileCode2,
  PanelRightClose,
  PanelRightOpen,
  Play,
  Shield,
  Sparkles,
} from "lucide-react";
import {
  ApiError,
  api,
  type Compilation,
  type ImpactReport,
  type PipelineConfig,
  type StepId,
  type Workspace as WorkspaceState,
} from "../api";
import { Badge, Button, Dialog, ScoreRing, Spinner } from "../components/ui";
import { PipelinePanel } from "./PipelinePanel";
import { CodePanel } from "./CodePanel";
import { InsightsPanel } from "./InsightsPanel";
import { BottomDock } from "./BottomDock";

/** Narrated phases during a compile. The server does not stream, so these are paced. */
const COMPILE_PHASES = [
  "Preparing the sandbox",
  "Validating the code",
  "Running preprocessing",
  "Analysing the result",
  "Scoring dataset health",
];

const DOWNLOADS: { key: string; label: string; note: string; needsCompile: boolean }[] = [
  { key: "processed", label: "Processed dataset", note: "CSV", needsCompile: true },
  { key: "original", label: "Original dataset", note: "CSV", needsCompile: false },
  { key: "code", label: "Python pipeline", note: "pipeline.py", needsCompile: false },
  { key: "config", label: "Pipeline configuration", note: "JSON", needsCompile: false },
  { key: "report", label: "Health report", note: "Markdown", needsCompile: true },
  { key: "project", label: "Complete project", note: "ZIP", needsCompile: true },
];

function DownloadMenu({ workspace }: { workspace: WorkspaceState }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const compiled = workspace.latest_compilation?.status === "success";

  useEffect(() => {
    if (!open) return;
    const close = (event: MouseEvent) => {
      if (!ref.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [open]);

  return (
    <div ref={ref} className="relative">
      <Button size="sm" variant="secondary" onClick={() => setOpen((was) => !was)}>
        <Download size={13} /> Download
        <ChevronDown size={12} className={open ? "rotate-180" : ""} />
      </Button>
      {open && (
        <div className="rise absolute right-0 z-40 mt-1.5 w-64 overflow-hidden rounded-lg border border-line-strong bg-surface shadow-2xl">
          {DOWNLOADS.map((item) => {
            const blocked = item.needsCompile && !compiled;
            return blocked ? (
              <div
                key={item.key}
                className="flex items-center gap-2 px-3 py-2.5 opacity-45"
                title="Compile the pipeline first"
              >
                <span className="flex-1 text-[13px] text-text">{item.label}</span>
                <span className="font-mono text-[10.5px] text-faint">compile first</span>
              </div>
            ) : (
              <a
                key={item.key}
                href={api.downloadUrl(workspace.id, item.key)}
                onClick={() => setOpen(false)}
                className="flex items-center gap-2 px-3 py-2.5 transition-colors hover:bg-raised"
              >
                <span className="flex-1 text-[13px] text-text">{item.label}</span>
                <span className="font-mono text-[10.5px] text-faint">{item.note}</span>
              </a>
            );
          })}
        </div>
      )}
    </div>
  );
}

function CompileOverlay({ phase }: { phase: number }) {
  return (
    <div className="fade absolute inset-0 z-40 flex items-center justify-center bg-void/75 backdrop-blur-sm">
      <div className="panel w-[320px] p-5">
        <div className="flex items-center gap-2.5">
          <Spinner size={15} />
          <span className="font-display text-[15px] text-text">Compiling pipeline</span>
        </div>
        <div className="sweep relative mt-4 h-1 overflow-hidden rounded-full bg-sunken">
          <div
            className="h-full rounded-full bg-accent transition-[width] duration-400"
            style={{ width: `${((phase + 1) / COMPILE_PHASES.length) * 100}%` }}
          />
        </div>
        <ul className="mt-4 space-y-1.5">
          {COMPILE_PHASES.map((text, index) => (
            <li key={text} className="flex items-center gap-2 text-[12.5px]">
              <span
                className={`flex h-3.5 w-3.5 shrink-0 items-center justify-center rounded-full border text-[8px] ${
                  index < phase
                    ? "border-good bg-good/15 text-good"
                    : index === phase
                      ? "border-accent text-accent"
                      : "border-line-strong text-faint"
                }`}
              >
                {index < phase ? "✓" : index === phase ? "●" : ""}
              </span>
              <span className={index <= phase ? "text-text" : "text-faint"}>{text}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

/** The banner shown for a few seconds after a run finishes. */
function CompileResult({
  compilation,
  onDismiss,
}: {
  compilation: Compilation;
  onDismiss: () => void;
}) {
  const ok = compilation.status === "success";
  const comparison = compilation.result?.comparison;

  return (
    <div
      role="status"
      className={`rise fixed right-4 bottom-4 z-40 w-[330px] rounded-xl border p-4 shadow-2xl ${
        ok
          ? "border-[color-mix(in_oklab,var(--color-good)_40%,transparent)] bg-surface"
          : "border-[color-mix(in_oklab,var(--color-danger)_45%,transparent)] bg-surface"
      }`}
    >
      <div className="flex items-center gap-2">
        {ok ? (
          <Check size={15} className="text-good" />
        ) : (
          <AlertTriangle size={15} className="text-danger" />
        )}
        <span className="flex-1 font-display text-[14px] font-medium text-text">
          {ok ? "Compilation successful" : "Compilation failed"}
        </span>
        <button
          onClick={onDismiss}
          aria-label="Dismiss"
          className="text-faint transition-colors hover:text-text"
        >
          ×
        </button>
      </div>

      {ok && comparison ? (
        <div className="mt-3 grid grid-cols-3 gap-2">
          {[
            ["Health", comparison.score.before, comparison.score.after],
            ["Rows", comparison.rows.before, comparison.rows.after],
            ["Columns", comparison.columns.before, comparison.columns.after],
          ].map(([label, before, after]) => (
            <div key={String(label)} className="rounded-lg bg-raised px-2 py-1.5">
              <p className="text-[10px] text-faint">{label as string}</p>
              <p className="tnum mt-0.5 font-mono text-[12.5px] text-text">
                {(before as number).toLocaleString()}
                <span className="mx-0.5 text-faint">→</span>
                {(after as number).toLocaleString()}
              </p>
            </div>
          ))}
        </div>
      ) : (
        <p className="mt-2 font-mono text-[12px] leading-relaxed text-danger">
          {compilation.error?.type}: {compilation.error?.message}
        </p>
      )}

      <p className="mt-2.5 font-mono text-[11px] text-faint">
        {compilation.duration_ms} ms · from the {compilation.source}
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------------

export function Workspace({
  datasetId,
  onExit,
}: {
  datasetId: string;
  onExit: () => void;
}) {
  const [workspace, setWorkspace] = useState<WorkspaceState | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [compiling, setCompiling] = useState(false);
  const [phase, setPhase] = useState(0);
  const [banner, setBanner] = useState<Compilation | null>(null);
  const [jumpToLine, setJumpToLine] = useState<number | null>(null);
  const [insightsOpen, setInsightsOpen] = useState(true);
  const [mobileTab, setMobileTab] = useState<"pipeline" | "code" | "insights" | "data">("pipeline");
  const [dockHeight, setDockHeight] = useState(240);
  const [sandboxOpen, setSandboxOpen] = useState(false);

  const [pendingToggle, setPendingToggle] = useState<{ step: StepId; enabled: boolean } | null>(null);
  const [impact, setImpact] = useState<ImpactReport | null>(null);
  const [impactLoading, setImpactLoading] = useState(false);

  // ------------------------------------------------------------------- loading
  useEffect(() => {
    api
      .openWorkspace(datasetId)
      .then(setWorkspace)
      .catch((caught) =>
        setLoadError(caught instanceof ApiError ? caught.message : "Could not open the workspace."),
      );
  }, [datasetId]);

  useEffect(() => {
    if (!compiling) return;
    setPhase(0);
    const timer = setInterval(
      () => setPhase((current) => Math.min(current + 1, COMPILE_PHASES.length - 1)),
      420,
    );
    return () => clearInterval(timer);
  }, [compiling]);

  // Warn before leaving with work that has not been run.
  useEffect(() => {
    if (!workspace?.has_uncompiled_changes) return;
    const warn = (event: BeforeUnloadEvent) => event.preventDefault();
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [workspace?.has_uncompiled_changes]);

  // ------------------------------------------------------------------- actions
  const changeConfig = useCallback(
    async (next: PipelineConfig, summary = "") => {
      if (!workspace) return;
      // Show the new configuration at once and reconcile with the server's answer.
      // A toggle that waits on a round trip before moving feels broken.
      setWorkspace({ ...workspace, config: next });
      try {
        setWorkspace(await api.updateConfig(workspace.id, next, summary));
      } catch {
        setWorkspace(await api.readWorkspace(workspace.id));
      }
    },
    [workspace],
  );

  const requestToggle = useCallback(
    async (step: StepId, enabled: boolean) => {
      if (!workspace) return;
      setPendingToggle({ step, enabled });
      setImpactLoading(true);
      setImpact(null);
      try {
        setImpact(await api.previewImpact(workspace.id, step, enabled));
      } catch {
        // If the preview cannot be produced, apply the change rather than blocking the
        // user behind a warning we failed to compute.
        setPendingToggle(null);
        const next = {
          ...workspace.config,
          [step]: { ...workspace.config[step], enabled },
        } as PipelineConfig;
        void changeConfig(next);
      } finally {
        setImpactLoading(false);
      }
    },
    [workspace, changeConfig],
  );

  const confirmToggle = useCallback(() => {
    if (!workspace || !pendingToggle) return;
    const { step, enabled } = pendingToggle;
    const next = {
      ...workspace.config,
      [step]: { ...workspace.config[step], enabled },
    } as PipelineConfig;
    setPendingToggle(null);
    setImpact(null);
    void changeConfig(next);
  }, [workspace, pendingToggle, changeConfig]);

  const compile = useCallback(async () => {
    if (!workspace || compiling) return;
    setCompiling(true);
    setBanner(null);
    try {
      const { compilation, workspace: updated } = await api.compile(workspace.id, "auto");
      setWorkspace(updated);
      setBanner(compilation);
      if (compilation.status === "success") {
        setTimeout(() => setBanner(null), 7000);
      }
    } catch (caught) {
      setLoadError(caught instanceof ApiError ? caught.message : "The compile request failed.");
    } finally {
      setCompiling(false);
    }
  }, [workspace, compiling]);

  const saveCode = useCallback(
    async (code: string) => {
      if (!workspace) return;
      setWorkspace(await api.updateCode(workspace.id, code));
    },
    [workspace],
  );

  const resetCode = useCallback(async () => {
    if (!workspace) return;
    setWorkspace(await api.resetCode(workspace.id));
  }, [workspace]);

  const restore = useCallback(
    async (entryId: number) => {
      if (!workspace) return;
      setWorkspace(await api.restore(workspace.id, entryId));
    },
    [workspace],
  );

  // Drag the dock's top edge to resize it.
  const startResize = useCallback((event: React.PointerEvent) => {
    event.preventDefault();
    const startY = event.clientY;
    const startHeight = dockHeight;
    const move = (moveEvent: PointerEvent) => {
      const next = startHeight - (moveEvent.clientY - startY);
      setDockHeight(Math.max(120, Math.min(window.innerHeight - 260, next)));
    };
    const stop = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop);
  }, [dockHeight]);

  // ------------------------------------------------------------------ rendering
  if (loadError && !workspace) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center gap-4 bg-void px-6 text-center">
        <AlertTriangle size={22} className="text-danger" />
        <p className="max-w-[46ch] text-[15px] text-text">{loadError}</p>
        <Button onClick={onExit}>Back to upload</Button>
      </div>
    );
  }

  if (!workspace) {
    return (
      <div className="flex min-h-screen items-center justify-center gap-2.5 bg-void">
        <Spinner size={16} />
        <span className="text-[14px] text-muted">Opening the workspace…</span>
      </div>
    );
  }

  const health =
    workspace.latest_compilation?.status === "success"
      ? (workspace.latest_compilation.health_after as { score: number }).score
      : workspace.baseline_health.score;

  // The inline error marker points at a line number. Once the code has changed since
  // that run, that line is no longer the line that failed, so the marker is worse than
  // nothing — it sends the reader to an innocent statement. The log keeps the failure
  // as history; only the in-editor marker is retracted.
  const compilationError =
    workspace.latest_compilation?.status === "error" && !workspace.has_uncompiled_changes
      ? workspace.latest_compilation.error
      : null;

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-void">
      {/* ----------------------------------------------------------------- top bar */}
      <header className="flex h-12 shrink-0 items-center gap-3 border-b border-line px-3">
        <button
          onClick={onExit}
          className="font-display text-[14px] font-semibold tracking-tight text-text transition-colors hover:text-accent"
        >
          ML<span className="text-accent">·</span>BenchX
        </button>

        <div className="mx-1 hidden h-4 w-px bg-line sm:block" />

        <div className="hidden min-w-0 items-center gap-2 sm:flex">
          <Database size={12} className="shrink-0 text-faint" />
          <span className="truncate font-mono text-[12.5px] text-text">
            {workspace.dataset.filename}
          </span>
          <span className="tnum hidden shrink-0 font-mono text-[11px] text-faint md:inline">
            {workspace.dataset.rows.toLocaleString()} × {workspace.dataset.columns}
          </span>
        </div>

        {workspace.has_uncompiled_changes && (
          <Badge tone="warn" className="shrink-0">
            <span className="h-1 w-1 rounded-full bg-warn" /> uncompiled changes
          </Badge>
        )}
        {workspace.code_is_authoritative && (
          <Badge tone="warn" className="hidden shrink-0 lg:inline-flex">
            <FileCode2 size={9} /> code is authoritative
          </Badge>
        )}

        <div className="ml-auto flex items-center gap-2">
          <button
            onClick={() => setSandboxOpen(true)}
            title="How your code is isolated"
            className="hidden items-center gap-1.5 rounded-md px-2 py-1 font-mono text-[11px]
              text-faint transition-colors hover:bg-raised hover:text-muted lg:flex"
          >
            <Shield size={11} />
            {workspace.sandbox.label}
          </button>

          <div className="flex items-center gap-2 rounded-lg border border-line bg-surface py-1 pr-2.5 pl-1">
            <ScoreRing score={health} size={26} label="Dataset health" />
            <div className="leading-none">
              <p className="text-[9.5px] text-faint">health</p>
              <p className="tnum font-mono text-[12px] text-text">{health}</p>
            </div>
          </div>

          <DownloadMenu workspace={workspace} />

          <Button variant="primary" size="sm" onClick={compile} disabled={compiling}>
            {compiling ? <Spinner size={12} /> : <Play size={12} />}
            <span className="hidden sm:inline">
              {compiling ? "Compiling…" : "Compile pipeline"}
            </span>
          </Button>

          <button
            onClick={() => setInsightsOpen((was) => !was)}
            aria-label={insightsOpen ? "Hide insights" : "Show insights"}
            className="hidden rounded-md p-1.5 text-muted transition-colors hover:bg-raised hover:text-text xl:block"
          >
            {insightsOpen ? <PanelRightClose size={15} /> : <PanelRightOpen size={15} />}
          </button>
        </div>
      </header>

      {/* ------------------------------------------------------- mobile tab bar */}
      <div className="flex shrink-0 border-b border-line lg:hidden">
        {(
          [
            ["pipeline", "Pipeline"],
            ["code", "Code"],
            ["insights", "Health"],
            ["data", "Data"],
          ] as const
        ).map(([key, label]) => (
          <button
            key={key}
            onClick={() => setMobileTab(key)}
            aria-current={mobileTab === key}
            className={`relative flex-1 py-2.5 text-[12.5px] font-medium transition-colors ${
              mobileTab === key ? "text-text" : "text-muted"
            }`}
          >
            {label}
            {mobileTab === key && (
              <span className="absolute inset-x-4 bottom-0 h-px bg-accent" aria-hidden />
            )}
          </button>
        ))}
      </div>

      {/* ------------------------------------------------------------- main area */}
      <div className="relative flex min-h-0 flex-1 flex-col">
        {compiling && <CompileOverlay phase={phase} />}

        <div className="flex min-h-0 flex-1">
          {/* pipeline */}
          <aside
            className={`w-full shrink-0 overflow-hidden border-line lg:w-[290px] lg:border-r xl:w-[320px] ${
              mobileTab === "pipeline" ? "block" : "hidden lg:block"
            }`}
          >
            <PipelinePanel
              workspace={workspace}
              onConfigChange={changeConfig}
              onRequestImpact={requestToggle}
              impact={pendingToggle ? impact : null}
              impactLoading={impactLoading}
              onConfirmImpact={confirmToggle}
              onCancelImpact={() => {
                setPendingToggle(null);
                setImpact(null);
              }}
            />
          </aside>

          {/* code */}
          <main
            className={`min-w-0 flex-1 ${mobileTab === "code" ? "block" : "hidden lg:block"}`}
          >
            <CodePanel
              workspace={workspace}
              compiling={compiling}
              error={compilationError}
              jumpToLine={jumpToLine}
              onCodeChange={saveCode}
              onReset={resetCode}
              onCompile={compile}
            />
          </main>

          {/* insights */}
          <aside
            className={`w-full shrink-0 overflow-hidden border-line lg:w-[290px] lg:border-l ${
              mobileTab === "insights"
                ? "block"
                : insightsOpen
                  ? "hidden lg:block"
                  : "hidden"
            }`}
          >
            <InsightsPanel workspace={workspace} />
          </aside>
        </div>

        {/* dock */}
        <div
          className={`shrink-0 border-t border-line ${
            mobileTab === "data" ? "flex-1" : "hidden lg:block"
          }`}
          style={{ height: mobileTab === "data" ? undefined : dockHeight }}
        >
          <div
            onPointerDown={startResize}
            role="separator"
            aria-label="Resize the bottom panel"
            className="hidden h-1 cursor-row-resize bg-transparent transition-colors hover:bg-accent/40 lg:block"
          />
          <div className="h-[calc(100%-4px)]">
            <BottomDock
              workspace={workspace}
              onJumpToLine={(line) => {
                setJumpToLine(line);
                setMobileTab("code");
                // Reset so the same line can be requested again later.
                setTimeout(() => setJumpToLine(null), 100);
              }}
              onRestore={restore}
            />
          </div>
        </div>
      </div>

      {banner && <CompileResult compilation={banner} onDismiss={() => setBanner(null)} />}

      <Dialog
        open={sandboxOpen}
        onClose={() => setSandboxOpen(false)}
        title={
          <span className="flex items-center gap-2">
            <Shield size={15} className="text-accent" /> How your code runs
          </span>
        }
        footer={<Button onClick={() => setSandboxOpen(false)}>Close</Button>}
      >
        <p className="text-[14px] leading-relaxed text-text">{workspace.sandbox.detail}</p>
        <ul className="mt-4 space-y-2">
          {[
            ["Network access", workspace.sandbox.network_blocked],
            ["Filesystem isolation", workspace.sandbox.filesystem_isolated],
          ].map(([label, enforced]) => (
            <li key={String(label)} className="flex items-center gap-2 text-[13.5px]">
              {enforced ? (
                <Check size={13} className="text-good" />
              ) : (
                <AlertTriangle size={13} className="text-warn" />
              )}
              <span className="text-muted">{label as string}</span>
              <span className={`ml-auto font-mono text-[12px] ${enforced ? "text-good" : "text-warn"}`}>
                {enforced ? "blocked" : "not enforced"}
              </span>
            </li>
          ))}
        </ul>
        <p className="mt-4 flex gap-2 text-[12.5px] leading-relaxed text-faint">
          <Sparkles size={12} className="mt-0.5 shrink-0" />
          Memory, CPU time and wall-clock limits apply on every platform. A run that
          exceeds them is stopped and reported rather than left to exhaust the machine.
        </p>
      </Dialog>
    </div>
  );
}
