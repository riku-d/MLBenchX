/**
 * Getting a dataset into the workspace.
 *
 * Two things matter here. The drop zone has to accept a keyboard as readily as a
 * pointer — drag-and-drop alone would make the entry point to the whole product
 * unreachable for some users — and the analysis that follows has to be narrated. A
 * spinner with no text leaves the user unable to tell a slow parse from a hung one.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { AlertCircle, FileSpreadsheet, FolderOpen, Upload as UploadIcon } from "lucide-react";
import { ApiError, api, type DatasetSummary } from "../api";
import { Button, Spinner } from "../components/ui";

const ACCEPT = ".csv,.tsv,.txt,.xlsx,.xls,.xlsm,.json,.jsonl,.ndjson,.parquet,.pq";

/** The phases the user is told about while a dataset is being read. */
const PHASES = [
  "Reading the file",
  "Detecting column types",
  "Checking for missing values",
  "Looking for duplicates and outliers",
  "Scoring dataset health",
];

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function UploadScreen({
  onReady,
  onBack,
}: {
  onReady: (dataset: DatasetSummary) => void;
  onBack: () => void;
}) {
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [phase, setPhase] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [recent, setRecent] = useState<DatasetSummary[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.listDatasets().then(setRecent).catch(() => setRecent([]));
  }, []);

  // Advance the phase labels while the server works. The server does not stream
  // progress, so these are paced rather than reported — and they stop at the last
  // phase rather than looping, so a stall is visible instead of disguised.
  useEffect(() => {
    if (!busy) return;
    const timer = setInterval(
      () => setPhase((current) => Math.min(current + 1, PHASES.length - 1)),
      520,
    );
    return () => clearInterval(timer);
  }, [busy]);

  const submit = useCallback(
    async (file: File) => {
      setError(null);
      setBusy(true);
      setProgress(0);
      setPhase(0);
      try {
        const dataset = await api.uploadDataset(file, setProgress);
        onReady(dataset);
      } catch (caught) {
        setError(
          caught instanceof ApiError
            ? caught.message
            : "Something went wrong reading that file.",
        );
        setBusy(false);
      }
    },
    [onReady],
  );

  return (
    <div className="min-h-screen bg-void">
      <header className="border-b border-line px-6">
        <div className="mx-auto flex h-14 max-w-[880px] items-center gap-3">
          <button
            onClick={onBack}
            className="font-display text-[15px] font-semibold tracking-tight text-text transition-colors hover:text-accent"
          >
            ML<span className="text-accent">·</span>BenchX
          </button>
        </div>
      </header>

      <main className="mx-auto max-w-[880px] px-6 py-14">
        <h1 className="font-display text-[28px] leading-tight font-semibold tracking-tight text-text">
          Load a dataset
        </h1>
        <p className="mt-2 max-w-[58ch] text-[15px] leading-relaxed text-muted">
          It is read on your machine and stored locally. Nothing is uploaded anywhere
          beyond the API running on this computer.
        </p>

        {busy ? (
          <div className="panel mt-8 p-6">
            <div className="flex items-center gap-2.5">
              <Spinner size={16} />
              <span className="font-display text-[15px] text-text">Analysing dataset</span>
            </div>

            <div className="mt-5 h-1 overflow-hidden rounded-full bg-sunken">
              <div
                className="h-full rounded-full bg-accent transition-[width] duration-300"
                style={{ width: `${Math.max(progress * 60 + phase * 8, 6)}%` }}
              />
            </div>

            <ul className="mt-5 space-y-2">
              {PHASES.map((text, index) => (
                <li key={text} className="flex items-center gap-2.5 text-[13.5px]">
                  <span
                    className={`flex h-4 w-4 shrink-0 items-center justify-center rounded-full border text-[9px] ${
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
        ) : (
          <>
            <div
              onDragOver={(event) => {
                event.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(event) => {
                event.preventDefault();
                setDragging(false);
                const file = event.dataTransfer.files?.[0];
                if (file) void submit(file);
              }}
              className={`mt-8 rounded-xl border-2 border-dashed transition-colors ${
                dragging
                  ? "border-accent bg-[color-mix(in_oklab,var(--color-accent)_7%,transparent)]"
                  : "border-line-strong bg-surface hover:border-faint"
              }`}
            >
              <button
                type="button"
                onClick={() => inputRef.current?.click()}
                className="flex w-full flex-col items-center justify-center px-6 py-14 text-center"
              >
                <UploadIcon size={22} className={dragging ? "text-accent" : "text-faint"} />
                <span className="mt-4 font-display text-[17px] text-text">
                  Drop a dataset here
                </span>
                <span className="mt-1.5 text-[13.5px] text-muted">
                  or click to browse — CSV, TSV, Excel, JSON or Parquet, up to 50 MB
                </span>
              </button>
              <input
                ref={inputRef}
                type="file"
                accept={ACCEPT}
                className="sr-only"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) void submit(file);
                }}
              />
            </div>

            {error && (
              <div
                role="alert"
                className="shake mt-4 flex gap-2.5 rounded-lg border border-[color-mix(in_oklab,var(--color-danger)_40%,transparent)] bg-[color-mix(in_oklab,var(--color-danger)_8%,transparent)] px-4 py-3"
              >
                <AlertCircle size={15} className="mt-0.5 shrink-0 text-danger" />
                <p className="text-[14px] leading-relaxed text-text">{error}</p>
              </div>
            )}

            {recent.length > 0 && (
              <section className="mt-10">
                <h2 className="font-display text-[11px] tracking-[0.13em] text-faint uppercase">
                  Already loaded
                </h2>
                <ul className="mt-3 space-y-1.5">
                  {recent.slice(0, 6).map((dataset) => (
                    <li key={dataset.id}>
                      <button
                        onClick={() => onReady(dataset)}
                        className="card flex w-full items-center gap-3 px-3.5 py-3 text-left transition-colors hover:border-line-strong"
                      >
                        <FileSpreadsheet size={15} className="shrink-0 text-faint" />
                        <span className="min-w-0 flex-1 truncate font-mono text-[13px] text-text">
                          {dataset.filename}
                        </span>
                        <span className="tnum hidden shrink-0 font-mono text-[12px] text-faint sm:inline">
                          {dataset.n_rows.toLocaleString()} × {dataset.n_columns}
                        </span>
                        <span className="tnum shrink-0 font-mono text-[12px] text-muted">
                          {formatSize(dataset.size_bytes)}
                        </span>
                        <span
                          className={`tnum shrink-0 font-mono text-[12px] ${
                            dataset.health_score >= 75 ? "text-good" : "text-warn"
                          }`}
                        >
                          {dataset.health_score}
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              </section>
            )}

            <div className="mt-10 flex items-center gap-2 text-[13px] text-faint">
              <FolderOpen size={13} />
              <span>
                No dataset to hand? The API ships with a deliberately messy example —
                upload any CSV to try the workspace.
              </span>
            </div>

            <Button variant="ghost" size="sm" className="mt-6" onClick={onBack}>
              ← Back
            </Button>
          </>
        )}
      </main>
    </div>
  );
}
