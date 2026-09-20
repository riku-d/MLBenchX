/**
 * The bottom dock: the data itself, the run log, and the trail of changes.
 *
 * The data table is the thing that makes the rest concrete. A step card saying "885
 * blanks filled" is a claim; a cell that was empty and now holds 28.0 is the evidence.
 * So blanks and outliers are marked in place, and the original and processed views sit
 * behind one switch rather than on separate screens.
 *
 * Highlighting is computed on the server. The browser holds one page of rows and cannot
 * know where a column's quartiles fall, so a client-side rule would mark a different
 * set of cells on every page — which is worse than marking none.
 */

import { useCallback, useEffect, useState } from "react";
import {
  AlertOctagon,
  ChevronLeft,
  ChevronRight,
  Clock,
  Database,
  RotateCcw,
  ScrollText,
  Search,
  Terminal,
} from "lucide-react";
import { api, type Compilation, type HistoryItem, type PreviewPage, type Workspace } from "../api";
import { Badge, EmptyState, Segmented, Skeleton, Spinner, Tabs } from "../components/ui";

const KIND_TONE: Record<string, "neutral" | "accent" | "warn" | "good"> = {
  created: "neutral",
  config: "accent",
  code: "warn",
  compile: "good",
  restore: "warn",
};

// ---------------------------------------------------------------------------------
// Data table
// ---------------------------------------------------------------------------------

function DataTable({
  workspace,
  hasProcessed,
}: {
  workspace: Workspace;
  hasProcessed: boolean;
}) {
  const [which, setWhich] = useState<"original" | "processed">("original");
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState("");
  const [descending, setDescending] = useState(false);
  const [data, setData] = useState<PreviewPage | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const view = hasProcessed ? which : "original";

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(
        await api.preview(workspace.id, {
          which: view, page, pageSize: 25, search, sort, descending,
        }),
      );
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not load the rows.");
      setData(null);
    } finally {
      setLoading(false);
    }
  }, [workspace.id, view, page, search, sort, descending]);

  useEffect(() => {
    const timer = setTimeout(load, search ? 300 : 0);
    return () => clearTimeout(timer);
  }, [load, search]);

  // A new compile means new processed rows; re-read rather than showing stale ones.
  useEffect(() => {
    if (view === "processed") void load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [workspace.latest_compilation?.id]);

  return (
    <div className="flex h-full flex-col">
      <div className="flex shrink-0 flex-wrap items-center gap-2 border-b border-line px-3 py-2">
        <Segmented
          size="sm"
          label="Dataset view"
          value={view}
          onChange={(next) => {
            setWhich(next);
            setPage(1);
          }}
          options={[
            { value: "original", label: "Original" },
            {
              value: "processed",
              label: hasProcessed ? "Processed" : "Processed (compile first)",
            },
          ]}
        />

        <div className="relative">
          <Search size={12} className="absolute top-1/2 left-2 -translate-y-1/2 text-faint" />
          <input
            value={search}
            onChange={(event) => {
              setSearch(event.target.value);
              setPage(1);
            }}
            placeholder="Search rows"
            aria-label="Search rows"
            className="h-7 w-44 rounded-md border border-line bg-sunken pr-2 pl-7 text-[12px]
              text-text placeholder:text-faint focus:border-accent focus:outline-none"
          />
        </div>

        {data && (
          <span className="tnum font-mono text-[11px] text-faint">
            {data.total_rows.toLocaleString()} rows · {data.columns.length} columns
          </span>
        )}

        <div className="ml-auto flex items-center gap-1">
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page <= 1}
            aria-label="Previous page"
            className="rounded-md p-1 text-muted transition-colors hover:bg-raised hover:text-text disabled:opacity-30"
          >
            <ChevronLeft size={13} />
          </button>
          <span className="tnum font-mono text-[11px] text-faint">
            {page} / {data?.total_pages ?? 1}
          </span>
          <button
            onClick={() => setPage((p) => Math.min(data?.total_pages ?? 1, p + 1))}
            disabled={!data || page >= data.total_pages}
            aria-label="Next page"
            className="rounded-md p-1 text-muted transition-colors hover:bg-raised hover:text-text disabled:opacity-30"
          >
            <ChevronRight size={13} />
          </button>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-auto">
        {error ? (
          <EmptyState icon={<AlertOctagon size={18} />} title="Could not load rows" description={error} />
        ) : loading && !data ? (
          <div className="space-y-1.5 p-3">
            {Array.from({ length: 6 }, (_, index) => (
              <Skeleton key={index} className="h-6 w-full" />
            ))}
          </div>
        ) : !data || data.rows.length === 0 ? (
          <EmptyState
            icon={<Database size={18} />}
            title="No rows to show"
            description={search ? `Nothing matches “${search}”.` : "This view is empty."}
          />
        ) : (
          <table className="w-full border-collapse text-[12px]">
            <thead className="sticky top-0 z-10 bg-surface">
              <tr>
                <th className="w-10 border-b border-line px-2 py-1.5 text-right font-mono text-[10px] font-normal text-faint">
                  #
                </th>
                {data.columns.map((column) => (
                  <th
                    key={column.name}
                    className="border-b border-line px-2.5 py-1.5 text-left font-normal"
                  >
                    <button
                      onClick={() => {
                        if (sort === column.name) setDescending((was) => !was);
                        else {
                          setSort(column.name);
                          setDescending(false);
                        }
                      }}
                      className="group flex max-w-[190px] items-center gap-1.5"
                      title={`Sort by ${column.name}`}
                    >
                      <span className="truncate font-mono text-[11.5px] text-text">
                        {column.name}
                      </span>
                      <span className="shrink-0 text-[9px] text-faint">
                        {sort === column.name ? (descending ? "▼" : "▲") : ""}
                      </span>
                    </button>
                    <div className="mt-0.5 flex items-center gap-1.5">
                      <span className="text-[9.5px] text-faint">{column.kind}</span>
                      {column.missing > 0 && (
                        <span className="tnum text-[9.5px] text-warn">
                          {column.missing} blank
                        </span>
                      )}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.rows.map((row) => (
                <tr key={row.index} className="transition-colors hover:bg-raised/60">
                  <td className="tnum border-b border-line/50 px-2 py-1 text-right font-mono text-[10px] text-faint">
                    {row.index + 1}
                  </td>
                  {data.columns.map((column) => {
                    const cell = row.cells[column.name];
                    return (
                      <td
                        key={column.name}
                        className={`max-w-[190px] truncate border-b border-line/50 px-2.5 py-1 font-mono text-[11.5px] ${
                          cell?.missing
                            ? "bg-[color-mix(in_oklab,var(--color-warn)_10%,transparent)] text-warn"
                            : cell?.outlier
                              ? "bg-[color-mix(in_oklab,var(--color-danger)_9%,transparent)] text-danger"
                              : "text-muted"
                        }`}
                        title={
                          cell?.missing
                            ? "Missing value"
                            : cell?.outlier
                              ? "Outside the usual range for this column"
                              : String(cell?.value ?? "")
                        }
                      >
                        {cell?.missing ? "—" : String(cell?.value ?? "")}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="flex h-6 shrink-0 items-center gap-3 border-t border-line px-3 font-mono text-[10.5px] text-faint">
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-sm bg-[color-mix(in_oklab,var(--color-warn)_35%,transparent)]" />
          missing
        </span>
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-sm bg-[color-mix(in_oklab,var(--color-danger)_30%,transparent)]" />
          outlier
        </span>
        {loading && <Spinner size={10} />}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------------
// Logs
// ---------------------------------------------------------------------------------

function LogView({
  compilation,
  onJumpToLine,
}: {
  compilation: Compilation | null;
  onJumpToLine: (line: number) => void;
}) {
  if (!compilation) {
    return (
      <EmptyState
        icon={<Terminal size={18} />}
        title="Nothing has run yet"
        description="Compile the pipeline and its output will appear here, line by line."
      />
    );
  }

  const error = compilation.error;

  return (
    <div className="h-full overflow-y-auto p-3">
      <div className="mb-2.5 flex flex-wrap items-center gap-2">
        <Badge tone={compilation.status === "success" ? "good" : "danger"}>
          {compilation.status === "success" ? "Succeeded" : "Failed"}
        </Badge>
        <span className="tnum font-mono text-[11px] text-faint">
          {compilation.duration_ms} ms
        </span>
        <span className="font-mono text-[11px] text-faint">from the {compilation.source}</span>
        <span className="font-mono text-[11px] text-faint">
          {compilation.sandbox === "os_sandbox" ? "OS sandbox" : "process limits"}
        </span>
      </div>

      {error && (
        <div className="shake mb-3 rounded-lg border border-[color-mix(in_oklab,var(--color-danger)_40%,transparent)] bg-[color-mix(in_oklab,var(--color-danger)_7%,transparent)] p-3">
          <div className="flex items-center gap-2">
            <AlertOctagon size={13} className="shrink-0 text-danger" />
            <span className="font-mono text-[12.5px] text-danger">{error.type}</span>
            {error.line && (
              <button
                onClick={() => onJumpToLine(error.line!)}
                className="ml-auto rounded-md border border-line px-2 py-0.5 font-mono text-[11px]
                  text-muted transition-colors hover:border-line-strong hover:text-text"
              >
                Go to line {error.line} →
              </button>
            )}
          </div>
          <p className="mt-2 font-mono text-[12px] leading-relaxed text-text">{error.message}</p>
          {error.cause && (
            <p className="mt-2.5 text-[12.5px] leading-relaxed text-muted">
              <span className="text-faint">Likely cause — </span>
              {error.cause}
            </p>
          )}
          {error.suggestion && (
            <p className="mt-1.5 text-[12.5px] leading-relaxed text-muted">
              <span className="text-faint">Try — </span>
              {error.suggestion}
            </p>
          )}
          {error.traceback && (
            <details className="mt-2.5">
              <summary className="cursor-pointer list-none text-[11.5px] text-faint hover:text-muted">
                Full traceback ›
              </summary>
              <pre className="mt-1.5 overflow-x-auto rounded-md bg-sunken p-2.5 font-mono text-[10.5px] leading-relaxed text-faint">
                {error.traceback}
              </pre>
            </details>
          )}
        </div>
      )}

      <pre className="font-mono text-[11.5px] leading-[1.9] whitespace-pre-wrap text-muted">
        {compilation.logs.map((line, index) => (
          <div key={index} className="flex gap-2.5">
            <span className="tnum w-6 shrink-0 text-right text-faint">{index + 1}</span>
            <span className={line.text.startsWith("[") ? "text-text" : ""}>{line.text}</span>
          </div>
        ))}
      </pre>
    </div>
  );
}

// ---------------------------------------------------------------------------------
// History
// ---------------------------------------------------------------------------------

function HistoryView({
  workspaceId,
  version,
  onRestore,
}: {
  workspaceId: string;
  version: string;
  onRestore: (entryId: number) => void;
}) {
  const [entries, setEntries] = useState<HistoryItem[] | null>(null);

  useEffect(() => {
    api.history(workspaceId).then(setEntries).catch(() => setEntries([]));
  }, [workspaceId, version]);

  if (!entries) {
    return (
      <div className="space-y-1.5 p-3">
        {Array.from({ length: 4 }, (_, index) => (
          <Skeleton key={index} className="h-9 w-full" />
        ))}
      </div>
    );
  }

  if (entries.length === 0) {
    return (
      <EmptyState
        icon={<Clock size={18} />}
        title="No changes yet"
        description="Every toggle, edit and compile is recorded here, and any of them can be restored."
      />
    );
  }

  return (
    <ol className="h-full overflow-y-auto p-3">
      {entries.map((entry, index) => (
        <li key={entry.id} className="relative flex gap-3 pb-3">
          <div className="flex flex-col items-center">
            <span
              className={`mt-1.5 h-2 w-2 shrink-0 rounded-full ${
                index === 0 ? "bg-accent" : "bg-line-strong"
              }`}
            />
            {index < entries.length - 1 && <span className="mt-1 w-px flex-1 bg-line" />}
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex items-baseline gap-2">
              <Badge tone={KIND_TONE[entry.kind] ?? "neutral"}>{entry.kind}</Badge>
              <span className="tnum font-mono text-[10.5px] text-faint">
                {new Date(entry.created_at).toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                })}
              </span>
              {entry.restorable && index > 0 && (
                <button
                  onClick={() => onRestore(entry.id)}
                  className="ml-auto flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px]
                    text-muted transition-colors hover:bg-raised hover:text-text"
                >
                  <RotateCcw size={10} /> Restore
                </button>
              )}
            </div>
            <p className="mt-1 text-[12.5px] leading-snug text-text">{entry.summary}</p>
          </div>
        </li>
      ))}
    </ol>
  );
}

// ---------------------------------------------------------------------------------

export function BottomDock({
  workspace,
  onJumpToLine,
  onRestore,
}: {
  workspace: Workspace;
  onJumpToLine: (line: number) => void;
  onRestore: (entryId: number) => void;
}) {
  const [tab, setTab] = useState<"data" | "logs" | "history">("data");
  const compilation = workspace.latest_compilation;
  const hasProcessed = compilation?.status === "success";

  // A failed run is the thing the user most needs to read, so surface it without
  // making them go looking for the right tab.
  useEffect(() => {
    if (compilation?.status === "error") setTab("logs");
  }, [compilation?.id, compilation?.status]);

  return (
    <div className="flex h-full flex-col">
      <div className="flex h-10 shrink-0 items-center gap-3 border-b border-line px-3">
        <Tabs
          label="Bottom panel"
          value={tab}
          onChange={setTab}
          options={[
            { value: "data", label: "Data" },
            { value: "logs", label: "Logs" },
            { value: "history", label: "History" },
          ]}
        />
        {compilation?.status === "error" && tab !== "logs" && (
          <button
            onClick={() => setTab("logs")}
            className="flex items-center gap-1.5 text-[11.5px] text-danger hover:underline"
          >
            <AlertOctagon size={11} /> the last run failed
          </button>
        )}
        <div className="ml-auto flex items-center gap-2 font-mono text-[10.5px] text-faint">
          <ScrollText size={11} />
          {workspace.dataset.filename}
        </div>
      </div>

      <div className="min-h-0 flex-1">
        {tab === "data" && <DataTable workspace={workspace} hasProcessed={!!hasProcessed} />}
        {tab === "logs" && <LogView compilation={compilation} onJumpToLine={onJumpToLine} />}
        {tab === "history" && (
          <HistoryView
            workspaceId={workspace.id}
            version={workspace.updated_at}
            onRestore={onRestore}
          />
        )}
      </div>
    </div>
  );
}
