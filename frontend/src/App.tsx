import { useCallback, useState } from "react";
import type { Analysis } from "./api";
import { analyse, analyseExample } from "./api";
import { Results } from "./components/Results";
import { Upload } from "./components/Upload";
import { Working } from "./components/Working";

/**
 * The server answers in tens of milliseconds. Holding the working state for this
 * long lets the four phase labels actually be read — without it the screen
 * flickers and the user cannot tell whether anything happened.
 */
const WORKING_FLOOR_MS = 1200;

/** What produced the current result, so changing the target can re-run it. */
type Source = { kind: "file"; file: File; goal: string } | { kind: "example"; goal: string };

export default function App() {
  const [phase, setPhase] = useState<"upload" | "working" | "results">("upload");
  const [data, setData] = useState<Analysis | null>(null);
  const [source, setSource] = useState<Source | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const run = useCallback(
    async (next: Source, target?: string, showWorking = true) => {
      setError(null);
      setBusy(true);
      if (showWorking) setPhase("working");

      const started = performance.now();
      try {
        const result =
          next.kind === "file"
            ? await analyse(next.file, next.goal, target)
            : await analyseExample(next.goal, target);

        if (showWorking) {
          const remaining = WORKING_FLOOR_MS - (performance.now() - started);
          if (remaining > 0) await new Promise((resolve) => setTimeout(resolve, remaining));
        }
        setData(result);
        setSource(next);
        setPhase("results");
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : "Something went wrong.");
        setPhase(data ? "results" : "upload");
      } finally {
        setBusy(false);
      }
    },
    [data],
  );

  if (phase === "working") return <Working />;

  if (phase === "results" && data) {
    return (
      <>
        {error && (
          <p
            role="alert"
            className="mx-auto mt-6 max-w-[800px] rounded-lg border border-ochre/35 bg-ochre/8 px-4 py-3 text-[15px] text-ochre"
          >
            {error}
          </p>
        )}
        <Results
          data={data}
          busy={busy}
          onRetarget={(column) => source && run(source, column, false)}
          onReset={() => {
            setData(null);
            setSource(null);
            setError(null);
            setPhase("upload");
          }}
        />
      </>
    );
  }

  return (
    <Upload
      error={error}
      onAnalyse={(file, goal) => run({ kind: "file", file, goal })}
      onExample={(goal) => run({ kind: "example", goal })}
    />
  );
}
