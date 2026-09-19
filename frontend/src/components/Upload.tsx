import { useRef, useState } from "react";

type Props = {
  onAnalyse: (file: File, goal: string) => void;
  onExample: (goal: string) => void;
  error: string | null;
};

const ACCEPT = ".csv,.tsv,.xlsx,.xls,.txt";

export function Upload({ onAnalyse, onExample, error }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [goal, setGoal] = useState("which students will fail");
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  function submit(event: React.FormEvent) {
    event.preventDefault();
    if (file) onAnalyse(file, goal);
  }

  return (
    <div className="mx-auto max-w-[640px] px-5 pt-20 pb-24 sm:pt-28">
      <header>
        <h1 className="font-display text-[40px] leading-none font-semibold tracking-tight text-ink sm:text-[52px]">
          ML-BenchX
        </h1>
        <p className="mt-3 max-w-[46ch] text-[19px] leading-snug text-slate">
          See exactly what happens to your data, every change in plain English, with the
          reason for it.
        </p>
      </header>

      <form onSubmit={submit} className="mt-10">
        <label
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            const dropped = e.dataTransfer.files?.[0];
            if (dropped) setFile(dropped);
          }}
          className={`flex min-h-[168px] cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-8 text-center transition-colors ${
            dragging ? "border-xray bg-paper" : "border-slate/30 bg-paper/60 hover:border-slate/50"
          }`}
        >
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPT}
            className="sr-only"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
          {file ? (
            <>
              <span className="font-mono text-[15px] break-all text-ink">{file.name}</span>
              <span className="mt-1.5 text-[14px] text-slate">
                {(file.size / 1024).toFixed(0)} KB · click to choose a different file
              </span>
            </>
          ) : (
            <>
              <span className="font-display text-[19px] text-ink">Drop a spreadsheet here</span>
              <span className="mt-1.5 text-[14px] text-slate">
                CSV or Excel · or click to browse
              </span>
            </>
          )}
        </label>

        <div className="mt-7">
          <label htmlFor="goal" className="block font-display text-[17px] text-ink">
            What do you want to predict?
          </label>
          <input
            id="goal"
            value={goal}
            onChange={(e) => setGoal(e.target.value)}
            placeholder="which students will fail"
            className="mt-2 w-full rounded-lg border border-slate/25 bg-paper px-3.5 py-3 font-sans text-[17px] text-ink placeholder:text-slate/60 focus:border-xray focus:outline-none"
          />
          <p className="mt-1.5 text-[14px] text-slate">
            Plain English is fine. We work out which column you mean and show you our guess.
          </p>
        </div>

        {error && (
          <p
            role="alert"
            className="mt-6 rounded-lg border border-ochre/35 bg-ochre/8 px-4 py-3 text-[15px] text-ochre"
          >
            {error}
          </p>
        )}

        <div className="mt-8 flex flex-wrap items-center gap-x-6 gap-y-3">
          <button
            type="submit"
            disabled={!file}
            className="rounded-lg bg-ink px-5 py-3 font-display text-[16px] text-paper transition-opacity disabled:cursor-not-allowed disabled:opacity-35"
          >
            Analyse my data
          </button>
          <button
            type="button"
            onClick={() => onExample(goal)}
            className="font-sans text-[16px] text-xray underline-offset-4 hover:underline"
          >
            Try an example →
          </button>
        </div>
      </form>
    </div>
  );
}
