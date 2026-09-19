import type { Strip } from "../api";

/**
 * Three real values from the user's own file, before and after.
 *
 * This is the one place the design spends its boldness. Everything else on the
 * page is grey on near-white; a changed value is a solid block of the accent
 * colour. That contrast is the whole argument — you can see, in your own data,
 * exactly which cell moved and what it became.
 */
export function TransformStrip({ strip }: { strip: Strip }) {
  const width = strip.before.length;

  return (
    <figure className="mt-5">
      <figcaption className="mb-2 flex items-baseline gap-2">
        <span className="font-mono text-[13px] text-ink">{strip.column}</span>
        <span className="text-[13px] text-slate">three rows from your file</span>
      </figcaption>

      <div className="overflow-x-auto rounded-lg border border-slate/20 bg-bone">
        <table className="w-full min-w-[380px] border-collapse">
          <caption className="sr-only">
            Values in {strip.column} before and after this step
          </caption>
          <tbody>
            <tr>
              <th
                scope="row"
                className="w-[4.5rem] border-r border-slate/15 px-3 py-2.5 text-left align-middle font-sans text-[12px] font-normal tracking-wide text-slate uppercase"
              >
                before
              </th>
              {strip.before.map((value, i) => (
                <td
                  key={i}
                  className="px-3 py-2.5 text-center font-mono text-[15px] text-slate tabular-nums"
                  style={{ width: `${100 / width}%` }}
                >
                  {value}
                </td>
              ))}
            </tr>
            <tr className="border-t border-slate/15">
              <th
                scope="row"
                className="border-r border-slate/15 px-3 py-2.5 text-left align-middle font-sans text-[12px] font-normal tracking-wide text-slate uppercase"
              >
                after
              </th>
              {strip.after.map((value, i) => (
                <td key={i} className="px-2 py-2 text-center">
                  {strip.changed[i] ? (
                    <span
                      className="inline-block rounded-md bg-xray px-2.5 py-1 font-mono text-[15px] font-medium text-paper tabular-nums"
                      title="this value changed"
                    >
                      {value}
                    </span>
                  ) : (
                    <span className="font-mono text-[15px] text-ink tabular-nums">{value}</span>
                  )}
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>
    </figure>
  );
}

/** The split card gets a proportion bar instead — no values changed, rows moved. */
export function SplitBar({ train, test, percent }: { train: number; test: number; percent: number }) {
  const trainPct = 100 - percent;
  return (
    <figure className="mt-5">
      <div className="flex h-11 w-full overflow-hidden rounded-lg border border-slate/20">
        <div
          className="flex items-center justify-center bg-xray font-mono text-[13px] font-medium text-paper"
          style={{ width: `${trainPct}%` }}
        >
          {train.toLocaleString()} learned from
        </div>
        <div
          className="flex items-center justify-center bg-bone font-mono text-[13px] text-slate"
          style={{
            width: `${percent}%`,
            backgroundImage:
              "repeating-linear-gradient(45deg, transparent, transparent 5px, rgba(100,116,139,.18) 5px, rgba(100,116,139,.18) 10px)",
          }}
        >
          {test.toLocaleString()}
        </div>
      </div>
      <figcaption className="mt-2 text-[13px] text-slate">
        The hatched part was set aside before anything was measured.
      </figcaption>
    </figure>
  );
}
