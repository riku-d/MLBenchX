/**
 * The centre panel: the generated pipeline, in a real editor.
 *
 * The editor is not a viewer with a cursor. Someone has to be able to open this, change
 * a line, run it, and see the result — so it carries find-and-replace, folding, an error
 * marker on the failing line, and a jump-to-line from the error panel.
 *
 * Two behaviours are worth stating because they are easy to get wrong:
 *
 * Saving is debounced, not immediate. A round trip per keystroke would make typing
 * stutter and would fill the history with noise; a second of quiet is the signal that a
 * thought is finished.
 *
 * Edits are never discarded silently. Changing a toggle regenerates the code *only*
 * while it is untouched. Once edited, the generated version is still available — shown
 * side by side on request, and restorable with one button — but nothing overwrites the
 * user's work on their behalf.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import Editor, { type Monaco } from "@monaco-editor/react";
import type { editor } from "monaco-editor";
import {
  Check,
  Copy,
  Download,
  FileCode2,
  GitCompare,
  Play,
  RotateCcw,
  Search,
  WrapText,
} from "lucide-react";
import type { CompileError, Workspace } from "../api";
import { Badge, Button, Spinner } from "../components/ui";

/** Our palette, handed to Monaco so the editor is part of the app rather than a guest. */
function defineTheme(monaco: Monaco) {
  monaco.editor.defineTheme("mlbenchx", {
    base: "vs-dark",
    inherit: true,
    rules: [
      { token: "comment", foreground: "565e6e", fontStyle: "italic" },
      { token: "string", foreground: "4ba97a" },
      { token: "number", foreground: "3dd6c4" },
      { token: "keyword", foreground: "6f8fd6" },
      { token: "identifier", foreground: "e8ecf2" },
      { token: "type", foreground: "e0a33c" },
    ],
    colors: {
      "editor.background": "#0a0c0f",
      "editor.foreground": "#e8ecf2",
      "editorLineNumber.foreground": "#3a4150",
      "editorLineNumber.activeForeground": "#8b93a3",
      "editor.selectionBackground": "#1d7a7055",
      "editor.lineHighlightBackground": "#14171d",
      "editorCursor.foreground": "#3dd6c4",
      "editorIndentGuide.background1": "#1f242e",
      "editorGutter.background": "#0a0c0f",
      "editorWidget.background": "#14171d",
      "editorWidget.border": "#2b323f",
      "input.background": "#0a0c0f",
      "editorError.foreground": "#e2685f",
      "scrollbarSlider.background": "#2b323f88",
      "scrollbarSlider.hoverBackground": "#565e6e88",
    },
  });
}

export function CodePanel({
  workspace,
  compiling,
  error,
  jumpToLine,
  onCodeChange,
  onReset,
  onCompile,
}: {
  workspace: Workspace;
  compiling: boolean;
  error: CompileError | null;
  jumpToLine: number | null;
  onCodeChange: (code: string) => void;
  onReset: () => void;
  onCompile: () => void;
}) {
  const editorRef = useRef<editor.IStandaloneCodeEditor | null>(null);
  const monacoRef = useRef<Monaco | null>(null);
  const [draft, setDraft] = useState(workspace.code);
  const [saved, setSaved] = useState(true);
  const [copied, setCopied] = useState(false);
  const [wrap, setWrap] = useState(false);
  const [showDiff, setShowDiff] = useState(false);

  // The server is the source of truth for the code. When it sends a different version —
  // because a toggle regenerated it, or a history entry was restored — the editor
  // follows. The guard stops that from fighting the user's own typing.
  //
  // Replacing the value resets Monaco's scroll position and cursor, which would throw
  // someone reading line 300 back to line 1 every time they touched a toggle. The view
  // state is saved and restored around the change so the file moves under the reader
  // rather than the reader being moved.
  useEffect(() => {
    if (workspace.code === draft || !saved) return;
    const instance = editorRef.current;
    const view = instance?.saveViewState() ?? null;
    setDraft(workspace.code);
    if (view) {
      // After React has handed the new value to Monaco.
      requestAnimationFrame(() => instance?.restoreViewState(view));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [workspace.code]);

  // Debounced save. One second of quiet means the thought is finished.
  useEffect(() => {
    if (draft === workspace.code) {
      setSaved(true);
      return;
    }
    setSaved(false);
    const timer = setTimeout(() => {
      onCodeChange(draft);
      setSaved(true);
    }, 1000);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draft]);

  // Put a marker on the line that failed, so the error panel and the editor agree.
  useEffect(() => {
    const monaco = monacoRef.current;
    const instance = editorRef.current;
    if (!monaco || !instance) return;
    const model = instance.getModel();
    if (!model) return;

    monaco.editor.setModelMarkers(
      model,
      "mlbenchx",
      error?.line
        ? [
            {
              severity: monaco.MarkerSeverity.Error,
              startLineNumber: error.line,
              endLineNumber: error.line,
              startColumn: 1,
              endColumn: model.getLineMaxColumn(Math.min(error.line, model.getLineCount())),
              message: `${error.type}: ${error.message}`,
            },
          ]
        : [],
    );
  }, [error]);

  useEffect(() => {
    const instance = editorRef.current;
    if (!jumpToLine || !instance) return;
    // Position first, then reveal: revealing moves the viewport, and setting the
    // cursor afterwards can scroll it somewhere else again.
    instance.setPosition({ lineNumber: jumpToLine, column: 1 });
    instance.setSelection({
      startLineNumber: jumpToLine,
      startColumn: 1,
      endLineNumber: jumpToLine,
      endColumn: instance.getModel()?.getLineMaxColumn(jumpToLine) ?? 1,
    });
    instance.revealLineInCenter(jumpToLine);
    instance.focus();
  }, [jumpToLine]);

  const copy = useCallback(async () => {
    await navigator.clipboard.writeText(draft);
    setCopied(true);
    setTimeout(() => setCopied(false), 1600);
  }, [draft]);

  const lineCount = draft.split("\n").length;

  return (
    <div className="flex h-full min-w-0 flex-col">
      {/* ------------------------------------------------------------ toolbar */}
      <div className="flex h-11 shrink-0 items-center gap-2 border-b border-line px-3">
        <FileCode2 size={13} className="shrink-0 text-muted" />
        <span className="font-mono text-[12.5px] text-text">pipeline.py</span>

        {workspace.code_is_edited ? (
          <Badge tone="warn">edited by you</Badge>
        ) : (
          <Badge tone="neutral">generated</Badge>
        )}

        <span className="ml-1 flex items-center gap-1.5 font-mono text-[11px] text-faint">
          {saved ? (
            <>
              <span className="h-1 w-1 rounded-full bg-good" /> saved
            </>
          ) : (
            <>
              <span className="h-1 w-1 rounded-full bg-warn" /> unsaved
            </>
          )}
        </span>

        <div className="ml-auto flex items-center gap-0.5">
          <button
            onClick={() => editorRef.current?.getAction("actions.find")?.run()}
            title="Find and replace (⌘F)"
            aria-label="Find and replace"
            className="rounded-md p-1.5 text-muted transition-colors hover:bg-raised hover:text-text"
          >
            <Search size={13} />
          </button>
          <button
            onClick={() => setWrap((was) => !was)}
            title={wrap ? "Stop wrapping lines" : "Wrap long lines"}
            aria-label="Toggle line wrapping"
            aria-pressed={wrap}
            className={`rounded-md p-1.5 transition-colors hover:bg-raised ${
              wrap ? "text-accent" : "text-muted hover:text-text"
            }`}
          >
            <WrapText size={13} />
          </button>
          <button
            onClick={() => setShowDiff((was) => !was)}
            disabled={!workspace.code_is_edited}
            title="Compare with the generated version"
            aria-label="Compare with generated"
            aria-pressed={showDiff}
            className={`rounded-md p-1.5 transition-colors hover:bg-raised disabled:opacity-35 ${
              showDiff ? "text-accent" : "text-muted hover:text-text"
            }`}
          >
            <GitCompare size={13} />
          </button>
          <button
            onClick={copy}
            title="Copy the whole file"
            aria-label="Copy code"
            className="rounded-md p-1.5 text-muted transition-colors hover:bg-raised hover:text-text"
          >
            {copied ? <Check size={13} className="text-good" /> : <Copy size={13} />}
          </button>
          <a
            href={`/api/workspaces/${workspace.id}/download/code`}
            title="Download pipeline.py"
            aria-label="Download code"
            className="rounded-md p-1.5 text-muted transition-colors hover:bg-raised hover:text-text"
          >
            <Download size={13} />
          </a>
          <button
            onClick={onReset}
            disabled={!workspace.code_is_edited}
            title="Discard your edits and regenerate from the pipeline"
            aria-label="Reset code"
            className="rounded-md p-1.5 text-muted transition-colors hover:bg-raised hover:text-text disabled:opacity-35"
          >
            <RotateCcw size={13} />
          </button>

          <div className="mx-1 h-4 w-px bg-line" />

          <Button variant="primary" size="sm" onClick={onCompile} disabled={compiling}>
            {compiling ? <Spinner size={12} /> : <Play size={12} />}
            {compiling ? "Compiling…" : "Compile"}
          </Button>
        </div>
      </div>

      {/* ------------------------------------------------------------- editor */}
      <div className="monaco-host relative min-h-0 flex-1">
        {showDiff && workspace.code_is_edited && (
          <div className="absolute top-2 right-3 z-10 rounded-md border border-line bg-raised px-2 py-1 font-mono text-[11px] text-muted">
            left: generated · right: yours
          </div>
        )}
        <Editor
          height="100%"
          language="python"
          theme="mlbenchx"
          value={draft}
          beforeMount={defineTheme}
          onMount={(instance, monaco) => {
            editorRef.current = instance;
            monacoRef.current = monaco;
            // ⌘/Ctrl+Enter compiles from anywhere in the editor: the action people
            // repeat most should not require reaching for the mouse.
            instance.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, onCompile);
          }}
          onChange={(value) => setDraft(value ?? "")}
          options={{
            fontSize: 12.5,
            fontFamily: "'IBM Plex Mono', ui-monospace, monospace",
            lineHeight: 1.75,
            minimap: { enabled: false },
            scrollBeyondLastLine: false,
            renderLineHighlight: "line",
            smoothScrolling: true,
            cursorBlinking: "smooth",
            padding: { top: 14, bottom: 14 },
            wordWrap: wrap ? "on" : "off",
            tabSize: 4,
            rulers: [88],
            bracketPairColorization: { enabled: true },
            folding: true,
            lineNumbersMinChars: 3,
            overviewRulerBorder: false,
            scrollbar: { verticalScrollbarSize: 10, horizontalScrollbarSize: 10 },
          }}
          loading={
            <div className="flex h-full items-center justify-center gap-2 text-[13px] text-faint">
              <Spinner size={14} /> Loading the editor…
            </div>
          }
        />
      </div>

      {/* ------------------------------------------------------------- status */}
      <div className="flex h-7 shrink-0 items-center gap-3 border-t border-line bg-surface px-3 font-mono text-[11px] text-faint">
        <span className="tnum">{lineCount} lines</span>
        <span>Python</span>
        {error?.line && (
          <span className="text-danger">error on line {error.line}</span>
        )}
        <span className="ml-auto">⌘↵ to compile</span>
      </div>
    </div>
  );
}
