/**
 * The small, shared pieces — the ones used often enough that a second copy would
 * eventually diverge from the first.
 *
 * Everything here is built on the tokens in `index.css` rather than raw colours, so a
 * change to the palette moves the whole app at once. Everything here is also
 * keyboard-reachable and labelled: a toggle that only a mouse can reach would make the
 * central interaction of this product unavailable to some of the people using it.
 */

import { useEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { X } from "lucide-react";

// ---------------------------------------------------------------------------------
// Buttons
// ---------------------------------------------------------------------------------

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";
type ButtonSize = "sm" | "md" | "lg";

const BUTTON_VARIANTS: Record<ButtonVariant, string> = {
  primary:
    "bg-accent text-void hover:bg-[color-mix(in_oklab,var(--color-accent)_88%,white)] " +
    "disabled:bg-accent-dim disabled:text-muted",
  secondary:
    "bg-raised text-text border border-line-strong hover:border-faint hover:bg-[color-mix(in_oklab,var(--color-raised)_70%,white_4%)]",
  ghost: "text-muted hover:text-text hover:bg-raised",
  danger:
    "bg-transparent text-danger border border-[color-mix(in_oklab,var(--color-danger)_40%,transparent)] " +
    "hover:bg-[color-mix(in_oklab,var(--color-danger)_12%,transparent)]",
};

const BUTTON_SIZES: Record<ButtonSize, string> = {
  sm: "h-8 px-3 text-[13px] gap-1.5 rounded-md",
  md: "h-9 px-3.5 text-[14px] gap-2 rounded-lg",
  lg: "h-11 px-5 text-[15px] gap-2 rounded-lg",
};

export function Button({
  variant = "secondary",
  size = "md",
  className = "",
  children,
  ...rest
}: {
  variant?: ButtonVariant;
  size?: ButtonSize;
  className?: string;
  children: ReactNode;
} & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      className={`inline-flex items-center justify-center font-medium whitespace-nowrap transition-colors
        disabled:cursor-not-allowed disabled:opacity-55
        ${BUTTON_VARIANTS[variant]} ${BUTTON_SIZES[size]} ${className}`}
      {...rest}
    >
      {children}
    </button>
  );
}

// ---------------------------------------------------------------------------------
// Toggle
// ---------------------------------------------------------------------------------

/**
 * The switch on every step card — the most-used control in the product.
 *
 * It is a real `<button role="switch">` with `aria-checked`, not a styled checkbox,
 * because a screen reader has to announce the state and because the page needs to be
 * able to intercept the change to ask "are you sure?" before it happens.
 */
export function Toggle({
  checked,
  onChange,
  disabled = false,
  label,
  size = "md",
}: {
  checked: boolean;
  onChange: (next: boolean) => void;
  disabled?: boolean;
  label: string;
  size?: "sm" | "md";
}) {
  const dimensions = size === "sm" ? "h-4.5 w-8" : "h-5 w-9";
  const knob = size === "sm" ? "h-3 w-3" : "h-3.5 w-3.5";
  const travel = size === "sm" ? "translate-x-3.5" : "translate-x-4";

  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={`relative inline-flex shrink-0 items-center rounded-full border transition-colors
        disabled:cursor-not-allowed disabled:opacity-45 ${dimensions}
        ${
          checked
            ? "border-accent bg-[color-mix(in_oklab,var(--color-accent)_28%,transparent)]"
            : "border-line-strong bg-sunken"
        }`}
    >
      <span
        className={`pointer-events-none ml-0.5 rounded-full transition-transform duration-200 ${knob}
          ${checked ? `${travel} bg-accent` : "translate-x-0 bg-faint"}`}
      />
    </button>
  );
}

// ---------------------------------------------------------------------------------
// Segmented control
// ---------------------------------------------------------------------------------

/** A short, mutually exclusive choice — scaler, outlier action, preview tab. */
export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
  size = "md",
}: {
  value: T;
  options: { value: T; label: string; hint?: string }[];
  onChange: (next: T) => void;
  label: string;
  size?: "sm" | "md";
}) {
  return (
    <div
      role="radiogroup"
      aria-label={label}
      className="inline-flex rounded-lg border border-line bg-sunken p-0.5"
    >
      {options.map((option) => {
        const active = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={active}
            title={option.hint}
            onClick={() => onChange(option.value)}
            className={`rounded-md font-medium transition-colors
              ${size === "sm" ? "px-2 py-1 text-[12px]" : "px-2.5 py-1.5 text-[13px]"}
              ${active ? "bg-raised text-text shadow-sm" : "text-muted hover:text-text"}`}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------------
// Select
// ---------------------------------------------------------------------------------

export function Select<T extends string>({
  value,
  options,
  onChange,
  label,
  id,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (next: T) => void;
  label: string;
  id?: string;
}) {
  return (
    <select
      id={id}
      aria-label={label}
      value={value}
      onChange={(event) => onChange(event.target.value as T)}
      className="h-8 rounded-md border border-line bg-sunken px-2 text-[13px] text-text
        transition-colors hover:border-line-strong focus:border-accent focus:outline-none"
    >
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  );
}

// ---------------------------------------------------------------------------------
// Dialog
// ---------------------------------------------------------------------------------

/**
 * A modal, in a portal, with focus trapped inside it.
 *
 * The trap matters here more than usual: the most important dialog in this app is the
 * one that asks whether you really want to disable a step, and a dialog you can tab
 * out of is one you can act behind without reading.
 */
export function Dialog({
  open,
  onClose,
  title,
  children,
  footer,
  width = "max-w-lg",
  tone = "neutral",
}: {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  children: ReactNode;
  footer?: ReactNode;
  width?: string;
  tone?: "neutral" | "warn" | "danger";
}) {
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const previous = document.activeElement as HTMLElement | null;
    panelRef.current?.focus();

    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        onClose();
        return;
      }
      if (event.key !== "Tab" || !panelRef.current) return;
      const focusable = panelRef.current.querySelectorAll<HTMLElement>(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
      );
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }

    document.addEventListener("keydown", onKeyDown);
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.body.style.overflow = "";
      previous?.focus();
    };
  }, [open, onClose]);

  if (!open) return null;

  const accentBorder = {
    neutral: "border-line-strong",
    warn: "border-[color-mix(in_oklab,var(--color-warn)_45%,transparent)]",
    danger: "border-[color-mix(in_oklab,var(--color-danger)_45%,transparent)]",
  }[tone];

  return createPortal(
    <div className="fade fixed inset-0 z-50 flex items-center justify-center p-4">
      <button
        aria-label="Close"
        tabIndex={-1}
        onClick={onClose}
        className="absolute inset-0 cursor-default bg-void/80 backdrop-blur-sm"
      />
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-label={typeof title === "string" ? title : undefined}
        tabIndex={-1}
        className={`rise relative max-h-[86vh] w-full overflow-y-auto rounded-xl border bg-surface
          shadow-2xl focus:outline-none ${accentBorder} ${width}`}
      >
        <div className="flex items-start justify-between gap-4 border-b border-line px-5 py-4">
          <h2 className="font-display text-[17px] leading-snug font-semibold text-text">{title}</h2>
          <button
            onClick={onClose}
            aria-label="Close"
            className="-mt-0.5 rounded-md p-1 text-faint transition-colors hover:bg-raised hover:text-text"
          >
            <X size={16} />
          </button>
        </div>
        <div className="px-5 py-4">{children}</div>
        {footer && (
          <div className="flex flex-wrap justify-end gap-2 border-t border-line px-5 py-4">
            {footer}
          </div>
        )}
      </div>
    </div>,
    document.body,
  );
}

// ---------------------------------------------------------------------------------
// Status and feedback
// ---------------------------------------------------------------------------------

export type Tone = "neutral" | "good" | "warn" | "danger" | "info" | "accent";

const TONE_CLASSES: Record<Tone, string> = {
  neutral: "border-line-strong text-muted",
  good: "border-[color-mix(in_oklab,var(--color-good)_40%,transparent)] text-good",
  warn: "border-[color-mix(in_oklab,var(--color-warn)_40%,transparent)] text-warn",
  danger: "border-[color-mix(in_oklab,var(--color-danger)_40%,transparent)] text-danger",
  info: "border-[color-mix(in_oklab,var(--color-info)_40%,transparent)] text-info",
  accent: "border-[color-mix(in_oklab,var(--color-accent)_40%,transparent)] text-accent",
};

export function Badge({
  tone = "neutral",
  children,
  className = "",
}: {
  tone?: Tone;
  children: ReactNode;
  className?: string;
}) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px]
        font-medium whitespace-nowrap ${TONE_CLASSES[tone]} ${className}`}
    >
      {children}
    </span>
  );
}

/**
 * A meter with the number beside it.
 *
 * The number is what people read; the bar is for comparing several at a glance. Both
 * are present because either alone is weaker — a bar with no value cannot be quoted,
 * and a column of bare numbers cannot be scanned.
 */
export function Meter({
  value,
  max = 100,
  tone = "accent",
  label,
  showValue = true,
}: {
  value: number;
  max?: number;
  tone?: Tone;
  label: string;
  showValue?: boolean;
}) {
  const fraction = Math.max(0, Math.min(1, value / max));
  const fill = {
    neutral: "var(--color-faint)",
    good: "var(--color-good)",
    warn: "var(--color-warn)",
    danger: "var(--color-danger)",
    info: "var(--color-info)",
    accent: "var(--color-accent)",
  }[tone];

  return (
    <div className="flex items-center gap-2.5">
      <div
        role="meter"
        aria-label={label}
        aria-valuenow={Math.round(value)}
        aria-valuemin={0}
        aria-valuemax={max}
        className="h-1.5 flex-1 overflow-hidden rounded-full bg-sunken"
      >
        <div
          className="h-full rounded-full transition-[width] duration-500 ease-out"
          style={{ width: `${fraction * 100}%`, background: fill }}
        />
      </div>
      {showValue && (
        <span className="tnum w-9 shrink-0 text-right font-mono text-[12px] text-muted">
          {Math.round(value)}
        </span>
      )}
    </div>
  );
}

/** Health as a ring — the one number the whole workspace is oriented around. */
export function ScoreRing({
  score,
  size = 96,
  label = "Dataset health",
}: {
  score: number;
  size?: number;
  label?: string;
}) {
  const stroke = size > 80 ? 7 : 5;
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const tone =
    score >= 90 ? "var(--color-good)"
    : score >= 75 ? "var(--color-accent)"
    : score >= 55 ? "var(--color-warn)"
    : "var(--color-danger)";

  return (
    <div
      className="relative shrink-0"
      style={{ width: size, height: size }}
      role="meter"
      aria-label={label}
      aria-valuenow={score}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <svg width={size} height={size} className="-rotate-90">
        <circle
          cx={size / 2} cy={size / 2} r={radius}
          fill="none" stroke="var(--color-line)" strokeWidth={stroke}
        />
        <circle
          cx={size / 2} cy={size / 2} r={radius}
          fill="none" stroke={tone} strokeWidth={stroke} strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={circumference * (1 - Math.max(0, Math.min(100, score)) / 100)}
          style={{ transition: "stroke-dashoffset 700ms cubic-bezier(0.22,1,0.36,1), stroke 400ms" }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span
          className="tnum font-mono leading-none font-medium text-text"
          style={{ fontSize: size > 80 ? 26 : 17 }}
        >
          {Math.round(score)}
        </span>
        {size > 80 && <span className="mt-1 text-[10px] tracking-wide text-faint">/ 100</span>}
      </div>
    </div>
  );
}

/** A signed change. Direction is carried by the arrow and the word, not by colour alone. */
export function Delta({
  value,
  suffix = "",
  invert = false,
}: {
  value: number;
  suffix?: string;
  invert?: boolean;
}) {
  if (value === 0) {
    return <span className="tnum font-mono text-[12px] text-faint">no change</span>;
  }
  const improved = invert ? value < 0 : value > 0;
  return (
    <span
      className={`tnum font-mono text-[12px] ${improved ? "text-good" : "text-danger"}`}
      title={improved ? "improved" : "worsened"}
    >
      {value > 0 ? "▲" : "▼"} {value > 0 ? "+" : ""}
      {value}
      {suffix}
    </span>
  );
}

// ---------------------------------------------------------------------------------
// Loading and empty states
// ---------------------------------------------------------------------------------

export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`pulse-soft rounded-md bg-raised ${className}`} />;
}

export function Spinner({ size = 14 }: { size?: number }) {
  return (
    <svg
      width={size} height={size} viewBox="0 0 24 24"
      className="spin shrink-0" aria-hidden
    >
      <circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" strokeWidth="3" opacity="0.2" />
      <path d="M21 12a9 9 0 0 0-9-9" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}

export function EmptyState({
  icon,
  title,
  description,
  action,
}: {
  icon?: ReactNode;
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center px-6 py-14 text-center">
      {icon && <div className="mb-3 text-faint">{icon}</div>}
      <p className="font-display text-[16px] font-medium text-text">{title}</p>
      <p className="mt-1.5 max-w-[42ch] text-[14px] leading-relaxed text-muted">{description}</p>
      {action && <div className="mt-5">{action}</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------------
// Layout helpers
// ---------------------------------------------------------------------------------

export function PanelHeading({
  children,
  aside,
}: {
  children: ReactNode;
  aside?: ReactNode;
}) {
  return (
    <div className="flex h-11 shrink-0 items-center justify-between gap-3 border-b border-line px-3.5">
      <h2 className="font-display text-[11px] tracking-[0.13em] text-faint uppercase">{children}</h2>
      {aside}
    </div>
  );
}

export function Tabs<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: { value: T; label: string; count?: number }[];
  onChange: (next: T) => void;
  label: string;
}) {
  return (
    <div role="tablist" aria-label={label} className="flex items-center gap-0.5">
      {options.map((option) => {
        const active = option.value === value;
        return (
          <button
            key={option.value}
            role="tab"
            aria-selected={active}
            onClick={() => onChange(option.value)}
            className={`relative rounded-md px-2.5 py-1.5 text-[13px] font-medium transition-colors
              ${active ? "text-text" : "text-muted hover:text-text"}`}
          >
            {option.label}
            {option.count !== undefined && (
              <span className="tnum ml-1.5 font-mono text-[11px] text-faint">{option.count}</span>
            )}
            {active && (
              <span className="absolute inset-x-2 -bottom-px h-px bg-accent" aria-hidden />
            )}
          </button>
        );
      })}
    </div>
  );
}

/** A label/value row, used wherever facts are listed. */
export function Fact({
  label,
  value,
  mono = true,
}: {
  label: string;
  value: ReactNode;
  mono?: boolean;
}) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-1">
      <span className="text-[13px] text-muted">{label}</span>
      <span className={`tnum text-[13px] text-text ${mono ? "font-mono" : ""}`}>{value}</span>
    </div>
  );
}

/**
 * A disclosure that keeps its contents out of the tab order while closed.
 *
 * Built on `<details>` rather than state so it works without JavaScript having
 * hydrated, and so the browser's own find-in-page can open it.
 */
export function Disclosure({
  summary,
  children,
  defaultOpen = false,
}: {
  summary: ReactNode;
  children: ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <details open={open} onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary className="cursor-pointer list-none text-[13px] text-muted transition-colors hover:text-text">
        {summary}
      </summary>
      <div className="fade mt-2">{children}</div>
    </details>
  );
}
